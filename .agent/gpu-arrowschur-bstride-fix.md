# [agent][gpu] Fused arrow-Schur B/Y block-stride bug (P_MAX·P_MAX vs P_MAX·R_TEMPLATE)

GPU device: 6 (Tesla V100-SXM2-32GB, CC 7.0, `compute_70`). CUDA_VISIBLE_DEVICES=6.

## The bug (real GPU correctness defect, found 2026-06-30)

The fused arrow-Schur Newton kernel
(`crates/gam-solve/src/gpu_kernels/arrow_schur.rs`,
`crates/gam-solve/src/gpu_kernels/arrow_schur_nvrtc.rs`) packs three per-row
strided blocks:

| block | semantic | allocated block stride | documented |
|-------|----------|-------------------------|------------|
| `d_stack` | `D_i = H_tt` | `P_MAX · P_MAX` | ✅ |
| `b_stack` | `B_i = H_tβ`  | `P_MAX · R_TEMPLATE` | ✅ |
| `y_out`   | `Y_i = L⁻¹B_i`| `P_MAX · R_TEMPLATE` | ✅ |

But the host PACKER (`pack_fused_host`, line ~2985) and the DEVICE kernel
(`arrow_schur_forward_pgroup`, lines ~219 load and ~276 store) index `b_stack` /
`y_out` with the **D-block stride** `((i*P_MAX + c) * P_MAX + r)` — copy-pasted
from the D block — instead of the B/Y stride `((i*P_MAX + c) * P_MAX + r)` …
wait, the per-element layout inside a block is `col*P_MAX + row` which is the
SAME for B and Y (P_MAX rows). The defect is the **per-ROW (per-i) block
stride**: B's block i must start at `i * P_MAX * R_TEMPLATE`, but the code uses
`i * P_MAX * P_MAX` (via `(i*P_MAX + col)*P_MAX`). When `p_max > r_template` the
column index `col ∈ 0..r_runtime` combined with the wrong i-multiplier walks
past the end of the smaller B/Y buffer.

### Trigger
`p_max != r_template`. `p_max = d.next_power_of_two().min(32)`,
`r_template = ceil_to_template_r(k)`. Equal only by coincidence. The V100
validation test `arrow_schur_gpu_multi_size_groups_match_reference`
(`d ∈ {10,16,30}`, `k=5` → `r_template=5`, `p_max ∈ {16,16,32}`) panics:
`index out of bounds: the len is 1280 but the index is 2048` at
arrow_schur.rs:2987 (host packer). On-device the kernel does the same OOB
read/write (silent UB / wrong result / illegal access).

## Plan
1. [ ] Fix host packer B-block i-stride: base must use `P_MAX·R_TEMPLATE` per i.
2. [ ] Fix device kernel `b_stack` load index (line ~219).
3. [ ] Fix device kernel `y_out` store index (line ~276) — must match the
       readback which already uses `i*P_MAX*R_TEMPLATE + c*P_MAX + r`.
4. [ ] Verify host readback (line ~3681 etc.) already correct → it's the oracle.
5. [ ] Run V100 validation suite → all green, GPU engaged (nvidia-smi mem/util).
6. [ ] CPU↔GPU parity at d≠r sizes; document tolerance.

## Verification log
- 2026-06-30: BUG 1 (B/Y stride) FIXED + verified on V100 (device 6):
  - `arrow_schur_gpu_multi_size_groups_match_reference`: FAIL→**PASS** (parity 1e-10).
  - `arrow_schur_gpu_fused_layer_d_matches_layer_a_b_c`: **PASS**.
  - 6/7 of the V100 suite pass. GPU engaged (dmon: SM 3-5%, fb mem 1→312 MB
    during runs). NVRTC compiled for compute_70.
- BUG 2 (pre-existing, separate root cause): `ridge_bump_required_on_non_pd_row_recovers_after_bump`
  FAILS on main too (confirmed). Root cause: the `RidgeBumpRequired{bump}`
  estimate is `scale·|pivot|·√ε·1024` where `pivot` is the cuSOLVER POTRF info
  = a 1-based ROW INDEX, not the pivot magnitude. So for a strongly non-PD
  block (test: htt=-I, needs ridge>1) the reported bump≈1.5e-5 and the outer
  10-step geometric escalation caps at ~0.0156 < 1 and never recovers. The CPU
  oracle (arrow_schur/factorization.rs) instead lifts the per-row ridge "just
  enough" internally. FIX: make the GPU report a deficit-aware bump from the
  host-resident H_tt block (Gershgorin lower bound on min eigenvalue ⇒ shift
  needed to make it PD), so one retry recovers.

## Bug 2 producer sites (all in arrow_schur.rs)
- 1356 (cuda::solve dense batched POTRF)   — pivot=info index, has htt block
- 1220/1224 (multi-GPU tile POTRF)         — pivot=info index, has htt block
- 3396 (fused path status)                 — pivot=status index, has htt block
- 482  (row-procedural host Cholesky)      — has the block, no magnitude
- 2610 (SAE device PCG host Cholesky)      — has the block, no magnitude
Shared helper `ridge_bump_to_make_pd(htt, ridge_t)` → all sites.
