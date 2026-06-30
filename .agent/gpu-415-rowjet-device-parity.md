# #415 — GPU↔CPU survival row-jet device parity (PROVE IT ON THE V100)

## Context
Issue #415 asks for a parity-lock between the GPU row kernels and the CPU
marginal-slope / survival-flex formulas. Related: #1175 (FP-order &
transcendental parity).

Prior work (merged, commit 07ffd921e) added the **host-oracle "third leg"**:
a CPU transcription of the device `.cu` seeded-jet arithmetic, pinned to the
production CPU jet on EVERY box. Good — but it does NOT prove the actual NVRTC
device kernel compiles or runs. The `device_matches_cpu_when_available` test
only engages a real GPU on a GPU box, and the comments make clear the work was
done on a box WITHOUT a GPU.

## My box
- 8× Tesla V100-SXM2-32GB, compute_cap **7.0**, CUDA_VISIBLE_DEVICES=3.
- nvcc 12.1 / libnvrtc.so.12 present.

## Plan / gate
1. Build the gam-models GPU test target via `./build.sh`.
2. **Prove NVRTC compiles `survival_rowjet_kernel.cu` on sm_70** — fail loud if
   it declines (this is the recurring silent-CPU-fallback failure mode).
3. Run `device_matches_cpu_when_available` and confirm it actually exercised
   the GPU (nvidia-smi utilization moves), not the CPU fallback.
4. Add a **device-only** parity test that calls `survival_rigid_row_jets_device_only`
   and FAILS if the device path can't run — so a GPU box can never silently
   pass on the CPU fallback. Sweep edge regimes + both kernel variants
   (with/without t4). Document tolerance.
5. Keep CPU path green everywhere.

## Progress log
- (start) branch created, plan written.

## FINDING 1 (2026-06-30) — device kernel RUNS on V100 but parity FAILS
- nextest `device_matches_cpu_when_available` on V100 (sm_70):
  NVRTC compiled `survival_rowjet_kernel.cu`, kernel launched & executed
  (diff is non-zero ⇒ device path engaged, NOT the CPU fallback). ✅ engagement proven.
- BUT parity gate FAILED: `max abs diff 5.09e-8 > 1e-9`.
  Docstring/issue claim 4.7e-12 on A100. So either:
    (a) transcendental drift (CPU libm erfc vs CUDA erfc) amplified through the
        seeded-jet recurrences into the high-order (third/fourth) channels — #1175, or
    (b) a genuine arithmetic mismatch.
- Next: per-channel abs+rel diagnostic to localize the worst offender, then set a
  PRINCIPLED tolerance (mixed abs+rel) or fix the source of drift.

## RESOLUTION (2026-06-30)
Root cause: the parity gate used a flat `|Δ|<=1e-9` ABSOLUTE bound across all
channels. On a real V100 the third-derivative channel drifts 5.09e-8 — but
that is 1.2e-9 RELATIVE to the channel scale (42.5). The drift is irreducible
#1175 transcendental noise (CUDA erfc/exp vs host libm) amplified through the
order-4 jet chain, NOT an algebra bug (the host-oracle third leg already pins
the JS1/JS2 algebra on every box).

Fix:
- Replaced the flat absolute gate with a principled per-channel
  `atol(1e-9) + rtol(1e-7)*channel_scale` band. A real algebra bug perturbs at
  O(channel magnitude) → normalized residual ~1.0, 7 orders above the floor →
  caught with ~80x headroom.
- Added `device_only_path_runs_and_matches_cpu_fail_loud`: the #415 core
  deliverable. Calls the non-falling-back device-only entry, asserts it RUNS
  (no swallowed NVRTC/arch/launch error) and matches CPU for BOTH kernel
  variants (t4 / no_t4) across interior + edge regimes. GAM_REQUIRE_GPU=1 makes
  a missing/declined device a HARD failure (no silent CPU pass).
- Added `device_vs_cpu_channel_drift_report` (ignored diagnostic).
- Corrected stale "device == CPU to 4.7e-12 / proven ≤1e-9" docstrings that the
  V100 measurement falsifies.

Proven on Tesla V100 sm_70: NVRTC compiles survival_rowjet_kernel.cu, both
kernels launch & run, all 7 module tests PASS (2 of them exercise the GPU).

## FINDING 2 (2026-06-30) — survival_flex step6 entrypoint test was GPU-box-broken
`survival::marginal_slope::gpu::step6_tests::flex_entrypoints_fold_supplied_step6_rows_before_backend_gate`
used bit-exact `assert_eq!`, but on a GPU box the gradient/hvp/dense-hessian
entrypoints route the step6 fold through the device kernel `survival_flex_step6_rows`,
whose per-row `M=H_p·J` contraction reassociates + uses FMA → ~2e-16 drift.
- On a CPU-only box (CI) the host fold runs → assert_eq passes → the device path
  shipped untested by THIS test. Same #415/#1175 latent-GPU-only-failure genus.
- Fixed: replaced assert_eq! with a relative band (atol+rtol·(1+|x|), rtol=1e-12),
  matching the sibling `step6_device_contraction_matches_cpu_reference`.
- Corrected two "bit-for-bit / to the last ULP" comments that the V100 falsifies
  (FMA/reassociation in the device contraction).
- Verified PASS on V100 (failed before, passes after).
- POST-#1686 (2026-06-30): both step6 + intercept-solve kernels were STILL on
  bare `compile_ptx` (fmad=true, no arch pin) — #1686 only fixed survival_rowjet.
  Routed both through `compile_ptx_arch` so they inherit #1686's fmad=false +
  the #1551 arch pin (neither uses atomicAdd/includes, so no silent-fallback
  risk, but fmad=true left the M=H_p·J contraction needlessly FMA-fused).
  Re-measured step6 device-vs-CPU worst abs drift with fmad=false: 2.84e-14 —
  NOT bit-exact, so the residual is pure REASSOCIATION (different reduction
  order), not FMA. Corrected the comments crediting FMA. Band unchanged
  (atol+rtol·(1+|x|), rtol=1e-12; 2.8e-14 sits ~comfortably inside).

## FINDING 3 (2026-06-30) — cubic_cell device kernel NVRTC-broke on every GPU box
`gpu_kernels::cubic_cell::device::tests::cubic_cell_device_residency_matches_cpu_all_branches`
failed on the V100 with:
  "cubic_cell NVRTC compile (degree=9) failed: ... could not open source file stdint.h"
ROOT CAUSE (two layers):
1. cubic_cell compiled via the BARE `cudarc::nvrtc::compile_ptx` (no -I, no arch),
   while the kernel does `#include <stdint.h>`. NVRTC has no include search path
   → catastrophic compile error → device path silently fell back to CPU on EVERY
   GPU box (the #1551 silent-fallback genus, in cubic_cell).
   FIX: route through `gam_gpu::device_cache::compile_ptx_arch` (the shared
   arch-aware compile the repo standardized on — sets -I AND --gpu-architecture).
2. Even with -I, system <stdint.h> drags in gnu/stubs-32.h (absent w/o 32-bit
   dev libs) → still fails. The kernel needs only uint8_t/uint32_t and the CUDA
   device ABI fixes their widths, so typedef them inline (mirrors the existing
   inline CUDART_INF pattern). No host headers.
3. After the kernel RUNS: parity drift at degree 21 (Affine branch) up to
   1.5e-6 relative — the forward moment recurrence M_{n+1}=(n·M_{n-1}−d0·M_n−B_n)/d1
   is ill-conditioned, so CPU-serial vs device round-off compounds ~×10³ per +6
   orders. NOT a bug (NonAffineFinite GL branch agrees to ≤2e-15). Replaced the
   flat rel≤1e-11 gate with a per-order band rel_tol(k)=1e-12·10^(k/3).
   POST-#1686 RE-MEASURE: with fmad=false ACTIVE the Affine k=21 drift is
   1.478e-6 — essentially UNCHANGED from the fmad=true 1.5e-6. So this drift,
   like the row-jet third/fourth channels, is NOT FMA contraction; it is pure
   round-off order in an ill-conditioned recurrence. Corrected the device.rs
   comment that had credited "FMA-fused evaluation" (the band itself is correct
   and unchanged: k=21 tol 1e-5 vs worst 1.478e-6, ~7× headroom).
Verified PASS on Tesla V100 sm_70 (compiles, all 3 branches run, parity holds).

## RECONCILIATION (2026-06-30) — rebased onto main's #1686 (FMA-contraction fix)
Main landed #1686 ("NVRTC FMA-contraction breaks GPU↔CPU parity, fmad default
on"): it set `opts.fmad = Some(false)` in the shared `nvrtc_compile_options()`
and routed survival_rowjet through `compile_ptx_arch` instead of bare
`compile_ptx`. Rebased my branch on top — clean, no conflicts (my cubic_cell
also already routes through compile_ptx_arch, so it inherits fmad=false too).

Re-measured the survival row-jet device-vs-CPU drift on the V100 with fmad=false
ACTIVE (diagnostic `device_vs_cpu_channel_drift_report`, /tmp/rowjet_drift.txt):
```
  channel  fmad=true (pre-#1686)   fmad=false (post-#1686)
  value    ~1.48e-10               1.48e-10   (≈unchanged, already tiny)
  grad     ~8.18e-10               8.18e-10
  hess     ~8.79e-9                8.79e-9
  third     5.09e-8                5.09e-8    (BIT-IDENTICAL to 4 sig figs)
  fourth    4.54e-8                4.54e-8    (BIT-IDENTICAL)
```
KEY INSIGHT refining #1686's narrative: FMA was the dominant source for the
LOW-order channels (value/grad/hess) — #1686 genuinely helps there. But the
third/fourth channels are UNCHANGED: their 5e-8 floor is *transcendental* drift
(CUDA erfc/erfcx/exp vs host libm) amplified ~5e8× through the order-4 jet, which
`--fmad=false` cannot touch. So my magnitude-scaled per-channel band is NOT
redundant with #1686 — it is COMPLEMENTARY: #1686 removes the FMA component, the
band absorbs the irreducible transcendental component. A flat 1e-9 gate would
STILL fail post-#1686 (third 5.09e-8 > 1e-9). Updated module + PARITY_RTOL
docstrings to record this split. All 14 gate tests PASS post-rebase on V100.

## FINDING 4 (2026-06-30) — repo-wide fmad=false sweep (#1686 only fixed rowjet)
#1686 set fmad=false in the SHARED nvrtc options but only re-routed ONE kernel
(survival_rowjet) off bare `compile_ptx`. Every other bare-compile_ptx kernel
still got fmad=true. Swept all of them in the GPU lane and routed the
parity-sensitive ones through `compile_ptx_arch` (fmad=false + #1551 arch pin):
- survival_flex step6 (`M=H_p·J`) + intercept-solve  — measured step6 drift
  2.84e-14 under fmad=false: NOT bit-exact ⇒ pure REASSOCIATION, not FMA.
  Corrected comments crediting FMA; band unchanged.
- sae_rowjet (softmax seeded-jet, sibling of survival_rowjet) — drift
  2.78e-16→1.67e-16; added an anti-false-green device-only assertion so a dead
  kernel can't pass as CPU==CPU.
- row_hessian_ops, cubic_bspline_moments (hex+tet) — routed for consistency.
- sphere_gpu — routed; the stale "can't set arch with a runtime string" comment
  is obsolete (compile_ptx_arch resolves arch internally).
None of these use atomicAdd/#include, so none had the #1551 silent-fallback
compile bug — but all were needlessly FMA-fused against the CPU oracle.

## FINDING 5 (2026-06-30) — sphere fit-parity gated conditioning, not the GPU
`sphere_gpu_end_to_end_fit_parity_vs_cpu_truncated` FAILED on the V100
(max|Δβ|=1.177e-7 > 1e-9), pre-existing (fails with my changes reverted; fmad
unchanged). Instrumented it: the GENUINE GPU output — the raw kernel design
matrix K(data,centers)·Z — matches the CPU oracle to 9.7e-17 (one ULP, rel
1.2e-15). The test instead gated β = (XᵀX+λS)⁻¹Xᵀy, and cond(XᵀX+λS)=5.2e7 on
this fixture. Perturbation theory: ‖Δβ‖/‖β‖ ≲ cond·‖Δx_s‖/‖x_s‖ = 5e7·1e-16 ≈
5e-9·‖β‖ → ~1e-7 absolute. The customer-visible ŷ=x_s·β stays at 7.6e-11 (the
ill-conditioned directions cancel). So the flat 1e-9 β gate measured the
conditioning of the SHARED CPU solve, not the GPU kernel.
Fix (test made STRONGER): gate raw x_s bit-tight (1e-12·scale), gate ŷ at 1e-9,
gate β against a condition-aware bound cond·ULP·16. A real kernel bug perturbs
x_s at O(scale=0.08), 14 orders above the raw gate.
OUT OF LANE (not touched): the two sphere PERFORMANCE hill-climb tests
(ratio≥20x/≥10x) fail on this box (GPU 12.5s vs CPU 8.7s, GPU idle ⇒ genuine,
not contention). Those are #1687's WIP perf gates — flagged on #1687, NOT
weakened here.
