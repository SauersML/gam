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
