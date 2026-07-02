# SAC (Sequential Atom Composition) — discriminating experiments

WS-A critical path. Tests the SAC thesis (SAC_PLAN Part 1): *K=1 curved fits
succeed on real data; the simultaneous cold-start joint fit of K atoms is the
wrong algorithm — build K from K=1.* The prototype (`examples/sac_prototype.py`)
replaces the single joint `sae_manifold_fit(K)` call in `compose_tiers` with
forward K=1 births + backfitting sweeps, over the existing gamfit 0.1.247 FFI
(no Rust build required).

## Method (prototype)

- **Forward births** (`sac_fit`, Phase 1): fit a K=1 manifold atom on the
  running residual `R` (passing `R` as `X` *is* residual-seeding — the fit's PCA
  seed is computed on whatever it's handed), under the whitened / isometry gauge
  (`isometry_weight=1.0`, `structured_residual_passes=2` so Σ applies from atom
  one). Accept iff the atom's marginal EV clears an explicit minimum-effect floor
  `ev_floor` (salience is a separate dial from evidence — at frontier scale
  evidence alone keeps trivial wiggles forever). Subtract, repeat; stop after two
  consecutive rejections. The joint-fit guard stack (collapse detector,
  reseed-all, separation barrier, arrival floors) is bypassed by construction —
  K=1 never trips it.
- **Backfitting** (Phase 2): per atom, form the leave-one-out residual and refit
  the K=1 atom warm-started from its current chart/gate (`t_init`/`a_init`, #357).
  At fixed smoothing this is exact block-coordinate descent on the joint
  penalized objective, hence monotone in combined EV; refits are accepted only if
  they do not worsen their block.
- **Terminal joint assembly** (Phase 3): a single evaluate-don't-optimize
  arrow-Schur pass for joint Laplace evidence / cross-atom covariance. Needs a
  `merge_tiers`/`frozen_evaluate` FFI verb that does not exist yet — documented
  TODO in the prototype; orthogonal to the reconstruction/structure claims here.

Honesty: this is temporary Python scaffolding (SPEC.md — math lives in Rust). It
contains no model math beyond residual subtraction and an EV ledger. No
wall-clock budgets anywhere.

## Experiments

- **Exp 1 — planted two-circles** (`sac_experiments.py`): two circles in
  orthogonal subspaces (n=1500, p=32). Joint `sae_manifold_fit(K=2)` vs SAC. SAC
  must match the planted truth the joint fit passes: recover both circles'
  per-circle EV and their cyclic ordering (recovered phase a rigid function of
  the planted angle).
- **Exp 2 — W6 OLMo K=8 kill-test** (`sac_w6_runner.py`, node2): the EXACT
  (500,128) top-128-PCA matrix of real OLMo-3-7B (layers 15–17) that the joint
  `sae_manifold_fit(K=8, d_atom=1, circle, ibp_map, isometry_weight=1.0)` fit
  **timed out** on (W6 `results_K8.json`: 3×1500 s TIMEOUT). SAC runs 8 forced
  sequential K=1 fits. **Kill-criterion (Part 5):** if these do not produce 8
  healthy atoms with climbing EV in well under 25 min, the SAC diagnosis is
  WRONG.
- **Exp 5 — W5 compose E2E** (if 1 & 2 pass): SAC as the T2 stage of the
  N=3000 compose repro that was co-collapsing at K=6; composed held-out EV vs the
  T1-only baseline.

## Reproduction

Laptop is CPU-starved by the concurrent fleet (load ~6); the K=1 outer loop
converges to EV≈0.99 in seconds but oscillates near the incumbent without
terminating fast, so fits run on node2 (128 cores) via Heimdall.

```bash
# node2 (venv_fable, gamfit 0.1.247); files under /dev/shm/sauers_gpu/
# Exp 1:
RAYON_NUM_THREADS=32 /models/sauers_build/venv_fable/bin/python -u \
  sac_experiments.py --n 1500 --p 32 --n-iter 20
# Exp 2 (kill-test):
RAYON_NUM_THREADS=32 /models/sauers_build/venv_fable/bin/python -u \
  sac_w6_runner.py            # reads /dev/shm/w6/cache_K8.npy
```

Launched via Heimdall (`gpus:0`, wrapped to exit 0): exp1 `eec261cfc830`,
exp2 `9dc88332b127`.

## Diagnosis update (2026-07-02) — these runs are the comparison arms

Per `STAGE1_DIAGNOSIS.md`, the K≥2 failure is now attributed to the **guard
stack** (the collapse bar compares a k-sparse dictionary against a *dense*
rank-q PCA ceiling, so cold seeds start below their own acceptance bar on real
data; the bar is checked at iteration 0 and its WALL cost flattens the outer ρ
objective → timeouts/oscillation), not the joint-fit architecture. SAC/growth
stays valuable as a production *mode* (cold PC-pair seeds are basin roulette
regardless of guards; growth makes EV monotone in K), but cold-start-at-K is now
a **comparison arm**. These experiments therefore run as joint-vs-sequential arms
**before and after** the guard fix.

## Results — PRE-fix baseline (current build, gamfit 0.1.247 @ node2)

The current build's **outer ρ loop does not converge in bounded time** on these
fits — uniformly, at every K, which is exactly the guard-WALL-flattened-objective
signature of the revised diagnosis:

| Fit | Data | Result on current build |
|---|---|---|
| joint K=8 | real OLMo 500×128 (`cache_K8.npy`) | **3×1500 s TIMEOUT** (W6 `results_K8.json`) |
| joint K=6 (compose T2) | planted N=20k p=64 | co-collapse (reseed-all ×3) → **#1026 oscillation >1 h @ EV≈0.51** (`e2e_smoke.log`) |
| single **K=1** (SAC birth) | real OLMo 500×128 | **>7 min, no termination** (this run) — sequential fitting does **not** rescue real data pre-fix |
| joint K=2 | planted two-circles n=1500 p=32 | holds a healthy incumbent EV≈0.969 but outer loop **oscillates >5.5 min without returning** |

**Key pre-fix finding:** a single K=1 fit on the real W6 activations is *also*
crippled (the guard WALL flattens the outer objective at K=1 too), so SAC does
not work around the guard on the unpatched build. This corroborates the revised
(guard-stack) diagnosis over the original (joint-architecture) one.

Separately found and worked around a real gamfit **bug** (independent of the
guard issue): the **parallel K=1 fit deadlocks** (0 % CPU, hangs) under
OMP/OPENBLAS/RAYON oversubscription on node2 — `OPENBLAS_NUM_THREADS=1` fixes it
(rayon cannot parallelize a single atom, so BLAS×rayon nesting deadlocks).

## Results — POST-fix (rerun target)

_Rerun exp1/exp2/exp5 on S1-guards' patched `.so` the moment it builds into
`/models/sauers_build/target_fable`. Acceptance: joint K=8 completes in minutes
with a real EV; SAC forward births produce healthy climbing-EV atoms; the two
arms are compared on held-out EV, per-circle recovery, and cyclic ordering._
