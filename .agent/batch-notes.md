<<<<<<< HEAD
# Batch triage — agent gam-closed-598-777 (closed issues #598–#777)

## Result
Triaged all 180 closed issues. **#759 is the single genuine improperly-closed issue
in the range.** Fixed in PR #1640 (non-draft, gam-linalg 216/216 green). Every other
close is a real root-cause fix with tests, or a correct not-a-bug closure.

## #759 — FIXED (PR #1640)
`trace_product_sparse` was parallelized by commit b7879667b ("perf(linalg):
parallelize trace_product_sparse over columns (#759)", 2026-06-05 08:09:31). Seven
seconds later, WIP-sweep commit 004d24499 ("wip(robust): periodic sweep of in-flight
team work (uncompiled WIP)", 08:09:38) swept in stale working-tree state that
overwrote the parallel loop back to the original serial `for col` scan. It stayed
serial through HEAD (incl. the a80fe6943 gam-linalg crate extraction, which faithfully
relocated the already-serial version). The issue was closed citing the now-reverted
commit; nobody noticed.

Fix: restored the rayon per-column reduction (`TakahashiInverse::get` is `&self` with a
`Mutex`-guarded `exact_columns` cache → concurrent cache-miss column solves are sound),
plus two regression tests pinning parallel-vs-dense `tr(H⁻¹ S)` agreement (including a
40-column system whose off-pattern S entries force concurrent cache-miss column solves)
and a determinism check. So a future sweep/refactor cannot silently drop it again.

Deliberately did NOT parallelize the hot production path (`takahashi_block_trace` /
`takahashi_left_multiply_block` in reml/.../sparse_cholesky_backends.rs): those already
run inside the rayon-parallel ρ-pair outer-Hessian loop (outer_derivatives/dense.rs,
with explicit nested-rayon guards) — inner parallelism there would oversubscribe.

## Verified-correctly-closed (checked code, NOT reopened)
- #638 — adaptive early-exit was DELIBERATELY removed (reweight.rs NOTE: non-determinism
  under CPU contention, gam#979; accepted iterates 10× outside tol). Correctly closed.
- #667 — `fit_multinomial` Python entry works; scalar `fit(family='multinomial')` gives a
  clear redirect error. Satisfies the issue's option (2).
- #650 — README no longer carries the n=4 quickstart; capacity-check error in place.
- #756 — cloglog `[-50,50]` clamp removed, `-expm1(-exp(η))` form, detailed reasoning + tests.
- #771 — Tweedie φ estimated (`EstimatedTweediePhi`) + regression tests.
- #774 — Firth/Jeffreys is the unconditional default in the marginal-slope path.

## WIP-sweep clobber audit (the ~144 "periodic sweep of in-flight team work" commits)
Dedicated audit: every other landed fix in #598–777 is intact on HEAD, including silent
perf ones (#640 apply_into overrides, #762/#766 Lanczos unification, #683/#739 BMS
caching). Maintainers had already detected + re-landed the other sweep regressions
(#770, #773, #757, #723). A clean, well-maintained range.
=======
# Batch triage notes — agent gam-closed-58-237 (issues 58–237)

## Assignment
CLOSED issues 58–237 on SauersML/gam. 81 are real issues (rest are PRs).

## Triage outcome (parallel agents, evidence-based vs current code)
Vast majority were genuinely FIXED. Three improperly closed (closing comments
describe fixes / commit SHAs that DO NOT exist in the tree):

- #178 — sae_manifold_fit: random_state ignored. The deterministic Python
  fast path `_fit_dense_periodic_ibp_lsq` (and `_fit_disjoint_periodic_top1`)
  accept `random_state` but never use it. The in-repo regression test
  `tests/test_sae_manifold_determinism.py::test_sae_fit_random_state_changes_output`
  exercises exactly this path (K=2, periodic, ibp_map, no top_k, p>=2K) and
  therefore FAILS at HEAD — an XFAIL-in-disguise, banned by SPEC.

- #159 — `assignment_prior=` alias never added; closing comment cites a
  `_ASSIGNMENT_ALIASES` map that does not exist.
- #160 — `n_atoms=` alias + conflict detection never added.

## Plan
1. #178: wire random_state into the fast-path seeds (deterministic LCG jitter
   on the assignment logits / PCA-phase init) so distinct seeds differ and
   equal seeds stay bit-identical. (this PR)
2. #159/#160: add `assignment_prior=` and `n_atoms=` kwargs with canonical
   normalization + eager conflict ValueError (separate PR).
>>>>>>> origin/agent/reopen-178/sae-random-state-ignored
