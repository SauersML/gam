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
