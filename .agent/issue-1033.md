# Issue #1033 — n-independent outer loop (sufficient-statistic architecture)

Autonomous WIP. Force-improved commit-by-commit.

## Issue summary
Architectural invariant: n-dependent work happens ONCE per fit (sufficient
statistics / streamed Grams). The rho/kappa/psi outer loop must manipulate only
k×k objects (O(k³) per trial). Acceptance: KAPPA-PHASE-SUMMARY eval cost at
n=320k independent of n past one initial pass.

## Current state (from issue history + repo scan)
- Mechanism (a) θ-invariant Gram cache (ρ-only Gaussian): DONE.
- Mechanism (b) PsiGramTensor value + gradient n-pass: DONE, certified on
  standardized geometry (node ladder 9..129), #1216/#1264 closed.
- Mechanism (c) frozen-W GLM lane (glm_sufficient_lane.rs): first step landed.
- Acceptance gate: kappa_loop_n_scaling.rs perf tests exist.

## Plan
1. Build + run the kappa/rho perf-scale + psi_gram tensor tests to find actual
   live state (green/red, whether the n-free skip fires).
2. Identify the true remaining gap and harden it (root cause, not symptom).
3. Add/strengthen tests + docs. Verify green with build.sh.
