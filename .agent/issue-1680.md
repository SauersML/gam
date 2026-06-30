# Issue #1680 — near-collinear additive smooths: gamfit ~4x worse truth-RMSE than mgcv

## Problem
Additive model with near-collinear nuisance covariates (pairwise corr ~0.985).
Signal lives in uncorrelated x1, x4; x2, x3 are near-rank-1 collinear nuisance.
gamfit recovers true function ~4x worse than mgcv at n=120, ~1.5x worse at n=400.
Points at instability in penalized fit / lambda selection under near-degenerate
additive design (x1/x2/x3 block ~rank-1).

## Plan
1. Reproduce with the repro script.
2. Trace where lambda selection / penalized fit destabilizes under collinearity.
3. Root-cause fix (not symptom): likely REML/identifiability/centering interaction.
4. Add regression tests; verify against mgcv-like baseline.

## Status
- [ ] Reproduce
- [ ] Root cause
- [ ] Fix
- [ ] Tests
