# Issue #1689 — PERF: ordinary P-spline s(x) and thin-plate s(x,z) Gaussian fits 2-10x slower than mgcv

## Problem
- 1-D P-spline `s(x)` n=400: gamfit 2.3-3.7x slower than mgcv at equal accuracy
- 2-D thin-plate `s(x,z)` n=1200: gamfit ~8.8-24.5x slower than mgcv
- Same predictive RMSE — the extra time buys nothing.
- Also: very heavy stderr logging by default (thousands of [OUTER ...]/[GAM ALO] lines).

## Plan
1. Reproduce the benchmark numbers locally.
2. Profile where the time goes (outer REML loop iterations, inner fits, basis construction, prediction).
3. Fix root-cause hot spots without weakening accuracy or violating SPEC.md.
4. Address default logging verbosity (verbosity flag / quieter default).
5. Add regression/perf tests.

## Status
- [ ] Reproduce
- [ ] Profile
- [ ] Fix
