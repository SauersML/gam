# Issue #1690 — Poisson (~28x) & Gamma (~7x) 1-D s(x) fits slower than mgcv

## Plan
- Poisson/Gamma do NOT pay Firth (unlike binomial #1575). Bottleneck is the
  outer-loop work count: seed-grid prepass (~20 full-n solves) + multistart +
  ARC Newton, plus possibly dispersion / family-specific overhead.
- Why is Poisson (28x) ~4x worse than Gamma (7x)? Investigate.
- Profile the actual Poisson/Gamma path (n=600, single smooth, k=12).
- Find a correctness-preserving lever specific to these families.

## Status: investigating
