# Issue #979 — marginal-slope (binary + survival) slowdown / hang

## Status snapshot (2026-06-29)
- 225+ comment omnibus. Original symptoms (binary slowdown, survival uncatchable
  hang) reportedly RESOLVED + regression-tested on main.
- Owner's last word: Large-scale CI "still failing on all three" — the three
  centers=24 marginal-slope methods (binary rigid, binary linkwiggle, survival).
- Recent large_scale runs: 50-min job-cap TIMEOUTS on the n=60000/centers=24
  fits (true #979 residual), PLUS a build break (ban-scanner violations in a
  #932 SAE file) blocking the latest scheduled runs from reaching benchmark.

## Plan
1. Establish local build/test ground truth on main HEAD.
2. Profile the large-scale margslope path (examples/large_scale_margslope_*).
3. Attack the genuine residual: inner joint-Newton grind on the ill-posed
   marginal<->logslope coupling at large centers; survival reparam parity.
4. Make the centers=24 fits complete fast + converge honestly (no fake
   converged=true, no wall-clock-as-correctness).

This is an autonomous WIP, force-improved commit-by-commit.
