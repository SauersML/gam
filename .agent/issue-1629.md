# Issue #1629 — matern() ~6x worse than thinplate()/tensor(); k= has no effect

## Plan
- Reproduce the matern under-recovery on the fine 2-D surface.
- Investigate matern length-scale optimization & basis construction.
- Compare against thinplate/tensor code paths to isolate the defect.
- Fix the root cause (likely length-scale converging to over-smoothed solution,
  and `k=` ignored for matern).
- Add regression tests; verify via build.sh + nextest.

This is an autonomous WIP; commits will ratchet quality upward.
