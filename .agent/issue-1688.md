# Issue #1688 — matern()/GP smooths 12-55x slower than mgcv s(.,bs="gp")

## Problem
`matern()` / GP smooths fit 12-55x slower than equivalent mgcv GP smooth at
identical predictive accuracy. Pure speed regression. Also: a firehose of stderr
logging (~84 lines/obs) by default.

## Plan
1. Reproduce the perf gap with a Rust-side benchmark / criterion or CLI timing.
2. Profile the matern/GP fit path to find the hot spot (outer REML loop iterations,
   per-iteration linear algebra, basis construction, etc).
3. Identify root cause — likely:
   - excessive outer-loop iterations
   - dense O(n^3) operations that should exploit structure
   - default logging cost
4. Fix root cause; verify accuracy unchanged and speed materially improved.
5. Kill the default stderr firehose (gate behind verbosity).

## Status
- [ ] reproduce
- [ ] profile
- [ ] root cause
- [ ] fix
- [ ] verify
