# Issue 979: constrained steps and cached coefficient accuracy

The full binary and survival acceptance checks are still outstanding. These
results establish two narrower defects and their corrections.

## Active-face curvature

Commit `62305b333` makes coordinate bounds use the same certified exact
active-face Newton calculation as general linear inequalities. Previously,
coordinate bounds bypassed that calculation and used an ambient convexified
Hessian, which can change curvature along feasible directions.

On acn112, with four Rayon threads, one BLAS thread, optimization level 1,
`repro979_survival_margslope 160 6` reached seven accepted outer iterations in
a 20-second diagnostic window after this change; the baseline reached zero.
Warm inner solves in the corrected run commonly converged in 2–4 cycles.
The full fit still exceeded a subsequent 60-second cap, so this is not a
full-fit performance pass. Both diagnostic processes terminated at their caps.

Logs in `/projects/standard/hsiehph/sauer354/y1-logs/`:

- `codex979-opt1-baseline-survival160.log`
- `codex979-opt1-face-survival160.log`
- `codex979-opt1-face-survival160-completion.log`

## Accuracy is part of a reusable mode certificate

The cache checked objective identity, convergence, Laplace artifacts, and fresh
curvature, but did not check the requested coefficient-solve accuracy. A mode
accepted at a coarse tolerance could bypass a later stricter solve. The result,
in-memory cache, and persisted cache now carry the producing solve tolerance.
Reuse requires that tolerance to be at least as strict as the new request.
An explicitly owned precomputed mode also needs correction when its accuracy
does not meet the criterion evaluation's requirement.

The independent nonlinear quartic regression solves coarsely, requests a tighter
solve at the identical smoothing parameter, and checks the analytic residual
falls below `1e-10`. It also verifies the tighter result remains reusable for a
looser request. On MSI it passes in 0.14 seconds. The spectral mixed-derivative
check, including repeated eigenvalues and moving gate/floor regimes, also passes
in 0.10 seconds after removal of the redundant eigendecomposition.

Current-source correctness log: `codex979-cache-accuracy-current.log` in the
same log directory. The warm build took 2m39s. Numerical execution and builds
were entirely on MSI; the cache regression used optimization level 0 for the
solver and custom-family crates and is not a release-performance measurement.
