The full public constrained-REML acceptance sweep now passes on MSI's freshly built Python extension. The fixes are published in `877b32ca5`, `18712e957`, and `a03cf2d0b`, included in main merge `657d12cf5`.

For the original seed-9, 50-row weighted cubic-spline fixture with its rank-deficient second-derivative penalty, all **16 coefficient halfspaces** pass: each of eight coefficients constrained in both orientations, with the wall moved `0.3` beyond the unconstrained coefficient. Each fit returns finite coefficients, reports active row `[0]`, satisfies `A beta >= b`, and binds the wall within `1e-8`. Repeating each fit with `S + 1e-10 I` changes fitted values by less than `1e-3`.

Both the affine-face profile and the terminal KKT recheck now retain data curvature in penalty-null directions through the same spectral geometry. The earlier fresh native run passed all six focused checks, including the analytic rotated-penalty KKT oracle at log strengths 0, 30, and 45 and the nonzero-affine-face backward finite-difference check.

The combined #2830/#2831 public run reports **146 passed, 73 deselected in 5.97 seconds**. Command: `python -m pytest -q tests/test_reml_rank_deficient_penalty_geometry.py -k 'block_reml or binding_constraint' --tb=short --show-capture=no -ra`, CPU affinity 112–115, four Rayon threads, isolated `.venv-python312-issues`.

Extension SHA-256: `fad6cf9fabb9c3859f27a8455dbd4b6189da1421e276d2aaaffae864f5b3f299`. Provenance: `bench/measurements/issue_triage_20260908/reml-public-146-provenance.json`. The public regression file is published in `fbc2eaa29`.

Closing: both halfspace orientations across every coefficient, active-face correctness, and negligible-ridge continuity have now been verified through the public API.
