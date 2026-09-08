Verified against a freshly built Linux Python extension on MSI and published in `18712e957` and `a03cf2d0b`, included in main merge `657d12cf5`.

All 130 public block-REML acceptance cases pass:

- Seed-5 weighted two-spline fixture: all 60 response rows perturbed by both `-1e-6` and `+1e-6` (120 fits). Every fit succeeds and fitted values stay within `1e-4` of the unperturbed fit.
- All 10 second-block starting log strengths `[5, 10, 15, 20, 25, 28, 30, 32, 35, 40]`, with the first at `-7.37`. Every fitted curve agrees with the default start within `1e-4`.

The production block profile uses the canonical penalty range/null basis, equilibrates independent modes before factorization, and derives its search domain from the data-relative penalty spectrum. The independent limiting-null-model and finite-difference regression also passed in the earlier focused native run.

The combined #2830/#2831 public run reports **146 passed, 73 deselected in 5.97 seconds**. Command: `python -m pytest -q tests/test_reml_rank_deficient_penalty_geometry.py -k 'block_reml or binding_constraint' --tb=short --show-capture=no -ra`, CPU affinity 112–115, four Rayon threads, isolated `.venv-python312-issues`.

Extension SHA-256: `fad6cf9fabb9c3859f27a8455dbd4b6189da1421e276d2aaaffae864f5b3f299`. Provenance: `bench/measurements/issue_triage_20260908/reml-public-146-provenance.json`. The public regression file is already published in `fbc2eaa29`.

Closing: both response-perturbation and starting-strength sweeps requested here now pass.
