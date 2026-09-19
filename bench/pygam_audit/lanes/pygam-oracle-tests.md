# pygam-oracle-tests

TITLE: Port pyGAM invariant tests as synthetic oracle tests (additivity, shape pdep, scale invariance, nesting)
WORK ITEM: Read audit/pygam_tests.md sections 4 and 5 (coverage holes and port list). This item adds new test files only.
Missing tests:
- eta == intercept + sum of partial dependences;
- pdep of a shape-constrained term obeys its shape;
- concave posterior draws (test_posterior_monotone_shape_constraint.py covers only inc, dec and convex);
- p-values/edf invariant to y*1e6 (Gaussian);
- intervals nested across levels (0.5 inside 0.9 inside 0.95);
- loglik ordering (saturated >= fit >= null).
Port pyGAM's test_pg_*.py to tests/pygam_oracle_*_test.py using synthetic data, with expectations derived from the invariant itself (not recorded from pyGAM outputs); pyGAM must not be imported at test time. Any test that exposes a real bug is fixed, never marked XFAIL (SPEC). Hand the bug to the owning branch and keep the test.
Coordinate: shape-constraints, shape-te-by, partial-effects, family-regressions, pvalue (for invariance only; no calibration assertions).
Acceptance: the new files run in CI. Each invariant test fails when an injected mutation is applied (e.g. dropping the intercept from the pdep sum). A grep asserts no `import pygam` in tests/pygam_oracle_*.
