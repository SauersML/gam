# Issue #1621 — debiased_functional point/contrast column-order bug

## Status on arrival
- Issue #1621 ALREADY FIXED + CLOSED on main (`fe1425a`, `debiased_query_design_full_schema`).
- Sibling #1622 (missing weighted Gram) also fixed + closed.
- Both regression tests present & passing.
- BUT: HEAD build was RED fleet-wide — 9 #932/SAE ban-scanner violations break the
  non-bypassable build.rs gate (maturin re-runs it). Owned by PRs #1642/#1639; YIELDED.

## What this PR adds (verified via local-only build-gate patch + maturin develop --release)

### 1. Fix: categorical bookkeeping column abort in point/contrast (root: the #1621 fix itself)
`debiased_query_design_full_schema` laid the x0/x1 row over EVERY training column and
strict-encoded the whole row against the saved schema, granting unseen-OK encoding only
to random-effect columns. A frame carrying an UNREFERENCED categorical bookkeeping column
(`DataFrame({y, x, group})` fit `y ~ s(x)`) filled `group` with the "0" placeholder →
`unseen level '0' in categorical column 'group'` abort — the #840 foot-gun `predict`
avoids via `project_frame_to_model_columns`.
- Fix: lenient-encode every column NOT in the model's required-prediction set (exactly the
  placeholder-filled columns). `crates/gam-pyffi/src/latent/reml_latent_fit_ffi.rs`.
- PROVEN: reverting the fix reproduces the abort; with the fix all pass.
- Tests: `tests/bug_hunt_debiased_point_contrast_bookkeeping_column_test.py` (5 cases:
  point, contrast, presence-invariance, multi-col far-from-placeholder, contrast over a
  categorical predictor with inert bookkeeping factor).

### 2. Fix: unsatisfiable #1120 average_derivative assertion (banned XFAIL-by-accident)
The sin arm used `sin(2*pi*x)` over a FULL period, where BOTH true mean value AND mean
derivative are ~0, so `abs(ad - truth_value) > 0.5` could never hold even though the code
is correct (it returns ad ~= truth_deriv, distinct from av). Switched to `sin(pi*x)`
half-period (mean value 2/pi ~= 0.637 vs mean deriv ~0). Pins #1120 honestly.

## Verification
- Green build via temporary working-tree patch of the 9 #932 ban violations (UNCOMMITTED,
  reverted after — not touching #1642/#1639 territory).
- `maturin develop --release`; full debiased suite **13 passed** (was 12 passed / 1 broken).
- 21 selected debiased/unseen/column-order/leave-one-group tests pass, regression-free.
