# Issue #1676 — scale_dimensions silent no-op for thinplate()

## Status: relaunch — prior fix already on main, but its test does not compile

The prior run's fix (commit a2ea1020c) is ALREADY merged on origin/main:
`enable_scale_dimensions` rewrites a multi-axis thin-plate term into the
mathematically-equivalent anisotropic s=0 Duchon spline, so the documented
per-axis anisotropy genuinely engages for bs="tp".

### Bug found in merged state
`tests/basis_smooth/smooths/thin_plate_scale_dimensions_1676.rs` imports
`SmoothBasisSpec` from `gam::basis`, but it lives in `gam::smooth`. The
`basis_smooth` test binary therefore FAILS TO COMPILE on main. Fixing.

## Plan
1. Fix the import so the regression test compiles + passes.
2. Re-verify the whole basis_smooth suite + adjacent thin-plate/duchon tests.
3. Run the issue's Python reproduction end-to-end.
4. Harden: edge cases (periodic tp, double_penalty, identifiability).
5. Open tracking PR, verify green, merge.
