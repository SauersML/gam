# Issue #1122 — matern(x1,x2) isotropic kappa optimizer stalls

## Symptom
`matern(x1, x2)` on ordinary smooth 2-D data: isotropic length-scale ("kappa")
optimizer exhausts 80-iter budget with large projected gradient
(final_grad_norm ~30-50), REML fit aborts with IntegrationError. Sibling radial
smooths (thinplate, duchon, te) fit the same data fine.

## Root cause (per issue + comment)
Operator-penalty log-kappa derivative for matern. The Duchon homogeneity
chain-rule was wrongly copied to matern; matern's exp factor breaks homogeneity.
Triplet-identity psi-derivative fix landed partially but is incomplete; gradient
never driven to zero.

## Key locations
- crates/gam-models/src/fit_orchestration/drivers/spatial_optimization.rs:4538
  ("did not converge after N iterations")
- :4436 SpatialHyperKind::Isotropic => "iso-kappa joint REML"
- design_construction.rs:6239 "spatial kappa optimization failed: {err}"

## Plan
1. Reproduce the failure (build python ext or rust test).
2. Locate matern psi-derivative / log-kappa penalty derivative.
3. Compare against the operator-penalty gradient (FD audit) to find the bug.
4. Fix the analytic derivative; verify grad_norm -> 0.
5. Add regression test (Rust + python).
