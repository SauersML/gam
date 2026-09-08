Found and fixed another occurrence of the same cancellation mechanism: the affine profile already used a spectral tangent solve, but active-face selection and the terminal KKT recheck still assembled `G + λS` in natural coordinates.

Both constrained quadratic solves now reuse the unconstrained fit's data-whitened penalty basis. With `BᵀGB = I`, `BᵀSB = diag(δ)`, and `β = B diag((1 + λδ)^(-1/2)) u`, the quadratic Hessian is the identity. The constraint rows and warm point are transformed into that frame; returned coefficients and active-row identities remain in the original problem. No numerical ridge or failed-endpoint suppression is introduced.

A new independent oracle uses a rotated rank-one penalty and a binding coefficient constraint at log strengths **0, 30, and 45**. Its constrained solution is available directly from three scalar normal equations, so it checks the production solver against an analytic answer, including the regime where natural-coordinate assembly loses the data curvature.

Fresh MSI verification of the current `gam-solve` source: **6 passed, 0 failed, 0.18 seconds**:

- New constrained KKT oracle at large strengths.
- Existing rank-deficient affine-profile limit and derivative checks.
- Complete nonzero-affine VJP finite-difference oracle.
- Weakly active constraint derivative refusal.
- Block REML penalty-nullspace limit oracle.
- Firth competing-strength trace oracle.

The test crate was compiled directly from the canonical source against the exact dependencies recorded by an existing warm Cargo test build, with four CPUs. Complete compiler arguments, dependency versions, source hashes, and output are recorded in `bench/measurements/issue_triage_20260908/reml-kkt-build.json` and `reml-kkt-tests.log`. No local build or test ran.

**Public API acceptance remains pending.** The authored suite now covers all 16 binding constraints and their negligible-ridge comparisons, plus the complete related block/Firth/factor-smooth sweeps. A fresh Python extension is still needed before claiming those results. Leaving this issue open.
