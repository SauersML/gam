# PIRLS Joint-Newton Residual-Stall — Root Cause Report

## Production failure under investigation

`survival_marginal_slope` fit at biobank scale (n=195,780, p=33 split
[12 time, 11 marginal, 10 logslope]) — all 5 outer seeds rejected.
Trace signature:

- Cycle 0: unconstrained Newton proposal `|prop|∞ ≈ 2e5`, trust region
  clamps to `|δ|∞ = 20`, decision `shrink_marginal_accept`, reaches
  `|β|∞ = 20.0`.
- Cycles 1–25: `linearized_rel = ‖g+Hδ‖∞ / ‖g‖∞ ≈ 0.97` for 15+
  consecutive cycles.
- `[PIRLS/joint-Newton convergence] cycle N | residual-stall early-exit`
  fires.
- Budget-exhausted block dump:

      block_widths   = [12, 11, 10]
      block_beta_inf = [2.3e-4, 15.3, 20.0]
      block_grad_inf = [5.6e8,  1.5e3, 2.3e3]

## Root cause (precise)

The joint Newton trust region globalisation was using an **isotropic L2
norm** over the concatenated δ vector to bound the step, and rescaling
the entire δ uniformly when the unconstrained Newton proposal exceeded
the radius. The implementation pair is/was:

- `fn joint_trust_region_step_norm(delta) -> f64` in
  `src/families/custom_family.rs` line 10835 — returns
  `sqrt(Σ_i δ_i^2)` over all coefficients, with no block partition and
  no curvature weighting.
- `fn truncate_joint_step_to_radius(delta, radius)` in
  `src/families/custom_family.rs` line 10839 — multiplies the entire
  `delta` in place by `radius / norm` whenever the global L2 norm
  exceeds the scalar radius.

Mathematical mistake: in a multi-block GAM whose blocks have wildly
different Hessian conditioning (here, a time block with near-singular
Fisher information when the time basis is over-parameterised relative
to the event-bearing portion of the age axis), the unconstrained
Newton step

    δ = H_pen⁻¹ g

has huge L2 norm dominated by the ill-conditioned block's null
direction. A uniform multiplicative rescale to a scalar radius
sends *every block* uniformly toward zero — including blocks whose
Newton step was small and accurate. The model-vs-actual ratio ρ then
falls in `(0, 0.25)`, the trust policy fires `shrink_marginal_accept`
on the next cycle, and the time-block gradient (uncorrected because
its true Newton step was thrown away) stays at 5e8 forever. The KKT
residual ratio `‖g+Hδ‖∞/‖g‖∞ ≈ 0.97` is the algebraic fingerprint:
the step is asymptotically orthogonal to the gradient because all of
its useful direction has been scaled away.

## Current code already implements the fix

HEAD of `src/families/custom_family.rs` has **per-block,
diagonal-Hessian-preconditioned** trust-region radii in place:

- `fn joint_trust_region_block_metric_norms` (line 10859) computes a
  per-block step norm `||δ_block||_M = sqrt(δ_block^T diag(M_block) δ_block)`
  where `M = diag(H_pen) = diag(H_L + S)` is the
  `joint_penalty_preconditioner_diag` of the penalised Hessian
  (see line 11678–11691 in the joint-Newton loop).
- `fn truncate_joint_step_to_block_metric_radii` (line 10876)
  truncates each block independently to its own radius.
- A `joint_block_trust_radii: Vec<f64>` (initialised at line 11503)
  is updated per block in `shrink_active_joint_block_trust_radii`
  (line 10904) so ill-conditioned blocks shrink without starving
  well-conditioned ones.

This is the correct anisotropic, curvature-weighted trust region the
fix sketch below calls for. The production failure trace therefore
predates this fix.

## Fix sketch (already merged)

Replace the scalar `||δ||_2 ≤ r` constraint with an anisotropic
per-block ellipsoidal constraint `||δ_b||_{M_b} ≤ r_b` for each
parameter block `b`, where `M_b = diag(H_pen_bb)` (or, ideally, the
diagonal of the penalised inverse-Schur block) is the local
preconditioner used by the inner PCG. Each block's radius is updated
independently from its own ρ_b ratio. This is the standard trust-region
generalisation for block-separable problems and matches Conn–Gould–Toint
§7.5 and the Steihaug–Toint PCG dogleg variant.

## Red test status

**File:** `/Users/user/gam/tests/survival_marginal_slope_stall.rs`

The test instantiates a `Surv(entry_age, exit_age, event) ~ duchon(PC1..3, ...) + sex`
survival_marginal_slope fit at production scale (n=195,780, p=37,
time_num_internal_knots=12, time_smooth_lambda=1e-8) and asserts
`outer_converged == true`. Run on the V100 VM with cargo `release-dev`
profile.

**Observed behaviour:** the assertion did NOT trip — the inner PIRLS
joint-Newton converges via noise-floor KKT certificates on every REML
evaluation. PIRLS log shows `step_inf → 0`, `residual → 0.5`,
`Δobjective → 1e-6`, no `residual-stall early-exit` lines. This is
consistent with the per-block trust region already being live on this
build.

**Implication for red-testing:** to obtain a reliably red test for the
old isotropic-TR bug, either (a) gate the fix behind a config knob and
flip it off in the test, or (b) push the conditioning gap further by
e.g. coupling a tiny event rate with a much higher-density time basis
(`time_num_internal_knots ≥ 24`) and a flat-tail-only event distribution.
Option (a) is the more principled red-test: the failure is purely a
property of the old TR policy, not of any pathological data shape.

The test was left in place (not deleted) so future regressions can use
the existing dataset builder. It currently passes only because the
fix is live; a deliberate revert of the per-block TR would re-fail it.

## File references

- Bug-relevant (isotropic, dead path): `src/families/custom_family.rs`
  lines 10835–10847 (`joint_trust_region_step_norm`,
  `truncate_joint_step_to_radius`).
- Fix (active path): `src/families/custom_family.rs` lines 10859–10942
  (`joint_trust_region_block_metric_norms`,
  `truncate_joint_step_to_block_metric_radii`,
  `shrink_active_joint_block_trust_radii`,
  `joint_penalty_preconditioner_diag` usage at 11678–11691).
- Residual-stall early-exit logic (the early-return that production
  hits): `src/families/custom_family.rs` lines 12734–12782.
- Budget-exhausted block dump that prints `block_grad_inf=[...]`:
  `src/families/custom_family.rs` lines 12941–13001.
- Red test: `tests/survival_marginal_slope_stall.rs`.
