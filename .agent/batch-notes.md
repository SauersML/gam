# Batch gam-closed-1318-1497 triage notes

## Primary reopen target: #1476
Double-penalty `s(x1)+s(x2)` over-shrinks a genuinely-supported smooth on the large
default basis (~2-3.6x worse RMSE-to-truth vs dp=False / vs mgcv select=TRUE).
The catastrophic λ-rail collapse was fixed (c3e16fc95), and the projector matrix
was fixed (26ab264e3), but a RUNTIME-WITH-R verification by cormundus shows the
recovery gate (tests/issue_1476_concurvity_double_penalty_collapse.rs) is STILL RED
on current main (e23ebdbbe): gam rmse 0.119 vs mgcv 0.033 (3.63x), edf 15.0 vs 11.85.

Mechanism (per cormundus + owner #1477 trace): the strong nullspace select-out PC
prior in relax_smoothing_rho_prior still forces a genuinely-supported linear/constant
component toward zero on the large basis; bending coordinate over-compensates.

## Plan
1. Reproduce with dp-toggle (no mgcv needed): gam dp=True vs dp=False on same basis.
2. Fix the residual over-shrink in the Gaussian double-penalty nullspace prior.
3. Verify dp=True recovery matches dp=False / no over-shrink.

## Other candidates flagged
- #1471 validation baseline not actually mgcv-matched (owner's own last comment)
- #1373 edf_per_class over-count, "no safe blind fix" handoff
