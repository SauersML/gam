# Issue #1610 — Manifold SAE magic-constant penalty strengths

## State at start (HEAD on main)
- `SAE_SEPARATION_BARRIER_STRENGTH = 10.0` is ALREADY DELETED (commit d3f63d62e).
  Replaced by data-derived `μ_C = K / reachable_rank` in
  `penalties.rs::separation_barrier_strength`.
- Decoder-repulsion strength derived as `RATIO · μ_C` (8381579).
- Collapse bar uses reachable geometric rank (339fe75).
- Norm-floor scale-invariant (fc20443d2).

## Remaining substantive ask (issue ask #1)
Owner: "It doesn't matter if they are scale relative. They are unprincipled and
arbitrary." `K/rank` is still a hand-chosen formula, NOT REML/evidence-derived.
Ask #1 explicitly: derive collapse-prevention strength from the EVIDENCE/REML
objective.

## Plan
1. Understand the SAE evidence/REML machinery (rho.rs, outer_objective.rs).
2. Make the barrier strength genuinely evidence-derived (or prove K/rank IS the
   evidence-optimal value).
3. Audit remaining SAE_* constants for principled replacement.
4. Crate-level tests; flag GPU/OLMo validation gaps honestly.
