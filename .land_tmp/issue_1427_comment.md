## Confirmed at source level — this is a double-centering basis-geometry bug, not the optimizer

I verified this against the **current tree** and it holds. This diagnosis supersedes the "joint-REML λ-selection" framing (including my own earlier optimizer analysis): the factor-by construction deletes one genuine nonconstant spline direction **per level, before REML ever runs**, and no choice of λ can recover a fit outside the resulting column space. Credit to the reporter — the OLS-floor falsification is the decisive argument.

### The double centering, in the live source

1. `src/terms/basis/types.rs:452-454` — a fresh B-spline defaults to `BSplineIdentifiability::WeightedSumToZero { weights: None }`.
2. `src/terms/smooth/term_specs.rs:6574-6589` — the `ByVariable` build clones the inner basis **verbatim** and runs `build_single_local_smooth_term` on the **full pooled data** *before* gating rows. So the inner B-spline's pooled weighted-sum-to-zero is applied first: raw `k` → `k-1`.
3. `src/terms/smooth/design_construction.rs:1040-1055` (`factor_by_level_gate` / `build_parametric_constraint_block_for_term`) — the global identifiability pass then centers the already-gated, already-centered block a **second** time, against the level indicator `c_g`: `k-1` → `k-2`.

The second centering is correct and necessary (it hands each level's constant to the treatment-coded factor main effect — the #900 convention). The bug is that the inner basis's **default model-space centering is not deferred** when it sits inside a factor-level `by=` wrapper, so two generically-independent constraints are imposed.

### Why it's `k-2`, and why it's not an optimizer problem

Let `B_g` be the raw `k`-column spline for level `g`, `m_g = B_g^\top w_g` its weighted column-sum, and `m = \sum_h m_h` the pooled one. The first transform forces `Z_0^\top m = 0`; the second forces `Z_0^\top m_g = 0`. The realized smooth coefficient space is `ker(m^\top) ∩ ker(m_g^\top)`, which has dimension `k-2` whenever `m_g` is not proportional to `m` (rank-nullity on two independent functionals). For `k=10`, that's **8 smooth columns per level instead of 9**, and the joint design is `1 + 2 + 3·8 = 27` columns rather than `30`.

The falsification is clean: OLS onto the malformed 8-column space already has essentially the fitted REML error, and every penalized/REML mean lies in `col(X)` — so no λ, line search, Hessian, or block solver can reach the deleted direction. That direction (`v_g = (I - m_g m_g^\top/\lVert m_g\rVert^2)\,m`) is generally nonconstant with large boundary amplitude, which is why the joint fit blows up at the boundaries and why the RMSE *ratio* grows with n (a fixed approximation-bias floor while the correctly-specified independent fit's variance vanishes).

### One honest caveat on the equal-support case

When every group shares an identical x grid, the pooled `m = \sum_h m_h` is proportional to each `m_g`, so the two constraints are (near-)redundant in exact arithmetic and the loss can collapse back to `k-1`. The unambiguous, generic trigger — and the one in this issue's own random-x repro — is **independent per-group support**, where `m_g ∦ m` and the column is genuinely lost under both `double_penalty=false` and `true`. The regression test must therefore use random-x-per-group and assert `k-1` columns per level (the proposed test does exactly this).

### Fix

Make factor-level identifiability **single-owner**: defer the inner basis's default model-space centering when it is wrapped by `ByVariable::Level`, and let the existing level-indicator centering be the sole constraint. The targeted patch (clone inner spec, downgrade only a default `WeightedSumToZero` to `None`, preserve explicit/frozen/structural transforms) does this; I'm adapting it to current line numbers and adding a structural test plus an end-to-end truth-recovery check.

Governing invariant worth enforcing repo-wide: **toggling `double_penalty` may change penalties/λ but must never change the unpenalized design column space or its dimension.** That invariant would have caught this immediately, and guards the same latent double-ownership bug for tp/Duchon/Matérn wrappers.

Secondary (not the root, lower priority): `s(x, by=g) + g` lowers to *two* `g` blocks — the auto-added unpenalized treatment main effect plus a separately-penalized categorical block — adding a spurious smoothing parameter (numerically inert here). Formula lowering should deduplicate or reserve a penalized intercept for explicit `group(g)`.

Close criteria I'll gate on: every `k=10` level block has 9 smooth columns under both `double_penalty` settings; `[group intercept | by-block]` matches the standalone `[intercept | s(x)]` column space; and the low-noise large-n joint fit recovers RMSE comparable to the independent fits (not merely "width is right").

*Verified against current `main`/HEAD source; the corrected-geometry REML recovery numbers are the reporter's (no Rust toolchain in that environment) — I'm reproducing them as part of the fix.*
