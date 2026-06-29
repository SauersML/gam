# Issue #1074 — spatial/tensor under-recovery

## Plan
Root-cause and fix the spatial/tensor truth-recovery gaps vs mgcv across
te/tp/duchon/matern/sz/factor-smooth tests. Owner already localized several
turnkey roots (needs build access, which I have):

1. Matern iso-κ ψ-floor cap clamps length_scale ≤ 0.15·r_max
   (crates/gam-terms/src/smooth/term_specs.rs:~3450) — over-smooths.
2. te/tp tensor penalty / REML λ selection null-space recovery gap.
3. pair_surface / thin-plate-2d (worst), rw2 held-out R².

## Approach
- Build warm, reproduce failing tests, log converged λ/κ vs truth-optimal.
- Fix real roots; recalibrate only provably-wrong bounds with derivation.
