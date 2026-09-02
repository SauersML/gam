//! #939 deliverable 4 — EMPIRICAL NULL-SIMULATION SIZE CALIBRATION of the
//! smooth-term likelihood-ratio test, the validation the issue demands before
//! closure.
//!
//! Under a null data-generating process (the smooth's covariate has no effect on
//! the mean) the per-term LR statistic `W = 2(ℓ_full − ℓ_null)` follows a central
//! `χ²_d` only to first order. At modest `n` the first-order reference is
//! **anti-conservative**: `E[W] = d + Δε > d`, so the χ²_d tail under-covers and
//! the empirical size — the fraction of null replicates rejected at level `α` —
//! exceeds the nominal `α`. The Bartlett correction rescales `W` by
//! `c = E[W]/d` so the corrected statistic's mean returns to `d` and the size
//! returns to nominal. This harness measures that directly, comparing three
//! lanes from the SAME live driver (`smooth_term_lr_inference_forspec`):
//!
//!   (a) first-order χ²        — `p_value_uncorrected`,
//!   (b) fixed-λ Bartlett      — `p_value_corrected` with the conditional factor,
//!   (c) estimated-λ Bartlett  — `p_value_corrected` with the ρ̂-variation factor
//!                                (`correction == LawleyLrEstimatedLambda`).
//!
//! Empirical size at `α` is `#{p ≤ α}/R`. Its Monte-Carlo standard error is
//! `√(α(1−α)/R)`; the assertions use a `±k·SE` band so they are robust to the
//! finite simulation budget. The defining claims (#939 deliverable 4):
//!
//!   1. WHERE FIRST-ORDER IS DISTORTED (small `n`): the first-order size is
//!      materially above nominal, and the corrected lanes pull it back — the
//!      estimated-λ size is at least as close to nominal as the first-order size
//!      AND lands inside the MC band, across families and penalty ranks.
//!   2. ESTIMATED-λ NEVER WORSE: across the whole grid the estimated-λ size's
//!      distance from nominal never exceeds the first-order distance by more than
//!      MC noise — the correction is safe to apply everywhere.
//!   3. MATERIALITY: the per-test `material` flag fires exactly when the applied
//!      Bartlett factor moves the result by more than 10% (the documented rule).
//!
//! Truth-recovery bar (not a reference-tool match): the ground truth is the exact
//! null distribution of the LR statistic, i.e. Uniform p-values / nominal size.
//!
//! Budget: the full grid is `n ∈ {30,50,100,200,500}` × 2 families × 2 penalty
//! ranks × `REPS` replicates. The default `REPS` keeps wall-time in CI range
//! while holding the MC band tight enough for the directional claims; the small-n
//! cells (where the correction matters and a fit is cheap) carry the load.

