//! Transport-law measurement tests: does layer-to-layer transport of a fitted
//! circle atom obey the phase-shift law?
//!
//! Two arms, both built as SYNTHETIC 2-layer crosscoders through the landed M1
//! driver [`SaeManifoldTerm::run_multiblock_reml_fit`] (mirroring the fixtures in
//! `tests_crosscoder_multiblock`):
//!
//! 1. **Planted phase shift** — layer 2 IS layer 1 reparameterized by a constant
//!    phase `φ0` (`Y_2(θ) = circle(θ + 2π·φ0)` on the SAME frame). The fitted
//!    transport must recover `φ0` with `phase_r2 > 0.95` and
//!    `smooth_r2 − phase_r2 < 0.02` — the law holds, the extra harmonics buy
//!    nothing.
//! 2. **Planted nonlinear transport** — layer 2 is a *squashed* image of the same
//!    circle (a different-shape ellipse, not a phase rotation), so projecting the
//!    round layer-1 image onto it is a genuinely nonlinear (2nd-harmonic) map. The
//!    verdict must FLIP: `smooth_r2 − phase_r2` clears a margin derived from the
//!    phase-shift arm's own gap.

