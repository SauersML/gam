//! Predict-side measure-jet honesty: the closed-form extrapolation variance
//! from the frame notes (`docs/measure_jet_frame.md` §5).
//!
//! The current Gaussian representers decay off-support toward the parametric
//! backbone with small posterior variance — confident reversion, which the
//! honesty contract forbids. The structural fix prices ignorance off the web from the SAME
//! fitted spectrum that smooths on it: every band level ℓ carries a fitted
//! amplitude λ̂_ℓ (prior precision of the level's innovations), and a query
//! that the level-ℓ kernel mass does not cover simply has an UNKNOWN level-ℓ
//! innovation — prior variance λ̂_ℓ⁻¹, collected in full.
//!
//! # The formula (and its algebraic relation to §5)
//!
//! With `q̄_ℓ = (Σ_i m_i q_ℓ(c_i)) / (Σ_i m_i)` the web-averaged scale-ℓ
//! support and `a_ℓ(x★) = min(q_ℓ(x★)/q̄_ℓ, 1)` the scale-correct on-web-ness
//! weight in `[0, 1]`, let
//! `ℓ★ = min{ℓ : q_ℓ(x★) ≥ coverage_floor · q̄_ℓ}` be the first covering
//! level (ε★ = ε_{ℓ★}). Then, for per-level spectra,
//!
//! ```text
//!   Var_extrap(x★) = Σ_{ℓ < ℓ★} λ̂_ℓ⁻¹  +  Σ_{ℓ ≥ ℓ★} (1 − a_ℓ(x★)) · λ̂_ℓ⁻¹
//!                  = Σ_ℓ λ̂_ℓ⁻¹  −  Σ_{ℓ: ε_ℓ ≥ ε★} a_ℓ(x★) · λ̂_ℓ⁻¹ .
//! ```
//!
//! The second line is the §5 statement: the total prior ignorance of the
//! spectrum minus the part the query's coverage recovers — the recovered sum
//! runs over the covered levels `ε_ℓ ≥ ε★` exactly as written in the charter.
//! In fused mode the band has one precision, so the same coverage idea reduces
//! to one charge: `λ_fused⁻¹` if no level clears its floor, otherwise
//! `(1 − max_ℓ a_ℓ(x★)) · λ_fused⁻¹`.
//! On-web queries (ε★ = ε_0, a_ℓ ≈ 1 everywhere) recover the full spectrum
//! and pay ≈ 0 extra; far-off queries recover (almost) nothing and pay the
//! full Σ_ℓ λ̂_ℓ⁻¹. Levels FINER than the first covering scale get no credit
//! for stray sub-floor kernel mass: below ε★ the prediction is a jet
//! extension, not an interpolation, so those innovations are charged as pure
//! ignorance.
//!
//! # Never-covered convention
//!
//! If no band level clears the coverage floor (ε★ lies past the band), the
//! covered set is EMPTY: in per-level mode every level contributes its full
//! λ̂_ℓ⁻¹ and `Var_extrap = Σ_ℓ λ̂_ℓ⁻¹`; in fused mode the single band
//! amplitude contributes once. The variance saturates at the spectrum's total
//! prior ignorance instead of growing without bound, which is the honest
//! statement: the model's coefficient prior is the only information it ever
//! claimed about such a point.
//!
//! # Monotonicity (the distance-honesty theorem)
//!
//! Claim: if `q ≤ q′` pointwise (the support row of the farther query is
//! nowhere larger), then `Var_extrap(q) ≥ Var_extrap(q′)`.
//!
//! Proof. `{ℓ : q_ℓ ≥ coverage_floor · q̄_ℓ} ⊆
//! {ℓ : q′_ℓ ≥ coverage_floor · q̄_ℓ}` for the scale-specific floors, so
//! `ℓ★(q) ≥ ℓ★(q′)`. Compare the per-level weights `w_ℓ`:
//! - `ℓ < ℓ★(q′)`: both weights are 1;
//! - `ℓ ≥ ℓ★(q)`: `w_ℓ(q) = 1 − a_ℓ(q) ≥ 1 − a_ℓ(q′) = w_ℓ(q′)`;
//! - `ℓ★(q′) ≤ ℓ < ℓ★(q)`: `w_ℓ(q) = 1 ≥ 1 − a_ℓ(q′) = w_ℓ(q′)`.
//! Every weight is no smaller and every `λ̂_ℓ⁻¹ > 0`, so the sum is no
//! smaller. ∎
//!
//! Since the Gaussian kernel mass `q_ℓ(x★)` is pointwise nonincreasing as
//! `x★` recedes from every center simultaneously, intervals widen
//! monotonically with distance from the web. The ε★ gate introduces the only
//! discontinuity, and it is bounded: a level crossing the floor changes its
//! weight by at most `a_ℓ ≤ coverage_floor`, so the total jump is at most
//! `coverage_floor · Σ_ℓ λ̂_ℓ⁻¹` and vanishes as the floor tightens.
//!
//! # Units
//!
//! The result is on the scale of physical `λ̂⁻¹`: callers must unnormalize the
//! fitted Frobenius-normalized precision first (`λ_phys = λ_tilde / c`). Family
//! dispersion scaling remains outside this pure spectrum-side kernel.

#[derive(Clone, Copy)]
pub enum MeasureJetExtrapolationSpectrum<'a> {
    /// One physical precision per band level.
    PerLevel(&'a [f64]),
    /// One physical precision for the fused band. It is charged once, with the
    /// band's best coverage fraction.
    Fused(f64),
}

