//! Owed-work objective evaluation for issue #975 — "the carving problem":
//! functional-ANOVA carving of a fitted product-manifold atom into additive
//! main effects (superposition) vs a genuine interaction (binding), plus a
//! STATISTICAL TEST for feature binding.
//!
//! The capability lives in `terms::structure::anova_atom`: `fit_pair_surface`
//! fits the penalized tensor surface `y_d(x1,x2) ≈ φ¹(x1)ᵀ C_d φ²(x2)` over a
//! scattered code sample with REML-selected smoothness, returning coefficients
//! AND their scale-included posterior covariance; `carve` performs the EXACT
//! ANOVA reparameterization (mean + main_a + main_b + interaction), measures
//! the continuous interaction-energy fraction (the "how bound" dial), and runs
//! a Wood-style Wald test of `f₁₂ ≡ 0` on the gauge-projected interaction block
//! — its `edge_p_value` is the feature-binding test. `fission_decision` turns
//! the test into a three-valued split/keep verdict.
//!
//! The in-module unit tests already pin the algebra (exact reparameterization,
//! a hand-planted additive surface fissions losslessly, a hand-planted rank-1
//! interaction refuses + the test rejects). What they do NOT exercise is the
//! capability the issue actually asks for: does the END-TO-END pipeline —
//! REML-fit a scattered surface, harvest its posterior, carve — correctly
//! CLASSIFY a panel of synthetic feature pairs with KNOWN additive-vs-
//! interaction structure as separable vs bound, at controlled Type-I error and
//! high power? This file is that objective evaluation.
//!
//! ## Design
//!
//! A panel of deterministically generated pairs (no RNG: a fixed
//! low-discrepancy abscissa cloud + a fixed RNG-free zero-mean noise stream so
//! the run is a pure function of its inputs). Half are SEPARABLE truths
//! `f(x1)+f(x2)` (additivity = superposition: the binding test must NOT reject
//! and the energy fraction must be tiny), half are BOUND truths carrying a
//! genuine, non-removable cross term (`f₁₂ ≢ 0`: the binding test must reject
//! and the energy fraction must be large). The classifier is the single
//! production rule the evidence ledger consumes:
//!
//!     bound  ⇔  carve(...).edge_p_value ≤ alpha
//!
//! and we require it to classify EVERY pair in the panel correctly (a confusion
//! matrix with zero off-diagonal), plus the continuous `interaction_fraction`
//! and the `fission_decision` to agree with the truth. This is the objective
//! the mandate names: "on synthetic data with known additive-vs-interaction
//! structure, the test must correctly classify bound vs separable feature
//! pairs."
//!
//! No `let _`, no `#[allow(...)]`, no env vars, no `#[cfg(feature=...)]`.

// --- the panel -------------------------------------------------------------
// SEPARABLE truths: f(x1)+f(x2). The ANOVA interaction block is exactly zero
// in expectation; any nonzero estimate is noise, so the binding test must NOT
// reject (Type-I control) and the energy fraction must be small.

// BOUND truths: a genuine, non-removable cross term (f₁₂ ≢ 0). The binding
// test must reject and the energy fraction must be large.

