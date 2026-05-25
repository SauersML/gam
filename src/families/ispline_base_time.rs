//! Structural monotonicity for the survival-marginal-slope time block via
//! I-spline base coefficients with `γ ≥ 0` enforced as simple coordinate
//! bounds.
//!
//! # Why this module exists
//!
//! The survival_marginal_slope time block is hybrid:
//!
//! * The **wiggle tail** (β slice `time_tail` from `time_wiggle_range()`)
//!   already uses an I-spline basis with non-negativity bound constraints —
//!   see `gamlss::monotone_wiggle_basis_from_knots` (which dispatches
//!   `BasisOptions::i_spline()`) and `gamlss::monotone_wiggle_nonnegative_constraints`.
//!   Positivity of the wiggle is **structural**: nonneg I-spline × nonneg
//!   coefficient ⇒ nonneg.
//!
//! * The **base portion** (`..p_base`) is an unconstrained additive
//!   `q_base(t) = D_base(t) · β_base + o_base(t)`. NOT structurally monotone.
//!   Monotonicity is enforced post-hoc — currently by row-wise linear
//!   inequality constraints `D β + o ≥ guard` built by
//!   `survival_location_scale::time_derivative_guard_constraints`. The
//!   `qd1-constraint-kkt` work integrates those rows into the active-set QP
//!   so a binding cliff produces a real KKT multiplier instead of the
//!   "phantom multiplier" that the deleted silent-projection path used to
//!   leave behind.
//!
//! Row-wise constraints are correct for the additive parameterization but
//! they keep monotonicity as a *runtime claim that the construction site is
//! supposed to honour*. This module replaces the base parameterization so
//! monotonicity becomes a **structural property of the basis** with the same
//! pattern the wiggle tail already uses:
//!
//! ```text
//! q_base(t)   = guard · t + C_base(t) · γ_base + o_base(t)
//! q_base'(t)  = guard     + I_base(t) · γ_base
//! γ_base_j   ≥ 0 enforced via coordinate bounds in the active-set QP
//! ```
//!
//! where `I_base(t)` is the I-spline row at `t` (nonneg piecewise polynomial)
//! and `C_base(t) = ∫_0^t I_base(s) ds` is its antiderivative, which is itself
//! a closed-form spline (a C-spline / B-spline of one higher order; Ramsay
//! 1988, `splines2` C-spline docs). All evaluations remain closed form.
//!
//! # What this preserves
//!
//! * **Closed-form `q_base(t)`** — `C_base(t)` is closed form.
//! * **Closed-form `q_base'(t)`** — `I_base(t)` is closed form.
//! * **Bilinear Hessian** — q' is linear in γ_base, so
//!   `∂²ℓ/∂γ∂γ = Σ_i a_i X_i X_iᵀ` with no chain-rule diagonal term.
//!   The softplus reparameterization (`γ_j = softplus(γ̃_j)`) would add a
//!   `diag(g_j σ'(γ̃_j))` term, breaking pure bilinearity and preventing
//!   `γ_j = 0` from being attained at finite parameter — strictly worse than
//!   the bound-constrained path used by the wiggle.
//! * **No parameter-dependent quadrature.** `time_wiggle_geometry`
//!   (`survival_marginal_slope.rs:4612`) does pointwise basis evaluations at
//!   the h0/h1 time coordinates only — there is no `∫ w(s, t) ds` kernel
//!   with parameter-dependent weights anywhere in the family that this
//!   module would have to integrate. Codex's caveat about `exp(-Λ(s))`-shape
//!   weights does not apply.
//!
//! # What this deletes
//!
//! * `survival_marginal_slope::project_time_qd1_feasible` — already deleted
//!   in `qd1-constraint-kkt`'s row-wise constraint work. This module
//!   confirms validation is sufficient.
//! * Under `TimeBlockMonotonicity::StructuralISpline` the row-wise
//!   `time_derivative_guard_constraints` becomes vacuous and is replaced
//!   by the wiggle's existing coordinate-bound emission path applied to
//!   the base coordinates as well.
//! * `max_feasible_time_step` becomes a sanity-check no-op (α ≈ 1 in
//!   steady state; any persistent strict shrink is a bug signal).
//!
//! # Wiring contract
//!
//! `TimeBlockInput` carries
//!
//! ```rust
//! pub enum TimeBlockMonotonicity {
//!     EnforcedByCoordinateCone,  // location-scale's existing coordinate-cone
//!     EnforcedByRowConstraint,   // current marginal-slope additive base + row-wise D β + o ≥ guard
//!     StructuralISpline,         // this module — I-spline base + γ ≥ 0 coordinate bounds
//! }
//! ```
//!
//! Under `StructuralISpline`, `build_survival_time_basis` populates
//! `design_entry / design_exit / design_derivative_exit` from this module's
//! C-spline / I-spline constructors and adds `guard · t` into the offset
//! slots. The constraint surface for the base coordinates becomes the same
//! coordinate-bound shape the wiggle already emits.
//!
//! # Stage status
//!
//! This file is the design note (Stage 1). The closed-form C-spline /
//! I-spline design builders, the enum migration on `TimeBlockInput`, and
//! the dispatch wiring follow in subsequent stages. Stage 1 lives here so
//! the module is registered in `src/families/mod.rs` and downstream
//! teammates can see the intended shape.
