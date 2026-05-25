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
//! Stage 1 (the design note above) and Stage 2 (closed-form C-spline values
//! plus `build_ispline_base_designs`) live in this file. The enum migration
//! on `TimeBlockInput` has already landed at `survival_location_scale.rs`
//! (`TimeBlockMonotonicity`). The dispatch wiring at the marginal-slope
//! construction sites follows in Stage 3.
//!
//! # Closed-form C-spline derivation (Stage 2 math)
//!
//! Setup. Knot vector `τ` of length `K`; I-spline degree `k`. I-splines are
//! built from degree-`(k+1)` B-splines on `τ`:
//!
//!   I_j^{(k)}(u) = Σ_{m=j+1}^{N_B-1} B_m^{(k+1)}(u; τ),   j = 0..N_I-1
//!
//! where `N_B = K − (k+2)` and `N_I = N_B − 1`, anchored to `I_j(τ[k+1]) = 0`
//! and `I_j(τ[N_B]) = 1` (the convention `create_ispline_dense` already uses).
//!
//! C-spline goal:
//!
//!   C_j(u) := ∫_{τ[k+1]}^{u} I_j^{(k)}(v) dv, anchored to C_j(τ[k+1]) = 0.
//!
//! Curry-Schoenberg integration identity. For a B-spline of degree `(k+1)`
//! on `τ`, the antiderivative is a degree-`(k+2)` spline expressible on an
//! extended knot vector `τ̃` (one extra repeat of `τ[0]` at the left and of
//! `τ[K-1]` at the right; length `K+2`):
//!
//!   ∫_{−∞}^{u} B_m^{(k+1)}(v; τ) dv
//!       = ((τ[m+k+2] − τ[m])/(k+2)) · T_m^{(k+2)}(u; τ̃)
//!
//! where `T_m^{(k+2)}(u; τ̃) := Σ_{i ≥ m} B_i^{(k+2)}(u; τ̃)` is the
//! degree-`(k+2)` right-cumulative tail. (Equivalent to Marsden / dual-spline
//! integration; see Schumaker 1981 §4 or de Boor *A Practical Guide to
//! Splines* eq. X.30 for the right-cumulative form.)
//!
//! Substituting and using linearity:
//!
//!   C_j(u) = Σ_{m=j+1}^{N_B-1} ((τ[m+k+2] − τ[m])/(k+2))
//!            · [ T_m^{(k+2)}(u; τ̃) − T_m^{(k+2)}(τ[k+1]; τ̃) ]
//!
//! The anchor subtraction `T_m^{(k+2)}(τ[k+1]; τ̃)` is a precomputable vector
//! of size `N_B`; once cached the per-data-point cost is one degree-`(k+2)`
//! B-spline evaluation on `τ̃` plus a right-cumulative sweep, identical to
//! the pattern `create_ispline_dense` already runs at degree `(k+1)`.
//!
//! Sanity checks built into the test in Stage 6 (`tests/ispline_base_time.rs`):
//! * `C_j(τ[k+1]) = 0` by construction (the subtracted anchor cancels).
//! * `dC_j/du = I_j^{(k)}(u)` pointwise — verified by analytic vs finite-
//!   difference comparison on a randomized grid of `u`.
//! * `C_j(u) > 0` strictly on `(τ[k+1], u]` whenever `I_j(u) > 0` on a
//!   positive-measure subset of `(τ[k+1], u]`.
//!
//! # Wiring (Stage 3, briefed here for completeness)
//!
//! `build_ispline_base_designs(log_t_entry, log_t_exit, age_entry, age_exit,
//! knots, internal_degree, derivative_guard)` returns
//!
//!   { design_entry, design_exit, design_derivative_exit,
//!     offset_residual_entry, offset_residual_exit,
//!     derivative_offset_residual_exit }
//!
//! with the conventions:
//!
//!   design_entry[i, j]            = C_j^{(k)}(log t_entry[i])
//!   design_exit[i, j]             = C_j^{(k)}(log t_exit[i])
//!   design_derivative_exit[i, j]  = (1 / age_exit[i]) · I_j^{(k)}(log t_exit[i])
//!   offset_residual_entry[i]      = guard · age_entry[i]
//!   offset_residual_exit[i]       = guard · age_exit[i]
//!   derivative_offset_residual_exit[i] = guard
//!
//! The caller is `prepare_survival_time_stack`, which already merges these
//! offset residuals additively into the existing offset slots via
//! `add_survival_time_derivative_guard_offset` — so the existing pipeline
//! continues to own offset accumulation.
