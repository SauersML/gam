//! Documentation pointer: the survival time block uses I-spline base
//! coefficients (`γ ≥ 0`) to enforce structural monotonicity of
//! `q(t)`, replacing the row-wise `D β + o ≥ guard` derivative-guard
//! constraints the marginal-slope family used to rely on.
//!
//! The canonical implementation lives at
//! [`crate::families::survival_construction::SurvivalTimeBasisConfig::ISpline`]
//! (see `survival_construction.rs:140-145` for the enum variant and
//! `1285-1505` for the design builder). It exposes:
//!
//! * `x_entry_time` / `x_exit_time` — I-spline value rows on the `log(t)`
//!   axis. Non-negative entries plus `γ ≥ 0` give a monotone-non-decreasing
//!   `q(t) = I_basis(log t) · γ`, the structural property the
//!   marginal-slope family needs.
//! * `x_derivative_time` — right-cumulative B-spline-derivative on
//!   `log(t)` scaled by `1/t`, again non-negative with `γ ≥ 0`, so
//!   `q'(t) ≥ 0` pointwise. The `derivative_guard` constant is added in
//!   externally by [`crate::families::survival_construction::add_survival_time_derivative_guard_offset`],
//!   leaving the derivative guarantee `q'(t) ≥ guard` exact.
//! * 2nd-difference penalty on the underlying degree-`(k+1)` B-spline
//!   coefficients, filtered through `keep_cols` for identifiability.
//!
//! `TimeBlockInput::time_monotonicity` declares to the consuming family
//! how monotonicity is enforced. The marginal-slope construction site
//! sets it to [`crate::families::survival_location_scale::TimeBlockMonotonicity::StructuralISpline`]
//! so the family skips row-wise `D β + o ≥ guard` constraint generation
//! and treats `γ ≥ 0` as the sole derivative-guard mechanism. The
//! universal `validate_time_qd1_feasible` safety net runs regardless.
//!
//! A previous iteration of this module proposed a separate C-spline
//! antiderivative parameterization that put `q'(t)` in the I-spline
//! space and `q(t)` in the integral-of-I-spline space. That was
//! mathematically equivalent but a strictly worse fit for the codebase
//! (extra basis degree, an extra antiderivative builder, an extra
//! identifiability path, an extra penalty). It has been removed in favor
//! of the canonical I-spline-value path described above.
