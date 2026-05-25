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

use ndarray::{Array1, Array2, ArrayView1};

use crate::terms::basis::{
    BasisOptions, KnotSource, SplineScratch, create_basis, evaluate_bspline_basis_scalar,
};

/// Errors emitted by the I-spline base-time design builders.
#[derive(Debug, thiserror::Error)]
pub enum ISplineBaseTimeError {
    /// The I-spline / C-spline degree must be `≥ 1` (B-splines of degree
    /// `k+1` must be well-defined).
    #[error("ispline_base_time: I-spline degree must be >= 1, got {degree}")]
    InvalidDegree { degree: usize },
    /// Knot vector too short for the requested degree.
    #[error(
        "ispline_base_time: knot vector of length {provided} is too short for I-spline degree {degree}; need >= {required}"
    )]
    InsufficientKnots {
        degree: usize,
        required: usize,
        provided: usize,
    },
    /// Entry and exit length mismatch when building paired designs.
    #[error("ispline_base_time: entry/exit length mismatch (entry={entry}, exit={exit})")]
    EntryExitMismatch { entry: usize, exit: usize },
    /// Bubbled error from the underlying spline routines.
    #[error("ispline_base_time: spline routine failed: {reason}")]
    Spline { reason: String },
    /// A requested time coordinate is non-finite or non-positive.
    #[error(
        "ispline_base_time: time coordinate at row {row} is not finite-and-positive (got {value})"
    )]
    InvalidTime { row: usize, value: f64 },
}

/// Result of `build_ispline_base_designs`: dense entry/exit/derivative designs
/// in the C-spline / scaled-I-spline basis, plus the additive `guard·t`
/// residuals that the caller folds into the existing offset slots.
#[derive(Debug, Clone)]
pub struct ISplineBaseDesigns {
    pub design_entry: Array2<f64>,
    pub design_exit: Array2<f64>,
    pub design_derivative_exit: Array2<f64>,
    pub offset_residual_entry: Array1<f64>,
    pub offset_residual_exit: Array1<f64>,
    pub derivative_offset_residual_exit: Array1<f64>,
}

/// Closed-form C-spline values `C_j^{(k)}(u) = ∫_{τ[k+1]}^{u} I_j^{(k)}(v) dv`
/// evaluated at the rows of `data`, for I-spline degree `k = degree` on the
/// knot vector `knots`.
///
/// Implementation follows the derivation in this module's doc comment:
/// integrates each constituent B-spline of degree `k+1` via the Curry-
/// Schoenberg formula, yielding a closed-form expression in degree-`(k+2)`
/// B-splines on the extended knot vector `τ̃`. The right-cumulative sweep
/// across `m` is the same shape `create_ispline_dense` already uses one
/// degree lower.
///
/// Returns an `(n_data, N_I)` matrix where `N_I = K − (k+3)` matches the
/// number of I-splines that `create_basis::<Dense>(.., BasisOptions::i_spline())`
/// produces on the same `(knots, degree)`.
pub fn cspline_basis_values(
    data: ArrayView1<'_, f64>,
    knots: &Array1<f64>,
    degree: usize,
) -> Result<Array2<f64>, ISplineBaseTimeError> {
    if degree < 1 {
        return Err(ISplineBaseTimeError::InvalidDegree { degree });
    }
    let k = degree;
    // The I-spline of degree k builds on B-splines of degree (k+1) over
    // `knots`, requiring at least 2*(k+2) knots; the C-spline integrates
    // those into degree-(k+2) B-splines on the *extended* knot vector
    // (one extra repeat at each boundary, length |knots| + 2). The
    // degree-(k+2) basis on the extended vector requires
    //   |knots| + 2 >= 2*(k+3) <=> |knots| >= 2*k + 4 = 2*(k+2).
    // i.e. exactly the same lower bound the I-spline already enforces.
    let required = 2 * (k + 2);
    if knots.len() < required {
        return Err(ISplineBaseTimeError::InsufficientKnots {
            degree: k,
            required,
            provided: knots.len(),
        });
    }
    // Extended knot vector τ̃: prepend τ[0] and append τ[K-1].
    let kk = knots.len();
    let mut tau_ext = Array1::<f64>::zeros(kk + 2);
    tau_ext[0] = knots[0];
    for i in 0..kk {
        tau_ext[i + 1] = knots[i];
    }
    tau_ext[kk + 1] = knots[kk - 1];

    let bs_deg = k + 2;
    // Number of degree-(k+2) B-splines on τ̃: |τ̃| − (bs_deg + 1).
    let n_b_ext = tau_ext.len() - (bs_deg + 1);
    // Number of degree-(k+1) B-splines on τ: same as `num_bspline_basis`
    // inside `create_ispline_dense`. This is the `m` index range used
    // below for the right-cumulative sum.
    let n_b_inner = knots.len() - (k + 2);
    // Number of I-splines (and therefore C-splines): one less than
    // the inner B-spline count, matching `create_ispline_dense`.
    let n_i = n_b_inner.saturating_sub(1);
    if n_i == 0 {
        return Ok(Array2::zeros((data.len(), 0)));
    }

    // Coefficient α_m = (τ[m+k+2] − τ[m]) / (k+2) for m = 0..n_b_inner-1.
    // This is the Curry-Schoenberg integration weight for B_m^{(k+1)}(·; τ).
    let mut alpha = Array1::<f64>::zeros(n_b_inner);
    for m in 0..n_b_inner {
        let lo = knots[m];
        let hi = knots[m + k + 2];
        alpha[m] = (hi - lo) / ((k + 2) as f64);
    }

    // Anchor: T_m^{(k+2)}(τ[k+1]; τ̃) for m = 0..n_b_inner-1.
    //
    // Evaluate degree-(k+2) B-spline values on τ̃ at the anchor point
    // `τ[k+1]` (the left boundary of the I-spline support), then form
    // right-cumulative tails of length n_b_inner. The tail at index `m`
    // is Σ_{i ≥ m} B_i^{(k+2)}(τ[k+1]; τ̃); we only need `m` up to
    // n_b_inner − 1 because that's the range we sum over below.
    let anchor_u = knots[k + 1];
    let mut anchor_b = vec![0.0_f64; n_b_ext];
    let mut anchor_scratch = SplineScratch::new(bs_deg);
    evaluate_bspline_basis_scalar(
        anchor_u,
        tau_ext.view(),
        bs_deg,
        &mut anchor_b,
        &mut anchor_scratch,
    )
    .map_err(|e| ISplineBaseTimeError::Spline {
        reason: format!("anchor evaluation failed: {e}"),
    })?;
    // Right-cumulative T_m^{(k+2)}(anchor) at the anchor: sum over i ≥ m.
    // We need indices m = 0..n_b_inner-1 inclusive; the index `i` ranges
    // over 0..n_b_ext. For correctness we cap at n_b_ext (the full degree-
    // (k+2) basis), then index by m below.
    let mut anchor_tail = vec![0.0_f64; n_b_ext.max(n_b_inner)];
    let mut running = 0.0_f64;
    for i in (0..n_b_ext).rev() {
        running += anchor_b[i];
        anchor_tail[i] = running;
    }
    // Tails for i ≥ n_b_ext are vacuously 0; the buffer's extra slots stay
    // at their initial zero, which is what we want.

    // Working buffers per data row.
    let mut row_b = vec![0.0_f64; n_b_ext];
    let mut row_scratch = SplineScratch::new(bs_deg);

    let mut out = Array2::<f64>::zeros((data.len(), n_i));
    for (row_idx, &u) in data.iter().enumerate() {
        if !u.is_finite() {
            return Err(ISplineBaseTimeError::InvalidTime {
                row: row_idx,
                value: u,
            });
        }
        // Evaluate degree-(k+2) B-splines on τ̃ at u.
        for slot in row_b.iter_mut() {
            *slot = 0.0;
        }
        evaluate_bspline_basis_scalar(u, tau_ext.view(), bs_deg, &mut row_b, &mut row_scratch)
            .map_err(|e| ISplineBaseTimeError::Spline {
                reason: format!("row {row_idx} evaluation failed: {e}"),
            })?;
        // Right-cumulative T_m^{(k+2)}(u) for m = 0..n_b_inner-1, computed
        // in the same backward sweep as the anchor.
        let mut running_u = 0.0_f64;
        // Antiderivative-of-B per-m value: A_m(u) = α_m · (T_m(u) − T_m(anchor)).
        // Then I-spline cumulative gives:
        //   C_j(u) = Σ_{m = j+1}^{n_b_inner - 1} A_m(u),
        // which is the right-cumulative tail of A across `m` indexed by `j`.
        let mut a_running = 0.0_f64;
        // Walk i (and therefore m) from high to low so we can build both
        // running sums in one pass.
        // First, accumulate the T_m(u) tail for i = n_b_ext-1 down to 0.
        // Then for i < n_b_inner we form A_m(u) and accumulate the j-tail.
        // We keep a separate running tail per role: `running_u` over i
        // (degree-(k+2) basis), `a_running` over m (= i restricted to the
        // inner range, used to fill C_j).
        for i in (0..n_b_ext).rev() {
            running_u += row_b[i];
            if i < n_b_inner {
                let m = i;
                let t_m_u = running_u;
                let t_m_anchor = anchor_tail[m];
                let a_m = alpha[m] * (t_m_u - t_m_anchor);
                a_running += a_m;
                // C_j with j = m - 1 (since the inner sum runs m = j+1..)
                if m >= 1 {
                    let j = m - 1;
                    if j < n_i {
                        out[[row_idx, j]] = a_running;
                    }
                }
            }
        }
    }

    // Numerical floor matching `create_ispline_dense` / `create_ispline_derivative_dense`.
    for val in out.iter_mut() {
        if val.abs() <= 1e-15 {
            *val = 0.0;
        }
    }
    Ok(out)
}

/// Build the structurally-monotone time-block designs for a survival time
/// axis. Entry/exit values use the C-spline basis at `log t`; the exit
/// derivative is `(1/t) · I-spline(log t)`; the additive `guard·t` linear
/// part is returned as offset residuals for the caller to fold into the
/// existing offset slots (e.g. via `add_survival_time_derivative_guard_offset`
/// in the survival construction pipeline).
///
/// `age_entry` / `age_exit` are clock times (in the same units the
/// derivative guard is expressed in). `knots` lives on the `log t` axis
/// and must already include the clamped-boundary repeats expected by
/// `create_basis::<Dense>(.., BasisOptions::i_spline())`.
pub fn build_ispline_base_designs(
    age_entry: ArrayView1<'_, f64>,
    age_exit: ArrayView1<'_, f64>,
    log_floor: f64,
    knots: &Array1<f64>,
    degree: usize,
    derivative_guard: f64,
) -> Result<ISplineBaseDesigns, ISplineBaseTimeError> {
    let n_entry = age_entry.len();
    let n_exit = age_exit.len();
    if n_entry != n_exit {
        return Err(ISplineBaseTimeError::EntryExitMismatch {
            entry: n_entry,
            exit: n_exit,
        });
    }
    let n = n_entry;
    let mut log_entry = Array1::<f64>::zeros(n);
    let mut log_exit = Array1::<f64>::zeros(n);
    for i in 0..n {
        let t_in = age_entry[i];
        let t_out = age_exit[i];
        if !t_in.is_finite() || t_in < 0.0 {
            return Err(ISplineBaseTimeError::InvalidTime {
                row: i,
                value: t_in,
            });
        }
        if !t_out.is_finite() || t_out < 0.0 {
            return Err(ISplineBaseTimeError::InvalidTime {
                row: i,
                value: t_out,
            });
        }
        log_entry[i] = t_in.max(log_floor).ln();
        log_exit[i] = t_out.max(log_floor).ln();
    }

    let design_entry = cspline_basis_values(log_entry.view(), knots, degree)?;
    let design_exit = cspline_basis_values(log_exit.view(), knots, degree)?;
    // I-spline values at exit; derivative wrt clock time picks up the
    // chain-rule factor 1/t. `create_basis` returns dense via Arc.
    let (ispline_arc, _) = create_basis::<crate::terms::basis::Dense>(
        log_exit.view(),
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::i_spline(),
    )
    .map_err(|e| ISplineBaseTimeError::Spline {
        reason: format!("I-spline value evaluation failed at exit: {e}"),
    })?;
    let mut design_derivative_exit = ispline_arc.as_ref().clone();
    for i in 0..n {
        let chain = 1.0 / age_exit[i].max(log_floor);
        for j in 0..design_derivative_exit.ncols() {
            design_derivative_exit[[i, j]] *= chain;
        }
    }

    // Offset residuals: the `guard·t` and `guard` linear pieces.
    let mut offset_residual_entry = Array1::<f64>::zeros(n);
    let mut offset_residual_exit = Array1::<f64>::zeros(n);
    let mut derivative_offset_residual_exit = Array1::<f64>::zeros(n);
    for i in 0..n {
        offset_residual_entry[i] = derivative_guard * age_entry[i];
        offset_residual_exit[i] = derivative_guard * age_exit[i];
        derivative_offset_residual_exit[i] = derivative_guard;
    }

    Ok(ISplineBaseDesigns {
        design_entry,
        design_exit,
        design_derivative_exit,
        offset_residual_entry,
        offset_residual_exit,
        derivative_offset_residual_exit,
    })
}
