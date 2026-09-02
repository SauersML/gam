use crate::estimate::EstimationError;
use faer::linalg::solvers::SolveLstsq;
use faer::Side;
use gam_linalg::faer_ndarray::{
    FaerArrayView, FaerCholesky, FaerLinalgError, FaerSvd, array1_to_col_matmut,
    default_rrqr_rank_alpha, rrqr_nullspace_basis,
};
use gam_linalg::utils::{KahanSum, StableSolver, array_is_finite};
use gam_problem::{
    ConstraintRowId, ConstraintSet, KhatriRaoConeConstraints, LinearInequalityConstraints,
};
use ndarray::{Array1, Array2, ArrayView1, s};
use serde::{Deserialize, Serialize};
use std::cell::Cell;
use std::collections::HashSet;

/// Primal-feasibility tolerance the inequality-constrained active-set Newton
/// solver guarantees on its returned iterate, measured in the unit-normalized
/// constraint-row metric `(b_i − a_i·β)/‖a_i‖` that
/// [`ConstraintSet::max_scaled_violation`] computes.
///
/// The solver accepts a step when the worst scaled violation is at or below
/// this threshold (see the acceptance gate in
/// `solve_linear_constrained_newton_step` and the KKT diagnostics in
/// `compute_constraint_kkt_diagnostics`).
///
/// The definition lives with the metric, in `gam_problem::constraint_set`, and
/// is re-exported here under the solver-facing name so that the tolerance and
/// the quantity it bounds cannot drift apart. Every gate, validator and step
/// rule that decides "is this feasible" must reference this one value; four
/// sites once re-spelled the literal `1e-8` beside it and one step rule
/// demanded exact feasibility instead (gam#2719).
pub use gam_problem::PRIMAL_FEASIBILITY_TOL as ACTIVE_SET_PRIMAL_FEASIBILITY_TOL;

/// Scaled slack tolerance for membership in an active working face.
///
/// This is intentionally tighter than the public primal-feasibility contract:
/// a row may be numerically feasible without being an equality at the current
/// point. Warm-start and terminal face provenance both use this value so a QP
/// endpoint row cannot remain active after globalization accepts an interior
/// subsegment of the endpoint chord.
pub const ACTIVE_SET_WORKING_FACE_TOL: f64 = 1e-10;

/// Stationarity tolerance for the strong-KKT acceptance gate: the projected
/// (working-set) gradient residual ‖∇L − Aᵀλ‖∞, either absolute or relative to
/// `max(1, ‖∇L‖∞)`, must fall below this to certify a constrained stationary
/// point. Matched against `ACTIVE_SET_KKT_COMPLEMENTARITY_TOL` so both KKT
/// residual channels are certified at compatible scales.
const ACTIVE_SET_KKT_STATIONARITY_TOL: f64 = 2e-6;

/// Complementarity-slackness tolerance for the KKT acceptance gate:
/// `max_i |λ_i · slack_i|` must fall below this for the
/// active-inactive partition to be consistent.
pub(crate) const ACTIVE_SET_KKT_COMPLEMENTARITY_TOL: f64 = 1e-6;

/// Dual-feasibility tolerance for the KKT acceptance gate: every working-set
/// multiplier must satisfy `λ_i ≥ −ACTIVE_SET_KKT_DUAL_FEASIBILITY_TOL` (a
/// strictly-negative multiplier means the constraint should be released).
const ACTIVE_SET_KKT_DUAL_FEASIBILITY_TOL: f64 = 1e-8;

/// Relaxed stationarity tolerance accepted only on a *genuinely degenerate
/// boundary face* — one whose active rows are linearly dependent
/// (`rank(A_active) < n_active`), so the active-row multipliers are non-unique
/// and the exact projected gradient cannot reach
/// `ACTIVE_SET_KKT_STATIONARITY_TOL`. Still requires primal feasibility,
/// complementarity, and a relative-stationarity backstop.
///
/// Public so the outer REML / PIRLS validation gate can apply the same
/// relaxation when the diagnostic reports a rank-deficient active face — a
/// strict 5e-6 check there would otherwise refuse iterates that the inner
/// active-set solver legitimately certified via its own `degenerate_boundary_ok`
/// clause.
///
/// NOTE: this is *not* the mechanism that fixes the `shape=concave` /
/// `shape=convex` cold-vs-warm cache divergence (#873). The B-spline shape path
/// reparameterizes curvature into independent *coordinate lower bounds*
/// `γ_j ≥ 0` (see `shape_lower_bounds_local`); any subset of those active rows
/// is full rank, so `working_set_rank_deficient` stays `false` and this
/// relaxation never fires for them — and must not be widened to. That bug is a
/// *seed* problem (a cold seed landing on the cone vertex with every curvature
/// row tight); it is fixed at the source by
/// `project_point_strictly_into_feasible_cone`, which starts the inner solve
/// strictly inside the cone so the strict tolerance is reachable.
pub(crate) const ACTIVE_SET_KKT_DEGENERATE_STATIONARITY_TOL: f64 = 1e-3;

/// KKT diagnostics for inequality-constrained Newton subproblems.
///
/// Constraints are represented as `A * beta >= b` in the same coefficient
/// coordinate system as the returned `beta`.
///
/// **Invariants** (held by all producers; not enforced at consumer boundary):
/// - `n_active <= n_constraints` (a row cannot be active twice).
/// - All four residual components (`primal_feasibility`, `dual_feasibility`,
///   `complementarity`, `stationarity`) are `>= 0.0` and finite.
/// - `active_tolerance >= 0.0` and finite.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConstraintKktDiagnostics {
    /// Number of inequality rows.
    pub n_constraints: usize,
    /// Number of rows considered active (`slack <= active_tolerance`).
    pub n_active: usize,
    /// Maximum primal feasibility violation: `max_i max(0, b_i - a_i^T beta)`.
    pub primal_feasibility: f64,
    /// Maximum dual feasibility violation: `max_i max(0, -lambda_i)`.
    pub dual_feasibility: f64,
    /// Maximum complementarity residual: `max_i |lambda_i * slack_i|`.
    pub complementarity: f64,
    /// Stationarity residual: `||grad - A^T lambda||_inf`.
    pub stationarity: f64,
    /// Tolerance used to classify active constraints from slacks.
    pub active_tolerance: f64,
    /// `true` when the active rows are linearly dependent (`rank(A_active) <
    /// n_active`) — a *degenerate boundary face*. On such a face the active-row
    /// multipliers are non-unique and the strict stationarity tolerance is
    /// unreachable by construction. The inner active-set solver certifies these
    /// iterates via its `ACTIVE_SET_KKT_DEGENERATE_STATIONARITY_TOL` relaxation;
    /// the outer validation gate must consult this flag to apply the matching
    /// relaxation, or it will refuse a legitimately-converged constrained
    /// optimum and abort the REML startup loop.
    ///
    /// NOTE: B-spline `shape=concave`/`shape=convex` faces are *not* degenerate
    /// — that path reparameterizes curvature into independent coordinate lower
    /// bounds `γ_j ≥ 0` (full-rank active subsets), so this flag stays `false`
    /// for them. Their cold-start fragility is a seed problem fixed by the
    /// strictly-interior seed, not by this relaxation.
    #[serde(default)]
    pub working_set_rank_deficient: bool,
    /// `true` when the cone projector REFUSED rather than returning
    /// multipliers, so `lambda` was never computed and `stationarity` is the
    /// raw gradient by default rather than by measurement.
    ///
    /// Without this, a refusal is indistinguishable from an answer of exactly
    /// zero: both leave `lambda = 0`, both make `stationarity == ‖grad‖∞`, and
    /// both print `stat_rel = 1.000e0`. #2601 reports precisely that on a
    /// fully active convexity face, and the two readings call for opposite
    /// responses — a genuine `λ = 0` says the point is not stationary, while a
    /// refusal says nothing was measured at all.
    #[serde(default)]
    pub cone_projection_refused: bool,
    /// Inf-norm of the (raw, unprojected) gradient at `beta`, `‖gradient‖∞` —
    /// the natural scale of the stationarity residual. A converged constrained
    /// optimum drives `stationarity = ‖grad − Aᵀλ‖∞` to zero *relative to* this
    /// scale, not to a fixed absolute floor: the profiled REML latent objective
    /// carries an O(n) gradient magnitude even at a genuine stationary point
    /// (issue #879), so a bare absolute stationarity gate is unreachable there
    /// by construction. The inner active-set solver already certifies
    /// convergence on the scale-invariant ratio
    /// `stationarity / max(gradient_scale, 1)` (its `stationarity_rel` path
    /// against `ACTIVE_SET_KKT_STATIONARITY_TOL`); the outer validation gate
    /// [`crate::estimate::reml::outer_eval`]`::enforce_constraint_kkt` consults this
    /// field to apply the identical relative test, so the two stop on the same
    /// contract instead of the gate spuriously aborting a constrained optimum
    /// the solver legitimately reached (issue #989). Defaults to `0.0` when
    /// deserialized from a model saved before this field existed, which makes
    /// `max(gradient_scale, 1) = 1` and recovers the bare absolute test.
    #[serde(default)]
    pub gradient_scale: f64,
}

impl ConstraintKktDiagnostics {
    /// The note a rendered KKT verdict must carry when the cone projector
    /// REFUSED instead of returning multipliers.
    ///
    /// `stationarity` is `‖grad − Aᵀλ‖∞`. On a refusal `λ` was never computed,
    /// so the number reported in that slot is the raw gradient BY DEFAULT rather
    /// than by measurement — and two failures with opposite remedies then render
    /// identically: a computed `λ = 0` says this point is not stationary, a
    /// refusal says nothing was projected at all.
    ///
    /// [`Self::cone_projection_refused`] has recorded which since #2601, and
    /// until this accessor existed every reader of the numbers dropped it — the
    /// outer REML gate's `ParameterConstraintViolation` and the post-fit
    /// feasibility audit's `KKT[...]` suffix both rendered `stat` with no way to
    /// say whether it had been projected. A flag that cannot change what anyone
    /// sees is not a diagnostic.
    ///
    /// Returns the separator too, so a non-refusal contributes nothing rather
    /// than a dangling one.
    pub fn cone_projection_note(&self) -> &'static str {
        if self.cone_projection_refused {
            "; cone_projection=REFUSED (no multipliers computed: stat is the UNPROJECTED gradient, \
             not a residual)"
        } else {
            ""
        }
    }
}

/// Inf-norm `‖g‖∞` used as the scale of the stationarity residual in the
/// relative KKT criterion shared by the inner active-set solver and the outer
/// validation gate (see [`ConstraintKktDiagnostics::gradient_scale`]).
fn gradient_inf_norm(gradient: &Array1<f64>) -> f64 {
    gradient.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()))
}

fn solve_newton_direction_dense(
    hessian: &Array2<f64>,
    gradient: &Array1<f64>,
    direction_out: &mut Array1<f64>,
) -> Result<(), EstimationError> {
    if direction_out.len() != gradient.len() {
        *direction_out = Array1::zeros(gradient.len());
    }

    let factor = StableSolver::new()
        .factorize(hessian)
        .map_err(EstimationError::LinearSystemSolveFailed)?;
    direction_out.assign(gradient);
    let mut rhsview = array1_to_col_matmut(direction_out);
    factor.solve_in_place(rhsview.as_mut());
    direction_out.mapv_inplace(|v| -v);
    if array_is_finite(direction_out) {
        return Ok(());
    }
    Err(EstimationError::LinearSystemSolveFailed(
        FaerLinalgError::FactorizationFailed {
            context: "active-set newton direction non-finite solve",
        },
    ))
}

fn solve_dense_system_via_pseudoinverse(
    matrix: &Array2<f64>,
    rhs: &Array1<f64>,
    out: &mut Array1<f64>,
) -> Result<(), EstimationError> {
    if matrix.nrows() != matrix.ncols() || rhs.len() != matrix.nrows() {
        crate::bail_invalid_estim!("dense pseudoinverse solve dimension mismatch");
    }

    let (u_opt, singular, vt_opt) = matrix.svd(true, true).map_err(|_| {
        EstimationError::InvalidInput("dense pseudoinverse solve SVD failed".to_string())
    })?;
    let (Some(u), Some(vt)) = (u_opt, vt_opt) else {
        crate::bail_invalid_estim!("dense pseudoinverse solve missing singular vectors");
    };

    let max_singular = singular.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let tol = 100.0
        * f64::EPSILON
        * (matrix.nrows().max(matrix.ncols()).max(1) as f64)
        * max_singular.max(1.0);
    let mut coeff = u.t().dot(rhs);
    for (idx, value) in coeff.iter_mut().enumerate() {
        let sigma = singular[idx];
        if sigma.abs() > tol {
            *value /= sigma;
        } else {
            *value = 0.0;
        }
    }
    let solution = vt.t().dot(&coeff);
    if !array_is_finite(&solution) {
        crate::bail_invalid_estim!("dense pseudoinverse solve produced non-finite values");
    }
    if out.len() != solution.len() {
        *out = Array1::zeros(solution.len());
    }
    out.assign(&solution);
    Ok(())
}

/// Least-squares `min_z ‖A z − b‖` for `A` of shape `(p, k)` and rhs `b`
/// (length `p`), returning `z` (length `k`) or `None` on numerical failure.
///
/// - Tall or square (`p ≥ k`): the rank-revealing col-pivoted QR (faer
///   `solve_lstsq`) — the exact prior behavior, byte-for-byte.
/// - Wide (`k > p`): the system is underdetermined. This arises on a DEGENERATE
///   active face where more constraint rows are active than the problem has
///   dimensions — e.g. a monotone coefficient cone plus many binding per-row
///   derivative guards. The minimum-norm solution `z = Aᵀ (A Aᵀ)⁺ b` is taken
///   via the SVD pseudoinverse of the square (possibly rank-deficient) Gram
///   `A Aᵀ`, matching what a wide-capable least-squares would return.
///
/// Faer's `solve_lstsq` asserts `nrows ≥ ncols`, so feeding it a wide matrix
/// panics — and here that panic would cross the Rust/Python FFI boundary,
/// violating the typed-error contract. Routing the wide case through this helper
/// keeps the failure typed: callers receive `None` and treat it as
/// "not certified" (conservative), never a process abort.
fn least_squares_min_norm_any_shape(a: &Array2<f64>, b: &Array1<f64>) -> Option<Array1<f64>> {
    let p = a.nrows();
    let k = a.ncols();
    if b.len() != p {
        return None;
    }
    if k == 0 {
        return Some(Array1::zeros(0));
    }
    if k <= p {
        let mut rhs = Array2::<f64>::zeros((p, 1));
        rhs.column_mut(0).assign(b);
        let a_view = FaerArrayView::new(a);
        let rhs_view = FaerArrayView::new(&rhs);
        let solved = a_view.as_ref().col_piv_qr().solve_lstsq(rhs_view.as_ref());
        let mut z = Array1::<f64>::zeros(k);
        for c in 0..k {
            let value = solved[(c, 0)];
            if !value.is_finite() {
                return None;
            }
            z[c] = value;
        }
        Some(z)
    } else {
        // Underdetermined: min-norm `z = Aᵀ (A Aᵀ)⁺ b`. `A Aᵀ` is `p × p`, so it
        // satisfies the square precondition of the SVD pseudoinverse solve, and
        // the pseudoinverse absorbs the rank deficiency of an over-complete face.
        let gram = a.dot(&a.t());
        let mut y = Array1::<f64>::zeros(p);
        solve_dense_system_via_pseudoinverse(&gram, b, &mut y).ok()?;
        let z = a.t().dot(&y);
        if z.iter().any(|value| !value.is_finite()) {
            return None;
        }
        Some(z)
    }
}

/// How much of a refused stationarity residual is even *closable* by a
/// multiplier, and how much is not.
///
/// The KKT stationarity residual `‖grad − Aᵀλ‖∞` is one number, and on its own
/// it cannot distinguish two completely different failures:
///
/// * **Unreachable** — the residual has a component ORTHOGONAL to
///   `span(A_activeᵀ)`. No multiplier vector, of any sign, can remove it: the
///   iterate is non-stationary along a direction the constraints do not touch,
///   i.e. the inner solve simply has not converged.
/// * **Blocked** — the residual lies inside `span(A_activeᵀ)` but cannot be
///   written with `λ ≥ 0`. The point is stationary in the free directions and
///   is pressed against the WRONG side of the cone; the working face is wrong,
///   not the convergence.
///
/// Reported only where a verdict is actually rendered (the refusal message), so
/// the O(rank · n_active · p) orthogonalization never runs on the hot path.
/// Returns `(unreachable_inf, blocked_inf)`; `None` when the geometry cannot be
/// formed (dimension mismatch, no active rows).
pub(crate) fn stationarity_residual_reachability(
    beta: &Array1<f64>,
    gradient: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
) -> Option<(f64, f64)> {
    let p = constraints.a.ncols();
    if beta.len() != p || gradient.len() != p {
        return None;
    }
    let face = active_face(beta, constraints)?;
    if face.active_idx.is_empty() {
        // Every direction is free: the whole residual is unreachable.
        let inf = gradient_inf_norm(gradient);
        return Some((inf, 0.0));
    }
    let (_, lambda_active) =
        project_stationarity_residual_on_constraint_cone(gradient, &face.a_active)?;
    let mut residual = gradient.to_owned();
    for (r, &value) in lambda_active.iter().enumerate() {
        if value != 0.0 {
            residual.scaled_add(-value, &face.a_active.row(r));
        }
    }

    // Orthonormal basis of the active ROW space by modified Gram–Schmidt. The
    // basis is at most `p`-dimensional, so the scan stops as soon as it is
    // complete no matter how many rows are active.
    let mut basis: Vec<Array1<f64>> = Vec::new();
    let drop_tol = 1e-12;
    for r in 0..face.a_active.nrows() {
        if basis.len() == p {
            break;
        }
        let mut v = face.a_active.row(r).to_owned();
        for q in &basis {
            let projection = q.dot(&v);
            v.scaled_add(-projection, q);
        }
        let norm = v.dot(&v).sqrt();
        if norm > drop_tol {
            v.mapv_inplace(|value| value / norm);
            basis.push(v);
        }
    }

    let mut orthogonal = residual.clone();
    for q in &basis {
        let projection = q.dot(&residual);
        orthogonal.scaled_add(-projection, q);
    }
    let unreachable = gradient_inf_norm(&orthogonal);
    let in_row_space = &residual - &orthogonal;
    Some((unreachable, gradient_inf_norm(&in_row_space)))
}

/// The active face at `beta`: per-row-scaled inequalities, their slacks, the
/// indices considered active, and the gathered active rows.
///
/// One derivation shared by [`compute_constraint_kkt_diagnostics`] and
/// [`stationarity_residual_reachability`], so the reported reachability split
/// can never describe a different face than the residual it explains.
struct ActiveFace {
    a_scaled: Array2<f64>,
    slack: Array1<f64>,
    primal_feasibility: f64,
    active_idx: Vec<usize>,
    a_active: Array2<f64>,
}

/// The unit-scaled rows of the constraints that BIND at `(beta, gradient)`:
/// primal-active rows whose multiplier in the Moreau cone projection of the
/// gradient is strictly positive. `k × p`, `k` possibly zero; `None` on a width
/// mismatch or when the projection refuses.
///
/// This is the face an exact Newton refinement of a constrained KKT point has
/// to stay on: with the binding multipliers carried by `∇L − Aᵀλ`, the residual
/// that is left to zero lives in `null(A_binding)`, so the refinement step is
/// the Newton step of the objective restricted to that null space.
///
/// A primal-active row with a zero multiplier is not part of the KKT face: the
/// stationarity residual it leaves is the full gradient component along its
/// normal, and a Newton step confined to that row's face cannot remove it.
/// Treating the row as free lets the refinement move along its normal — inward
/// when the objective wants to leave the boundary — while the caller's primal
/// feasibility check still refuses any step that would cross it.
pub(crate) fn binding_constraint_rows(
    beta: &Array1<f64>,
    gradient: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
) -> Option<Array2<f64>> {
    let face = active_face(beta, constraints)?;
    let p = constraints.a.ncols();
    if face.active_idx.is_empty() || gradient.len() != p {
        return Some(Array2::<f64>::zeros((0, p)));
    }
    let (_, lambda_active) =
        project_stationarity_residual_on_constraint_cone(gradient, &face.a_active)?;
    let keep: Vec<usize> = (0..face.active_idx.len())
        .filter(|&row| lambda_active.get(row).is_some_and(|value| *value > 0.0))
        .collect();
    let mut rows = Array2::<f64>::zeros((keep.len(), p));
    for (position, &row) in keep.iter().enumerate() {
        rows.row_mut(position).assign(&face.a_active.row(row));
    }
    Some(rows)
}

fn active_face(
    beta: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
) -> Option<ActiveFace> {
    let m = constraints.a.nrows();
    let p = constraints.a.ncols();
    if beta.len() != p {
        return None;
    }
    // Measure feasibility in the *scaled* (geometric) coordinate system the
    // solver's tolerance is expressed in — see the note in
    // `compute_constraint_kkt_diagnostics`.
    let mut a_scaled = constraints.a.clone();
    let mut b_scaled = constraints.b.clone();
    for i in 0..m {
        let n_i = constraints.a.row(i).dot(&constraints.a.row(i)).sqrt();
        if n_i > 0.0 {
            let inv = 1.0 / n_i;
            a_scaled.row_mut(i).mapv_inplace(|v| v * inv);
            b_scaled[i] *= inv;
        }
    }
    let mut slack = Array1::<f64>::zeros(m);
    let mut primal_feasibility: f64 = 0.0;
    for i in 0..m {
        let s_i = a_scaled.row(i).dot(beta) - b_scaled[i];
        slack[i] = s_i;
        primal_feasibility = primal_feasibility.max((-s_i).max(0.0));
    }
    let active_idx: Vec<usize> = (0..m)
        .filter(|&i| slack[i] <= ACTIVE_SET_PRIMAL_FEASIBILITY_TOL)
        .collect();
    let mut a_active = Array2::<f64>::zeros((active_idx.len(), p));
    for (r, &idx) in active_idx.iter().enumerate() {
        a_active.row_mut(r).assign(&a_scaled.row(idx));
    }
    Some(ActiveFace {
        a_scaled,
        slack,
        primal_feasibility,
        active_idx,
        a_active,
    })
}

pub(crate) fn compute_constraint_kkt_diagnostics(
    beta: &Array1<f64>,
    gradient: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
) -> ConstraintKktDiagnostics {
    let m = constraints.a.nrows();
    let active_tolerance = ACTIVE_SET_PRIMAL_FEASIBILITY_TOL;

    // Feasibility is measured in the *scaled* (geometric) coordinate system the
    // solver's tolerance is expressed in — see `active_face`, which owns that
    // derivation for this function and for
    // [`stationarity_residual_reachability`].
    let p = constraints.a.ncols();
    let Some(ActiveFace {
        a_scaled,
        slack,
        primal_feasibility,
        active_idx,
        a_active,
    }) = active_face(beta, constraints)
    else {
        // `beta` does not match the constraint system's coefficient width. No
        // face can be formed, so no KKT claim can be made: report the raw
        // gradient scale and an empty active set rather than inventing one.
        return ConstraintKktDiagnostics {
            n_constraints: m,
            n_active: 0,
            primal_feasibility: f64::INFINITY,
            dual_feasibility: 0.0,
            complementarity: 0.0,
            stationarity: gradient_inf_norm(gradient),
            active_tolerance,
            working_set_rank_deficient: false,
            cone_projection_refused: false,
            gradient_scale: gradient_inf_norm(gradient),
        };
    };

    let mut lambda = Array1::<f64>::zeros(m);
    let mut working_set_rank_deficient = false;
    // A refusal and an answer of zero are not the same fact.
    //
    // `project_stationarity_residual_on_constraint_cone` returns `None` on a
    // non-finite target, a width mismatch, or the Lawson-Hanson `3m + 30`
    // guard being reached -- and this `if let` used to leave `lambda` at its
    // zeros in every one of those cases, with no trace. That is
    // indistinguishable from NNLS genuinely reporting `λ = 0`: both give
    // `stationarity == ‖grad‖∞` and print `stat_rel = 1.000e0`.
    //
    // #2601 reports exactly that on a fully active convexity face. A measured
    // check (`the_kkt_cone_convention_is_grad_equals_a_transpose_lambda_2601`)
    // rules out the obvious explanation: a gradient in the POLAR cone still
    // recruits rows and leaves `stat_rel = 0.5` on that face, not 1.0. So a
    // flipped sign cannot produce the observed number and a silent refusal can.
    let mut cone_projection_refused = false;
    if !active_idx.is_empty() {
        let n_active = active_idx.len();
        match project_stationarity_residual_on_constraint_cone(gradient, &a_active) {
            Some((_, lambda_active)) => {
                for (r, &idx) in active_idx.iter().enumerate() {
                    lambda[idx] = lambda_active[r];
                }
            }
            None => cone_projection_refused = true,
        }
        // Rank-deficiency detection on the (scaled) active rows. Per-row
        // positive scaling is rank-preserving, so this answers the same
        // question the finite dual solver's reduced face does —
        // `rank(A_active) < n_active`. For curvature
        // constraints the second-difference operator forces dependence
        // whenever more than `p` rows bind, and for monotonicity the
        // first-difference operator does so beyond a similar count. The
        // diagnostic exposes the flag so the outer validation gate can apply
        // the same `ACTIVE_SET_KKT_DEGENERATE_STATIONARITY_TOL` relaxation
        // the inner solver does, instead of refusing the iterate at strict
        // `ACTIVE_SET_KKT_STATIONARITY_TOL`.
        working_set_rank_deficient = if n_active > p {
            true
        } else if n_active > 1 {
            let groups: Vec<Vec<usize>> = (0..n_active).map(|i| vec![i]).collect();
            let b_dummy = Array1::<f64>::zeros(n_active);
            let (reduced_a, _, _, _) =
                rank_reduce_rows_pivoted_qr_with_dependence(a_active, b_dummy, groups);
            reduced_a.nrows() < n_active
        } else {
            false
        };
    }

    let mut dual_feasibility: f64 = 0.0;
    let mut complementarity: f64 = 0.0;
    for i in 0..m {
        dual_feasibility = dual_feasibility.max((-lambda[i]).max(0.0));
        complementarity = complementarity.max((lambda[i] * slack[i]).abs());
    }
    let stationarity = {
        let mut resid = gradient.to_owned();
        resid -= &a_scaled.t().dot(&lambda);
        resid.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()))
    };

    ConstraintKktDiagnostics {
        n_constraints: m,
        n_active: active_idx.len(),
        primal_feasibility,
        dual_feasibility,
        complementarity,
        stationarity,
        active_tolerance,
        working_set_rank_deficient,
        cone_projection_refused,
        gradient_scale: gradient_inf_norm(gradient),
    }
}

/// Operator-native Lawson–Hanson projection onto a finitely generated cone.
///
/// `row_values(r)` returns every raw row product `A r`; `gather_rows(ids)`
/// materializes only the named rows. The passive set of an NNLS solution has
/// at most coefficient-space rank, so a factored cone can scan millions of
/// generators while gathering only `O(p²)` storage. Entering rows use ascending
/// original row id as the exact-tie break, so the unique projection is
/// independent of warm-start history.
fn nonnegative_cone_projection_by_rows<RowValues, GatherRows>(
    row_norms: &[f64],
    target: &Array1<f64>,
    row_values: RowValues,
    gather_rows: GatherRows,
) -> Option<(Vec<(usize, f64)>, Array1<f64>)>
where
    RowValues: Fn(&Array1<f64>) -> Option<Array1<f64>>,
    GatherRows: Fn(&[usize]) -> Option<Array2<f64>>,
{
    let p = target.len();
    let m = row_norms.len();
    if m == 0 {
        return Some((Vec::new(), target.clone()));
    }
    if target.iter().any(|v| !v.is_finite())
        || row_norms
            .iter()
            .any(|norm| !norm.is_finite() || *norm < 0.0)
    {
        return None;
    }
    let target_inf = target.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    if target_inf == 0.0 {
        return Some((Vec::new(), target.clone()));
    }
    // Gradient tolerance in λ-space: with unit rows, `w_i = a_î·r` is bounded
    // by ‖r‖, so a relative band on the target scale is dimensionless.
    let tol_w = 1e-10 * target_inf;
    let lambda_floor = 1e-14 * target_inf;

    let mut lambda_unit = Array1::<f64>::zeros(m);
    let mut passive: Vec<usize> = Vec::new();
    let mut in_passive = vec![false; m];
    let mut residual = target.clone();
    // Rows whose trial coefficient collapsed to zero at the current residual;
    // re-eligible as soon as the residual moves. Prevents an add/drop loop on
    // exactly degenerate geometry.
    let mut banned = vec![false; m];

    let solve_passive = |passive: &[usize]| -> Option<Array1<f64>> {
        let k = passive.len();
        // `design` is `p × k` (each column is a unit active row). On a degenerate
        // over-complete face `k` can exceed `p` (more active rows than
        // dimensions); the min-norm least-squares helper handles that wide case
        // instead of panicking inside faer's tall-only `solve_lstsq`.
        let mut design = Array2::<f64>::zeros((p, k));
        let rows = gather_rows(passive)?;
        if rows.nrows() != k || rows.ncols() != p || rows.iter().any(|value| !value.is_finite()) {
            return None;
        }
        for (col, &row) in passive.iter().enumerate() {
            let norm = row_norms[row];
            if !(norm > 0.0) {
                return None;
            }
            design
                .column_mut(col)
                .assign(&(&rows.row(col) / norm));
        }
        least_squares_min_norm_any_shape(&design, target)
    };

    let max_outer = m.saturating_mul(3).saturating_add(30);
    for _ in 0..max_outer {
        // Most-ascent candidate among non-passive, non-banned rows.
        let values = row_values(&residual)?;
        if values.len() != m || values.iter().any(|value| !value.is_finite()) {
            return None;
        }
        let mut best: Option<(usize, f64)> = None;
        for i in 0..m {
            if in_passive[i] || banned[i] || row_norms[i] <= 0.0 {
                continue;
            }
            let w = values[i] / row_norms[i];
            if w > tol_w && best.map(|(_, best_w)| w > best_w).unwrap_or(true) {
                best = Some((i, w));
            }
        }
        let Some((entering, _)) = best else {
            break;
        };
        passive.push(entering);
        in_passive[entering] = true;

        let mut inner_ok = false;
        for _ in 0..(m + 2) {
            let Some(z) = solve_passive(&passive) else {
                return None;
            };
            let min_z = z.iter().copied().fold(f64::INFINITY, f64::min);
            if min_z > lambda_floor {
                for (pos, &row) in passive.iter().enumerate() {
                    lambda_unit[row] = z[pos];
                }
                inner_ok = true;
                break;
            }
            // Interpolate toward z until the first coefficient hits zero,
            // then drop every zeroed row from the passive set.
            let mut alpha = 1.0_f64;
            for (pos, &row) in passive.iter().enumerate() {
                if z[pos] <= lambda_floor {
                    let current = lambda_unit[row];
                    let denom = current - z[pos];
                    if denom > 0.0 {
                        alpha = alpha.min((current / denom).clamp(0.0, 1.0));
                    } else {
                        alpha = 0.0;
                    }
                }
            }
            for (pos, &row) in passive.iter().enumerate() {
                lambda_unit[row] += alpha * (z[pos] - lambda_unit[row]);
            }
            let mut retained = Vec::with_capacity(passive.len());
            for &row in &passive {
                if lambda_unit[row] > lambda_floor {
                    retained.push(row);
                } else {
                    lambda_unit[row] = 0.0;
                    in_passive[row] = false;
                    // The row failed at THIS residual; ban it until the
                    // residual moves so a degenerate add/drop pair cannot
                    // cycle within one outer round.
                    banned[row] = true;
                }
            }
            if retained.len() == passive.len() {
                // Nothing dropped despite a non-positive trial coefficient:
                // numerically stuck; stop refining this passive set.
                inner_ok = true;
                for (pos, &row) in passive.iter().enumerate() {
                    lambda_unit[row] = z[pos].max(0.0);
                }
                break;
            }
            passive = retained;
            if passive.is_empty() {
                break;
            }
        }
        // Refresh the residual; any movement re-enables banned rows.
        let mut fitted = Array1::<f64>::zeros(p);
        let passive_rows = gather_rows(&passive)?;
        if passive_rows.nrows() != passive.len()
            || passive_rows.ncols() != p
            || passive_rows.iter().any(|value| !value.is_finite())
        {
            return None;
        }
        for (position, &row) in passive.iter().enumerate() {
            fitted.scaled_add(
                lambda_unit[row] / row_norms[row],
                &passive_rows.row(position),
            );
        }
        let new_residual = target - &fitted;
        let moved = new_residual
            .iter()
            .zip(residual.iter())
            .any(|(a, b)| (a - b).abs() > 1e-15 * target_inf);
        residual = new_residual;
        if moved {
            banned.iter_mut().for_each(|b| *b = false);
        } else if !inner_ok {
            break;
        }
    }

    // Exact Moreau/KKT exit: the residual must lie in the polar cone, i.e.
    // every unit generator has non-positive correlation (within the same
    // scale-relative tolerance used for entering). This distinguishes normal
    // Lawson–Hanson termination from exhausting the floating-point pivot cap;
    // a capped non-polar iterate is not a projection and must never reach a
    // stationarity certificate or projected-gradient direction.
    let final_values = row_values(&residual)?;
    if final_values.len() != m
        || final_values.iter().any(|value| !value.is_finite())
        || (0..m).any(|row| {
            row_norms[row] > 0.0 && final_values[row] / row_norms[row] > tol_w
        })
    {
        return None;
    }

    let multipliers: Vec<(usize, f64)> = passive
        .into_iter()
        .filter_map(|row| {
            let lambda = lambda_unit[row] / row_norms[row];
            (lambda > 0.0).then_some((row, lambda))
        })
        .collect();
    if multipliers.iter().any(|(_, value)| !value.is_finite())
        || !array_is_finite(&residual)
    {
        return None;
    }
    Some((multipliers, residual))
}

/// Lawson–Hanson nonnegative least squares onto a dense finitely generated
/// cone.
///
/// Solves `min_{λ ≥ 0} ‖rowsᵀ λ − target‖₂` for a row block `rows` (`m × p`,
/// original row units) and returns `(λ, projected)` with
/// `projected = target − rowsᵀ λ`. By the Moreau decomposition `rowsᵀ λ` is
/// the Euclidean projection of `target` onto the cone generated by the rows,
/// so `projected` is the projection onto that cone's polar.
///
/// This is the existence-form dual-feasibility certificate for degenerate
/// working faces: multipliers on a rank-deficient face are non-unique, and
/// any single reconstruction (KKT least-squares, per-group attribution) can
/// carry huge canceling ± components — reporting `dual ≫ 0` at a point where
/// a different `λ ≥ 0` closes stationarity exactly (#2298 survival
/// monotonicity faces, #979 CTN Khatri–Rao faces). NNLS answers the right
/// question: does ANY nonnegative multiplier close stationarity?
///
/// Rows are unit-normalized internally so pivot ordering and tolerances are
/// scale-invariant; the returned `λ` is in original row units. Zero rows
/// carry `λ = 0`. Classic LH terminates after finitely many passive-set
/// changes; a `3m + 30` outer guard bounds float pathologies, and the terminal
/// full-row polarity check refuses rather than returning a non-KKT iterate if
/// that guard is ever reached.
pub(crate) fn nonnegative_cone_multipliers(
    rows: &Array2<f64>,
    target: &Array1<f64>,
) -> Option<(Array1<f64>, Array1<f64>)> {
    let p = target.len();
    let m = rows.nrows();
    if rows.ncols() != p {
        return None;
    }
    let norms: Vec<f64> = (0..m)
        .map(|row| rows.row(row).dot(&rows.row(row)).sqrt())
        .collect();
    let (sparse, projected) = nonnegative_cone_projection_by_rows(
        &norms,
        target,
        |residual| Some(rows.dot(residual)),
        |ids| {
            let mut gathered = Array2::<f64>::zeros((ids.len(), p));
            for (position, &row) in ids.iter().enumerate() {
                gathered.row_mut(position).assign(&rows.row(row));
            }
            Some(gathered)
        },
    )?;
    let mut lambda = Array1::<f64>::zeros(m);
    for (row, value) in sparse {
        lambda[row] = value;
    }
    Some((lambda, projected))
}

pub fn project_stationarity_residual_on_constraint_cone(
    residual: &Array1<f64>,
    active_a: &Array2<f64>,
) -> Option<(Array1<f64>, Array1<f64>)> {
    let p = residual.len();
    if active_a.ncols() != p {
        return None;
    }
    if active_a.nrows() == 0 {
        return Some((residual.clone(), Array1::zeros(0)));
    }
    // Projection onto a finitely generated cone IS nonnegative least squares
    // (Moreau): `projected = residual − Aᵀλ*` with
    // `λ* = argmin_{λ≥0} ‖residual − Aᵀλ‖`. Use that definition directly.
    // The former two-algorithm cascade first ran a primal working-set QP and
    // silently substituted Lawson–Hanson when it cycled. A stationarity
    // projector must be one deterministic map, not a success-dependent choice
    // between algorithms (#2432).
    nonnegative_cone_multipliers(active_a, residual).map(|(lambda, projected)| (projected, lambda))
}

pub(crate) fn feasible_point_for_linear_constraints(
    constraints: &LinearInequalityConstraints,
    p: usize,
) -> Option<Array1<f64>> {
    if constraints.a.ncols() != p
        || constraints.a.nrows() == 0
        || constraints.b.len() != constraints.a.nrows()
    {
        return None;
    }
    // The zero-vector shortcut must compare `b` in GEOMETRIC (per-row-scaled)
    // units: on raw `b` alone, `1e-20·β ≥ 1e-20` — the same half-space as
    // `β ≥ 1` — would accept `β = 0`. A numerically-zero row is vacuous when
    // `b_i ≤ 0` and infeasible (no seed exists) when `b_i > 0`.
    let mut all_scaled_b_tiny = true;
    for i in 0..constraints.a.nrows() {
        let norm = constraints.a.row(i).dot(&constraints.a.row(i)).sqrt();
        if norm > 0.0 {
            if constraints.b[i].abs() > 1e-14 * norm {
                all_scaled_b_tiny = false;
            }
        } else if constraints.b[i] > 0.0 {
            return None;
        }
    }
    if all_scaled_b_tiny {
        return Some(Array1::zeros(p));
    }

    let gram = constraints.a.dot(&constraints.a.t());
    let (u_opt, singular, vt_opt) = gram.svd(true, true).ok()?;
    let (Some(u), Some(vt)) = (u_opt, vt_opt) else {
        return None;
    };
    let max_singular = singular.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    // Rank tolerance relative to the LARGEST singular value only — an absolute
    // `max(σ_max, 1)` floor declares a uniformly small (but perfectly
    // well-conditioned) system rank-deficient purely because of its units.
    let tol = 100.0 * f64::EPSILON * constraints.a.nrows().max(1) as f64 * max_singular;
    let mut coeff = u.t().dot(&constraints.b);
    for (idx, value) in coeff.iter_mut().enumerate() {
        let sigma = singular[idx];
        if sigma.abs() > tol {
            *value /= sigma;
        } else {
            *value = 0.0;
        }
    }
    let dual = vt.t().dot(&coeff);
    let beta = constraints.a.t().dot(&dual);
    if beta.len() != p || beta.iter().any(|v| !v.is_finite()) {
        return None;
    }
    // Accept on per-row GEOMETRIC slack (raw slack over ‖a_i‖), the same
    // scale-invariant metric the active-set gates use.
    let feasible = (0..constraints.a.nrows()).all(|i| {
        let norm = constraints.a.row(i).dot(&constraints.a.row(i)).sqrt();
        if norm > 0.0 {
            (constraints.a.row(i).dot(&beta) - constraints.b[i]) / norm >= -1e-8
        } else {
            constraints.b[i] <= 0.0
        }
    });
    if feasible { Some(beta) } else { None }
}

/// Strictly-interior margin (in per-row geometric / scaled-slack units) required
/// of the projected cold-start seed produced by
/// [`project_point_strictly_into_feasible_cone`]. Each constraint row is shifted
/// to `a_iᵀβ ≥ b_i + ACTIVE_SET_INTERIOR_SEED_MARGIN·‖a_i‖` so that, scaled by
/// `‖a_i‖`, every row of the returned seed has slack `≥` this margin. The value
/// is far above the active-set activation threshold (`tol_active = 1e-10`) so the
/// initial working set the QP step solver builds from the seed is **empty** — no
/// row is mistaken for "on the boundary" — yet small enough that the seed stays a
/// negligible distance from the data-driven projection it is derived from.
const ACTIVE_SET_INTERIOR_SEED_MARGIN: f64 = 1e-6;

/// The strictly-interior cold-start margin (scaled-slack units) that
/// [`project_point_strictly_into_feasible_cone`] guarantees on its returned
/// seed. Exposed so the P-IRLS seed builder can decide, on the same scale,
/// whether the current seed is already strictly interior (and may be used as-is)
/// or sits on / outside the cone boundary (and must be projected).
#[inline]
pub(crate) fn interior_seed_margin() -> f64 {
    ACTIVE_SET_INTERIOR_SEED_MARGIN
}

/// Maximum nesting depth of the strictly-interior feasibility repair before the
/// solver stops re-projecting and surfaces an honest constraint-violation error.
///
/// The operator strict-interior projection and its feasibility repair are
/// mutually recursive: a failed projected-gradient repair can request another
/// inward-shifted identity-metric solve. On a well-conditioned cone the repair
/// converges at depth 0–1. But on near-anti-parallel rows (the clamped / anchored
/// monotone time-warp constraints an interval-censored survival fit emits, which
/// are only *near* — not exactly — anti-parallel and so slip past the zero-width
/// equality lift below), successive inward shifts may never certify. A cone that
/// cannot be certified within this many levels is degenerate; the projection
/// surfaces [`EstimationError::ParameterConstraintViolation`] instead of
/// exhausting the worker stack. Dense strict QPs no longer participate in this
/// cycle: they use the finite dual metric projection directly (#2432).
const MAX_FEASIBILITY_REPAIR_DEPTH: u32 = 16;

thread_local! {
    /// Current nesting depth of the `solve ↔ project` feasibility-repair cycle on
    /// this thread. Every recursive projected-gradient repair re-enters one of
    /// the strict-interior projection entry points, so the shared counter bounds
    /// the whole cycle. Per-thread because each solve runs to completion on a
    /// single call stack; independent solves on other worker threads carry their
    /// own counter.
    static FEASIBILITY_REPAIR_DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// RAII depth counter for the feasibility-repair recursion. [`enter`] increments
/// the per-thread depth and returns a guard whose `Drop` restores it on every
/// exit path — including the projection's many `return None` branches — so the
/// counter can never leak. It yields `None` once
/// [`MAX_FEASIBILITY_REPAIR_DEPTH`] is reached, so the caller bails out of the
/// recursion instead of descending another level.
///
/// [`enter`]: FeasibilityRepairGuard::enter
struct FeasibilityRepairGuard;

impl FeasibilityRepairGuard {
    fn enter() -> Option<Self> {
        FEASIBILITY_REPAIR_DEPTH.with(|depth| {
            let current = depth.get();
            if current >= MAX_FEASIBILITY_REPAIR_DEPTH {
                None
            } else {
                depth.set(current + 1);
                Some(Self)
            }
        })
    }
}

impl Drop for FeasibilityRepairGuard {
    fn drop(&mut self) {
        FEASIBILITY_REPAIR_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

/// Project `point` to a *strictly interior* feasible point of the polyhedron
/// `{β : A·β ≥ b}`: the solution of `min_β ½‖β − point‖²` subject to the
/// margin-shifted system `A·β ≥ b + δ·‖a_i‖`, with `δ =
/// ACTIVE_SET_INTERIOR_SEED_MARGIN`.
///
/// This is the principled feasible cold-start seed for a shape-constrained
/// (convex / concave / monotone) smooth. It is qualitatively different from
/// `feasible_point_for_linear_constraints`, which returns the *minimum-norm*
/// feasible point — for a homogeneous cone (`b = 0`, as the second-difference
/// convexity / concavity constraints are) that minimum-norm point is the cone
/// **vertex** `β = 0` (a flat line) where every constraint row is tight. A
/// shape-constrained P-IRLS launched from that vertex hands the inner active-set
/// QP an all-rows-active working set (every row's slack is `0`), and the QP then
/// stalls on a degenerate, non-stationary face of the cone. The fit's success
/// then depends on whether a warm-start seed happens to drop it into the right
/// basin, so the same fit silently diverges (or aborts) between a cold and a
/// warm cache (#873).
///
/// Requiring a strictly-positive margin on every row makes the returned seed an
/// interior point: the QP step solver starts from an **empty** active set and
/// adds only the genuinely binding rows, converging to the certified constrained
/// stationary point regardless of cache state. The projection is the
/// identity-Hessian instance of [`solve_quadratic_with_linear_constraints`]
/// (`H = I`, `rhs = point` ⇒ minimizing `½‖β − point‖²`), so the interior seed is
/// also the *nearest* strictly-interior point to the supplied data-driven
/// `point` — it inherits whatever curvature `point` already carries. Returns
/// `None` if the constraints are malformed or the QP cannot certify a feasible
/// solution.
pub fn project_point_strictly_into_feasible_cone(
    point: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
) -> Option<Array1<f64>> {
    // Bound the mutually-recursive `solve ↔ project` feasibility repair. Every
    // recursion path re-enters here, so a too-deep call returns `None` (a
    // degenerate cone the strictly-interior QP cannot certify) instead of
    // recursing until the worker stack overflows. The guard restores the
    // per-thread depth on every early return via its `Drop`.
    let repair_guard = FeasibilityRepairGuard::enter()?;
    let p = point.len();
    let m = constraints.a.nrows();
    if constraints.a.ncols() != p || m == 0 || constraints.b.len() != m {
        return None;
    }
    let norms: Vec<f64> = (0..m)
        .map(|i| constraints.a.row(i).dot(&constraints.a.row(i)).sqrt())
        .collect();

    // Classify rows. An *anti-parallel pair* with ~zero scaled feasible-slab
    // width is an EQUALITY `rᵀβ = t` encoded as `{rᵀβ ≥ t, −rᵀβ ≥ −t}` (the
    // canonical encoding emitted by a clamped / anchored boundary condition).
    // Representing an equality as two opposing inequalities makes the inequality
    // active-set QP CYCLE: it adds one side, the equality-split multiplier turns
    // the other negative, it removes it, and the working set repeats until cycle
    // detection aborts the solve — so the projection would fail and the caller
    // would fall back to the cone vertex, silently reintroducing the #873 seed
    // for the *combined* case (`shape=concave`/`convex` with `bc=clamped`). So we
    // lift such pairs out as genuine equalities, eliminate them through the null
    // space, and run the strictly-interior QP only on the one-sided rows. A pure
    // shape cone has no anti-parallel rows, so `equality_rows` is empty and this
    // reduces to the original single-QP path verbatim.
    const ANTIPARALLEL_COS_TOL: f64 = -1.0 + 1e-9;
    const EQUALITY_WIDTH_TOL: f64 = 1e-9;
    let mut is_equality_member = vec![false; m];
    let mut equality_rows: Vec<usize> = Vec::new();
    let mut margin = vec![ACTIVE_SET_INTERIOR_SEED_MARGIN; m];
    for i in 0..m {
        if norms[i] == 0.0 {
            margin[i] = 0.0;
            continue;
        }
        for j in (i + 1)..m {
            if norms[j] == 0.0 {
                continue;
            }
            let cos = constraints.a.row(i).dot(&constraints.a.row(j)) / (norms[i] * norms[j]);
            if cos > ANTIPARALLEL_COS_TOL {
                continue;
            }
            // Anti-parallel rows â and −â: row i is `âᵀβ ≥ b_i/‖a_i‖`, row j is
            // `âᵀβ ≤ −b_j/‖a_j‖`. Scaled feasible-slab width:
            let width = -constraints.b[j] / norms[j] - constraints.b[i] / norms[i];
            if width.abs() <= EQUALITY_WIDTH_TOL {
                // Zero width ⇒ equality. Record it once (row i's orientation) and
                // exclude both rows from the one-sided interior shift.
                if !is_equality_member[i] && !is_equality_member[j] {
                    equality_rows.push(i);
                }
                is_equality_member[i] = true;
                is_equality_member[j] = true;
            } else {
                // Genuine (wide) two-sided bound: cap each side's inward shift at
                // `w/3` so the shifted slab `s_i + s_j ≤ w` stays non-empty.
                let cap = (width / 3.0).max(0.0);
                margin[i] = margin[i].min(cap);
                margin[j] = margin[j].min(cap);
            }
        }
    }

    // One-sided rows (everything not lifted into an equality), shifted strictly
    // inward by `margin·‖a‖`.
    let ineq_rows: Vec<usize> = (0..m).filter(|&i| !is_equality_member[i]).collect();
    let mut a_ineq = Array2::<f64>::zeros((ineq_rows.len(), p));
    let mut b_ineq = Array1::<f64>::zeros(ineq_rows.len());
    for (r, &i) in ineq_rows.iter().enumerate() {
        a_ineq.row_mut(r).assign(&constraints.a.row(i));
        b_ineq[r] = constraints.b[i] + margin[i] * norms[i];
    }

    let beta = if equality_rows.is_empty() {
        // No equalities: the original single strictly-interior QP
        // (`min ½‖β − point‖²` s.t. the margin-shifted one-sided rows).
        let interior = LinearInequalityConstraints::new(a_ineq, b_ineq)
            .expect("shifted interior constraint shape invariant");
        let identity = Array2::<f64>::eye(p);
        solve_quadratic_with_linear_constraints(&identity, point, point, &interior, None)
            .ok()?
            .0
    } else {
        // Eliminate `E β = e` through its null space. From the thin SVD
        // `E = U Σ Vᵀ` (rank `r`): the row space is `span(v_0..v_{r-1})`, the
        // minimum-norm particular solution is `β_p = Σ_{i<r} (uᵢᵀe / σᵢ) vᵢ`, and
        // an orthonormal null basis `Z` (p × (p−r)) is the complement of the row
        // space (built by Gram-Schmidt of the standard axes — `p` is a single
        // smooth-term width, so this is cheap and exact). Writing `β = β_p + Z u`
        // and using `ZᵀZ = I`, the projection becomes the reduced strictly-
        // interior QP `min ½‖u − Zᵀ(point − β_p)‖²` s.t. `(A_ineq Z) u ≥ b_ineq −
        // A_ineq β_p`, whose rows carry no anti-parallel pair, so it can't cycle.
        let k = equality_rows.len();
        let mut e_mat = Array2::<f64>::zeros((k, p));
        let mut e_rhs = Array1::<f64>::zeros(k);
        for (r, &i) in equality_rows.iter().enumerate() {
            e_mat.row_mut(r).assign(&constraints.a.row(i));
            e_rhs[r] = constraints.b[i];
        }
        let (u_opt, sing, vt_opt) = e_mat.svd(true, true).ok()?;
        let (u_mat, vt) = (u_opt?, vt_opt?);
        let smax = sing.iter().fold(0.0_f64, |acc, &v| acc.max(v));
        let rank_tol = smax.max(1.0) * (k.max(p) as f64) * f64::EPSILON * 100.0;
        let rank = sing.iter().filter(|&&s| s > rank_tol).count();
        if rank == 0 || rank >= p {
            return None;
        }
        let mut beta_p = Array1::<f64>::zeros(p);
        for idx in 0..rank {
            let coeff = u_mat.column(idx).dot(&e_rhs) / sing[idx];
            beta_p.scaled_add(coeff, &vt.row(idx));
        }
        // Orthonormal null basis: Gram-Schmidt the standard axes against the row
        // space `vt[0..rank]` and the null vectors collected so far.
        let mut basis: Vec<Array1<f64>> = (0..rank).map(|i| vt.row(i).to_owned()).collect();
        let mut z = Array2::<f64>::zeros((p, p - rank));
        let mut collected = 0usize;
        for axis in 0..p {
            if collected == p - rank {
                break;
            }
            let mut v = Array1::<f64>::zeros(p);
            v[axis] = 1.0;
            for q in basis.iter() {
                let c = q.dot(&v);
                v.scaled_add(-c, q);
            }
            let nrm = v.dot(&v).sqrt();
            if nrm > 1e-8 {
                v /= nrm;
                z.column_mut(collected).assign(&v);
                basis.push(v);
                collected += 1;
            }
        }
        if collected != p - rank {
            return None;
        }
        let a_red = a_ineq.dot(&z);
        let b_red = &b_ineq - &a_ineq.dot(&beta_p);
        let u0 = z.t().dot(&(point - &beta_p));
        let reduced = LinearInequalityConstraints::new(a_red, b_red)
            .expect("reduced constraint shape invariant");
        let identity = Array2::<f64>::eye(z.ncols());
        let (u_sol, _active) =
            solve_quadratic_with_linear_constraints(&identity, &u0, &u0, &reduced, None).ok()?;
        &beta_p + &z.dot(&u_sol)
    };

    if beta.len() != p || beta.iter().any(|v| !v.is_finite()) {
        return None;
    }
    // Certify against the ORIGINAL constraints: every genuine one-sided row must
    // clear (most of) its requested margin so the QP step solver sees no spurious
    // active rows; equality-pair rows need only be feasible — they are
    // legitimately tight.
    const SEED_FEASIBILITY_TOL: f64 = 1e-9;
    for i in 0..m {
        let s = scaled_constraint_slack(&beta, constraints, i);
        let lower = if is_equality_member[i] {
            -SEED_FEASIBILITY_TOL
        } else {
            0.5 * margin[i] - SEED_FEASIBILITY_TOL
        };
        if s < lower {
            return None;
        }
    }
    // All mutually-recursive `solve ↔ project` calls are complete; release the
    // per-thread recursion-depth guard explicitly on the success path (early
    // returns above release it via `Drop`). Named + dropped (not `let _guard`)
    // to satisfy the underscore-binding ban without changing its lifetime.
    drop(repair_guard);
    Some(beta)
}

/// Per-row signed scaled slack: `(a_i·beta - b_i) / ‖a_i‖`. A degenerate row
/// with `‖a_i‖ = 0` carries no direction, but it is NOT free of content: for
/// `b_i > 0` the row `0ᵀβ ≥ b_i` is unconditionally violated (−∞ slack), and
/// only for `b_i ≤ 0` is it vacuously satisfied (+∞ slack). Returning zero for
/// both let an impossible row report zero violation and pass every gate.
#[inline]
fn scaled_constraint_slack(
    beta: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
    i: usize,
) -> f64 {
    let norm = constraints.a.row(i).dot(&constraints.a.row(i)).sqrt();
    if norm > 0.0 {
        (constraints.a.row(i).dot(beta) - constraints.b[i]) / norm
    } else if constraints.b[i] > 0.0 {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    }
}

struct ActiveEqualityResidualCertificate {
    worst_row: usize,
    residual: f64,
    allowed: f64,
}

impl ActiveEqualityResidualCertificate {
    fn is_certified(&self) -> bool {
        self.residual.is_finite() && self.allowed.is_finite() && self.residual <= self.allowed
    }
}

/// Certify an active affine face in the normalized geometry in which it was
/// solved.
///
/// The bottom block is `a_i' d = r_i`. Its representable residual is governed
/// by the standard length-`p` dot-product roundoff bound
///
/// `gamma_(p+1) * (sum_j |a_ij d_j| + |r_i|)`,
///
/// where the extra operation is the final subtraction. This is a forward
/// equality certificate, not only the normwise backward-error certificate for
/// the whole (potentially very stiff) saddle system: an O(1e-8) equality drift
/// can be backward-stable against a huge Hessian block while still moving the
/// constrained quadratic by O(1e-3).
///
/// The bound also carries the scale at which `direction` was COMPUTED, not only
/// the scale of the products this row happens to sum. `direction` comes out of a
/// linear solve, so each component carries an absolute error of order
/// `eps·‖direction‖`, never `eps·|d_j|`: cancellation inside one row does not buy
/// that row a smaller input error. Bounding by `sum_j |a_ij d_j|` alone makes the
/// tolerance shrink with exactly the cancellation it exists to tolerate, and on a
/// degenerate face it shrinks below anything f64 can deliver. A factored cone
/// reaches that face routinely — when one coefficient block goes to zero, every
/// observation row over that block becomes tight while its products underflow, so
/// the row-local scale is ~1e-65 and the certificate demands an equality residual
/// no arithmetic can produce. The `‖a_i‖₁·‖direction‖_∞` term is that floor.
fn certify_active_equalities(
    active_a: &Array2<f64>,
    rhs: &Array1<f64>,
    direction: &Array1<f64>,
) -> ActiveEqualityResidualCertificate {
    let p = active_a.ncols();
    let m = active_a.nrows();
    let operations = p.saturating_add(1).max(1);
    let roundoff = operations as f64 * f64::EPSILON;
    let gamma = roundoff / (1.0 - roundoff);
    let direction_scale = direction
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let mut worst = ActiveEqualityResidualCertificate {
        worst_row: 0,
        residual: 0.0,
        allowed: f64::MIN_POSITIVE,
    };
    let mut worst_ratio = 0.0_f64;
    for active_row in 0..m {
        let mut dot = KahanSum::default();
        let mut magnitude = KahanSum::default();
        let mut row_magnitude = KahanSum::default();
        for column in 0..p {
            let entry = active_a[[active_row, column]];
            let product = entry * direction[column];
            dot.add(product);
            magnitude.add(product.abs());
            row_magnitude.add(entry.abs());
        }
        let residual = (rhs[active_row] - dot.sum()).abs();
        let solve_scale = row_magnitude.sum() * direction_scale;
        let allowed = (gamma * (magnitude.sum() + rhs[active_row].abs() + solve_scale))
            .max(f64::MIN_POSITIVE);
        if !residual.is_finite() || !allowed.is_finite() {
            return ActiveEqualityResidualCertificate {
                worst_row: active_row,
                residual,
                allowed,
            };
        }
        let ratio = residual / allowed;
        if ratio > worst_ratio {
            worst_ratio = ratio;
            worst = ActiveEqualityResidualCertificate {
                worst_row: active_row,
                residual,
                allowed,
            };
        }
    }
    worst
}

/// Compute `rhs - A * direction` with compensated row reductions.
fn compensated_active_residual(
    active_a: &Array2<f64>,
    rhs: &Array1<f64>,
    direction: &Array1<f64>,
) -> Array1<f64> {
    Array1::from_shape_fn(active_a.nrows(), |row| {
        let mut dot = KahanSum::default();
        for column in 0..active_a.ncols() {
            dot.add(active_a[[row, column]] * direction[column]);
        }
        rhs[row] - dot.sum()
    })
}

fn minimum_norm_from_svd(
    u: &Array2<f64>,
    singular: &Array1<f64>,
    vt: &Array2<f64>,
    rank: usize,
    rhs: &Array1<f64>,
) -> Array1<f64> {
    let mut solution = Array1::<f64>::zeros(vt.ncols());
    for index in 0..rank {
        let coefficient = u.column(index).dot(rhs) / singular[index];
        solution.scaled_add(coefficient, &vt.row(index));
    }
    solution
}

fn transposed_minimum_norm_from_svd(
    u: &Array2<f64>,
    singular: &Array1<f64>,
    vt: &Array2<f64>,
    rank: usize,
    rhs: &Array1<f64>,
) -> Array1<f64> {
    let mut solution = Array1::<f64>::zeros(u.nrows());
    for index in 0..rank {
        let coefficient = vt.row(index).dot(rhs) / singular[index];
        solution.scaled_add(coefficient, &u.column(index));
    }
    solution
}

/// Solve the equality-constrained strictly-convex quadratic
///
/// `min_d 1/2 d' H d + g' d  subject to A d = r`
///
/// in an orthonormal null-space coordinate system.
///
/// The bordered KKT representation `[H A'; A 0]` mixes the scale of a stiff
/// positive-definite metric with unit-normalized active equations in one
/// indefinite factor. On the #979 CTN face that finite LBLT answer missed an
/// active equation by `6.384e-4`, and even its residual-correction solve became
/// non-finite. The null-space representation never forms that saddle matrix:
///
/// * normalize each active equation;
/// * use a thin SVD `A = U S V'` for a minimum-norm affine point `d_p` and a
///   rank-revealing Householder QR of `A'` for its full orthonormal null basis
///   `Z`;
/// * solve the positive-definite reduced problem
///   `(Z' H Z) z = -Z' (g + H d_p)`; and
/// * recover active multipliers from the stationarity equation.
///
/// This is algebraically the same constrained minimizer. Rank-deficient active
/// equations use the RRQR rank consistently in both decompositions; an
/// inconsistent affine right-hand side is rejected by the forward equality
/// certificate rather than hidden by a pseudoinverse rank drop.
pub(crate) fn solve_kkt_direction(
    hessian: &Array2<f64>,
    gradient: &Array1<f64>,
    active_a: &Array2<f64>,
    active_residual: Option<&Array1<f64>>,
) -> Result<(Array1<f64>, Array1<f64>), EstimationError> {
    let p = hessian.nrows();
    let m = active_a.nrows();
    if hessian.ncols() != p || gradient.len() != p || active_a.ncols() != p {
        crate::bail_invalid_estim!("null-space constrained solve dimension mismatch");
    }
    if let Some(residual) = active_residual
        && residual.len() != m
    {
        crate::bail_invalid_estim!(
            "active-equality residual length mismatch: got {}, expected {}",
            residual.len(),
            m
        );
    }
    if m == 0 {
        let mut d = Array1::<f64>::zeros(p);
        solve_newton_direction_dense(hessian, gradient, &mut d)?;
        return Ok((d, Array1::zeros(0)));
    }

    let mut scaled_a = active_a.clone();
    let mut scaled_rhs = active_residual
        .cloned()
        .unwrap_or_else(|| Array1::<f64>::zeros(m));
    let mut row_norms = Array1::<f64>::zeros(m);
    for row in 0..m {
        let norm = active_a.row(row).dot(&active_a.row(row)).sqrt();
        if !(norm.is_finite() && norm > 0.0) {
            crate::bail_invalid_estim!(
                "active equality row {row} has invalid norm {norm}"
            );
        }
        row_norms[row] = norm;
        let inverse = 1.0 / norm;
        scaled_a.row_mut(row).mapv_inplace(|value| value * inverse);
        scaled_rhs[row] *= inverse;
    }

    let (u_opt, singular, vt_opt) = scaled_a.svd(true, true).map_err(|_| {
        EstimationError::InvalidInput(
            "null-space constrained quadratic active-equation SVD failed".to_string(),
        )
    })?;
    let (Some(u), Some(vt)) = (u_opt, vt_opt) else {
        crate::bail_invalid_estim!(
            "null-space constrained quadratic SVD omitted singular vectors"
        );
    };
    let (mut null_basis, rank) =
        rrqr_nullspace_basis(&scaled_a.t(), default_rrqr_rank_alpha()).map_err(|_| {
            EstimationError::InvalidInput(
                "null-space constrained quadratic active-equation RRQR failed".to_string(),
            )
        })?;
    if rank == 0 {
        crate::bail_invalid_estim!(
            "null-space constrained quadratic active equations have numerical rank zero"
        );
    }
    if rank > singular.len()
        || !singular[rank - 1].is_finite()
        || singular[rank - 1] <= 0.0
    {
        crate::bail_invalid_estim!(
            "null-space constrained quadratic RRQR rank {rank} has no positive SVD pivot"
        );
    }
    let nullity = p.saturating_sub(rank);
    if null_basis.dim() != (p, nullity) {
        crate::bail_invalid_estim!(
            "null-space constrained quadratic RRQR basis has shape {}x{}, expected {}x{}",
            null_basis.nrows(),
            null_basis.ncols(),
            p,
            nullity,
        );
    }
    // Make the null basis orthogonal to the row space WITHOUT dividing by a
    // singular value.
    //
    // `vt.row(0..rank)` is already an ORTHONORMAL basis of `row(scaled_a)`, so
    // the row-space component of a column `z` is `sum_i <v_i, z> v_i` and
    // removing it is Gram-Schmidt against an orthonormal set. In exact
    // arithmetic that is the same map as the previous correction
    // (`V Σ⁻¹ Uᵀ (A z)` equals `V Vᵀ z`), but in floating point the two are not
    // remotely equal: the old form first FORMS `A z` -- catastrophic
    // cancellation exactly when `z` is nearly in the null space, which is the
    // only case that matters here -- and then multiplies by `1/σ`, amplifying
    // whatever survived by `1/σ_min`. On a near-rank-deficient face that is
    // where the basis loses its accuracy.
    //
    // This matters because the reduced Newton solve below zeroes the gradient on
    // `span(Z)`, while the KKT certificate measures stationarity against the
    // TRUE tangent, the complement of `row(A)`. If those two subspaces differ,
    // the solve faithfully zeroes the wrong one and the endpoint is accepted
    // carrying real tangent mass -- which is what #2592 measures.
    //
    // Two passes: one sweep of classical Gram-Schmidt loses orthogonality when
    // the input is already nearly dependent, and repeating it once is the
    // standard remedy ("twice is enough").
    for column in 0..nullity {
        for _ in 0..2 {
            let mut basis_column = null_basis.column(column).to_owned();
            for index in 0..rank {
                let projection = vt.row(index).dot(&basis_column);
                basis_column.scaled_add(-projection, &vt.row(index));
            }
            null_basis.column_mut(column).assign(&basis_column);
        }
        // Keep the columns unit-scaled so `ZᵀHZ` inherits `H`'s conditioning
        // rather than the basis's. A column that collapses under the projection
        // was not independent of the row space to begin with; leave it as the
        // rank checks above left it rather than amplifying noise.
        let norm = null_basis.column(column).dot(&null_basis.column(column)).sqrt();
        if norm.is_finite() && norm > 0.0 {
            null_basis.column_mut(column).mapv_inplace(|value| value / norm);
        }
    }
    if !array_is_finite(&null_basis) {
        crate::bail_invalid_estim!(
            "null-space constrained quadratic refined RRQR basis is non-finite"
        );
    }

    let mut particular = minimum_norm_from_svd(&u, &singular, &vt, rank, &scaled_rhs);
    if !array_is_finite(&particular) {
        crate::bail_invalid_estim!(
            "null-space constrained quadratic affine solution is non-finite"
        );
    }

    let initial_affine_residual =
        compensated_active_residual(&scaled_a, &scaled_rhs, &particular);
    let affine_correction =
        minimum_norm_from_svd(&u, &singular, &vt, rank, &initial_affine_residual);
    particular += &affine_correction;

    let mut direction = particular.clone();
    if nullity > 0 {
        let mut reduced_hessian = null_basis.t().dot(hessian).dot(&null_basis);
        for row in 0..nullity {
            for column in (row + 1)..nullity {
                let average =
                    0.5 * (reduced_hessian[[row, column]] + reduced_hessian[[column, row]]);
                reduced_hessian[[row, column]] = average;
                reduced_hessian[[column, row]] = average;
            }
        }
        let affine_gradient = gradient + &hessian.dot(&particular);
        let reduced_rhs = -null_basis.t().dot(&affine_gradient);
        let factor = reduced_hessian
            .cholesky(Side::Lower)
            .map_err(EstimationError::LinearSystemSolveFailed)?;
        let mut reduced_solution = factor.solvevec(&reduced_rhs);
        if !array_is_finite(&reduced_solution) {
            crate::bail_invalid_estim!(
                "null-space constrained quadratic reduced solve is non-finite"
            );
        }
        // THIS solve is what makes the endpoint stationary on its face: it sets
        // `Zᵀ(H beta + g) = 0`, and every downstream KKT certificate depends on
        // it holding. Its residual was never looked at.
        //
        // `Z` comes from an RRQR null-space basis that is itself corrected in a
        // loop above, so `ZᵀHZ` is a product of three inexact factors; when the
        // face is close to rank-deficient it is ill-conditioned, and a single
        // Cholesky substitution returns a `w` whose reduced residual is nowhere
        // near roundoff. The endpoint then satisfies its active equalities
        // exactly (that IS certified, two blocks below) and is all-row feasible,
        // so the walk accepts it -- and the failure only surfaces three frames
        // later as "failed stationarity certification", where it reads as a
        // solver mystery rather than an unchecked linear solve.
        //
        // Measured (#2592): transformation-normal seed 1 reached that refusal
        // with `residual=1.600e1` against `gradient_scale=9.908e2`, and the
        // smallest residual ANY multipliers could achieve was `1.600e1` too --
        // i.e. the whole 1.6% lived in the face TANGENT, which is precisely the
        // quantity this solve is responsible for zeroing.
        //
        // One step of iterative refinement in the same factorization. It costs a
        // matvec and a substitution, is the standard remedy for exactly this,
        // and cannot move a solve that was already exact (its correction is then
        // zero to roundoff).
        let reduced_residual = &reduced_rhs - &reduced_hessian.dot(&reduced_solution);
        let reduced_correction = factor.solvevec(&reduced_residual);
        if array_is_finite(&reduced_correction) {
            reduced_solution += &reduced_correction;
        }
        direction += &null_basis.dot(&reduced_solution);
    }

    let initial_certificate =
        certify_active_equalities(&scaled_a, &scaled_rhs, &direction);
    if !initial_certificate.is_certified() {
        let affine_residual =
            compensated_active_residual(&scaled_a, &scaled_rhs, &direction);
        let correction =
            minimum_norm_from_svd(&u, &singular, &vt, rank, &affine_residual);
        if !correction.iter().all(|value| value.is_finite()) {
            return Err(EstimationError::ParameterConstraintViolation(format!(
                "null-space active-equality correction produced a non-finite value \
                 (active_row={}, residual={:.3e}, roundoff_bound={:.3e})",
                initial_certificate.worst_row,
                initial_certificate.residual,
                initial_certificate.allowed,
            )));
        }
        direction += &correction;
        let refined_certificate =
            certify_active_equalities(&scaled_a, &scaled_rhs, &direction);
        if !refined_certificate.is_certified() {
            return Err(EstimationError::ParameterConstraintViolation(format!(
                "null-space active equality is unresolved after affine correction \
                 (active_row={}, residual={:.3e}, roundoff_bound={:.3e}; \
                 initial_active_row={}, initial_residual={:.3e}, \
                 initial_roundoff_bound={:.3e})",
                refined_certificate.worst_row,
                refined_certificate.residual,
                refined_certificate.allowed,
                initial_certificate.worst_row,
                initial_certificate.residual,
                initial_certificate.allowed,
            )));
        }
    }

    let stationarity_rhs = -(gradient + &hessian.dot(&direction));
    let scaled_multiplier =
        transposed_minimum_norm_from_svd(&u, &singular, &vt, rank, &stationarity_rhs);
    let multiplier = &scaled_multiplier / &row_norms;
    if !array_is_finite(&multiplier) {
        crate::bail_invalid_estim!(
            "null-space constrained quadratic multiplier recovery is non-finite"
        );
    }
    Ok((direction, multiplier))
}

/// One dependent row of the WORKING SET expressed against its representative:
/// `a_dep ≈ coeff · a_rep`.
///
/// Recorded ONLY for exactly-parallel (positively-aligned scalar-multiple)
/// dependents; a general-position dependent is dropped from the working set with
/// NO entry and re-enters via the next feasibility scan (it never receives a
/// distributed/phantom multiplier).
///
/// `active_pos` is an ACTIVE-SET POSITION: an index into the caller's `active`
/// slice, so the original constraint id is `active[active_pos]`. Working-face
/// rank reduction seeds each group with these positions before collapsing
/// dependent rows. It is NOT a constraint-row id and NOT a coefficient index.
/// The reduced-face op reports its dependents in constraint-row space instead
/// and therefore uses its own [`ConstraintRowDependence`] — the two must not be
/// interchanged.
#[derive(Clone, Copy, Debug)]
pub struct ActiveRowDependence {
    pub active_pos: usize,
    pub coeff: f64,
}

/// One tight row of a REDUCED FACE expressed against its representative:
/// `a_dep ≈ coeff · a_rep`, with the dependent named in constraint-row space.
///
/// Same `(A)`-strict recording rule as [`ActiveRowDependence`], different index
/// space: `row` is the dependent's [`ConstraintRowId`] in the reduced set's own
/// row space. The representative is identified by the index of the owning
/// [`ReducedFace::dependence`] slot, which is aligned with
/// [`ReducedFace::representatives`].
#[derive(Clone, Copy, Debug)]
pub struct ConstraintRowDependence {
    pub row: ConstraintRowId,
    pub coeff: f64,
}

/// The result of reducing a tight active face to a minimal independent set — the
/// shared output of the `ConstraintSet` reduced-face op (Dense arm =
/// [`dense_reduced_face`]; KhatriRaoCone / BlockDiagonal arms produce the same
/// shape). Determinism: representatives are the lowest-flat-index row per
/// independent direction, ascending, with no float tie-break.
///
/// INDEX SPACE: every id here is a [`ConstraintRowId`] in the reduced set's own
/// constraint-row space (`0..nrows()`), addressing `values()` / `bound()` /
/// `row_norm()`. It is NOT a coefficient index; to reach β coordinates go
/// through [`gam_problem::ConstraintSet::row_column_support`].
#[derive(Clone, Debug)]
pub struct ReducedFace {
    /// Kept independent rows — the lowest-flat-index representative per direction,
    /// ascending. Flat id space is `0..nrows` (Dense) / `slot*n + obs` (cone) /
    /// the concatenation of the member row spaces (block-diagonal).
    pub representatives: Vec<ConstraintRowId>,
    /// Per-representative parallel-dependent map, index-aligned with
    /// `representatives`. `dependence[i]` lists the exactly-parallel dependents of
    /// `representatives[i]` (empty when it has none); general-position dependents
    /// are absent (dropped, re-enter on the next feasibility scan).
    pub dependence: Vec<Vec<ConstraintRowDependence>>,
    /// The full tight set that was reduced, ascending flat ids.
    pub tight_rows: Vec<ConstraintRowId>,
}

/// Reduce the tight active face of a Khatri–Rao monotonicity cone to its minimal
/// independent set — the `KhatriRaoCone` arm of the `ConstraintSet` reduced-face
/// op (gam#2306; the Dense arm is [`dense_reduced_face`]).
///
/// A cone row `(slot, i)` has normal `e_{k} ⊗ ψ_i` (`k = coupled_rows[slot]`),
/// so two normals' inner product is `δ_{slot,slot'}·(ψ_iᵀ ψ_{i'})`: cross-block
/// normals are ALWAYS orthogonal and never redundant, and redundancy occurs only
/// WITHIN a shape block among linearly dependent covariate rows `ψ_i`. The
/// reduction therefore decomposes into independent per-block Gram–Schmidt scans
/// over the block's tight `ψ_i` rows — never forming the `n·|coupled|` system.
///
/// Contract (matches the Dense arm): FULL rank cut (every dependent row is
/// dropped from `representatives`, parallel OR general-position); the dependence
/// map records `(A)`-strict — ONLY exactly-parallel dependents
/// (`|cos(ψ_dep, ψ_rep)| ≥ 1 − 1e-9`) get a [`ConstraintRowDependence`] against their
/// single representative (`coeff = ψ_depᵀψ_rep / ‖ψ_rep‖²`, so `a_dep ≈ coeff·a_rep`);
/// general-position drops get no entry and re-enter via the next feasibility
/// scan. Representatives are the lowest-flat-index row per direction (ascending
/// obs within a block), host-deterministic with no float tie-break. The rank
/// tolerance mirrors the Dense scan (`100·ε·max(n_tight, p_cov)·max‖ψ‖`), so the
/// two arms cut to the same numerical rank. Flat id is `slot*n + obs`, matching
/// [`KhatriRaoConeConstraints::values`].
pub fn khatri_rao_cone_reduced_face(
    cone: &KhatriRaoConeConstraints,
    beta: ndarray::ArrayView1<'_, f64>,
    membership_tol: f64,
) -> Result<ReducedFace, EstimationError> {
    let psi = cone.factor();
    let n = psi.nrows();
    let p_cov = psi.ncols();
    let coupled = cone.coupled_rows();
    let values = cone.values(beta).map_err(|error| {
        EstimationError::ParameterConstraintViolation(format!(
            "Khatri-Rao cone reduced-face values: {error}"
        ))
    })?;

    // ‖ψ_i‖ is shared across coupled slots (the same covariate factor).
    let row_norms: Vec<f64> = (0..n)
        .map(|i| {
            let row = psi.row(i);
            row.dot(&row).sqrt()
        })
        .collect();

    const RANK_ALPHA: f64 = 100.0;
    // Exactly-parallel threshold, matching the Dense scan's ±1e-9 cosine band.
    const PARALLEL_COS_TOL: f64 = 1.0 - 1e-9;

    let mut representatives: Vec<ConstraintRowId> = Vec::new();
    let mut dependence: Vec<Vec<ConstraintRowDependence>> = Vec::new();
    let mut tight_rows: Vec<ConstraintRowId> = Vec::new();

    for slot in 0..coupled.len() {
        // Tight obs in this block, ascending. A zero-norm ψ_i is a vacuous row
        // (0ᵀβ ≥ 0 always holds) — never a constraint direction, never a rep.
        let mut tight_obs: Vec<usize> = Vec::new();
        for i in 0..n {
            let norm_i = row_norms[i];
            if norm_i <= 0.0 {
                continue;
            }
            let scaled_slack = values[slot * n + i] / norm_i;
            if scaled_slack <= membership_tol {
                tight_rows.push(ConstraintRowId(slot * n + i));
                tight_obs.push(i);
            }
        }
        if tight_obs.is_empty() {
            continue;
        }

        let max_norm = tight_obs
            .iter()
            .map(|&i| row_norms[i])
            .fold(0.0_f64, f64::max);
        let rank_tol =
            RANK_ALPHA * f64::EPSILON * (tight_obs.len().max(p_cov).max(1) as f64) * max_norm;

        let mut ortho_basis: Vec<Array1<f64>> = Vec::new();
        // Kept representatives in THIS block: (obs, ψ_obs, index into representatives).
        let mut kept: Vec<(usize, Array1<f64>, usize)> = Vec::new();
        for &i in &tight_obs {
            let psi_i = psi.row(i).to_owned();
            let mut resid = psi_i.clone();
            for q in &ortho_basis {
                let proj = resid.dot(q);
                resid.scaled_add(-proj, q);
            }
            let resid_norm = resid.dot(&resid).sqrt();
            let flat = ConstraintRowId(slot * n + i);
            if resid_norm > rank_tol {
                ortho_basis.push(&resid / resid_norm);
                let out_idx = representatives.len();
                representatives.push(flat);
                dependence.push(Vec::new());
                kept.push((i, psi_i, out_idx));
            } else {
                // (A)-strict: record ONLY an exactly-parallel single-representative
                // dependence; general-position drops carry no multiplier.
                let mut best_abs_cos = 0.0_f64;
                let mut best: Option<(usize, f64)> = None;
                for (rep_obs, rep_psi, rep_out_idx) in &kept {
                    let rep_norm = row_norms[*rep_obs];
                    let dot = psi_i.dot(rep_psi);
                    let cos = if rep_norm > 0.0 {
                        dot / (row_norms[i] * rep_norm)
                    } else {
                        0.0
                    };
                    if cos.abs() > best_abs_cos {
                        best_abs_cos = cos.abs();
                        best = Some((*rep_out_idx, dot / (rep_norm * rep_norm)));
                    }
                }
                if best_abs_cos >= PARALLEL_COS_TOL {
                    if let Some((out_idx, coeff)) = best {
                        dependence[out_idx].push(ConstraintRowDependence {
                            row: flat,
                            coeff,
                        });
                    }
                }
            }
        }
    }

    Ok(ReducedFace {
        representatives,
        dependence,
        tight_rows,
    })
}

/// Dense arm of the reduced-face op: reduce the tight rows of an explicit
/// `A x ≥ b` set at `beta` to a minimal independent set. Mirrors
/// [`khatri_rao_cone_reduced_face`] exactly — ascending-index greedy MGS,
/// `RANK_ALPHA·ε·max(n_tight,p)·max‖a‖` tolerance, (A)-strict parallel-only
/// dependence (|cos| ≥ 1−1e-9, `coeff = a_depᵀa_rep/‖a_rep‖²`, `row` = the
/// dependent row's flat id) — so both carriers produce the same `ReducedFace`
/// contract. Flat id = the constraint row index. A zero-norm row is vacuous
/// (never a direction, never a representative).
pub fn dense_reduced_face(
    lin: &LinearInequalityConstraints,
    beta: ndarray::ArrayView1<'_, f64>,
    membership_tol: f64,
) -> Result<ReducedFace, EstimationError> {
    let a = &lin.a;
    let b = &lin.b;
    let n = a.nrows();
    let p = a.ncols();

    let row_norms: Vec<f64> = (0..n)
        .map(|i| {
            let row = a.row(i);
            row.dot(&row).sqrt()
        })
        .collect();

    const RANK_ALPHA: f64 = 100.0;
    const PARALLEL_COS_TOL: f64 = 1.0 - 1e-9;

    // The scan runs in raw row indices (they address `a` / `row_norms`); the
    // ids are wrapped into constraint-row space once, at the return boundary.
    let mut tight: Vec<usize> = Vec::new();
    for i in 0..n {
        let norm_i = row_norms[i];
        if norm_i <= 0.0 {
            continue;
        }
        let scaled_slack = (a.row(i).dot(&beta) - b[i]) / norm_i;
        if scaled_slack <= membership_tol {
            tight.push(i);
        }
    }

    let mut representatives: Vec<ConstraintRowId> = Vec::new();
    let mut dependence: Vec<Vec<ConstraintRowDependence>> = Vec::new();
    if tight.is_empty() {
        return Ok(ReducedFace {
            representatives,
            dependence,
            tight_rows: Vec::new(),
        });
    }

    let max_norm = tight
        .iter()
        .map(|&i| row_norms[i])
        .fold(0.0_f64, f64::max);
    let rank_tol = RANK_ALPHA * f64::EPSILON * (tight.len().max(p).max(1) as f64) * max_norm;

    let mut ortho_basis: Vec<Array1<f64>> = Vec::new();
    // Kept representatives: (row, a_row, index into `representatives`).
    let mut kept: Vec<(usize, Array1<f64>, usize)> = Vec::new();
    for &i in &tight {
        let a_i = a.row(i).to_owned();
        let mut resid = a_i.clone();
        for q in &ortho_basis {
            let proj = resid.dot(q);
            resid.scaled_add(-proj, q);
        }
        let resid_norm = resid.dot(&resid).sqrt();
        if resid_norm > rank_tol {
            ortho_basis.push(&resid / resid_norm);
            let out_idx = representatives.len();
            representatives.push(ConstraintRowId(i));
            dependence.push(Vec::new());
            kept.push((i, a_i, out_idx));
        } else {
            // (A)-strict: record ONLY an exactly-parallel single-representative
            // dependence; a general-position drop carries no multiplier and
            // re-enters via the next feasibility scan.
            let mut best_abs_cos = 0.0_f64;
            let mut best: Option<(usize, f64)> = None;
            for (rep_row, rep_a, rep_out_idx) in &kept {
                let rep_norm = row_norms[*rep_row];
                let dot = a_i.dot(rep_a);
                let cos = if rep_norm > 0.0 {
                    dot / (row_norms[i] * rep_norm)
                } else {
                    0.0
                };
                if cos.abs() > best_abs_cos {
                    best_abs_cos = cos.abs();
                    best = Some((*rep_out_idx, dot / (rep_norm * rep_norm)));
                }
            }
            if best_abs_cos >= PARALLEL_COS_TOL {
                if let Some((out_idx, coeff)) = best {
                    dependence[out_idx].push(ConstraintRowDependence {
                        row: ConstraintRowId(i),
                        coeff,
                    });
                }
            }
        }
    }

    Ok(ReducedFace {
        representatives,
        dependence,
        tight_rows: tight.into_iter().map(ConstraintRowId).collect(),
    })
}

/// Lift a MEMBER's constraint-row id into the JOINT block-diagonal row space.
///
/// Derivation: `ConstraintSet::BlockDiagonal` stacks its members' constraint
/// ROWS in block order — `ConstraintSet::values` writes member `m`'s values into
/// `out[off .. off + m.nrows()]`, and `bound` / `row_norm` decode a joint row by
/// walking the same running `nrows()` sum (`block_for_row`). So the joint id of
/// member row `local` is `local + Σ_{earlier m} m.nrows()`, which is what
/// `row_offset` accumulates.
///
/// The offset is deliberately NOT `col_start`. That is the COEFFICIENT offset,
/// and it advances by `ncols()`. Using one for the other is only invisible while
/// every member is square (`nrows() == ncols()`); the moment a member constrains
/// fewer rows than it has coefficients, the two sequences diverge and the ids
/// silently name the wrong block. To go from these ids to β coordinates, use
/// `ConstraintSet::row_column_support` — never arithmetic on the id.
#[inline]
fn lift_member_row(local: ConstraintRowId, row_offset: usize) -> ConstraintRowId {
    ConstraintRowId(local.index() + row_offset)
}

/// The shared tight-face reduction op over the `ConstraintSet` carrier union.
/// An extension trait (not an inherent method) so the numeric reduction
/// stays in gam-solve where the solvers consume it, keeping `gam-problem` a pure
/// data crate. All three arms produce the same `ReducedFace` contract.
pub trait ConstraintSetReducedFace {
    fn reduced_face(
        &self,
        beta: ndarray::ArrayView1<'_, f64>,
        membership_tol: f64,
    ) -> Result<ReducedFace, EstimationError>;
}

impl ConstraintSetReducedFace for ConstraintSet {
    fn reduced_face(
        &self,
        beta: ndarray::ArrayView1<'_, f64>,
        membership_tol: f64,
    ) -> Result<ReducedFace, EstimationError> {
        match self {
            ConstraintSet::Dense(lin) => dense_reduced_face(lin, beta, membership_tol),
            ConstraintSet::KhatriRaoCone(cone) => {
                khatri_rao_cone_reduced_face(cone, beta, membership_tol)
            }
            ConstraintSet::BlockDiagonal { blocks, .. } => {
                // Compose per inner block. TWO independent offsets are in play and
                // they are NOT interchangeable:
                //   * `block.col_start` slices β — COEFFICIENT space, advancing by
                //     each member's `ncols()`;
                //   * `row_offset` lifts the returned ids — CONSTRAINT-ROW space,
                //     advancing by each member's `nrows()`.
                // They coincide only when every member is a square carrier, which
                // is why a mixed block (a constrained sub-basis alongside
                // unconstrained intercept/covariate columns, `nrows() < ncols()`)
                // is the case that separates them. See `lift_member_row`.
                let mut representatives: Vec<ConstraintRowId> = Vec::new();
                let mut dependence: Vec<Vec<ConstraintRowDependence>> = Vec::new();
                let mut tight_rows: Vec<ConstraintRowId> = Vec::new();
                let mut row_offset = 0usize;
                for block in blocks {
                    let start = block.col_start;
                    let end = start + block.set.ncols();
                    let beta_block = beta.slice(ndarray::s![start..end]);
                    let sub = block.set.reduced_face(beta_block, membership_tol)?;
                    for r in sub.representatives {
                        representatives.push(lift_member_row(r, row_offset));
                    }
                    for deps in sub.dependence {
                        dependence.push(
                            deps.into_iter()
                                .map(|d| ConstraintRowDependence {
                                    row: lift_member_row(d.row, row_offset),
                                    coeff: d.coeff,
                                })
                                .collect(),
                        );
                    }
                    for t in sub.tight_rows {
                        tight_rows.push(lift_member_row(t, row_offset));
                    }
                    row_offset += block.set.nrows();
                }
                Ok(ReducedFace {
                    representatives,
                    dependence,
                    tight_rows,
                })
            }
        }
    }
}

fn identity_multiplier_dependence(groups: &[Vec<usize>]) -> Vec<Vec<ActiveRowDependence>> {
    groups
        .iter()
        .map(|group| {
            group
                .iter()
                .copied()
                .map(|active_pos| ActiveRowDependence {
                    active_pos,
                    coeff: 1.0,
                })
                .collect()
        })
        .collect()
}

pub fn rank_reduce_rows_pivoted_qr_with_dependence(
    a: Array2<f64>,
    b: Array1<f64>,
    groups: Vec<Vec<usize>>,
) -> (
    Array2<f64>,
    Array1<f64>,
    Vec<Vec<usize>>,
    Vec<Vec<ActiveRowDependence>>,
) {
    let k = a.nrows();
    let p = a.ncols();
    if k <= 1 {
        let multiplier_dependence = identity_multiplier_dependence(&groups);
        return (a, b, groups, multiplier_dependence);
    }

    // DETERMINISTIC, host-independent representative selection. The former faer
    // `col_piv_qr` pivots by largest column norm; on an equal-norm tie
    // (near-parallel / identical active rows — the degenerate-face case) faer's
    // internal tie-break can differ across CPU arch / SIMD width / library
    // version, which would record a host-dependent active face and re-introduce
    // the nondeterministic cross-host certification the face feeds. Instead do a
    // greedy ASCENDING-original-index independence scan: iterate rows in index
    // order and keep row r iff its residual after projecting onto the orthonormal
    // span of the already-kept rows exceeds the rank tolerance; otherwise record
    // it dependent. This yields the lowest-index representative per independent
    // direction with no float-comparison tie-break.
    //
    // Rank tolerance is relative to the largest row norm — the same |R00| scale
    // (= largest column norm of Aᵀ = largest row norm of A) the pivoted QR used —
    // so the accepted COUNT matches the prior numerical rank; only WHICH
    // representative is chosen among tied near-parallel rows changes, and it
    // changes deterministically. The scale carries NO absolute floor, preserving
    // the unit-robustness of the prior tolerance (a perfectly independent system
    // in tiny units, e.g. A = 1e-20·I, keeps full rank rather than being dropped).
    const RANK_ALPHA: f64 = 100.0;
    let max_row_norm = (0..k)
        .map(|r| {
            let row = a.row(r);
            row.dot(&row).sqrt()
        })
        .fold(0.0_f64, f64::max);
    let tol = RANK_ALPHA * f64::EPSILON * (k.max(p).max(1) as f64) * max_row_norm;

    let mut ortho_basis: Vec<Array1<f64>> = Vec::new();
    let mut kept_orig: Vec<usize> = Vec::new();
    let mut dropped_orig: Vec<usize> = Vec::new();
    for r in 0..k {
        let mut resid = a.row(r).to_owned();
        // TWO passes, and the second one is the rank decision (gam#2600).
        //
        // A single modified-Gram-Schmidt sweep leaves the computed residual
        // contaminated by the accumulated loss of orthogonality in
        // `ortho_basis`, which is `O(eps * cond(A_kept))` — NOT `O(eps)`. On an
        // ill-conditioned face that contamination is larger than `tol`, so a
        // row that is genuinely in the span of the kept rows reports a residual
        // above the floor and is kept as "independent". The face then carries a
        // numerically redundant row, and every consumer that factors it
        // afterwards disagrees with this scan about the rank.
        //
        // Measured on #2600's `transformation_normal_held_out_pit_is_uniform`
        // fixture: 90 reduced-face solves where the block this scan returned had
        // `nrows = svd_rank + 1`, EVERY time — `face_rows=39, rank=38` with the
        // retained singular values down to `3.86e-5` and the 39th at `~1e-15`,
        // against this scan's `tol = 1.07e-12`. `cond(A_kept) ~ 6e4` puts the
        // single-pass contamination at `~1.3e-11`, an order above `tol`, which
        // is exactly the observed miss. The physical reduced face then handed
        // that block to an SVD that refused it as a singular affine system and
        // ended the whole fit.
        //
        // Reorthogonalizing once ("twice is enough", Kahan) drops the residual
        // error back to `O(eps)*||a_r||`, so this scan's rank and a
        // rank-revealing factorization of the same block agree. It is the same
        // deliberate second pass `active_constraint_tangent_geometry` already
        // runs when it completes a null basis, for the same reason.
        for _ in 0..2 {
            for q in &ortho_basis {
                let proj = resid.dot(q);
                resid.scaled_add(-proj, q);
            }
        }
        let resid_norm = resid.dot(&resid).sqrt();
        if resid_norm > tol {
            kept_orig.push(r);
            ortho_basis.push(&resid / resid_norm);
        } else {
            dropped_orig.push(r);
        }
    }
    let rank = kept_orig.len();
    if rank >= k {
        let multiplier_dependence = identity_multiplier_dependence(&groups);
        return (a, b, groups, multiplier_dependence);
    }
    if rank == 0 {
        log::debug!(
            "rank-reduced active constraints from {} to 0 rows (all active rows numerically zero)",
            k
        );
        return (
            Array2::<f64>::zeros((0, p)),
            Array1::<f64>::zeros(0),
            Vec::new(),
            Vec::new(),
        );
    }

    let mut orig_to_out = std::collections::HashMap::with_capacity(rank);
    let mut a_out = Array2::<f64>::zeros((rank, p));
    let mut b_out = Array1::<f64>::zeros(rank);
    let mut groups_out: Vec<Vec<usize>> = Vec::with_capacity(rank);
    let mut multiplier_dependence: Vec<Vec<ActiveRowDependence>> = Vec::with_capacity(rank);
    for (out_idx, &orig_idx) in kept_orig.iter().enumerate() {
        a_out.row_mut(out_idx).assign(&a.row(orig_idx));
        b_out[out_idx] = b[orig_idx];
        groups_out.push(groups[orig_idx].clone());
        multiplier_dependence.push(
            groups[orig_idx]
                .iter()
                .copied()
                .map(|active_pos| ActiveRowDependence {
                    active_pos,
                    coeff: 1.0,
                })
                .collect(),
        );
        orig_to_out.insert(orig_idx, out_idx);
    }

    // (A)-strict merge, matching the shared `dense_reduced_face` /
    // `khatri_rao_cone_reduced_face` `ReducedFace` contract. A dropped row joins
    // a representative's group — and receives a distributed multiplier — ONLY
    // when it is exactly PARALLEL to that representative (the same half-space up
    // to positive scale). A GENERAL-POSITION dependent — dependent only because
    // more normals bind than the face dimension (e.g. three normals inside a 2-D
    // coupled block) — is dropped outright with NO group entry and NO
    // multiplier: it re-enters the working set via the next feasibility scan and
    // is never conflated with a different half-space's dual (#979). The former
    // `best_positive_align` merge folded such a row into whichever kept row it
    // was most positively aligned with, silently truncating a general-position
    // active row out of the enforced face and pinning the wrong vertex (#2378).
    const PARALLEL_COS_TOL: f64 = 1.0 - 1e-9;
    for &dropped_idx in &dropped_orig {
        let dropped_row = a.row(dropped_idx);
        let dropped_norm = dropped_row.dot(&dropped_row).sqrt();
        let mut best_abs_cos = 0.0_f64;
        let mut best_target: Option<(usize, f64)> = None;
        for &kept_idx in &kept_orig {
            let kept_row = a.row(kept_idx);
            let kept_norm = kept_row.dot(&kept_row).sqrt();
            let dot = kept_row.dot(&dropped_row);
            let cos = if kept_norm > 0.0 && dropped_norm > 0.0 {
                dot / (kept_norm * dropped_norm)
            } else {
                0.0
            };
            let coeff = if kept_norm > 0.0 {
                dot / (kept_norm * kept_norm)
            } else {
                0.0
            };
            if cos.abs() > best_abs_cos {
                best_abs_cos = cos.abs();
                best_target = Some((kept_idx, coeff));
            }
        }
        // Only an exactly-parallel dependent is recorded; a general-position
        // drop carries no phantom distributed dual. The group (whose whole-set
        // release the working-set loop drives) additionally requires POSITIVE
        // parallelism — same constraint up to positive scale — so an opposing
        // (anti-parallel) tight row is never released together with it.
        if best_abs_cos >= PARALLEL_COS_TOL {
            if let Some((target, coeff)) = best_target {
                let &out_idx = orig_to_out
                    .get(&target)
                    .expect("merge target must be a kept row");
                for &active_pos in &groups[dropped_idx] {
                    multiplier_dependence[out_idx].push(ActiveRowDependence { active_pos, coeff });
                }
                if coeff > 0.0 {
                    groups_out[out_idx].extend_from_slice(&groups[dropped_idx]);
                }
            }
        }
    }

    for group in &mut groups_out {
        group.sort_unstable();
        group.dedup();
    }
    for dependencies in &mut multiplier_dependence {
        dependencies.sort_unstable_by_key(|dependency| dependency.active_pos);
        dependencies.dedup_by_key(|dependency| dependency.active_pos);
    }

    let mut row_order: Vec<usize> = (0..groups_out.len()).collect();
    row_order.sort_by_key(|&idx| groups_out[idx].first().copied().unwrap_or(usize::MAX));
    if row_order.iter().enumerate().any(|(idx, &orig)| idx != orig) {
        let mut a_sorted = Array2::<f64>::zeros((rank, p));
        let mut b_sorted = Array1::<f64>::zeros(rank);
        let mut groups_sorted = Vec::with_capacity(rank);
        let mut dependence_sorted = Vec::with_capacity(rank);
        for (out_idx, orig_idx) in row_order.into_iter().enumerate() {
            a_sorted.row_mut(out_idx).assign(&a_out.row(orig_idx));
            b_sorted[out_idx] = b_out[orig_idx];
            groups_sorted.push(groups_out[orig_idx].clone());
            dependence_sorted.push(multiplier_dependence[orig_idx].clone());
        }
        a_out = a_sorted;
        b_out = b_sorted;
        groups_out = groups_sorted;
        multiplier_dependence = dependence_sorted;
    }

    if rank < k {
        log::debug!(
            "rank-reduced active constraints from {} to {} rows (rank deficiency {})",
            k,
            rank,
            k - rank
        );
    }

    (a_out, b_out, groups_out, multiplier_dependence)
}

// ============================================================================
// Operator constraint geometry and finite dual projection
//
// The factored Khatri-Rao monotonicity cone has `n · p_shape` rows over
// `p_resp · p_cov` coefficients; its dense materialization is gigabytes while
// every operation the primal active-set method performs factors through the
// `n × p_cov` covariate design. Every full-row-set sweep (activation scan,
// ratio test, violation gate) runs on batched constraint values, never on
// explicit rows. Strict quadratic entry points for both Dense and factored
// carriers use the finite dual metric projection below; the primal loop remains
// only for the operator strict-interior construction that permits a feasible
// tangent chord.
// ============================================================================

/// Batched full-row-set geometry for a [`ConstraintSet`].
///
/// `scaled_margin` shifts every non-vacuous row inward by that amount in
/// scaled (geometric) units — `a_iᵀβ ≥ b_i + scaled_margin·‖a_i‖` — which is
/// exactly the uniform interior-seed shift the dense strict projection
/// applies. The main QP solve uses `scaled_margin = 0`.
struct ConstraintSetOps<'a> {
    set: &'a ConstraintSet,
    norms: Vec<f64>,
    bounds: Vec<f64>,
    scaled_margin: f64,
}

impl<'a> ConstraintSetOps<'a> {
    fn new(set: &'a ConstraintSet, scaled_margin: f64) -> Result<Self, EstimationError> {
        let m = set.nrows();
        let mut norms = Vec::with_capacity(m);
        let mut bounds = Vec::with_capacity(m);
        for row in 0..m {
            norms.push(set.row_norm(row).map_err(|e| {
                EstimationError::ParameterConstraintViolation(format!(
                    "constraint-set row norm: {e}"
                ))
            })?);
            bounds.push(set.bound(row).map_err(|e| {
                EstimationError::ParameterConstraintViolation(format!(
                    "constraint-set row bound: {e}"
                ))
            })?);
        }
        Ok(Self {
            set,
            norms,
            bounds,
            scaled_margin,
        })
    }

/// Operator view of only the rows that are tight at `beta`. Inactive rows
    /// do not constrain the tangent cone, so make them vacuous by zeroing both
    /// their cached norm and bound while retaining the original row indexing.
    /// This avoids materializing the potentially enormous tight submatrix and
    /// keeps returned active ids in the parent [`ConstraintSet`] coordinates.
    fn tangent_face(set: &'a ConstraintSet, beta: &Array1<f64>) -> Result<Self, EstimationError> {
        let mut ops = Self::new(set, 0.0)?;
        let values = ops.values(beta)?;
        for row in 0..ops.nrows() {
            if ops.norms[row] <= 0.0 {
                if ops.bounds[row] > 0.0 {
                    crate::bail_invalid_estim!(
                        "infeasible zero-norm constraint row {} entered tangent-face projection",
                        row
                    );
                }
                ops.bounds[row] = 0.0;
                continue;
            }
            let is_tight = ops.scaled_slack(&values, row) <= ACTIVE_SET_PRIMAL_FEASIBILITY_TOL;
            // Tangent directions are homogeneous even when the original
            // feasible set is affine: a_i^T d >= 0 on a tight row.
            ops.bounds[row] = 0.0;
            if !is_tight {
                ops.norms[row] = 0.0;
            }
        }
        Ok(ops)
    }

    fn nrows(&self) -> usize {
        self.norms.len()
    }

    fn values(&self, x: &Array1<f64>) -> Result<Array1<f64>, EstimationError> {
        self.set.values(x.view()).map_err(|e| {
            EstimationError::ParameterConstraintViolation(format!("constraint-set values: {e}"))
        })
    }

    /// Signed scaled slack of one row given the batched raw values, with the
    /// same ±∞ zero-norm semantics as [`scaled_constraint_slack`].
    #[inline]
    fn scaled_slack(&self, values: &Array1<f64>, row: usize) -> f64 {
        let norm = self.norms[row];
        if norm > 0.0 {
            (values[row] - self.bounds[row]) / norm - self.scaled_margin
        } else if self.bounds[row] > 0.0 {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        }
    }

    /// Gather the working rows as an explicit UNIT-normalized system (the
    /// per-row scale the dense path reaches via up-front canonicalization),
    /// with the margin shift folded into `b`. Zero-norm rows are rejected —
    /// they are vacuous and must never enter a working set.
    fn gather_unit_rows(
        &self,
        rows: &[usize],
    ) -> Result<LinearInequalityConstraints, EstimationError> {
        let mut gathered = self.set.gather_rows(rows).map_err(|e| {
            EstimationError::ParameterConstraintViolation(format!(
                "constraint-set working-row gather: {e}"
            ))
        })?;
        for (out_row, &row) in rows.iter().enumerate() {
            let norm = self.norms[row];
            if norm <= 0.0 {
                crate::bail_invalid_estim!(
                    "vacuous zero-norm constraint row {} entered the working set",
                    row
                );
            }
            let inv = 1.0 / norm;
            gathered.a.row_mut(out_row).mapv_inplace(|v| v * inv);
            gathered.b[out_row] = self.bounds[row] * inv + self.scaled_margin;
        }
        Ok(gathered)
    }

}

/// Add every geometrically independent violated separator available at one
/// operator iterate, in descending scaled-violation order.
///
/// A factored cone can expose `m ≫ p` violated observation rows after a
/// globalized Newton step leaves the previous endpoint face. Adding one row
/// and re-solving the conditioned `p`-dimensional KKT system after every
/// separator makes face discovery cost `O(p)` dense factorizations. The #979
/// CTN witness had `m=24_000`, `p=144`, and only 24 point-tight warm rows; that
/// serial path spent the remainder of a 300-second command inside one metric
/// projection.
///
/// This routine performs one full value scan (already required by the primal
/// gate), orders candidates by their geometric violation, and streams their
/// unit normals in coefficient-sized chunks. Modified Gram--Schmidt extends
/// the current active normal basis until no coefficient-space direction
/// remains or the rank reaches `p`. At most `p` dense rows are retained and no
/// `m × p` matrix is materialized.
fn independent_violated_operator_rows(
    ops: &ConstraintSetOps<'_>,
    values: &Array1<f64>,
    active: &[usize],
    is_active: &[bool],
    banned: &[bool],
    max_new: usize,
) -> Result<Vec<usize>, EstimationError> {
    let p = ops.set.ncols();
    if max_new == 0 {
        return Ok(Vec::new());
    }
    if values.len() != ops.nrows()
        || is_active.len() != ops.nrows()
        || banned.len() != ops.nrows()
    {
        crate::bail_invalid_estim!(
            "operator batch-separation dimension mismatch: values={}, active_mask={}, \
             banned_mask={}, constraints={}",
            values.len(),
            is_active.len(),
            banned.len(),
            ops.nrows(),
        );
    }

    let mut candidates = Vec::<(usize, f64)>::new();
    for row in 0..ops.nrows() {
        if is_active[row] || banned[row] || ops.norms[row] <= 0.0 {
            continue;
        }
        let violation = (-ops.scaled_slack(values, row)).max(0.0);
        if violation > ACTIVE_SET_PRIMAL_FEASIBILITY_TOL {
            candidates.push((row, violation));
        }
    }
    candidates.sort_unstable_by(|(left_row, left_violation), (right_row, right_violation)| {
        right_violation
            .total_cmp(left_violation)
            .then_with(|| left_row.cmp(right_row))
    });
    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    // Every gathered row is unit-normalized. Use the same relative rank scale
    // as the working-face reducer, with `p` (the maximum attainable rank) in
    // place of a row-count-dependent tolerance.
    let rank_tolerance = 100.0 * f64::EPSILON * p.max(1) as f64;
    let mut basis = Vec::<Array1<f64>>::with_capacity(p);
    if !active.is_empty() {
        let active_rows = ops.gather_unit_rows(active)?;
        for row in active_rows.a.rows() {
            extend_operator_normal_basis(&mut basis, row, rank_tolerance);
        }
    }

    let chunk_size = p.max(32);
    let mut selected = Vec::with_capacity(max_new.min(p.saturating_sub(basis.len())));
    for chunk in candidates.chunks(chunk_size) {
        let chunk_ids = chunk.iter().map(|(row, _)| *row).collect::<Vec<_>>();
        let gathered = ops.gather_unit_rows(&chunk_ids)?;
        for (position, &row) in chunk_ids.iter().enumerate() {
            if extend_operator_normal_basis(
                &mut basis,
                gathered.a.row(position),
                rank_tolerance,
            ) {
                selected.push(row);
                if selected.len() == max_new || basis.len() == p {
                    return Ok(selected);
                }
            }
        }
    }
    Ok(selected)
}

/// Reorthogonalized modified Gram--Schmidt append for one unit constraint
/// normal. Returns true exactly when the row adds a resolved normal-space
/// direction.
fn extend_operator_normal_basis(
    basis: &mut Vec<Array1<f64>>,
    row: ArrayView1<'_, f64>,
    rank_tolerance: f64,
) -> bool {
    let mut residual = row.to_owned();
    // The second pass prevents a long, nearly dependent active basis from
    // manufacturing a false new direction through first-pass roundoff.
    for _ in 0..2 {
        for direction in basis.iter() {
            let projection = residual.dot(direction);
            residual.scaled_add(-projection, direction);
        }
    }
    let residual_norm = residual.dot(&residual).sqrt();
    if !(residual_norm.is_finite() && residual_norm > rank_tolerance) {
        return false;
    }
    residual /= residual_norm;
    basis.push(residual);
    true
}

/// Retain only candidate row ids that are genuinely tight at `beta`.
///
/// Active-face provenance is point-local. A constrained QP reports its full
/// endpoint face, while trust-region globalization may accept a strict
/// subsegment whose endpoint-only rows are still slack. This helper is the
/// shared handoff for warm starts and terminal tangent-space evidence: it uses
/// the carrier's exact row scaling, preserves canonical input order, and never
/// scans rows outside the sparse candidate face.
pub fn constraint_set_rows_tight_at_point(
    set: &ConstraintSet,
    beta: &Array1<f64>,
    candidate_rows: &[usize],
) -> Result<Vec<usize>, EstimationError> {
    if set.ncols() != beta.len() {
        crate::bail_invalid_estim!(
            "active-face point dimension mismatch: set has {} columns, beta has {}",
            set.ncols(),
            beta.len()
        );
    }
    let mut seen = HashSet::with_capacity(candidate_rows.len());
    let mut unique = Vec::with_capacity(candidate_rows.len());
    for &row in candidate_rows {
        if row < set.nrows() && seen.insert(row) {
            unique.push(row);
        }
    }
    if unique.is_empty() {
        return Ok(Vec::new());
    }
    let gathered = set.gather_rows(&unique).map_err(|error| {
        EstimationError::ParameterConstraintViolation(format!(
            "active-face candidate-row gather failed: {error}"
        ))
    })?;
    let mut tight = Vec::with_capacity(unique.len());
    for (position, &row) in unique.iter().enumerate() {
        let constraint_row = gathered.a.row(position);
        let norm = constraint_row.dot(&constraint_row).sqrt();
        if norm > 0.0 {
            let scaled_slack = (constraint_row.dot(beta) - gathered.b[position]) / norm;
            if scaled_slack <= ACTIVE_SET_WORKING_FACE_TOL {
                tight.push(row);
            }
        }
    }
    Ok(tight)
}

/// Project a stationarity residual onto the normal cone of an operator-carried
/// constraint set without materializing its complete tight face.
///
/// Lawson–Hanson discovers the required generators through batched operator
/// products and gathers only its `O(p)` passive rows. Selection depends only
/// on the current face geometry, never on a warm active-set history.
/// `seed_active` is output provenance only: tight seed rows are retained in the
/// returned sparse face even when their KKT multiplier is zero, but they never
/// enter the Lawson–Hanson pivot order or alter the projected vector.
pub fn project_stationarity_residual_on_constraint_set(
    residual: &Array1<f64>,
    beta: &Array1<f64>,
    set: &ConstraintSet,
    seed_active: &[usize],
) -> Option<(Array1<f64>, Vec<usize>)> {
    let p = residual.len();
    if beta.len() != p || set.ncols() != p {
        return None;
    }
    match set {
        ConstraintSet::KhatriRaoCone(cone) if cone.p_left() != 1 || cone.coupled_rows() != &[0] => {
            // Each coupled response row occupies a disjoint `p_cov` slice,
            // and the projection Hessian is identity. The global projection is
            // therefore the exact direct sum of small row projections. Solving
            // all response rows in one `p_left*p_cov` KKT system needlessly
            // pays cubic global algebra at the all-tight CTN vertex.
            let p_cov = cone.factor().ncols();
            let n = cone.factor().nrows();
            let mut projected = residual.clone();
            let mut active = Vec::new();
            for (slot, &coefficient_row) in cone.coupled_rows().iter().enumerate() {
                let start = coefficient_row * p_cov;
                let end = start + p_cov;
                let local_residual = residual.slice(s![start..end]).to_owned();
                let local_beta = beta.slice(s![start..end]).to_owned();
                let local_set = ConstraintSet::KhatriRaoCone(cone.single_coupled_slot(slot).ok()?);
                let row_start = slot * n;
                let row_end = row_start + n;
                let local_seed: Vec<usize> = seed_active
                    .iter()
                    .copied()
                    .filter(|&row| row >= row_start && row < row_end)
                    .map(|row| row - row_start)
                    .collect();
                let (local_projected, local_active) =
                    project_stationarity_residual_on_constraint_set(
                        &local_residual,
                        &local_beta,
                        &local_set,
                        &local_seed,
                    )?;
                projected.slice_mut(s![start..end]).assign(&local_projected);
                active.extend(local_active.into_iter().map(|row| row_start + row));
            }
            Some((projected, active))
        }
        ConstraintSet::BlockDiagonal { blocks, .. } => {
            // The same direct-sum identity applies to explicitly placed blocks;
            // columns outside all blocks are unconstrained and retain their
            // original residual components.
            let mut projected = residual.clone();
            let mut active = Vec::new();
            let mut row_offset = 0usize;
            for block in blocks {
                let width = block.set.ncols();
                let start = block.col_start;
                let end = start + width;
                let local_residual = residual.slice(s![start..end]).to_owned();
                let local_beta = beta.slice(s![start..end]).to_owned();
                let row_end = row_offset + block.set.nrows();
                let local_seed: Vec<usize> = seed_active
                    .iter()
                    .copied()
                    .filter(|&row| row >= row_offset && row < row_end)
                    .map(|row| row - row_offset)
                    .collect();
                let (local_projected, local_active) =
                    project_stationarity_residual_on_constraint_set(
                        &local_residual,
                        &local_beta,
                        &block.set,
                        &local_seed,
                    )?;
                projected.slice_mut(s![start..end]).assign(&local_projected);
                active.extend(local_active.into_iter().map(|row| row_offset + row));
                row_offset = row_end;
            }
            Some((projected, active))
        }
        _ => project_stationarity_residual_on_constraint_set_undivided(
            residual,
            beta,
            set,
            seed_active,
        ),
    }
}

fn project_stationarity_residual_on_constraint_set_undivided(
    residual: &Array1<f64>,
    beta: &Array1<f64>,
    set: &ConstraintSet,
    seed_active: &[usize],
) -> Option<(Array1<f64>, Vec<usize>)> {
    let ops = ConstraintSetOps::tangent_face(set, beta).ok()?;
    let (multipliers, projected) = nonnegative_cone_projection_by_rows(
        &ops.norms,
        residual,
        |candidate| ops.values(candidate).ok(),
        |rows| ops.set.gather_rows(rows).ok().map(|gathered| gathered.a),
    )?;
    let mut active: Vec<usize> = multipliers.into_iter().map(|(row, _)| row).collect();
    for &row in seed_active {
        if row < ops.nrows() && ops.norms[row] > 0.0 && !active.contains(&row) {
            active.push(row);
        }
    }
    Some((projected, active))
}

/// Strictly-interior projection onto a [`ConstraintSet`]: the operator
/// analogue of [`project_point_strictly_into_feasible_cone`]. Dense sets
/// delegate to the dense projection (including its anti-parallel equality
/// lift); the factored cone is homogeneous and one-sided, so the projection
/// is a single identity-Hessian QP against the margin-shifted rows. The QP uses
/// the same finite dual active-set solver as production metric projection:
/// coefficient dimension bounds its dense work, while carrier row count enters
/// only through operator scans.
///
/// A refusal is a typed [`EstimationError::ParameterConstraintViolation`]
/// naming the failing condition (dimension mismatch, non-finite iterate, or
/// the specific row whose half-margin the projection could not clear), never a
/// bare `None`: the caller decides whether that refusal is fatal or a soft
/// fallback, but it is never a silent one.
pub fn project_point_strictly_into_feasible_constraint_set(
    point: &Array1<f64>,
    set: &ConstraintSet,
) -> Result<Array1<f64>, EstimationError> {
    match set {
        ConstraintSet::Dense(dense) => {
            // The dense arm keeps its `Option` contract (it has other callers);
            // its refusal is retyped here so this seam carries a diagnostic
            // rather than a bare `None`.
            project_point_strictly_into_feasible_cone(point, dense).ok_or_else(|| {
                EstimationError::ParameterConstraintViolation(
                    "dense strict-interior projection could not certify a feasible point"
                        .to_string(),
                )
            })
        }
        _ => {
            let p = point.len();
            if set.ncols() != p {
                return Err(EstimationError::ParameterConstraintViolation(format!(
                    "strict-interior projection dimension mismatch: point length {p} != constraint columns {}",
                    set.ncols()
                )));
            }
            let ops = ConstraintSetOps::new(set, ACTIVE_SET_INTERIOR_SEED_MARGIN)?;
            let identity = Array2::<f64>::eye(p);
            // min ½‖β − point‖² ⇒ Hessian = I and rhs = point. This is the
            // same strictly convex operator projection solved by the finite
            // Goldfarb--Idnani dual path below, not the retired primal add/drop
            // direction loop. The latter bounded work by the carrier's row
            // count; a tiny roundoff repair on the 320k-row CTN cone could
            // therefore spend the entire command budget enumerating row ids.
            // Here the face rank and transition bounds depend only on p, and m
            // contributes only full operator scans.
            let factor = identity.cholesky(Side::Lower).map_err(|error| {
                EstimationError::InvalidInput(format!(
                    "strict-interior identity metric could not be factored: {error}"
                ))
            })?;
            let (beta, _) = solve_operator_metric_projection_dual_active_set(
                &identity,
                point,
                point,
                &factor,
                &ops,
                &[],
            )?;
            if beta.iter().any(|v| !v.is_finite()) {
                return Err(EstimationError::ParameterConstraintViolation(
                    "strict-interior projection produced a non-finite iterate".to_string(),
                ));
            }
            // Certify against the ORIGINAL (unshifted) rows with half-margin
            // clearance, mirroring the dense projection's exit contract.
            const SEED_FEASIBILITY_TOL: f64 = 1e-9;
            let unshifted = ConstraintSetOps::new(set, 0.0)?;
            let values = unshifted.values(&beta)?;
            let half_margin = 0.5 * ACTIVE_SET_INTERIOR_SEED_MARGIN - SEED_FEASIBILITY_TOL;
            for row in 0..unshifted.nrows() {
                if unshifted.norms[row] <= 0.0 {
                    continue;
                }
                let slack = unshifted.scaled_slack(&values, row);
                if slack < half_margin {
                    return Err(EstimationError::ParameterConstraintViolation(format!(
                        "strict-interior projection could not clear the half-margin at row {row}: \
                         scaled slack {slack:.3e} < {half_margin:.3e}"
                    )));
                }
            }
            Ok(beta)
        }
    }
}

/// Re-solve one discovered passive face in the original metric.
///
/// The terminal KKT contract admits multipliers down to
/// `-ACTIVE_SET_KKT_DUAL_FEASIBILITY_TOL`; the transition rule must use that
/// same numerical cone. Treating every roundoff-sized negative multiplier as a
/// leaving pivot makes a degenerate face alternate between equivalent bases
/// that the eventual certificate would accept. Materially negative rows leave
/// one at a time in Bland order (lowest constraint id), so the conditioned
/// phase has a deterministic anti-cycling pivot rule.
fn refine_operator_metric_face(
    hessian: &Array2<f64>,
    rhs: &Array1<f64>,
    unconstrained: &Array1<f64>,
    ops: &ConstraintSetOps<'_>,
    active: &mut Vec<usize>,
    is_active: &mut [bool],
    transitions: &mut usize,
) -> Result<(Array1<f64>, Array1<f64>), EstimationError> {
    loop {
        if active.is_empty() {
            return Ok((unconstrained.clone(), Array1::zeros(0)));
        }
        let rows = ops.gather_unit_rows(active)?;
        // Solve for the absolute endpoint, not a correction to the free point.
        // The correction is routinely the negation of an O(1) free point while
        // the constrained endpoint is O(eps) or smaller. Forming
        // `unconstrained + correction` then loses absolute digits to
        // cancellation that no subsequent face solve can recover, and an
        // active row can drift by O(eps) even though its endpoint-scale
        // representable residual is O(eps²). The original quadratic is
        //
        //     1/2 beta' H beta - rhs' beta,
        //
        // so solving it directly with `A beta = b` is algebraically identical
        // and preserves the endpoint's own scale.
        let objective_gradient = -rhs;
        let (candidate, system_multipliers) = solve_kkt_direction(
            hessian,
            &objective_gradient,
            &rows.a,
            Some(&rows.b),
        )?;
        let refined_multipliers = -system_multipliers;
        let leaving_position = refined_multipliers
            .iter()
            .enumerate()
            .filter(|(_, value)| {
                !value.is_finite() || **value < -ACTIVE_SET_KKT_DUAL_FEASIBILITY_TOL
            })
            .min_by_key(|(position, _)| active[*position])
            .map(|(position, _)| position);
        let Some(leaving_position) = leaving_position else {
            return Ok((candidate, refined_multipliers));
        };
        let leaving_row = active.remove(leaving_position);
        is_active[leaving_row] = false;
        *transitions += 1;
    }
}

/// Numerical dependence floor, in the whitened metric, for "the entering
/// constraint normal is already spanned by the active face".
///
/// The dual step direction of the entering row `p` is `z = H⁻¹(n_p − Nᵀr)`,
/// whose whitened form is the component of `L⁻¹n_p` orthogonal to the whitened
/// active normals. `n_pᵀz = ‖z_w‖²` is therefore the *rate* at which a step
/// closes row `p`'s violation, and it vanishes exactly when `n_p ∈ span(N)`. The
/// test is relative to `‖L⁻¹n_p‖`, so it is invariant to the metric's scale, and
/// it sits far above `eps`: a row that differs from the face only by roundoff
/// must be classified DEPENDENT (and handled by a dual drop), never handed a
/// spurious enormous step length.
const ACTIVE_SET_DUAL_DEPENDENCE_TOL: f64 = 1e-11;

/// Thin QR by reorthogonalized modified Gram--Schmidt: `columns = Q R` with `Q`
/// orthonormal (returned as its column list) and `R` upper triangular.
///
/// Returns `None` when a column is numerically dependent on its predecessors.
/// The dual active-set solve maintains an independent face by construction, so
/// `None` is a genuine numerical breakdown rather than an expected branch, and
/// the caller converts it into a typed refusal.
fn thin_qr_reorthogonalized(
    columns: &[Array1<f64>],
    rank_tolerance: f64,
) -> Option<(Vec<Array1<f64>>, Array2<f64>)> {
    let k = columns.len();
    let mut q: Vec<Array1<f64>> = Vec::with_capacity(k);
    let mut r = Array2::<f64>::zeros((k, k));
    for (column_index, column) in columns.iter().enumerate() {
        let scale = column.dot(column).sqrt();
        let mut residual = column.clone();
        // Two passes: one sweep leaves a long, nearly dependent basis able to
        // manufacture a false orthogonal direction out of first-pass roundoff.
        for _ in 0..2 {
            for (basis_index, basis) in q.iter().enumerate() {
                let projection = residual.dot(basis);
                r[[basis_index, column_index]] += projection;
                residual.scaled_add(-projection, basis);
            }
        }
        let norm = residual.dot(&residual).sqrt();
        if !(norm.is_finite() && scale.is_finite() && norm > rank_tolerance * scale.max(1.0)) {
            return None;
        }
        r[[column_index, column_index]] = norm;
        residual /= norm;
        q.push(residual);
    }
    Some((q, r))
}

/// Back substitution against an upper-triangular `R` (`R x = y`).
fn upper_triangular_back_substitution(r: &Array2<f64>, y: &Array1<f64>) -> Option<Array1<f64>> {
    let k = y.len();
    if r.nrows() != k || r.ncols() != k {
        return None;
    }
    let mut x = Array1::<f64>::zeros(k);
    for row in (0..k).rev() {
        let mut sum = y[row];
        for column in (row + 1)..k {
            sum -= r[[row, column]] * x[column];
        }
        let pivot = r[[row, row]];
        if !(pivot.is_finite() && pivot != 0.0) {
            return None;
        }
        x[row] = sum / pivot;
    }
    if array_is_finite(&x) { Some(x) } else { None }
}

/// One violated row plus its scaled violation, as produced by a full scan.
struct ViolatedConstraintRow {
    row: usize,
    violation: f64,
}

/// Full-set primal scan.
///
/// `worst` owns the one-sided public feasibility decision over EVERY row, while
/// `inactive` contains only rows the dual iteration can admit. Active rows obey
/// a stronger, two-sided equality contract that is certified separately after
/// the original-metric face solve. Keeping primal feasibility and separator
/// admission in distinct fields prevents an empty inactive queue from being
/// mistaken for an all-row certificate (#979).
struct OperatorViolationScan {
    worst: ViolatedConstraintRow,
    inactive: Vec<ViolatedConstraintRow>,
}

impl OperatorViolationScan {
    fn is_primal_feasible(&self) -> bool {
        self.worst.violation <= ACTIVE_SET_PRIMAL_FEASIBILITY_TOL
    }
}

fn scan_operator_violations(
    ops: &ConstraintSetOps<'_>,
    values: &Array1<f64>,
    is_active: &[bool],
) -> Result<OperatorViolationScan, EstimationError> {
    if values.len() != ops.nrows() || is_active.len() != ops.nrows() {
        crate::bail_invalid_estim!(
            "operator violation scan dimension mismatch: values={}, active_mask={}, rows={}",
            values.len(),
            is_active.len(),
            ops.nrows(),
        );
    }
    let mut worst = 0.0_f64;
    let mut worst_row = 0usize;
    let mut inactive = Vec::<ViolatedConstraintRow>::new();
    for row in 0..ops.nrows() {
        if ops.norms[row] <= 0.0 {
            // A vacuous row constrains nothing unless its bound is positive, in
            // which case the feasible set is empty and no projection exists.
            if ops.bounds[row] > 0.0 {
                return Err(EstimationError::ParameterConstraintViolation(format!(
                    "operator metric projection has an infeasible zero-norm constraint row {row} \
                     with bound {:.3e}",
                    ops.bounds[row]
                )));
            }
            continue;
        }
        let violation = (-ops.scaled_slack(values, row)).max(0.0);
        if violation > worst {
            worst = violation;
            worst_row = row;
        }
        if violation > ACTIVE_SET_PRIMAL_FEASIBILITY_TOL && !is_active[row] {
            inactive.push(ViolatedConstraintRow { row, violation });
        }
    }
    Ok(OperatorViolationScan {
        worst: ViolatedConstraintRow {
            row: worst_row,
            violation: worst,
        },
        inactive,
    })
}

/// Goldfarb--Idnani dual active-set solve of the strictly convex operator
/// metric projection
///
/// ```text
/// minimize  ½ βᵀHβ − rhsᵀβ    subject to   Aβ ≥ b,   H ≻ 0.
/// ```
///
/// # Why this algorithm and not an add/drop primal face iteration
///
/// The predecessor solved the equality-constrained subproblem on a working
/// face, added violated rows, dropped negative-multiplier rows, and repeated.
/// That iteration has **no monotone merit function**: nothing forbids it from
/// returning to a face it has already left, so its only termination argument was
/// an exact-state memo plus a temporary ban set — and on a degenerate face
/// (`m ≫ p` with many parallel rows, exactly the shape-constrained
/// transformation carriers this solver exists for) it duly cycled and refused
/// (#2432: 565 / 390 / 210 / 70 dual transitions across three shapes).
///
/// The dual method removes that failure mode *structurally*:
///
/// * between separators, `(β, μ)` is the exact minimizer subject to the active
///   rows held as EQUALITIES, with `μ ≥ 0`; during a separator pivot, its
///   cumulative pending multiplier is part of that same primal/dual
///   representation until the row is fully admitted;
/// * an entering row is closed by a step `t = min(t₁, t₂)` along
///   `z = H⁻¹(n_p − Nᵀr)`, where `t₂` reaches that row's boundary and `t₁` is the
///   largest step keeping every multiplier nonnegative;
/// * a **full** step (`t = t₂ > 0`) strictly increases the dual objective, so no
///   active set can ever be revisited;
/// * a **partial** step (`t = t₁ < t₂`) drops exactly one row and keeps the same
///   entering row, so at most `|A| ≤ p` of them can occur consecutively without
///   an intervening strict increase.
///
/// Together those are a finite-termination proof that does not assume
/// non-degeneracy: no anti-cycling rule, no ban set and no state memo is needed,
/// because a cycle is not representable. Linear independence of the active
/// normals is maintained rather than assumed — a row dependent on the face
/// (`‖z_w‖ ≈ 0`) can only be admitted after a dual drop makes room for it, and
/// if no drop is available the feasible set is empty and the solve refuses with
/// that diagnosis instead of grinding.
///
/// # Operator-native cost
///
/// Full constraint scans (`ops.values`) happen only when the candidate queue
/// empties; an individual entering row costs one single-row gather plus an
/// `O(p·k²)` face factorization with `k ≤ p`. Observation-row cardinality
/// therefore affects linear scans only, never the size or number of dense
/// systems — the #979 property, preserved. The caller's warm active set and the
/// batched independent-separator scan feed the queue as *ordering hints only*:
/// the returned minimizer is a function of `(H, rhs, A, b)` alone, so warm-start
/// history can change how fast this solve runs but never what it returns.
fn solve_operator_metric_projection_dual_active_set(
    hessian: &Array2<f64>,
    rhs: &Array1<f64>,
    unconstrained: &Array1<f64>,
    factor: &gam_linalg::faer_ndarray::FaerCholeskyFactor,
    ops: &ConstraintSetOps<'_>,
    warm_rows: &[usize],
) -> Result<(Array1<f64>, Vec<usize>), EstimationError> {
    use gam_linalg::triangular::{
        back_substitution_lower_transpose, forward_substitution_lower_vector,
    };

    let p = unconstrained.len();
    let m = ops.nrows();
    let lower = factor.lower_triangular();
    let face_rank_tolerance = 100.0 * f64::EPSILON * (p.max(1) as f64);

    let mut beta = unconstrained.clone();
    let mut active = Vec::<usize>::new();
    let mut is_active = vec![false; m];
    // Whitened active normals `L⁻¹n_i`, index-parallel to `active`.
    let mut whitened_active = Vec::<Array1<f64>>::new();
    let mut multipliers = Vec::<f64>::new();
    let mut queue = std::collections::VecDeque::<usize>::new();
    for &row in warm_rows {
        if row < m && ops.norms[row] > 0.0 && !queue.contains(&row) {
            queue.push_back(row);
        }
    }

    // Operational bounded-work limits, not a finite-termination theorem. Exact
    // arithmetic cannot revisit a face after a strict dual improvement, but the
    // number of distinct faces is not bounded by a polynomial in `p`. Reaching
    // either limit therefore surfaces an explicit refusal without diagnosing
    // its cause as floating-point breakdown.
    let max_transitions = 8usize
        .saturating_mul(p.saturating_add(2))
        .saturating_mul(p.saturating_add(2))
        .saturating_add(64);
    let max_refills = 4usize.saturating_mul(p).saturating_add(32);
    let mut transitions = 0usize;
    let mut refills = 0usize;

    let (candidate, refined_multipliers) = loop {
        'dual: loop {
            let Some(entering) = queue.pop_front() else {
                let values = ops.values(&beta)?;
                let scan = scan_operator_violations(ops, &values, &is_active)?;
                if scan.inactive.is_empty() {
                    // No row remains that this phase can admit. An active row
                    // may have accumulated forward-error drift, so this is a
                    // transition to original-metric face conditioning, not a
                    // claim that the all-row primal certificate passed.
                    break 'dual;
                }
                refills += 1;
                if refills > max_refills {
                    return Err(EstimationError::ParameterConstraintViolation(format!(
                        "operator metric projection reached its bounded-work limit of \
                         {max_refills} separator scans with {} rows still violated \
                         (worst {:.3e})",
                        scan.inactive.len(),
                        scan.inactive
                            .iter()
                            .map(|entry| entry.violation)
                            .fold(0.0_f64, f64::max),
                    )));
                }
                // Prefer a batch of mutually independent separators: each is
                // admissible without an intervening drop, so one scan can rebuild
                // a whole face. When the face already spans every violated normal
                // the batch is empty and plain most-violated order is queued —
                // those rows are dependent, and the dual drop rule makes room.
                let no_bans = vec![false; m];
                let batch = independent_violated_operator_rows(
                    ops,
                    &values,
                    &active,
                    &is_active,
                    &no_bans,
                    p.saturating_sub(active.len()),
                )?;
                if batch.is_empty() {
                    let mut ordered = scan.inactive;
                    ordered.sort_unstable_by(|left, right| {
                        right
                            .violation
                            .total_cmp(&left.violation)
                            .then_with(|| left.row.cmp(&right.row))
                    });
                    queue.extend(
                        ordered
                            .iter()
                            .take(p.saturating_add(8))
                            .map(|entry| entry.row),
                    );
                } else {
                    queue.extend(batch);
                }
                continue 'dual;
            };
            if is_active[entering] || ops.norms[entering] <= 0.0 {
                continue 'dual;
            }
            let entering_rows = ops.gather_unit_rows(&[entering])?;
            let normal = entering_rows.a.row(0).to_owned();
            let bound = entering_rows.b[0];
            let whitened_normal = forward_substitution_lower_vector(lower.view(), normal.view());
            let whitened_scale = whitened_normal.dot(&whitened_normal).sqrt();
            if !(array_is_finite(&whitened_normal) && whitened_scale > 0.0) {
                crate::bail_invalid_estim!(
                    "operator metric projection whitened entering row {entering} to a degenerate \
                     normal (scale {whitened_scale:.3e})"
                );
            }

            // Inner dual loop: hold `entering` fixed and take dual steps until it
            // is admitted (full step) or proven inadmissible (no drop available).
            //
            // A partial step already adds a positive multiplier for `entering`
            // to the primal representation while releasing one old row. The
            // separator must therefore remain pending across every partial
            // drop, even if its remaining violation falls inside the public
            // primal tolerance. Abandoning it there leaves `beta` carrying an
            // unrecorded normal component; terminal face conditioning erases
            // that component and recreates the same violation (#979).
            let mut remaining_violation = bound - normal.dot(&beta);
            if remaining_violation <= ACTIVE_SET_PRIMAL_FEASIBILITY_TOL {
                // The iterate moved since this row was queued; no dual step
                // for this separator has begun, so it is safe to discard.
                continue 'dual;
            }
            let mut entering_multiplier = 0.0_f64;
            loop {
                let (dual_direction, tangent) = if active.is_empty() {
                    (Array1::<f64>::zeros(0), whitened_normal.clone())
                } else {
                    let Some((q, r)) =
                        thin_qr_reorthogonalized(&whitened_active, face_rank_tolerance)
                    else {
                        crate::bail_invalid_estim!(
                            "operator metric projection lost independence of its {} active normals",
                            active.len()
                        );
                    };
                    let projections =
                        Array1::from_iter(q.iter().map(|basis| basis.dot(&whitened_normal)));
                    let Some(dual_direction) =
                        upper_triangular_back_substitution(&r, &projections)
                    else {
                        crate::bail_invalid_estim!(
                            "operator metric projection could not solve its {}-row dual direction",
                            active.len()
                        );
                    };
                    let mut tangent = whitened_normal.clone();
                    for (basis, projection) in q.iter().zip(projections.iter()) {
                        tangent.scaled_add(-projection, basis);
                    }
                    (dual_direction, tangent)
                };

                // `n_pᵀ z = ‖tangent‖²`: the rate at which a unit dual step
                // closes this row's violation, zero exactly on a dependent normal.
                let rate = tangent.dot(&tangent);
                let dependence_floor = ACTIVE_SET_DUAL_DEPENDENCE_TOL * whitened_scale;
                let full_step = if rate > dependence_floor * dependence_floor {
                    remaining_violation / rate
                } else {
                    f64::INFINITY
                };

                // Largest step preserving `μ ≥ 0`, with the lowest constraint id
                // breaking exact ties so the drop rule is deterministic.
                let mut partial_step = f64::INFINITY;
                let mut blocking: Option<usize> = None;
                for (position, &direction) in dual_direction.iter().enumerate() {
                    if !(direction > 0.0) {
                        continue;
                    }
                    let ratio = (multipliers[position] / direction).max(0.0);
                    let replaces = match blocking {
                        None => true,
                        Some(current) => {
                            ratio < partial_step
                                || (ratio == partial_step && active[position] < active[current])
                        }
                    };
                    if replaces {
                        partial_step = ratio;
                        blocking = Some(position);
                    }
                }

                if !full_step.is_finite() && blocking.is_none() {
                    return Err(EstimationError::ParameterConstraintViolation(format!(
                        "operator metric projection proved its constraint set infeasible: row \
                         {entering} is violated by {remaining_violation:.3e} and lies in the span \
                         of the {} active normals with no releasable multiplier",
                        active.len(),
                    )));
                }
                let step = full_step.min(partial_step);
                if !step.is_finite() {
                    crate::bail_invalid_estim!(
                        "operator metric projection produced a non-finite dual step for row \
                         {entering}"
                    );
                }
                if step > 0.0 {
                    let primal_direction =
                        back_substitution_lower_transpose(lower.view(), tangent.view());
                    beta.scaled_add(step, &primal_direction);
                    if !array_is_finite(&beta) {
                        crate::bail_invalid_estim!(
                            "operator metric projection iterate left the finite range"
                        );
                    }
                    for (multiplier, direction) in
                        multipliers.iter_mut().zip(dual_direction.iter())
                    {
                        *multiplier = (*multiplier - step * direction).max(0.0);
                    }
                }
                entering_multiplier += step;
                if !entering_multiplier.is_finite() {
                    crate::bail_invalid_estim!(
                        "operator metric projection accumulated a non-finite multiplier for \
                         entering row {entering}"
                    );
                }

                transitions += 1;
                if transitions > max_transitions {
                    return Err(EstimationError::ParameterConstraintViolation(format!(
                        "operator metric projection reached its bounded-work limit of \
                         {max_transitions} active-set transitions with {} active rows",
                        active.len(),
                    )));
                }

                if full_step <= partial_step {
                    active.push(entering);
                    is_active[entering] = true;
                    whitened_active.push(whitened_normal);
                    multipliers.push(entering_multiplier);
                    continue 'dual;
                }
                // For a partial step, `step < remaining_violation / rate`,
                // hence this quantity is strictly positive in exact
                // arithmetic. Track that algebraic residual instead of
                // re-reading a cancellation-prone row dot-product; terminal
                // original-metric conditioning owns the forward-error repair.
                remaining_violation =
                    (-step).mul_add(rate, remaining_violation).max(0.0);
                let leaving = blocking.expect("a finite partial step names a blocking row");
                let leaving_row = active.remove(leaving);
                whitened_active.remove(leaving);
                multipliers.remove(leaving);
                is_active[leaving_row] = false;
            }
        }

        // Terminal conditioning. Every dual step preserved `Nβ = b_N` in exact
        // arithmetic, so re-deriving the endpoint from the face's KKT system in
        // the ORIGINAL metric changes nothing mathematically — it removes only
        // the drift accumulated along the step path, which is exactly what an
        // absolute feasibility certificate stated in un-whitened row units
        // measures. The refinement also releases any multiplier that
        // conditioning pushed materially negative; a release re-enters the dual
        // iteration from the conditioned face rather than being certified.
        let refined = refine_operator_metric_face(
            hessian,
            rhs,
            unconstrained,
            ops,
            &mut active,
            &mut is_active,
            &mut transitions,
        )?;
        if transitions > max_transitions {
            return Err(EstimationError::ParameterConstraintViolation(format!(
                "operator metric projection reached its bounded-work limit of \
                 {max_transitions} active-set transitions during original-metric face \
                 conditioning with {} active rows",
                active.len(),
            )));
        }
        if !active.is_empty() {
            let active_rows = ops.gather_unit_rows(&active)?;
            let equality =
                certify_active_equalities(&active_rows.a, &active_rows.b, &refined.0);
            if !equality.is_certified() {
                return Err(EstimationError::ParameterConstraintViolation(format!(
                    "operator metric projection conditioned face failed its active-equality \
                     certificate at constraint row {} (active position {}): absolute residual \
                     {:.3e} exceeds roundoff bound {:.3e} over {} active rows",
                    active[equality.worst_row],
                    equality.worst_row,
                    equality.residual,
                    equality.allowed,
                    active.len(),
                )));
            }
        }
        let values = ops.values(&refined.0)?;
        let scan = scan_operator_violations(ops, &values, &is_active)?;
        if scan.is_primal_feasible() {
            break refined;
        }
        if scan.inactive.is_empty() {
            return Err(EstimationError::ParameterConstraintViolation(format!(
                "operator metric projection found all-row scaled violation {:.3e} at row {} \
                 after certifying {} active equalities, but found no inactive separator",
                scan.worst.violation,
                scan.worst.row,
                active.len(),
            )));
        }
        // The conditioned face solve is itself a valid dual state — `β` is the
        // exact equality-constrained minimizer and every multiplier is
        // nonnegative — so the dual iteration resumes from it directly. There
        // is no separate conditioning-round budget: every nonterminal scan now
        // contains an inactive separator, and the pending-separator invariant
        // guarantees that processing it records a transition or returns a typed
        // refusal. The counters above bound operational work; they are not the
        // exact-arithmetic finite-termination proof.
        beta = refined.0;
        multipliers = refined.1.to_vec();
        whitened_active.clear();
        if !active.is_empty() {
            let face_rows = ops.gather_unit_rows(&active)?;
            for position in 0..active.len() {
                whitened_active.push(forward_substitution_lower_vector(
                    lower.view(),
                    face_rows.a.row(position),
                ));
            }
        }
        queue.clear();
        queue.extend(scan.inactive.iter().map(|entry| entry.row));
    };

    let active_ids = active.clone();
    let gradient = hessian.dot(&candidate) - rhs;
    let (stationarity, complementarity, dual_violation) = if active_ids.is_empty() {
        (gradient_inf_norm(&gradient), 0.0, 0.0)
    } else {
        let rows = ops.gather_unit_rows(&active_ids)?;
        let residual = &gradient - &rows.a.t().dot(&refined_multipliers);
        let complementarity = refined_multipliers
            .iter()
            .enumerate()
            .map(|(position, multiplier)| {
                let slack = rows.a.row(position).dot(&candidate) - rows.b[position];
                (multiplier * slack).abs()
            })
            .fold(0.0_f64, f64::max);
        let dual_violation = refined_multipliers
            .iter()
            .map(|multiplier| (-multiplier).max(0.0))
            .fold(0.0_f64, f64::max);
        (
            gradient_inf_norm(&residual),
            complementarity,
            dual_violation,
        )
    };
    let gradient_scale = gradient_inf_norm(&gradient).max(1.0);
    if stationarity > ACTIVE_SET_KKT_STATIONARITY_TOL
        && stationarity / gradient_scale > ACTIVE_SET_KKT_STATIONARITY_TOL
    {
        // WHICH of the two things failed is not recoverable from `residual`
        // alone, and they need opposite repairs (#2592).
        //
        // Stationarity here is `|| g - A_Aᵀ μ ||` for the multipliers this walk
        // recovered. That number is large either because
        //
        //   (a) the point is not stationary on its own face -- `g` has real
        //       mass in the face TANGENT, which no multipliers can absorb, so
        //       the solve stopped somewhere that is not the constrained
        //       minimizer; or
        //   (b) the point IS stationary and the MULTIPLIERS are wrong -- `g`
        //       lies (nearly) in the row space of `A_A`, and the walk simply
        //       failed to recover the `μ` that represents it, e.g. on a
        //       rank-deficient face.
        //
        // The discriminator is the smallest residual ANY multipliers could
        // achieve, `min_μ || g - A_Aᵀ μ ||`, which is exactly the norm of `g`
        // projected onto the face tangent. Report it beside the achieved one:
        // near zero means (b), near `residual` means (a). A ridge keeps the
        // normal-equation solve defined on a rank-deficient face, which is the
        // very case this diagnostic exists to name; it only makes the reported
        // achievable residual an UPPER bound, so it can never turn (a) into (b).
        let achievable = if active_ids.is_empty() {
            Some(gradient_inf_norm(&gradient))
        } else {
            ops.gather_unit_rows(&active_ids).ok().and_then(|rows| {
                let gram = rows.a.dot(&rows.a.t());
                let ridge = 1.0e-12 * gram.diag().iter().fold(0.0_f64, |m, v| m.max(v.abs()));
                let mut regularized = gram;
                for i in 0..regularized.nrows() {
                    regularized[[i, i]] += ridge.max(f64::MIN_POSITIVE);
                }
                regularized.cholesky(Side::Lower).ok().map(|factor| {
                    let least_squares = factor.solvevec(&rows.a.dot(&gradient));
                    gradient_inf_norm(&(&gradient - &rows.a.t().dot(&least_squares)))
                })
            })
        };
        let achievable_report = achievable.map_or_else(
            || "unmeasured".to_string(),
            |value| format!("{value:.3e}"),
        );
        let verdict = match achievable {
            Some(value) if value > 0.5 * stationarity => {
                "the face TANGENT carries the residual, so this point is not the \
                 constrained minimizer of its own face"
            }
            Some(_) => {
                "the residual lies in the face ROW SPACE, so the point is stationary \
                 and the recovered multipliers do not represent its gradient"
            }
            None => "achievable residual unmeasured (face gram not factorizable)",
        };
        return Err(EstimationError::ParameterConstraintViolation(format!(
            "operator metric projection failed stationarity certification: \
             residual={stationarity:.3e}, relative={:.3e}, active={}, transitions={transitions}, \
             achievable={achievable_report} (best over all multipliers), gradient_scale={gradient_scale:.3e}; \
             {verdict}",
            stationarity / gradient_scale,
            active_ids.len(),
        )));
    }
    if dual_violation > ACTIVE_SET_KKT_DUAL_FEASIBILITY_TOL
        || complementarity > ACTIVE_SET_KKT_COMPLEMENTARITY_TOL
    {
        return Err(EstimationError::ParameterConstraintViolation(format!(
            "operator metric projection failed dual/complementarity certification: \
             dual={dual_violation:.3e}, complementarity={complementarity:.3e}, active={}",
            active_ids.len(),
        )));
    }
    Ok((candidate, active_ids))
}

/// Constrained quadratic solve: minimize `½ βᵀHβ − rhsᵀβ` subject to the
/// [`ConstraintSet`]. Dense and factored carriers use the same dual active-set
/// metric projection; only row products and row gathering differ.
///
/// Same public feasibility contract as
/// [`solve_quadratic_with_linear_constraints`]: the returned point is feasible
/// to [`ACTIVE_SET_PRIMAL_FEASIBILITY_TOL`] or the solve errors. For an operator
/// carrier the Hessian must be strictly positive definite; the unique minimizer
/// then satisfies
///
/// ```text
/// β = u + H⁻¹ Cᵀ μ,   Cβ − d ≥ 0,   μ ≥ 0,   μ ⊙ (Cβ − d) = 0
/// ```
///
/// for `u = H⁻¹rhs` and unit-scaled rows `C` with bounds `d`, and
/// [`solve_operator_metric_projection_dual_active_set`] computes it exactly.
/// This is also a semantic boundary: a quadratic-projection API either returns
/// the certified minimizer or an error. It never substitutes a generic feasible
/// descent direction after exhausting a working-set path (#979).
fn solve_strictly_convex_quadratic_with_constraint_set_dual(
    hessian: &Array2<f64>,
    rhs: &Array1<f64>,
    beta_start: &Array1<f64>,
    set: &ConstraintSet,
    warm_active_set: Option<&[usize]>,
) -> Result<(Array1<f64>, Vec<usize>), EstimationError> {
    let p = rhs.len();
    if p == 0
        || hessian.nrows() != p
        || hessian.ncols() != p
        || beta_start.len() != p
        || set.ncols() != p
        || hessian.iter().any(|value| !value.is_finite())
        || rhs.iter().any(|value| !value.is_finite())
        || beta_start.iter().any(|value| !value.is_finite())
    {
        crate::bail_invalid_estim!("operator metric-projection dimension/finite contract failed");
    }
    let factor = hessian.cholesky(Side::Lower).map_err(|error| {
        EstimationError::InvalidInput(format!(
            "operator metric projection requires a strictly positive-definite Hessian: {error}"
        ))
    })?;
    let unconstrained = factor.solvevec(rhs);
    if !array_is_finite(&unconstrained) {
        crate::bail_invalid_estim!("operator metric-projection free solve is non-finite");
    }

    let ops = ConstraintSetOps::new(set, 0.0)?;
    // Warm faces are point-local: globalization may accept only part of the
    // previous QP chord, so retain only cached rows that remain tight at this
    // cycle's accepted `beta_start`. They are an ENTERING ORDER, not a preset
    // basis — the dual solve admits each through the same certified step as any
    // other row, so a stale hint costs a skipped queue pop and nothing else.
    let warm_tight =
        constraint_set_rows_tight_at_point(set, beta_start, warm_active_set.unwrap_or(&[]))?;
    let (candidate, dual_basis) = solve_operator_metric_projection_dual_active_set(
        hessian,
        rhs,
        &unconstrained,
        &factor,
        &ops,
        &warm_tight,
    )?;

    // ── what "active" means to a caller ─────────────────────────────────────
    //
    // `dual_basis` is what the dual walk ADMITTED, and it admits a row only
    // when the running candidate VIOLATES it. A row the caller handed in as its
    // current face, which the answer still lies on, is therefore never admitted
    // and never listed. Every consumer reads this set as the FACE the answer
    // lies on: it nulls the set's tangent before certifying curvature, seeds the
    // next cycle's working set from it, and decides which blockers were
    // resolved. A silently dropped face row makes the consumer count a direction
    // free that the polytope pins — which is precisely how a PHANTOM saddle is
    // manufactured, a failure `resolve_constrained_converged_mode` already names
    // and repairs with `widen_active_sets_to_tight_face` at one call site
    // (#2589; #2432 turned six reduced-face tests red on exactly this, each
    // reporting `[1]` where the face is `[0, 1]`).
    //
    // So report the rows the solve actually established the answer on: the ones
    // it admitted, plus the caller's incoming face rows that the ANSWER is tight
    // on. Note that is tight at the ANSWER, not at `beta_start`: a warm row can
    // be slack at entry and pinned at the endpoint, which is exactly the
    // projection case (`beta_start = 1e-9`, answer `0`), so `warm_tight` above
    // is the wrong set to widen with.
    //
    // Rank-reduce the union, because a degenerate vertex can hand back four
    // parallel rows for one geometric face and this is a face description, not a
    // row list. Deliberately NOT a rescan of every row: an operator carrier can
    // present thousands of equivalent rows (a Khatri-Rao cone tight at one
    // coefficient point) and expanding to all of them is what
    // `operator_cone_does_not_materialize_a_whole_tight_face` forbids. The union
    // is bounded by the rows this solve was already given or found.
    let dual_basis_rows = dual_basis.len();
    let mut face_candidates = dual_basis;
    for row in constraint_set_rows_tight_at_point(set, &candidate, warm_active_set.unwrap_or(&[]))?
    {
        if !face_candidates.contains(&row) {
            face_candidates.push(row);
        }
    }
    if face_candidates.len() == dual_basis_rows {
        return Ok((candidate, face_candidates));
    }
    // The dual's own rows lead, so rank reduction keeps them as the
    // representatives when an incoming row is parallel to one of them.
    let gathered = ops.gather_unit_rows(&face_candidates)?;
    let groups: Vec<Vec<usize>> = face_candidates.iter().map(|row| vec![*row]).collect();
    let (_, _, kept, _) =
        rank_reduce_rows_pivoted_qr_with_dependence(gathered.a, gathered.b, groups);
    // One id per kept group, not every member of it. Rank reduction MERGES a
    // parallel class into a single group rather than dropping its members, so
    // flattening would hand back all four rows of a degenerate vertex and
    // undo the reduction. The lowest index is the canonical representative,
    // which is also what `accepted_face_is_canonical_across_degenerate_qp_row_bases`
    // expects of a face that must not carry the path that produced it.
    let mut face_rows: Vec<usize> = kept
        .into_iter()
        .filter_map(|group| group.into_iter().min())
        .collect();
    face_rows.sort_unstable();
    face_rows.dedup();
    Ok((candidate, face_rows))
}

pub fn solve_quadratic_with_constraint_set(
    hessian: &Array2<f64>,
    rhs: &Array1<f64>,
    beta_start: &Array1<f64>,
    set: &ConstraintSet,
    warm_active_set: Option<&[usize]>,
) -> Result<(Array1<f64>, Vec<usize>), EstimationError> {
    match set {
        ConstraintSet::Dense(dense) => solve_quadratic_with_linear_constraints(
            hessian,
            rhs,
            beta_start,
            dense,
            warm_active_set,
        ),
        _ => {
            if hessian.ncols() != hessian.nrows()
                || rhs.len() != hessian.nrows()
                || beta_start.len() != hessian.nrows()
                || set.ncols() != hessian.nrows()
            {
                crate::bail_invalid_estim!(
                    "operator-constrained quadratic solve: system dimension mismatch"
                );
            }
            solve_strictly_convex_quadratic_with_constraint_set_dual(
                hessian,
                rhs,
                beta_start,
                set,
                warm_active_set,
            )
        }
    }
}

pub(crate) fn solve_newton_direction_with_linear_constraints(
    hessian: &Array2<f64>,
    gradient: &Array1<f64>,
    beta: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
    direction_out: &mut Array1<f64>,
    active_hint: Option<&mut Vec<usize>>,
) -> Result<(), EstimationError> {
    if hessian.nrows() != hessian.ncols()
        || gradient.len() != hessian.nrows()
        || beta.len() != hessian.nrows()
        || constraints.a.ncols() != hessian.nrows()
    {
        crate::bail_invalid_estim!("linear-constrained Newton system dimension mismatch");
    }
    // `gradient = H·beta - rhs` for the local quadratic model, hence
    // `rhs = H·beta - gradient`. Solve the strict-convex QP itself rather than
    // asking the legacy primal face walk for a merely feasible descent chord:
    // a Newton-step API that returns before KKT stationarity makes the outer
    // solver optimize a different object on the next cycle (#2366/#2432).
    let rhs = hessian.dot(beta) - gradient;
    let warm_active = active_hint.as_ref().map(|hint| hint.as_slice());
    let (candidate, active) = solve_quadratic_with_linear_constraints(
        hessian,
        &rhs,
        beta,
        constraints,
        warm_active,
    )?;
    if direction_out.len() != beta.len() {
        *direction_out = Array1::zeros(beta.len());
    }
    direction_out.assign(&(&candidate - beta));
    if let Some(hint) = active_hint {
        hint.clear();
        hint.extend(active);
    }
    Ok(())
}

pub fn solve_quadratic_with_linear_constraints(
    hessian: &Array2<f64>,
    rhs: &Array1<f64>,
    beta_start: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
    warm_active_set: Option<&[usize]>,
) -> Result<(Array1<f64>, Vec<usize>), EstimationError> {
    if hessian.ncols() != hessian.nrows()
        || rhs.len() != hessian.nrows()
        || beta_start.len() != hessian.nrows()
        || constraints.a.ncols() != hessian.nrows()
    {
        crate::bail_invalid_estim!("constrained quadratic solve: system dimension mismatch");
    }
    // Canonicalize at the chokepoint: reject non-finite / infeasible-zero rows
    // and unit-normalize every row, so all downstream slack, activation, and
    // rank tolerances are geometric (scale-free) regardless of the units the
    // caller expressed the constraints in. Row order is preserved, so
    // `warm_active_set` indices and the returned active ids stay valid.
    let constraints = constraints.canonicalized().map_err(|e| {
        EstimationError::ParameterConstraintViolation(format!(
            "constrained quadratic solve: invalid constraint system: {e}"
        ))
    })?;
    // Dense and factored carriers are the same mathematical problem. The old
    // Dense arm used a separate primal add/drop walk with no monotone merit
    // function and, on the competing-risks fixture, stopped on a feasible but
    // nonstationary 3/332 face. Route it through the finite dual metric
    // projection already used by operator carriers: every admitted face is an
    // exact equality-constrained minimizer, multipliers remain nonnegative, and
    // a full pivot strictly increases the dual objective. Warm rows affect
    // ordering only, never the unique answer.
    let set = ConstraintSet::Dense(constraints);
    solve_strictly_convex_quadratic_with_constraint_set_dual(
        hessian,
        rhs,
        beta_start,
        &set,
        warm_active_set,
    )
}

#[cfg(test)]
mod tests {
    use super::{
        ACTIVE_SET_INTERIOR_SEED_MARGIN, ACTIVE_SET_KKT_DUAL_FEASIBILITY_TOL,
        ACTIVE_SET_PRIMAL_FEASIBILITY_TOL, ConstraintRowId, ConstraintSet, ConstraintSetOps,
        ConstraintSetReducedFace, LinearInequalityConstraints,
        array_is_finite, certify_active_equalities, compute_constraint_kkt_diagnostics,
        constraint_set_rows_tight_at_point,
        independent_violated_operator_rows,
        khatri_rao_cone_reduced_face, least_squares_min_norm_any_shape,
        nonnegative_cone_multipliers,
        project_point_strictly_into_feasible_cone,
        project_point_strictly_into_feasible_constraint_set,
        project_stationarity_residual_on_constraint_cone,
        project_stationarity_residual_on_constraint_set,
        rank_reduce_rows_pivoted_qr_with_dependence,
        scaled_constraint_slack, scan_operator_violations, solve_kkt_direction,
        solve_newton_direction_with_linear_constraints, solve_quadratic_with_constraint_set,
        solve_quadratic_with_linear_constraints,

    };
    use crate::estimate::EstimationError;
    use approx::assert_relative_eq;
    use gam_problem::KhatriRaoConeConstraints;
    use ndarray::{Array1, Array2, array};

    fn gather_linear_constraint_rows(
        constraints: &LinearInequalityConstraints,
        rows: &[usize],
    ) -> Result<LinearInequalityConstraints, EstimationError> {
        let p = constraints.a.ncols();
        let mut a = Array2::<f64>::zeros((rows.len(), p));
        let mut b = Array1::<f64>::zeros(rows.len());
        for (out, &row) in rows.iter().enumerate() {
            if row >= constraints.a.nrows() {
                crate::bail_invalid_estim!(
                    "active constraint row {} out of bounds for {} rows",
                    row,
                    constraints.a.nrows()
                );
            }
            a.row_mut(out).assign(&constraints.a.row(row));
            b[out] = constraints.b[row];
        }
        LinearInequalityConstraints::new(a, b)
            .map_err(|error| EstimationError::ParameterConstraintViolation(error.to_string()))
    }

    fn moreau_projection_via_strict_qp(
        residual: &Array1<f64>,
        active_a: &Array2<f64>,
    ) -> Option<(Array1<f64>, Array1<f64>)> {
        let p = residual.len();
        let m = active_a.nrows();
        let constraints =
            LinearInequalityConstraints::new(active_a.clone(), Array1::<f64>::zeros(m))
                .ok()?
                .canonicalized()
                .ok()?;

        // Independent oracle: solve the strictly convex primal tangent-cone QP
        // and reconstruct its canonical-face multipliers.
        let identity = Array2::<f64>::eye(p);
        let origin = Array1::<f64>::zeros(p);
        let rhs = -residual;
        let (tangent_direction, tangent_active) = solve_quadratic_with_linear_constraints(
            &identity,
            &rhs,
            &origin,
            &constraints,
            None,
        )
        .ok()?;
        if !array_is_finite(&tangent_direction) {
            return None;
        }
        let projected = -&tangent_direction;

        let mut lambda_canonical = Array1::<f64>::zeros(m);
        if !tangent_active.is_empty() {
            let gathered = gather_linear_constraint_rows(&constraints, &tangent_active).ok()?;
            let design = gathered.a.t().to_owned();
            let solved =
                least_squares_min_norm_any_shape(&design, &(residual + &tangent_direction))?;
            let scale = residual
                .iter()
                .fold(0.0_f64, |acc, &value| acc.max(value.abs()))
                .max(1.0);
            let tol = 100.0 * f64::EPSILON * (p.max(m) as f64) * scale;
            for (position, &row) in tangent_active.iter().enumerate() {
                let value = solved[position];
                if !value.is_finite() || value < -tol {
                    return None;
                }
                lambda_canonical[row] = value.max(0.0);
            }
        }
        let reconstructed = residual - &constraints.a.t().dot(&lambda_canonical);
        let reconstruction_error = reconstructed
            .iter()
            .zip(projected.iter())
            .fold(0.0_f64, |acc, (&left, &right)| {
                acc.max((left - right).abs())
            });
        let scale = residual
            .iter()
            .fold(0.0_f64, |acc, &value| acc.max(value.abs()))
            .max(1.0);
        if reconstruction_error > 1e-8 * scale || !array_is_finite(&lambda_canonical) {
            return None;
        }

        let mut lambda = Array1::<f64>::zeros(m);
        for row in 0..m {
            let norm = active_a.row(row).dot(&active_a.row(row)).sqrt();
            if norm > 0.0 {
                lambda[row] = lambda_canonical[row] / norm;
            }
        }
        Some((projected, lambda))
    }

    #[test]
    fn active_equality_certificate_rejects_public_tolerance_band_drift() {
        // The #979 production endpoint was accepted by the public 1e-8 primal
        // gate while carrying this much active-equality drift. Against a
        // unit-normalized active row, that is many orders above representational
        // roundoff and must not seed the next reduced-face quadratic.
        let active_a = array![[1.0, 0.0]];
        let rhs = array![0.0];
        let direction = array![8.604942e-9, 0.0];
        let certificate = certify_active_equalities(&active_a, &rhs, &direction);
        assert!(
            !certificate.is_certified(),
            "a tolerance-band endpoint is not a roundoff-resolved active equality"
        );
        assert_eq!(certificate.worst_row, 0);
        assert_relative_eq!(certificate.residual, 8.604942e-9, epsilon = 0.0);
        assert!(certificate.residual > 1.0e6 * certificate.allowed);
    }

    #[test]
    fn active_equality_certificate_uses_the_solve_scale_not_the_collapsed_row_scale() {
        // A degenerate face: row 0 is supported only on coordinate 2, and the
        // solve drove that coordinate to a pure-underflow residue while the rest
        // of the direction stayed O(1). This is the ordinary state of a factored
        // cone whose coefficient block has gone to zero — every observation row
        // over that block is tight at once.
        //
        // Bounding the row by `sum_j |a_ij d_j|` alone makes the tolerance
        // collapse WITH the coordinate (~1e-48 here), so the certificate demands
        // an equality residual no f64 arithmetic can produce and the face can
        // never be certified. That is the observed #979 refusal signature:
        // residual/allowed pinned at ~1/eps regardless of the actual geometry.
        let active_a = array![[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]];
        let rhs = array![0.0, 0.5];
        let collapsed = array![0.5, 0.3, 1.0e-33, 0.0];
        let certificate = certify_active_equalities(&active_a, &rhs, &collapsed);
        assert!(
            certificate.is_certified(),
            "an equality residual {:.3e} that is 1e-33 of the solve scale is \
             roundoff-resolved, not a face defect (allowed {:.3e})",
            certificate.residual,
            certificate.allowed
        );

        // …and the ambient scale does NOT become a blanket loosening: a drift in
        // the public tolerance band on the same face is still refused, because
        // `gamma · ||a_i||_1 · ||d||_inf` is ~1e-16 here, not ~1e-9.
        let drifted = array![0.5, 0.3, 1.0e-9, 0.0];
        let certificate = certify_active_equalities(&active_a, &rhs, &drifted);
        assert!(
            !certificate.is_certified(),
            "a 1e-9 equality drift against an O(1) solve scale is a real defect"
        );
        assert_eq!(certificate.worst_row, 0);
    }

    #[test]
    fn stiff_null_space_solve_returns_roundoff_resolved_active_equality() {
        // A strongly anisotropic SPD metric coupled to an oblique equality.
        // Normwise backward stability against the 1e16 Hessian entry alone is
        // insufficient: the active equality itself must resolve to its
        // length-p dot-product floor.
        let hessian = array![[1.0e16, 1.0e8], [1.0e8, 2.0]];
        let gradient = array![1.0e8, -3.0];
        let active_a = array![[0.6, 0.8]];
        let active_residual = array![1.0e-4];
        let (direction, multiplier) =
            solve_kkt_direction(&hessian, &gradient, &active_a, Some(&active_residual))
                .expect("stiff null-space constrained solve");

        let certificate =
            certify_active_equalities(&active_a, &active_residual, &direction);
        assert!(
            certificate.is_certified(),
            "active equality residual {:.3e} exceeds its roundoff bound {:.3e}",
            certificate.residual,
            certificate.allowed,
        );
        assert!(multiplier.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn dependent_active_equalities_share_one_null_space() {
        // Two scaled copies of one equality describe one geometric face. The
        // SVD must retain that rank-one row space, optimize in its orthogonal
        // complement, and return a direction satisfying both original rows.
        let hessian = array![
            [1.0e12, 0.0, 0.0],
            [0.0, 3.0, 0.5],
            [0.0, 0.5, 2.0],
        ];
        let gradient = array![2.0e5, -4.0, 1.0];
        let active_a = array![[1.0, 2.0, 0.0], [2.0, 4.0, 0.0]];
        let active_residual = array![1.0e-4, 2.0e-4];
        let (direction, multiplier) =
            solve_kkt_direction(&hessian, &gradient, &active_a, Some(&active_residual))
                .expect("rank-deficient active face must have one certified null space");

        let residual = &active_a.dot(&direction) - &active_residual;
        assert!(
            residual.iter().all(|value| value.abs() <= 1.0e-14),
            "dependent active equations were not resolved: {residual:?}"
        );
        assert!(multiplier.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn warm_face_rows_are_point_local_for_dense_and_operator_constraints() {
        // The previous QP endpoint was x=0, where x>=0 binds, but a trust
        // step accepted only an interior point x=1. Reusing row 0 as an
        // equality at x=1 would solve the wrong problem and drive x back to
        // the stale boundary. The actual quadratic has its feasible minimizer
        // at x=2, so both carriers must discard the slack warm row and return
        // the unconstrained interior minimizer with an empty face.
        let hessian = array![[1.0_f64]];
        let rhs = array![2.0_f64];
        let interior = array![1.0_f64];
        let dense = LinearInequalityConstraints::new(array![[1.0]], array![0.0])
            .expect("one-dimensional half-line");
        let (dense_solution, dense_active) =
            solve_quadratic_with_linear_constraints(&hessian, &rhs, &interior, &dense, Some(&[0]))
                .expect("dense stale-face solve");
        assert_relative_eq!(dense_solution[0], 2.0, epsilon = 1e-12);
        assert!(dense_active.is_empty());

        let factor = std::sync::Arc::new(array![[1.0_f64]]);
        let cone = KhatriRaoConeConstraints::new(factor, vec![0], 1)
            .expect("one-dimensional factored half-line");
        let operator = ConstraintSet::KhatriRaoCone(cone);
        let stale_terminal_face = constraint_set_rows_tight_at_point(&operator, &interior, &[0])
            .expect("terminal face classification");
        assert!(stale_terminal_face.is_empty());
        let (operator_solution, operator_active) =
            solve_quadratic_with_constraint_set(&hessian, &rhs, &interior, &operator, Some(&[0]))
                .expect("operator stale-face solve");
        assert_relative_eq!(operator_solution[0], 2.0, epsilon = 1e-12);
        assert!(operator_active.is_empty());
    }

    /// A `β = 0` seed sits on the boundary of EVERY row of a homogeneous
    /// (`b = 0`) convex/concave second-difference cone — it is the cone vertex.
    /// The strict-interior projection must move it to a point with a strictly
    /// positive scaled slack on every row, so the inner active-set QP starts
    /// from an EMPTY working set rather than an all-rows-active degenerate face
    /// (the #873 cache-dependence root cause). The zero seed is the worst case:
    /// the nearest interior point is unique up to the margin, and a buggy
    /// "min-norm" feasibility fallback would return `0` again.
    #[test]
    fn strict_interior_projection_lifts_vertex_seed_off_every_constraint_row() {
        // Signed second-difference rows of a 5-coefficient concave smooth:
        // -(β_{i+2} - 2β_{i+1} + β_i) ≥ 0 for i = 0..3.
        let p = 5usize;
        let rows = p - 2;
        let mut a = Array2::<f64>::zeros((rows, p));
        for i in 0..rows {
            a[[i, i]] = -1.0;
            a[[i, i + 1]] = 2.0;
            a[[i, i + 2]] = -1.0;
        }
        let constraints = LinearInequalityConstraints::new(a, Array1::zeros(rows))
            .expect("test constraint shape invariant");

        let vertex = Array1::<f64>::zeros(p);
        // The vertex is feasible (all rows exactly tight) but on every boundary.
        for i in 0..rows {
            assert!(
                scaled_constraint_slack(&vertex, &constraints, i).abs() < 1e-12,
                "vertex seed should sit exactly on row {i}"
            );
        }

        let interior = project_point_strictly_into_feasible_cone(&vertex, &constraints)
            .expect("strict-interior projection of the vertex must succeed");
        let min_slack = (0..rows)
            .map(|i| scaled_constraint_slack(&interior, &constraints, i))
            .fold(f64::INFINITY, f64::min);
        assert!(
            min_slack >= 0.5 * ACTIVE_SET_INTERIOR_SEED_MARGIN,
            "projected seed must be strictly interior on every row; min scaled slack = {min_slack:.3e}"
        );
    }

    /// Mirrors `s(x, shape=concave, bc=clamped)`: shape curvature reparameterized
    /// to independent coordinate lower bounds `γ_j ≥ 0` (genuine one-sided rows),
    /// MERGED with a boundary condition encoded as an anti-parallel inequality
    /// PAIR `{r·β ≥ t, −r·β ≥ −t}` (an equality `r·β = t`). A naive
    /// shift-every-row-inward projection turns that pair into the empty set
    /// `t+δ ≤ r·β ≤ t−δ`, fails, and the caller falls back to the cone vertex —
    /// silently reintroducing the #873 seed for the combined case. The
    /// anti-parallel-aware margin must leave the equality pair tight while still
    /// pushing the genuine shape rows strictly interior.
    #[test]
    fn strict_interior_projection_keeps_equality_pairs_tight_with_shape_bounds() {
        let p = 5usize;
        // Rows 0..3: shape lower bounds γ_2,γ_3,γ_4 ≥ 0 (homogeneous, b = 0).
        // Rows 3,4: endpoint equality β_0 = 0 as {e_0·β ≥ 0, −e_0·β ≥ 0}.
        let m = 3 + 2;
        let mut a = Array2::<f64>::zeros((m, p));
        a[[0, 2]] = 1.0;
        a[[1, 3]] = 1.0;
        a[[2, 4]] = 1.0;
        a[[3, 0]] = 1.0;
        a[[4, 0]] = -1.0;
        let constraints = LinearInequalityConstraints::new(a, Array1::zeros(m))
            .expect("test constraint shape invariant");

        // A seed that violates the shape bounds (negative curvature coords) and
        // the equality (β_0 ≠ 0).
        let point = Array1::from_vec(vec![0.7, -0.2, -0.5, -0.3, -0.1]);
        let seed = project_point_strictly_into_feasible_cone(&point, &constraints).expect(
            "strict-interior projection must succeed when an equality pair is present, \
             not collapse to the empty set and fall back to the vertex",
        );

        // Genuine one-sided shape rows are pushed strictly interior.
        for i in 0..3 {
            assert!(
                scaled_constraint_slack(&seed, &constraints, i)
                    >= 0.4 * ACTIVE_SET_INTERIOR_SEED_MARGIN,
                "shape row {i} not strictly interior: scaled slack = {:.3e}",
                scaled_constraint_slack(&seed, &constraints, i)
            );
        }
        // The equality pair stays tight (β_0 ≈ 0), i.e. the seed is projected
        // onto the boundary hyperplane rather than shifted off it.
        assert!(
            seed[0].abs() <= 1e-6,
            "boundary equality must be enforced, got β_0 = {:.3e}",
            seed[0]
        );
    }

    /// A seed that already carries genuine (concave) curvature and clears the
    /// interior margin is returned essentially unchanged — the projection only
    /// nudges boundary/violating seeds, it does not discard usable curvature.
    #[test]
    fn strict_interior_projection_preserves_a_curvature_carrying_seed() {
        let p = 5usize;
        let rows = p - 2;
        let mut a = Array2::<f64>::zeros((rows, p));
        for i in 0..rows {
            a[[i, i]] = -1.0;
            a[[i, i + 1]] = 2.0;
            a[[i, i + 2]] = -1.0;
        }
        let constraints = LinearInequalityConstraints::new(a, Array1::zeros(rows))
            .expect("test constraint shape invariant");
        // A strictly concave coefficient profile (-(j-2)^2): every second
        // difference is -(-2) = +2 > 0 after the concave sign flip, well above
        // the interior margin.
        let seed = Array1::from_iter((0..p).map(|j| -((j as f64 - 2.0).powi(2))));
        let projected = project_point_strictly_into_feasible_cone(&seed, &constraints)
            .expect("already-interior seed must project");
        let max_move = seed
            .iter()
            .zip(projected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_move < 1e-3,
            "strictly-interior curvature-carrying seed should be preserved; max move = {max_move:.3e}"
        );
    }

    #[test]
    fn dense_dual_newton_returns_the_exact_boundary_solution() {
        let hessian = array![[1.0]];
        let gradient = array![-1.0];
        let beta = array![0.0];
        let constraints = LinearInequalityConstraints {
            a: array![[-1.0]],
            b: array![-0.1],
        };
        let mut direction = Array1::zeros(1);
        let mut active_hint = Vec::new();

        solve_newton_direction_with_linear_constraints(
            &hessian,
            &gradient,
            &beta,
            &constraints,
            &mut direction,
            Some(&mut active_hint),
        )
        .expect("finite dual solve should return the unique boundary solution");

        assert_relative_eq!(direction[0], 0.1, epsilon = 1e-12);
        assert_eq!(active_hint, vec![0]);
    }

    #[test]
    fn dense_dual_releases_a_boundary_with_negative_multiplier() {
        // At x=0 under x>=0, gradient=-1 points toward increasing x: the
        // equality multiplier is negative and the exact constrained minimizer
        // leaves the face. A warm row is an ordering hint, not permission to
        // retain that non-KKT boundary.
        let hessian = array![[1.0_f64]];
        let beta = array![0.0_f64];
        let gradient = array![-1.0_f64];
        let constraints =
            LinearInequalityConstraints::new(array![[1.0]], array![0.0]).expect("one-sided bound");
        let mut direction = Array1::<f64>::zeros(1);
        let mut active = vec![0];
        solve_newton_direction_with_linear_constraints(
            &hessian,
            &gradient,
            &beta,
            &constraints,
            &mut direction,
            Some(&mut active),
        )
        .expect("negative-multiplier face must be released");

        assert_relative_eq!(direction[0], 1.0, epsilon = 1e-12);
        assert!(gradient.dot(&direction) < 0.0);
        assert!(active.is_empty(), "descent moves strictly into the cone");
    }

    #[test]
    fn rank_reduce_agrees_with_a_rank_revealing_factorization_on_an_ill_conditioned_face_2600() {
        // Six unit rows sampled from a smooth basis at closely spaced nodes —
        // the shape a spline/Khatri-Rao constraint face actually has — plus one
        // row that is EXACTLY their fourth finite difference. The block's rank
        // is six and the seventh row is redundant by construction, not by
        // tolerance: it is a linear combination of rows already present.
        //
        // What makes it a gate is the SIZE of that combination's coefficients.
        // The kept six run `sigma` from `2.8` down to `3.6e-7`, and a fourth
        // difference of near-collinear rows cancels down to `2e-3`, so
        // Gram-Schmidt subtracts O(1) projections to leave an O(1e-3) residual
        // and then has to see that the rest is zero. A single modified sweep
        // leaves contamination above this scan's own
        // `tol = 100*eps*8 = 1.78e-13`, so the row clears the floor and the
        // scan reports rank SEVEN. Reorthogonalizing once brings it back to
        // `O(eps)` and the scan reports six.
        //
        // That is #2600's pit arm in miniature: `face_rows = rank + 1` on 90
        // consecutive reduced-face solves, handed to a factorization that
        // refused the block as a singular affine system and ended the fit.
        //
        // The assertion is deliberately NOT a hardcoded six. It is that this
        // scan and a rank-revealing factorization of the same block, at the
        // same floor, agree — which is the invariant the physical reduced face
        // needs and the one that was violated.
        use gam_linalg::faer_ndarray::FaerSvd;

        const NODE_SPACING: f64 = 0.05;
        let p = 8usize;
        let independent = 6usize;
        let mut rows = Array2::<f64>::zeros((independent + 1, p));
        for index in 0..independent {
            let node = 1.0 + NODE_SPACING * index as f64;
            let mut power = 1.0_f64;
            for column in 0..p {
                rows[[index, column]] = power;
                power *= node;
            }
            let norm = rows.row(index).dot(&rows.row(index)).sqrt();
            let normalized = &rows.row(index).to_owned() / norm;
            rows.row_mut(index).assign(&normalized);
        }
        // Fourth finite difference of the first five rows: coefficients
        // 1, -4, 6, -4, 1 — exactly in the span, with an O(1e-3) norm.
        let difference_weights = [1.0_f64, -4.0, 6.0, -4.0, 1.0];
        let mut combination = Array1::<f64>::zeros(p);
        for (index, weight) in difference_weights.iter().enumerate() {
            combination.scaled_add(*weight, &rows.row(index));
        }
        let combination_norm = combination.dot(&combination).sqrt();
        rows.row_mut(independent)
            .assign(&(&combination / combination_norm));

        let (_u, singular, _vt) = rows.svd(false, false).expect("face SVD");
        let singular_max = singular.iter().copied().fold(0.0_f64, f64::max);
        let svd_floor = 100.0 * f64::EPSILON * (rows.nrows().max(p) as f64) * singular_max;
        let svd_rank = singular.iter().filter(|&&s| s > svd_floor).count();
        assert_eq!(
            svd_rank, independent,
            "the fixture must be rank-deficient by construction, not by tolerance \
             (singular values {singular:?})"
        );

        let b = Array1::<f64>::zeros(independent + 1);
        let groups: Vec<Vec<usize>> = (0..=independent).map(|row| vec![row]).collect();
        let (reduced, _b_out, groups_out, _dependence) =
            rank_reduce_rows_pivoted_qr_with_dependence(rows, b, groups);

        assert_eq!(
            reduced.nrows(),
            svd_rank,
            "the independence scan kept {} rows where a rank-revealing factorization \
             of the same block at the same floor finds {svd_rank}; a face carrying a \
             redundant row is what #2600's affine solve then refused",
            reduced.nrows()
        );
        assert_eq!(groups_out.len(), svd_rank);
    }

    #[test]
    fn rank_reduce_zero_rows_returns_empty_working_set() {
        let a = array![[0.0, 0.0], [0.0, 0.0],];
        let b = array![0.0, 0.0];
        let groups = vec![vec![0], vec![1]];

        let (a_out, b_out, groups_out, _) =
            rank_reduce_rows_pivoted_qr_with_dependence(a, b, groups);

        assert_eq!(a_out.nrows(), 0);
        assert_eq!(a_out.ncols(), 2);
        assert_eq!(b_out.len(), 0);
        assert!(groups_out.is_empty());
    }

    #[test]
    fn cone_projection_solves_nonnegative_least_squares_not_one_way_pruning() {
        let active_a = array![
            [0.85258593, -0.77270261],
            [-1.22152485, 2.05129351],
            [0.22794844, 1.56987265],
        ];
        let residual = array![-0.50524761, -1.10104911];

        let (projected, multipliers) =
            project_stationarity_residual_on_constraint_cone(&residual, &active_a)
                .expect("cone projection should solve");

        let row0 = active_a.row(0);
        let expected_mu0 = row0.dot(&residual) / row0.dot(&row0);
        assert_relative_eq!(multipliers[0], expected_mu0, epsilon = 1e-8);
        assert_relative_eq!(multipliers[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(multipliers[2], 0.0, epsilon = 1e-10);

        let raw_norm2 = residual.dot(&residual);
        let projected_norm2 = projected.dot(&projected);
        assert!(
            projected_norm2 < raw_norm2 - 0.1,
            "NNLS projection should keep the improving active row: raw={raw_norm2:.6e}, projected={projected_norm2:.6e}"
        );
        let dual = active_a.dot(&projected);
        for (idx, (&mu, &w)) in multipliers.iter().zip(dual.iter()).enumerate() {
            if mu <= 1e-10 {
                assert!(
                    w <= 1e-8,
                    "inactive cone generator {idx} has positive reduced gradient {w:.3e}"
                );
            }
        }
    }

    /// The direct Lawson–Hanson route must agree with the strict-QP Moreau
    /// projection wherever the latter succeeds: both compute the projection
    /// of `residual` onto the polar of the generated cone.
    #[test]
    fn nnls_moreau_projection_matches_strict_qp_route() {
        let cases: Vec<(Array2<f64>, Array1<f64>)> = vec![
            (
                array![
                    [0.85258593, -0.77270261],
                    [-1.22152485, 2.05129351],
                    [0.22794844, 1.56987265],
                ],
                array![-0.50524761, -1.10104911],
            ),
            (array![[1.0, 0.0], [0.0, 1.0]], array![3.0, -2.0]),
            (
                array![[1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [2.0, 2.0, 0.0]],
                array![1.5, 0.25, -0.75],
            ),
        ];
        for (rows, target) in cases {
            let qp = moreau_projection_via_strict_qp(&target, &rows)
                .expect("strict QP route must solve these well-posed instances");
            let (lambda, projected) = nonnegative_cone_multipliers(&rows, &target)
                .expect("LH route must solve the same instances");
            for (left, right) in qp.0.iter().zip(projected.iter()) {
                assert_relative_eq!(left, right, epsilon = 1e-8);
            }
            // λ ≥ 0 and exact reconstruction by construction.
            assert!(lambda.iter().all(|&v| v >= 0.0));
            let reconstructed = &target - &rows.t().dot(&lambda);
            for (left, right) in reconstructed.iter().zip(projected.iter()) {
                assert_relative_eq!(left, right, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn nnls_projects_axis_cone_exactly() {
        let rows = array![[1.0, 0.0], [0.0, 1.0]];
        let target = array![3.0, -2.0];
        let (lambda, projected) =
            nonnegative_cone_multipliers(&rows, &target).expect("axis cone NNLS");
        assert_relative_eq!(lambda[0], 3.0, epsilon = 1e-10);
        assert_relative_eq!(lambda[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(projected[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(projected[1], -2.0, epsilon = 1e-10);
    }

    /// A dependent active row that is weakly aligned with every kept row
    /// individually (`a3 = (a1 + a2)/(2ε)`, pairwise alignment ≈ ε) breaks the
    /// single-target multiplier attribution: `λ/coeff` explodes by `1/ε` and
    /// manufactures phantom huge duals. The existence-form certificate sees the
    /// exact nonnegative closure `g = 1·a3` and must certify stationarity.
    #[test]
    fn nnls_closes_stationarity_on_weakly_aligned_dependent_face() {
        let eps = 1e-8_f64;
        let rows = array![[1.0, eps], [-1.0, eps], [0.0, 1.0]];
        let target = array![0.0, 1.0];
        let (lambda, projected) =
            nonnegative_cone_multipliers(&rows, &target).expect("dependent-face NNLS");
        let closure = projected.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        assert!(
            closure <= 1e-10,
            "λ = e3 closes stationarity exactly; got closure {closure:.3e}"
        );
        assert!(lambda.iter().all(|&v| v >= 0.0));
    }

    /// End-to-end: the constrained Newton solve on the same weakly-aligned
    /// degenerate face must certify the vertex instead of chasing phantom
    /// negative duals into a working-set cycle and refusing (#2298 survival
    /// monotonicity faces, #979 CTN faces).
    #[test]
    fn degenerate_face_with_weak_alignment_certifies_instead_of_cycling() {
        let eps = 1e-8_f64;
        let a = array![[1.0, eps], [-1.0, eps], [0.0, 1.0]];
        let b = array![0.0, 0.0, 0.0];
        let constraints = LinearInequalityConstraints::new(a.clone(), b).expect("constraints");
        let hessian = Array2::<f64>::eye(2);
        // KKT at d* = 0 with λ = e3 ≥ 0: gradient = A^T e3 = a3.
        let gradient = array![0.0, 1.0];
        let beta = array![0.0, 0.0];
        let mut direction = Array1::<f64>::zeros(2);
        solve_newton_direction_with_linear_constraints(
            &hessian,
            &gradient,
            &beta,
            &constraints,
            &mut direction,
            None,
        )
        .expect("the vertex is a certified KKT point; refusal is the #2298 defect");
        let step = direction.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        assert!(
            step <= 1e-8,
            "optimum is the vertex itself; got |d|∞ = {step:.3e}"
        );
    }

    #[test]
    fn cone_projection_preserves_original_multiplier_units_after_row_canonicalization() {
        let residual = array![2.0, -1.0];
        let unit_row = array![[1.0, 0.0]];
        let scaled_row = array![[4.0, 0.0]];

        let (projected_unit, multiplier_unit) =
            project_stationarity_residual_on_constraint_cone(&residual, &unit_row)
                .expect("unit-row cone projection should solve");
        let (projected_scaled, multiplier_scaled) =
            project_stationarity_residual_on_constraint_cone(&residual, &scaled_row)
                .expect("scaled-row cone projection should solve");

        assert_relative_eq!(projected_unit[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(projected_unit[1], -1.0, epsilon = 1e-12);
        assert_relative_eq!(projected_scaled[0], projected_unit[0], epsilon = 1e-12);
        assert_relative_eq!(projected_scaled[1], projected_unit[1], epsilon = 1e-12);
        assert_relative_eq!(multiplier_unit[0], 2.0, epsilon = 1e-12);
        assert_relative_eq!(multiplier_scaled[0], 0.5, epsilon = 1e-12);

        let reconstructed_unit = &residual - &unit_row.t().dot(&multiplier_unit);
        let reconstructed_scaled = &residual - &scaled_row.t().dot(&multiplier_scaled);
        assert_relative_eq!(reconstructed_unit[0], projected_unit[0], epsilon = 1e-12);
        assert_relative_eq!(
            reconstructed_scaled[0],
            projected_scaled[0],
            epsilon = 1e-12
        );
    }

    // #2601 — a cone-projection REFUSAL must be visible in what a reader is
    // shown, not only in a field nobody rendered.
    //
    // `project_stationarity_residual_on_constraint_cone` returns `None` on a
    // non-finite target, so a non-finite gradient on a fully active face drives
    // the production diagnostic down the real refusal path. Both facts then have
    // to hold: the flag is set, and the note that carries it into a rendered
    // verdict is non-empty and says the residual is unprojected.
    #[test]
    fn a_refused_cone_projection_is_named_in_the_rendered_verdict_2601() {
        // β on the boundary of both rows, so the face is fully active and the
        // projector is actually reached (it is skipped on an empty face).
        let beta = array![0.0, 0.0];
        let constraints = LinearInequalityConstraints {
            a: array![[1.0, 0.0], [0.0, 1.0]],
            b: array![0.0, 0.0],
        };

        let finite = compute_constraint_kkt_diagnostics(&beta, &array![1.0, 2.0], &constraints);
        assert!(
            !finite.cone_projection_refused,
            "a finite gradient on a full-rank active face must be projected, not refused"
        );
        assert_eq!(
            finite.cone_projection_note(),
            "",
            "a projection that happened must contribute no note"
        );

        let refused =
            compute_constraint_kkt_diagnostics(&beta, &array![f64::NAN, 2.0], &constraints);
        assert_eq!(
            refused.n_active, 2,
            "the fixture must reach the projector: an empty active face skips it entirely"
        );
        assert!(
            refused.cone_projection_refused,
            "a non-finite target is one of the projector's documented refusals"
        );
        assert!(
            refused.cone_projection_note().contains("REFUSED"),
            "the rendered note must name the refusal; got {:?}",
            refused.cone_projection_note()
        );
        assert!(
            refused.cone_projection_note().contains("UNPROJECTED"),
            "the note must say the reported stat was never projected; got {:?}",
            refused.cone_projection_note()
        );
    }

    // #500: the KKT primal residual must be the *geometric* distance to the
    // constraint hyperplane — invariant to how the constraint row is scaled.
    // A B-spline endpoint-derivative clamp carries a large row norm, so the
    // raw slack `a·β − b` of a near-feasible iterate is inflated by ‖a‖ and a
    // downstream raw primal gate would spuriously refuse it. The same geometry
    // expressed with a unit-norm row must yield the same primal.
    #[test]
    fn kkt_primal_is_per_row_scale_invariant() {
        // β sits 2.071e-8 on the infeasible side of the hyperplane `row·β ≥ 0`
        // (the exact geometric residual reported in #500's startup abort).
        let geometric_violation = 2.071e-8_f64;
        let gradient = Array1::<f64>::zeros(2);

        // Unit-norm row: raw slack == geometric distance.
        let beta_unit = array![-geometric_violation, 0.0];
        let unit = LinearInequalityConstraints {
            a: array![[1.0, 0.0]],
            b: array![0.0],
        };
        let diag_unit = compute_constraint_kkt_diagnostics(&beta_unit, &gradient, &unit);

        // Same hyperplane, row scaled ×1000: raw slack would be 2.071e-5, but
        // the *scaled* primal must still equal the geometric distance.
        let beta_big = array![-geometric_violation, 0.0];
        let big = LinearInequalityConstraints {
            a: array![[1000.0, 0.0]],
            b: array![0.0],
        };
        let diag_big = compute_constraint_kkt_diagnostics(&beta_big, &gradient, &big);

        assert_relative_eq!(
            diag_unit.primal_feasibility,
            geometric_violation,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            diag_big.primal_feasibility,
            geometric_violation,
            epsilon = 1e-14
        );
        // The scaled diagnostic must NOT report the ‖a‖-inflated raw slack.
        assert!(
            diag_big.primal_feasibility < 1e-7,
            "scaled primal {:.3e} should pass a 1e-7 gate; raw slack would be {:.3e}",
            diag_big.primal_feasibility,
            1000.0 * geometric_violation
        );
    }

    // A B-spline `bc=clamped`/`bc=anchored` constraint is an EQUALITY
    // `a·β = b` encoded as two opposing inequalities `a·β ≥ b` and
    // `−a·β ≥ −b`. The active-set solver must drive the unconstrained
    // optimum back onto the hyperplane `a·β = b`. This is the isolated
    // analogue of the `bc=clamped` startup-validation abort: the exact
    // validation solve left `a·β ≈ 7.76` instead of 0, so the KKT primal
    // residual blew past tolerance and every seed was refused.
    #[test]
    fn opposing_inequality_pair_pins_equality_to_target() {
        // Minimize ½‖β‖² − rhs·β  (H = I) ⇒ unconstrained optimum β* = rhs.
        // rhs = [5,5,0,0] ⇒ a·β* = 10 with a = [1,1,0,0].
        // The opposing pair must pull a·β back to the target 0.
        let hessian = array![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let rhs = array![5.0, 5.0, 0.0, 0.0];
        let beta_start = Array1::<f64>::zeros(4);
        let constraints = LinearInequalityConstraints {
            a: array![[1.0, 1.0, 0.0, 0.0], [-1.0, -1.0, 0.0, 0.0]],
            b: array![0.0, 0.0],
        };

        let (beta, _active) = solve_quadratic_with_linear_constraints(
            &hessian,
            &rhs,
            &beta_start,
            &constraints,
            None,
        )
        .expect("opposing-inequality equality QP must solve");

        let a_dot_beta = beta[0] + beta[1];
        assert!(
            a_dot_beta.abs() < 1e-8,
            "opposing inequalities must pin a·β to 0, got {a_dot_beta:.6e} (β = {beta:?})"
        );
    }

    // Same as above but with a non-zero target and a large row norm — the
    // exact shape of a B-spline endpoint-derivative clamp, whose rows carry
    // ‖a‖ ≫ 1. The equality must still be pinned in geometric coordinates.
    #[test]
    fn opposing_inequality_pair_pins_scaled_equality_to_nonzero_target() {
        let hessian = array![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let rhs = array![5.0, 5.0, 0.0, 0.0];
        let beta_start = Array1::<f64>::zeros(4);
        // Row scaled ×1000 (mimics a derivative-clamp row norm) with target 3000
        // ⇒ geometric target a·β = 3.0 in unit coordinates.
        let constraints = LinearInequalityConstraints {
            a: array![[1000.0, 1000.0, 0.0, 0.0], [-1000.0, -1000.0, 0.0, 0.0]],
            b: array![3000.0, -3000.0],
        };

        let (beta, _active) = solve_quadratic_with_linear_constraints(
            &hessian,
            &rhs,
            &beta_start,
            &constraints,
            None,
        )
        .expect("scaled opposing-inequality equality QP must solve");

        let a_dot_beta = 1000.0 * (beta[0] + beta[1]);
        assert!(
            (a_dot_beta - 3000.0).abs() < 1e-5,
            "opposing inequalities must pin a·β to 3000, got {a_dot_beta:.6e} (β = {beta:?})"
        );
    }

    // `bc=clamped` at BOTH ends produces TWO opposing-inequality equalities
    // (4 rows total). The real abort reports `active=2/4` — only ONE of the
    // two equalities is being pinned. Reproduce two independent equalities
    // and require BOTH to be driven to their targets.
    #[test]
    fn two_opposing_inequality_equalities_both_pinned() {
        let hessian = array![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let rhs = array![5.0, 5.0, 5.0, 5.0];
        let beta_start = Array1::<f64>::zeros(4);
        // Equality A: β0 + β1 = 0 (rows 0,1). Equality B: β2 + β3 = 0 (rows 2,3).
        let constraints = LinearInequalityConstraints {
            a: array![
                [1.0, 1.0, 0.0, 0.0],
                [-1.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, -1.0, -1.0],
            ],
            b: array![0.0, 0.0, 0.0, 0.0],
        };

        let (beta, _active) = solve_quadratic_with_linear_constraints(
            &hessian,
            &rhs,
            &beta_start,
            &constraints,
            None,
        )
        .expect("two-equality QP must solve");

        assert!(
            (beta[0] + beta[1]).abs() < 1e-8,
            "equality A not pinned: β0+β1 = {:.6e}",
            beta[0] + beta[1]
        );
        assert!(
            (beta[2] + beta[3]).abs() < 1e-8,
            "equality B not pinned: β2+β3 = {:.6e}",
            beta[2] + beta[3]
        );
    }

    // Faithful to the failing fit: the penalized IRLS Hessian `X'WX + λS`
    // with λ at the over-smoothing ceiling is severely ill-conditioned — the
    // penalty `S` is rank-deficient (null space = the unpenalized polynomial
    // part), so directions in null(S) are governed by a tiny `X'WX` block
    // while penalized directions carry a huge λ. The opposing-inequality
    // equalities must STILL be pinned under this conditioning.
    #[test]
    fn opposing_inequality_equalities_pinned_under_ill_conditioned_penalty() {
        // H = diag(1, 1, λ, λ) with λ = 1e8 — penalized directions 2,3 are
        // ~1e8 stiffer than the data directions 0,1.
        let lam = 1.0e8_f64;
        let hessian = array![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, lam, 0.0],
            [0.0, 0.0, 0.0, lam],
        ];
        let rhs = array![5.0, 5.0, 5.0, 5.0];
        let beta_start = Array1::<f64>::zeros(4);
        // Two equalities that COUPLE a stiff and a soft coordinate, like a
        // B-spline derivative row spanning penalized and unpenalized parts:
        // A: β0 + β2 = 0, B: β1 + β3 = 0.
        let constraints = LinearInequalityConstraints {
            a: array![
                [1.0, 0.0, 1.0, 0.0],
                [-1.0, 0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, -1.0, 0.0, -1.0],
            ],
            b: array![0.0, 0.0, 0.0, 0.0],
        };

        let (beta, _active) = solve_quadratic_with_linear_constraints(
            &hessian,
            &rhs,
            &beta_start,
            &constraints,
            None,
        )
        .expect("ill-conditioned two-equality QP must solve");

        assert!(
            (beta[0] + beta[2]).abs() < 1e-6,
            "equality A not pinned under ill-conditioning: β0+β2 = {:.6e}",
            beta[0] + beta[2]
        );
        assert!(
            (beta[1] + beta[3]).abs() < 1e-6,
            "equality B not pinned under ill-conditioning: β1+β3 = {:.6e}",
            beta[1] + beta[3]
        );
    }

    // ==== gam#2306: operator (ConstraintSet) solver vs dense oracle ====

    /// Small Khatri-Rao cone whose dense materialization is exact: Ψ is
    /// 4 × 2, coefficient block is 3 × 2 (row 0 unconstrained location,
    /// rows 1–2 coupled), so p = 6 and the cone has 8 rows.
    fn small_cone() -> KhatriRaoConeConstraints {
        let psi = array![[1.0_f64, 0.2], [1.0, -0.4], [1.0, 1.3], [1.0, 0.8],];
        KhatriRaoConeConstraints::new(std::sync::Arc::new(psi), vec![1, 2], 3).expect("small cone")
    }

    /// Parallel tight rows collapse to the lowest-index representative, and the
    /// duplicate is recorded in the dependence map with its scalar ratio.
    #[test]
    fn cone_reduced_face_collapses_parallel_rows_to_lowest_index() {
        // ψ_2 = 2·ψ_0 (parallel); ψ_1 independent. p_cov=2, one coupled row.
        let psi = array![[1.0_f64, 0.0], [0.0, 1.0], [2.0, 0.0]];
        let cone = KhatriRaoConeConstraints::new(std::sync::Arc::new(psi), vec![1], 2)
            .expect("parallel cone");
        // β = 0 ⇒ every Γ = 0 ⇒ every row tight.
        let beta = Array1::<f64>::zeros(2 * 2);
        let face = khatri_rao_cone_reduced_face(&cone, beta.view(), 1e-8).expect("reduce");
        assert_eq!(face.tight_rows, rows(&[0, 1, 2]));
        // Rank 2: reps are the two independent directions at their lowest obs.
        assert_eq!(face.representatives, rows(&[0, 1]));
        assert_eq!(face.dependence.len(), 2);
        // ψ_2 (flat id 2) is parallel to representative ψ_0 (rep index 0), coeff 2.
        assert_eq!(face.dependence[0].len(), 1);
        assert_eq!(face.dependence[0][0].row.index(), 2);
        assert!((face.dependence[0][0].coeff - 2.0).abs() < 1e-12);
        assert!(face.dependence[1].is_empty());
    }

    /// A full-rank tight face keeps every row and records no dependence.
    #[test]
    fn cone_reduced_face_full_rank_has_no_dependence() {
        let psi = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let cone = KhatriRaoConeConstraints::new(std::sync::Arc::new(psi), vec![1], 2)
            .expect("full-rank cone");
        let beta = Array1::<f64>::zeros(2 * 2);
        let face = khatri_rao_cone_reduced_face(&cone, beta.view(), 1e-8).expect("reduce");
        assert_eq!(face.representatives, rows(&[0, 1]));
        assert!(face.dependence.iter().all(|d| d.is_empty()));
        assert_eq!(face.tight_rows, rows(&[0, 1]));
    }

    /// A general-position dependent (in the span but parallel to no single rep)
    /// is dropped from the working set (full rank cut) but gets NO dependence
    /// entry — the (A)-strict contract that avoids a phantom distributed dual.
    #[test]
    fn cone_reduced_face_general_combination_gets_no_dependence_entry() {
        // ψ_2 = ψ_0 + ψ_1: in the span, but cos with each rep is 1/√2 < 1.
        let psi = array![[1.0_f64, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let cone = KhatriRaoConeConstraints::new(std::sync::Arc::new(psi), vec![1], 2)
            .expect("general-combo cone");
        let beta = Array1::<f64>::zeros(2 * 2);
        let face = khatri_rao_cone_reduced_face(&cone, beta.view(), 1e-8).expect("reduce");
        assert_eq!(face.representatives, rows(&[0, 1])); // ψ_2 dropped
        assert_eq!(face.tight_rows, rows(&[0, 1, 2])); // but still in the tight set
        assert!(
            face.dependence.iter().all(|d| d.is_empty()),
            "a general-position drop must carry no distributed multiplier"
        );
    }

    /// Cross-block cone rows are automatically orthogonal (e_k ⊥ e_{k'}), so each
    /// shape block reduces independently — no cross-block dependence, and flat
    /// ids stay in the slot*n+obs space.
    #[test]
    fn cone_reduced_face_reduces_each_shape_block_independently() {
        let psi = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let cone = KhatriRaoConeConstraints::new(std::sync::Arc::new(psi), vec![1, 2], 3)
            .expect("two-block cone");
        let beta = Array1::<f64>::zeros(3 * 2);
        let face = khatri_rao_cone_reduced_face(&cone, beta.view(), 1e-8).expect("reduce");
        // Block 0 → flat 0,1; block 1 → flat 2,3 (slot*n+obs, n=2). All independent.
        assert_eq!(face.representatives, rows(&[0, 1, 2, 3]));
        assert!(face.dependence.iter().all(|d| d.is_empty()));
        assert_eq!(face.tight_rows, rows(&[0, 1, 2, 3]));
    }

    /// The Dense arm of the `ConstraintSet::reduced_face` dispatcher matches the
    /// cone arm's contract: parallel tight rows collapse to the lowest-index
    /// representative with the scalar ratio recorded; flat id = the row index.
    #[test]
    fn dense_reduced_face_via_dispatcher_collapses_parallel_rows() {
        // Row 2 = 2·row 0 (parallel); row 1 independent. b = 0 ⇒ every row tight
        // at β = 0 (scaled slack 0).
        let a = array![[1.0_f64, 0.0], [0.0, 1.0], [2.0, 0.0]];
        let set = ConstraintSet::Dense(
            LinearInequalityConstraints::new(a, Array1::<f64>::zeros(3)).expect("dense"),
        );
        let beta = Array1::<f64>::zeros(2);
        let face = set.reduced_face(beta.view(), 1e-8).expect("reduce");
        assert_eq!(face.tight_rows, rows(&[0, 1, 2]));
        assert_eq!(face.representatives, rows(&[0, 1]));
        assert_eq!(face.dependence[0].len(), 1);
        assert_eq!(face.dependence[0][0].row.index(), 2);
        assert!((face.dependence[0][0].coeff - 2.0).abs() < 1e-12);
        assert!(face.dependence[1].is_empty());
    }

    /// Constraint-row ids for the `ReducedFace` assertions below.
    fn rows(ids: &[usize]) -> Vec<ConstraintRowId> {
        ids.iter().copied().map(ConstraintRowId).collect()
    }

    /// A block-diagonal set whose FIRST member constrains fewer rows than it has
    /// coefficients — one `β₀ ≥ 0` row over a 3-wide block whose remaining two
    /// coordinates are unconstrained (intercept / covariate columns) — followed
    /// by a square 2×2 member at `col_start = 3`. This is the configuration that
    /// separates the constraint-row offset (`nrows`: 1) from the coefficient
    /// offset (`col_start`: 3); every pre-existing multi-block test used square
    /// members, where the two coincide and nothing can be distinguished.
    fn mixed_width_block_diagonal() -> ConstraintSet {
        let narrow = gam_problem::PlacedConstraintBlock {
            col_start: 0,
            set: ConstraintSet::Dense(
                LinearInequalityConstraints::new(
                    array![[1.0_f64, 0.0, 0.0]],
                    Array1::<f64>::zeros(1),
                )
                .expect("narrow block"),
            ),
        };
        let square = gam_problem::PlacedConstraintBlock {
            col_start: 3,
            set: ConstraintSet::Dense(
                LinearInequalityConstraints::new(
                    array![[1.0_f64, 0.0], [2.0, 0.0]],
                    Array1::<f64>::zeros(2),
                )
                .expect("square block"),
            ),
        };
        ConstraintSet::block_diagonal(vec![narrow, square], 5).expect("block-diagonal")
    }

    /// Every id a mixed-width block-diagonal reduced face emits addresses the
    /// JOINT CONSTRAINT-ROW space: it indexes `values()` and resolves through
    /// `bound()` / `row_norm()` to the member row it came from, and the tight
    /// rows really are tight there. This pins the id space that #2368 questioned
    /// — the running-`nrows()` shift is the one consistent with the rest of the
    /// `ConstraintSet` row API (`values` layout, `block_for_row` decoding).
    #[test]
    fn block_diagonal_reduced_face_row_ids_address_the_joint_constraint_row_space() {
        let set = mixed_width_block_diagonal();
        let beta = Array1::<f64>::zeros(5);
        let values = set.values(beta.view()).expect("values");
        let face = set.reduced_face(beta.view(), 1e-8).expect("reduce");

        // Joint rows: block 0 contributes row 0; block 1 contributes rows 1, 2.
        assert_eq!(set.nrows(), 3);
        assert_eq!(face.tight_rows, rows(&[0, 1, 2]));
        // Block 1's row 1 = 2·row 0, so it collapses onto joint representative 1.
        assert_eq!(face.representatives, rows(&[0, 1]));
        assert_eq!(face.dependence[1][0].row.index(), 2);

        for id in &face.tight_rows {
            let row = id.index();
            assert!(row < set.nrows(), "id {row} outside the joint row space");
            let norm = set.row_norm(row).expect("row norm resolves");
            let bound = set.bound(row).expect("bound resolves");
            assert!(
                (values[row] - bound) / norm <= 1e-8,
                "row {row} reported tight but has slack {}",
                (values[row] - bound) / norm
            );
        }
    }

    /// The same face, read as COEFFICIENT positions, is wrong — which is exactly
    /// why the ids are typed and why `row_column_support` exists.
    ///
    /// Block 1's representative is joint row 1, but it acts on β coordinate 3.
    /// Coordinate 1 is block 0's second column: an UNCONSTRAINED coefficient
    /// owned by a different block. A consumer that identified row ids with β
    /// positions (to build a free/pinned mask) would pin the wrong coordinate in
    /// the wrong block; the conversion recovers the right one.
    #[test]
    fn block_diagonal_reduced_face_row_ids_are_not_beta_coordinates() {
        let set = mixed_width_block_diagonal();
        let beta = Array1::<f64>::zeros(5);
        let face = set.reduced_face(beta.view(), 1e-8).expect("reduce");

        let block1_rep = face.representatives[1];
        assert_eq!(block1_rep.index(), 1);
        assert_eq!(
            set.row_column_support(block1_rep).expect("support"),
            vec![3],
            "block 1's row acts on the joint column 3 (col_start 3 + local 0)"
        );
        // The naive identity map would have named coordinate 1, which lies in
        // block 0's column range [0, 3) — a different block entirely.
        assert!(block1_rep.index() < 3, "id 1 falls inside block 0's columns");

        // Block 0's row is the one case where the two spaces agree; the
        // conversion must still be the thing that says so.
        assert_eq!(
            set.row_column_support(face.representatives[0])
                .expect("support"),
            vec![0]
        );
    }

    /// The BlockDiagonal arm composes member reductions and concatenates their
    /// row ids in order (each member's flat ids shift by the running member row
    /// count), so a parallel dependent in the second block reports its global id.
    #[test]
    fn block_diagonal_reduced_face_concatenates_member_row_ids() {
        // Two Dense blocks over disjoint columns; each: row0 independent, row1 =
        // 2·row0. b = 0 ⇒ all tight. Block 1's rows shift by block 0's 2 rows.
        let make = |c0: usize| gam_problem::PlacedConstraintBlock {
            col_start: c0,
            set: ConstraintSet::Dense(
                LinearInequalityConstraints::new(
                    array![[1.0_f64, 0.0], [2.0, 0.0]],
                    Array1::<f64>::zeros(2),
                )
                .expect("dense block"),
            ),
        };
        let set = ConstraintSet::block_diagonal(vec![make(0), make(2)], 4).expect("block-diagonal");
        let beta = Array1::<f64>::zeros(4);
        let face = set.reduced_face(beta.view(), 1e-8).expect("reduce");
        assert_eq!(face.tight_rows, rows(&[0, 1, 2, 3]));
        assert_eq!(face.representatives, rows(&[0, 2]));
        assert_eq!(face.dependence[0][0].row.index(), 1);
        assert_eq!(face.dependence[1][0].row.index(), 3);
    }

    /// Deterministic PD Hessian with off-diagonal coupling so active-set
    /// choices are not axis-trivial.
    fn coupled_pd_hessian(p: usize) -> Array2<f64> {
        let mut h = Array2::<f64>::eye(p) * 2.0;
        for i in 0..p {
            for j in 0..p {
                if i != j {
                    h[[i, j]] = 0.3 / (1.0 + (i as f64 - j as f64).abs());
                }
            }
        }
        h
    }

    #[test]
    fn operator_cone_qp_matches_dense_oracle_when_constraints_bind() {
        let cone = small_cone();
        let set = ConstraintSet::KhatriRaoCone(cone.clone());
        let dense = cone.to_dense().expect("dense oracle");
        let p = set.ncols();
        let hessian = coupled_pd_hessian(p);
        // rhs pulls the coupled rows negative so the unconstrained optimum
        // violates the cone and several rows must bind.
        let rhs = array![0.5_f64, -0.3, -2.0, 1.0, -1.5, -0.7];
        // Feasible start: coupled coefficient rows give strictly positive
        // functionals under every Ψ row (constant 1 with small slope loads).
        let beta_start = array![0.0_f64, 0.0, 1.0, 0.1, 1.0, 0.1];

        let (beta_op, mut active_op) =
            solve_quadratic_with_constraint_set(&hessian, &rhs, &beta_start, &set, None)
                .expect("operator solve");
        let (beta_dense, mut active_dense) =
            solve_quadratic_with_linear_constraints(&hessian, &rhs, &beta_start, &dense, None)
                .expect("dense solve");

        for j in 0..p {
            assert!(
                (beta_op[j] - beta_dense[j]).abs() < 1e-7,
                "operator/dense coefficient {j} mismatch: {} vs {}",
                beta_op[j],
                beta_dense[j]
            );
        }
        // The binding face must agree GEOMETRICALLY: both carriers land on the
        // same point (asserted above), so every reported active row must be
        // tight there, and both must carry the same number of independent
        // rows. Exact row-id equality is too strong — the fixture's coupled
        // rows admit alternate representations of the same face, and which
        // redundant row a carrier keeps is a tie-break, not semantics.
        active_op.sort_unstable();
        active_dense.sort_unstable();
        let values_at_solution = set.values(beta_op.view()).expect("values at solution");
        let tight_at_solution: Vec<usize> = (0..set.nrows())
            .filter(|&row| {
                let norm = set.row_norm(row).expect("norm");
                norm > 0.0 && values_at_solution[row] / norm <= 1e-7
            })
            .collect();
        for &row in active_op.iter().chain(active_dense.iter()) {
            assert!(
                tight_at_solution.contains(&row),
                "reported active row {row} is not tight at the common solution \
                 (op face {active_op:?}, dense face {active_dense:?}, tight {tight_at_solution:?})"
            );
        }
        assert_eq!(
            active_op.len(),
            active_dense.len(),
            "carriers disagree on the face dimension: op {active_op:?} vs dense {active_dense:?}"
        );
        assert!(
            !active_op.is_empty(),
            "fixture must actually bind at least one cone row"
        );
        // And the operator answer must be feasible on the full cone.
        let values = set.values(beta_op.view()).expect("values");
        let (worst, _) = set.max_scaled_violation(beta_op.view()).expect("violation");
        assert!(worst <= 1e-8, "operator answer infeasible: {worst:.3e}");
        assert_eq!(values.len(), 8);
    }

    #[test]
    fn operator_metric_dual_solves_the_non_diagonal_projection() {
        // Nonnegative quadrant with a genuinely coupled metric. The free
        // solution violates x>=0. On the binding face x=0, the exact minimizer
        // is y=1 and its row multiplier is 2:
        //
        // H [0,1]' - rhs = [2,0]'.
        //
        // An identity-metric Moreau projection would return a different point,
        // so this pins the H-metric dual rather than only cone feasibility.
        let psi = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let cone =
            KhatriRaoConeConstraints::new(std::sync::Arc::new(psi), vec![0], 1)
                .expect("nonnegative quadrant");
        let set = ConstraintSet::KhatriRaoCone(cone);
        let hessian = array![[4.0_f64, 1.0], [1.0, 2.0]];
        let rhs = array![-1.0_f64, 2.0];
        let beta_start = array![0.0_f64, 0.0];

        let (candidate, active) =
            solve_quadratic_with_constraint_set(&hessian, &rhs, &beta_start, &set, None)
                .expect("strict metric projection");

        assert_relative_eq!(candidate[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(candidate[1], 1.0, epsilon = 1e-12);
        assert_eq!(active, vec![0]);
        let gradient = hessian.dot(&candidate) - rhs;
        assert_relative_eq!(gradient[0], 2.0, epsilon = 1e-12);
        assert_relative_eq!(gradient[1], 0.0, epsilon = 1e-12);
    }

    /// A KKT tolerance is an acceptance bound, not permission for a warm face to
    /// change the unique minimizer of a strictly convex QP. Here the free
    /// optimum is `epsilon` inside the cone while the warm boundary face has a
    /// negative multiplier whose magnitude is only half the dual tolerance.
    /// The history-independent Moreau solve must return the interior optimum,
    /// not retain the numerically admissible but suboptimal boundary point.
    #[test]
    fn operator_metric_dual_uses_the_certificate_multiplier_cone_2432() {
        let psi = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let cone =
            KhatriRaoConeConstraints::new(std::sync::Arc::new(psi), vec![0], 1)
                .expect("nonnegative quadrant");
        let set = ConstraintSet::KhatriRaoCone(cone);
        let hessian = Array2::<f64>::eye(2);
        let epsilon = 0.5 * ACTIVE_SET_KKT_DUAL_FEASIBILITY_TOL;
        let rhs = array![epsilon, 1.0];
        let beta_start = array![0.0_f64, 0.0];

        let (candidate, active) = solve_quadratic_with_constraint_set(
            &hessian,
            &rhs,
            &beta_start,
            &set,
            Some(&[0]),
        )
        .expect("warm face must not perturb the unique cone projection");

        assert!(
            active.is_empty(),
            "the exact interior optimum has no active cone row"
        );
        assert_relative_eq!(candidate[0], epsilon, epsilon = 1e-14);
        assert_relative_eq!(candidate[1], 1.0, epsilon = 1e-14);
        let gradient = hessian.dot(&candidate) - rhs;
        assert_relative_eq!(gradient[0], 0.0, epsilon = 1e-14);
        assert_relative_eq!(gradient[1], 0.0, epsilon = 1e-14);
    }

    /// Exact reduced analogue of the public competing-risks failure shared by
    /// #2366/#2432. The supplied point is feasible and exactly three of 332 rows
    /// are tight, but its face certificate has the observed signature:
    /// negative dual `6.987e-1` and tangential stationarity residual `1.130e1`.
    /// Feasibility is therefore not a QP solution. Both a cold solve and a warm
    /// solve seeded with that wrong face must return the same unique KKT point,
    /// with strictly-positive multipliers on the two rows that truly bind.
    #[test]
    fn dense_metric_dual_leaves_feasible_nonstationary_three_of_332_face_2432() {
        let p = 5usize;
        let m = 332usize;
        let mut a = Array2::<f64>::zeros((m, p));
        let mut b = Array1::<f64>::from_elem(m, -100.0);
        for row in 0..3 {
            a[[row, row]] = 1.0;
            b[row] = 0.0;
        }
        // The remaining rows are deliberately slack but non-vacuous. They pin
        // the production cardinality without changing the five-dimensional
        // geometry of the bad face.
        for row in 3..m {
            a[[row, (row - 3) % p]] = 1.0;
        }
        let constraints =
            LinearInequalityConstraints::new(a, b).expect("332-row dense constraint system");
        let hessian = Array2::from_diag(&array![1.0_f64, 2.0, 3.0, 1.0, 4.0]);
        let rhs = array![0.6987_f64, -2.0, -3.0, -11.3, 0.0];
        let wrong_face_point = Array1::<f64>::zeros(p);

        let (cold, cold_active) = solve_quadratic_with_linear_constraints(
            &hessian,
            &rhs,
            &wrong_face_point,
            &constraints,
            None,
        )
        .expect("cold finite dual solve");
        let (warm, warm_active) = solve_quadratic_with_linear_constraints(
            &hessian,
            &rhs,
            &wrong_face_point,
            &constraints,
            Some(&[0, 1, 2]),
        )
        .expect("wrong-face warm hint must affect ordering only");

        assert!(
            cold.iter()
                .zip(warm.iter())
                .all(|(&left, &right)| left.to_bits() == right.to_bits()),
            "strictly-convex QP answer must be bitwise warm-history independent: \
             cold={cold:?}, warm={warm:?}"
        );
        assert_eq!(cold_active, vec![1, 2]);
        assert_eq!(warm_active, vec![1, 2]);
        let expected = array![0.6987_f64, 0.0, 0.0, -11.3, 0.0];
        for (&actual, &target) in cold.iter().zip(expected.iter()) {
            assert_relative_eq!(actual, target, epsilon = 1e-13);
        }

        let gradient = hessian.dot(&cold) - &rhs;
        let active_rows = LinearInequalityConstraints::new(
            constraints.a.select(ndarray::Axis(0), &[1, 2]),
            constraints.b.select(ndarray::Axis(0), &[1, 2]),
        )
        .expect("true active face");
        let (_, system_multipliers) =
            solve_kkt_direction(&hessian, &gradient, &active_rows.a, None)
                .expect("true-face multiplier reconstruction");
        let multipliers = -system_multipliers;
        assert_relative_eq!(multipliers[0], 2.0, epsilon = 1e-13);
        assert_relative_eq!(multipliers[1], 3.0, epsilon = 1e-13);
        assert!(
            multipliers.iter().all(|&value| value > 0.0),
            "the returned face must carry nonnegative KKT multipliers"
        );
        let certified = compute_constraint_kkt_diagnostics(&cold, &gradient, &constraints);
        assert!(certified.primal_feasibility <= 1e-14);
        assert!(certified.dual_feasibility <= 1e-14);
        assert!(certified.complementarity <= 1e-14);
        assert!(certified.stationarity <= 1e-13);
    }

    /// A globalized CTN step can retain only a small subset of the previous
    /// endpoint face. The next H-metric projection must recover every missing
    /// independent normal direction in one separator batch, not pay one dense
    /// face solve per observation-row id.
    #[test]
    fn operator_metric_projection_batches_a_partial_warm_face_979() {
        let rows = 24_000;
        let p = 24;
        let psi = Array2::from_shape_fn((rows, p), |(row, column)| {
            if column == row % p { 1.0 } else { 0.0 }
        });
        let cone =
            KhatriRaoConeConstraints::new(std::sync::Arc::new(psi), vec![0], 1)
                .expect("many-row coordinate cone");
        let set = ConstraintSet::KhatriRaoCone(cone);
        let hessian = Array2::<f64>::eye(p);
        let rhs = Array1::<f64>::from_elem(p, -1.0);
        let beta_start = Array1::<f64>::zeros(p);
        let warm = [0usize, 1, 2, 3];

        let ops = ConstraintSetOps::new(&set, 0.0).expect("operator geometry");
        let unconstrained = rhs.clone();
        let values = ops.values(&unconstrained).expect("free values");
        let mut is_active = vec![false; rows];
        for &row in &warm {
            is_active[row] = true;
        }
        let banned = vec![false; rows];
        let selected = independent_violated_operator_rows(
            &ops,
            &values,
            &warm,
            &is_active,
            &banned,
            p - warm.len(),
        )
        .expect("batch separation");
        assert_eq!(
            selected.len(),
            p - warm.len(),
            "one scan must recover every coefficient-space direction missing from the warm face"
        );

        let (candidate, active) = solve_quadratic_with_constraint_set(
            &hessian,
            &rhs,
            &beta_start,
            &set,
            Some(&warm),
        )
        .expect("batched metric projection");
        assert!(
            candidate.iter().all(|value| value.abs() <= 1e-12),
            "projection onto the repeated coordinate cone must be the origin: {candidate:?}"
        );
        assert_eq!(
            active.len(),
            p,
            "the returned face must contain one representative per independent coordinate"
        );
    }

    /// The strict-interior repair is an identity-metric projection, so a
    /// carrier with many repeated observation rows must cost one finite dual
    /// face in coefficient space. Row cardinality may increase operator-scan
    /// work; it must never restore the retired row-count pivot budget that made
    /// the 320k-row CTN repair grind until the command timeout.
    #[test]
    fn operator_strict_interior_projection_is_coefficient_bounded_on_repeated_rows_979() {
        let rows = 24_000;
        let p = 24;
        let psi = Array2::from_shape_fn((rows, p), |(row, column)| {
            if column == row % p { 1.0 } else { 0.0 }
        });
        let cone =
            KhatriRaoConeConstraints::new(std::sync::Arc::new(psi), vec![0], 1)
                .expect("many-row coordinate cone");
        let set = ConstraintSet::KhatriRaoCone(cone);
        let point = Array1::<f64>::from_elem(p, -1.0);

        let projected = project_point_strictly_into_feasible_constraint_set(&point, &set)
            .expect("finite dual strict-interior projection");
        let values = set.values(projected.view()).expect("projected values");
        for row in 0..set.nrows() {
            let norm = set.row_norm(row).expect("row norm");
            let scaled_slack = values[row] / norm;
            assert!(
                scaled_slack >= 0.5 * ACTIVE_SET_INTERIOR_SEED_MARGIN - 1e-9,
                "row {row} missed the certified interior: {scaled_slack:.3e}"
            );
        }
        for (column, value) in projected.iter().enumerate() {
            assert!(
                *value < ACTIVE_SET_INTERIOR_SEED_MARGIN + 1e-8,
                "identity projection moved coordinate {column} past its nearest interior face: {value:.3e}"
            );
        }
    }

    #[test]
    fn operator_scan_separates_primal_feasibility_from_active_equality_979() {
        let psi = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let cone = KhatriRaoConeConstraints::new(std::sync::Arc::new(psi), vec![0], 1)
            .expect("two-row operator cone");
        let set = ConstraintSet::KhatriRaoCone(cone);
        let ops = ConstraintSetOps::new(&set, 0.0).expect("operator geometry");
        // Row zero has drifted to the feasible side of its nominal equality.
        // This is public-primal feasible but not a certified active face: the
        // two contracts must remain distinct.
        let beta = array![2.0 * ACTIVE_SET_PRIMAL_FEASIBILITY_TOL, 1.0];
        let values = ops.values(&beta).expect("operator values");
        let scan = scan_operator_violations(&ops, &values, &[true, false])
            .expect("full-set violation scan");

        assert!(
            scan.inactive.is_empty(),
            "an active equality is not an admissible entering separator"
        );
        assert!(
            scan.is_primal_feasible(),
            "positive active-row slack is feasible for the one-sided public contract"
        );
        assert_relative_eq!(scan.worst.violation, 0.0, epsilon = 0.0);

        let active_rows = ops
            .gather_unit_rows(&[0])
            .expect("one-row active equality");
        let equality = certify_active_equalities(&active_rows.a, &active_rows.b, &beta);
        assert_eq!(
            equality.worst_row, 0,
            "the only active row must own the equality residual"
        );
        assert!(
            !equality.is_certified(),
            "feasible-side drift must still fail the two-sided active-equality certificate"
        );
        assert!(
            equality.residual > equality.allowed,
            "active equality residual {:.3e} must exceed its roundoff bound {:.3e}",
            equality.residual,
            equality.allowed,
        );
    }

    #[test]
    fn operator_metric_projection_finishes_a_separator_after_partial_drop_979() {
        // Unit normals n0=(1,0) and n1=(c,s), with H=I. The free point is
        // chosen so admitting n0 first and then pivoting toward n1 releases n0
        // with only half the public feasibility tolerance left on n1.
        //
        // The pre-fix dual loop abandoned n1 at that point. Its partial step
        // had already added a positive n1 multiplier to beta, but n1 was not
        // recorded in the active state; terminal conditioning therefore reset
        // to the free point and row-order refill repeated the same two pivots
        // through every conditioning round. The true unique optimum has only
        // n1 active and n0 strictly feasible.
        let sine = 0.1_f64;
        let cosine = (1.0 - sine * sine).sqrt();
        let residual_after_drop = 0.5 * ACTIVE_SET_PRIMAL_FEASIBILITY_TOL;
        let psi = array![[1.0_f64, 0.0], [cosine, sine]];
        let cone = KhatriRaoConeConstraints::new(std::sync::Arc::new(psi), vec![0], 1)
            .expect("partial-drop operator cone");
        let set = ConstraintSet::KhatriRaoCone(cone);
        let hessian = Array2::<f64>::eye(2);
        let unconstrained = array![
            -1.0_f64,
            -sine / cosine - residual_after_drop / sine
        ];
        let beta_start = Array1::<f64>::zeros(2);

        let (candidate, active) = solve_quadratic_with_constraint_set(
            &hessian,
            &unconstrained,
            &beta_start,
            &set,
            Some(&[0]),
        )
        .expect("a pending separator must survive its partial dual drop");

        let normal = array![cosine, sine];
        let multiplier = -normal.dot(&unconstrained);
        let expected = &unconstrained + &(&normal * multiplier);
        assert!(
            residual_after_drop <= ACTIVE_SET_PRIMAL_FEASIBILITY_TOL,
            "the fixture must leave the pending separator inside the public tolerance after \
             its partial drop"
        );
        assert!(
            multiplier > 1.0,
            "the true multiplier must be cumulative, not the tolerance-sized final step: \
             {multiplier:.3e}"
        );
        for (&actual, &oracle) in candidate.iter().zip(expected.iter()) {
            assert_relative_eq!(actual, oracle, epsilon = 5e-14);
        }
        let gradient = &candidate - &unconstrained;
        let expected_gradient = &normal * multiplier;
        for (&actual, &oracle) in gradient.iter().zip(expected_gradient.iter()) {
            assert_relative_eq!(actual, oracle, epsilon = 5e-14);
        }
        assert_eq!(
            active,
            vec![1],
            "the unique optimum is supported by the second normal only"
        );
        let values = set.values(candidate.view()).expect("candidate values");
        let scan = scan_operator_violations(
            &ConstraintSetOps::new(&set, 0.0).expect("operator geometry"),
            &values,
            &[false, true],
        )
        .expect("candidate feasibility");
        assert!(scan.is_primal_feasible());
        assert!(
            candidate[0] > 0.0,
            "released row zero must be strictly feasible at the optimum"
        );
    }

    #[test]
    fn operator_cone_qp_takes_unconstrained_path_when_interior() {
        let cone = small_cone();
        let set = ConstraintSet::KhatriRaoCone(cone);
        let p = set.ncols();
        let hessian = coupled_pd_hessian(p);
        // rhs pushing every coupled functional UP: unconstrained optimum is
        // strictly interior, so the operator path must equal the plain solve.
        let rhs = array![0.2_f64, 0.1, 3.0, 0.2, 2.5, 0.1];
        let beta_start = array![0.0_f64, 0.0, 1.0, 0.0, 1.0, 0.0];
        let (beta_op, active_op) =
            solve_quadratic_with_constraint_set(&hessian, &rhs, &beta_start, &set, None)
                .expect("operator solve");
        // Dense unconstrained oracle: H β = rhs.
        let mut beta_unconstrained = Array1::<f64>::zeros(p);
        super::solve_newton_direction_dense(
            &hessian,
            &(hessian.dot(&beta_start) - &rhs),
            &mut beta_unconstrained,
        )
        .expect("unconstrained newton");
        let beta_unconstrained = &beta_start + &beta_unconstrained;
        for j in 0..p {
            assert!(
                (beta_op[j] - beta_unconstrained[j]).abs() < 1e-8,
                "interior operator solve must match unconstrained optimum at {j}"
            );
        }
        assert!(
            active_op.is_empty(),
            "interior optimum must have empty face"
        );
    }

    #[test]
    fn operator_projection_returns_strictly_interior_point() {
        let cone = small_cone();
        let set = ConstraintSet::KhatriRaoCone(cone);
        // Infeasible point: coupled row 1 loaded negative everywhere.
        let point = array![0.4_f64, -0.2, -1.0, -0.5, 0.3, 0.05];
        let projected = project_point_strictly_into_feasible_constraint_set(&point, &set)
            .expect("projection must succeed on a one-sided homogeneous cone");
        let values = set.values(projected.view()).expect("values");
        for row in 0..set.nrows() {
            let norm = set.row_norm(row).expect("norm");
            if norm <= 0.0 {
                continue;
            }
            let slack = values[row] / norm;
            assert!(
                slack >= 0.5 * ACTIVE_SET_INTERIOR_SEED_MARGIN - 1e-9,
                "projected point not strictly interior on row {row}: slack {slack:.3e}"
            );
        }
        // The location coordinates (unconstrained) must be untouched by the
        // projection objective's optimum only if already optimal; at minimum
        // they must remain finite and close to the input (they carry no
        // constraint rows, and the identity-Hessian QP has no incentive to
        // move them).
        assert!((projected[0] - point[0]).abs() < 1e-8);
        assert!((projected[1] - point[1]).abs() < 1e-8);
    }

    /// #2378 regression, independent oracle. The operator strict-interior
    /// projection onto an OVER-COMPLETE cone face (three of a 2-D block's four
    /// half-spaces try to bind — rank 2) must not merely land on *a* feasible
    /// point; it must be the correct Euclidean projection. The former loose
    /// rank-reduction truncated the true binding extreme (row 2) out of the
    /// enforced face and refused the fit; a regression in the over-complete-face
    /// exchange would release the wrong representative and land on a different
    /// feasible vertex. Both are caught by matching the dense oracle over the
    /// same materialized rows AND by pinning which pair binds ({1,2}, not {1,3}).
    #[test]
    fn operator_projection_adjudicates_the_over_complete_face_2378() {
        let cone = small_cone();
        let set = ConstraintSet::KhatriRaoCone(cone.clone());
        // The #2378 witness point: coupled block 1 = coords[2..4] = (-1, -0.5)
        // is over-complete; block 2 is left feasible.
        let point = array![0.4_f64, -0.2, -1.0, -0.5, 0.3, 0.05];
        let projected = project_point_strictly_into_feasible_constraint_set(&point, &set)
            .expect("operator projection must certify the over-complete-face vertex");

        // Ground-truth oracle: the SAME projection over the dense materialization
        // of the cone rows, through the independent dense arm.
        let dense = ConstraintSet::Dense(cone.to_dense().expect("dense oracle"));
        let dense_proj = project_point_strictly_into_feasible_constraint_set(&point, &dense)
            .expect("dense projection oracle");
        for j in 0..point.len() {
            assert!(
                (projected[j] - dense_proj[j]).abs() < 1e-7,
                "operator projection diverged from the dense oracle at {j}: \
                 op={:.9e} dense={:.9e}",
                projected[j],
                dense_proj[j]
            );
        }

        // The correct binding pair is block-1 rows {1,2} (flat ids 1 and 2 in
        // slot 0). Row 2 — the extreme the old code truncated — must be TIGHT,
        // not violated. Rows are `slot*n + obs`, n = 4 Ψ rows, coupled slot 0.
        let values = set.values(projected.view()).expect("values");
        let scaled = |row: usize| values[row] / set.row_norm(row).expect("norm");
        // Rows 1 and 2 bind at (near) the strict-interior margin floor…
        for row in [1usize, 2] {
            assert!(
                scaled(row) < ACTIVE_SET_INTERIOR_SEED_MARGIN + 1e-7,
                "block-1 row {row} should bind, scaled slack {:.3e}",
                scaled(row)
            );
        }
        // …while rows 0 and 3 stay strictly slacker than the binding pair.
        for row in [0usize, 3] {
            assert!(
                scaled(row) > scaled(2) + 1e-9,
                "non-binding row {row} (slack {:.3e}) must exceed the binding \
                 row 2 (slack {:.3e})",
                scaled(row),
                scaled(2)
            );
        }
    }

    /// #2378 regression at the QP level (non-identity Hessian): operator-native
    /// metric projection must reach the same constrained minimizer as the dense
    /// oracle when a coupled block is loaded so that three of its half-spaces
    /// contend at the optimum.
    #[test]
    fn operator_cone_qp_over_complete_face_matches_dense_oracle_2378() {
        let cone = small_cone();
        let set = ConstraintSet::KhatriRaoCone(cone.clone());
        let dense = cone.to_dense().expect("dense oracle");
        let p = set.ncols();
        let hessian = coupled_pd_hessian(p);
        // Drive block-1's unconstrained optimum deep into the infeasible corner
        // where the extreme Ψ rows 1 and 2 both contend (the over-complete face),
        // and pin block-2 with its own mild load.
        let rhs = array![0.3_f64, -0.1, -2.5, -1.2, -0.4, 0.2];
        let beta_start = array![0.0_f64, 0.0, 1.0, 0.1, 1.0, 0.1];

        let (beta_op, active_op) =
            solve_quadratic_with_constraint_set(&hessian, &rhs, &beta_start, &set, None)
                .expect("operator QP solve over an over-complete face");
        let (beta_dense, _active_dense) =
            solve_quadratic_with_linear_constraints(&hessian, &rhs, &beta_start, &dense, None)
                .expect("dense QP oracle");

        for j in 0..p {
            assert!(
                (beta_op[j] - beta_dense[j]).abs() < 1e-7,
                "operator/dense coefficient {j} mismatch: {} vs {}",
                beta_op[j],
                beta_dense[j]
            );
        }
        // The operator answer is feasible on the full factored cone.
        let values = set.values(beta_op.view()).expect("values");
        for row in 0..set.nrows() {
            let norm = set.row_norm(row).expect("norm");
            if norm > 0.0 {
                assert!(
                    values[row] / norm >= -ACTIVE_SET_PRIMAL_FEASIBILITY_TOL,
                    "row {row} violated at the operator optimum: {:.3e}",
                    values[row] / norm
                );
            }
        }
        assert!(
            active_op.len() <= p,
            "operator passive face must contain at most one row per coefficient-space direction: \
             active={}, p={p}",
            active_op.len()
        );
    }

    #[test]
    fn operator_cone_does_not_materialize_a_whole_tight_face() {
        // All 4,096 observation rows describe the same half-space.  At the
        // cone vertex every row is tight, but one generator completely
        // describes the dual support. The operator solver must return that
        // compact support instead of gathering/rank-reducing all 4,096
        // redundant rows (the large-scale CTN cycle-2 stall from #979).
        let mut psi = Array2::<f64>::zeros((4096, 2));
        psi.column_mut(0).fill(1.0);
        let cone = KhatriRaoConeConstraints::new(std::sync::Arc::new(psi), vec![1], 2)
            .expect("repeated-row cone");
        let set = ConstraintSet::KhatriRaoCone(cone);
        let hessian = Array2::<f64>::eye(4);
        let rhs = array![0.3_f64, -0.2, -1.0, 0.0];
        let beta_start = Array1::<f64>::zeros(4);

        // Seed a non-first representative. The operator arm must consume this
        // point-tight warm face instead of rescanning 4,096 equivalent rows and
        // deterministically rediscovering row zero.
        let warm_row = 2048usize;
        let (beta, active) = solve_quadratic_with_constraint_set(
            &hessian,
            &rhs,
            &beta_start,
            &set,
            Some(&[warm_row]),
        )
        .expect("vertex solve");

        assert_eq!(
            active,
            vec![warm_row],
            "the compact point-tight warm representative was discarded or redundant rows entered"
        );
        assert!(beta[2].abs() <= ACTIVE_SET_PRIMAL_FEASIBILITY_TOL);
        assert!((beta[0] - 0.3).abs() < 1e-10);
        assert!((beta[1] + 0.2).abs() < 1e-10);
    }

    /// The KKT diagnostic's sign convention, pinned on the exact face that
    /// produces `stat_rel = 1.000e0` in #2601.
    ///
    /// `[convex]` on clean linear data reports `primal=8.4e-13`,
    /// `comp=4.4e-16`, `active=10/10`, every multiplier zero, and a
    /// stationarity residual equal to the whole gradient. Multipliers
    /// identically zero on a fully active face is not "slightly off": NNLS
    /// leaves `λ ≡ 0` exactly when NO active row has positive correlation with
    /// the target, so all ten rows failed the entry test at once. A gradient
    /// merely falling outside a pointed cone would normally still recruit some
    /// rows and leave `stat_rel < 1`. All of them failing together is what a
    /// gradient sitting in the POLAR cone looks like — i.e. a flipped sign.
    ///
    /// Before hunting the sign at the four call sites, establish which sign the
    /// projector itself expects. Constraints are `A β ≥ b`, so for a MINIMISED
    /// objective the Lagrangian is `f − λᵀ(Aβ − b)` and stationarity is
    /// `∇f = Aᵀλ, λ ≥ 0`. This test asserts exactly that, and asserts the
    /// negation produces the observed pathology — so if the projector is ever
    /// "fixed" by flipping it, this fails and says which direction is which.
    ///
    /// The face is the #2601 face: second-difference (convexity) rows, β
    /// affine so every row is tight to the last bit and `A_active` is
    /// rank-deficient by construction.
    #[test]
    fn the_kkt_cone_convention_is_grad_equals_a_transpose_lambda_2601() {
        let p = 6usize;
        let m = p - 2;
        let mut a = Array2::<f64>::zeros((m, p));
        for i in 0..m {
            a[[i, i]] = 1.0;
            a[[i, i + 1]] = -2.0;
            a[[i, i + 2]] = 1.0;
        }
        // Affine β has exactly zero second differences, so all m rows are
        // tight and the active face is the whole constraint set.
        let beta = Array1::from_shape_fn(p, |j| 0.5 + 2.0 * (j as f64));
        let constraints =
            LinearInequalityConstraints::new(a.clone(), Array1::<f64>::zeros(m))
                .expect("second-difference constraints");

        // The diagnostic works in per-row-normalised units, so build the
        // gradient from the SAME scaled rows its λ will be expressed in.
        let row_norm = 6.0_f64.sqrt(); // ‖[1, −2, 1]‖
        let a_scaled = a.mapv(|v| v / row_norm);
        let lambda_true = Array1::from_vec(vec![0.25, 1.5, 0.0, 3.0]);
        assert_eq!(lambda_true.len(), m);

        let aligned = a_scaled.t().dot(&lambda_true);
        let diag = compute_constraint_kkt_diagnostics(&beta, &aligned, &constraints);
        assert_eq!(
            diag.n_active, m,
            "an affine β must make every second-difference row tight"
        );
        assert!(
            diag.stationarity <= 1e-12 * diag.gradient_scale.max(1.0),
            "∇f = Aᵀλ with λ ≥ 0 IS the stationarity condition for A β ≥ b; the cone \
             projector must absorb it entirely (stat={:.6e}, ‖g‖∞={:.6e}, active={}/{})",
            diag.stationarity,
            diag.gradient_scale,
            diag.n_active,
            diag.n_constraints,
        );
        assert!(
            diag.dual_feasibility <= 1e-12,
            "recovered multipliers must stay nonnegative (dual={:.6e})",
            diag.dual_feasibility,
        );

        // Now the negation, which is what a flipped sign would look like.
        //
        // I expected `λ ≡ 0` and `stat_rel = 1.0` here -- the #2601 signature --
        // and the measurement said otherwise: `stat = 1.224745` against
        // `‖g‖∞ = 2.449490`, exactly HALF absorbed. A gradient in the polar
        // cone still recruits rows on this face, because the cone generated by
        // second-difference rows is not self-polar: some `−a_iᵀ(Aᵀλ_true)` are
        // still positive and enter the passive set.
        //
        // That refutes the sign explanation for #2601. A flipped sign cannot
        // produce a stationarity residual EQUAL to the gradient on this face,
        // so whatever leaves `λ ≡ 0` there is not a convention error. What
        // remains is a refusal: `project_stationarity_residual_on_constraint_cone`
        // returns `None` on a non-finite target, a width mismatch, or the
        // Lawson-Hanson guard, and the caller then leaves `λ` at its zeros --
        // which IS exactly `stat_rel = 1.0`. Hence `cone_projection_refused`.
        //
        // The `0.5` is asserted as a fact about this face, not as a target: it
        // is what makes the sign hypothesis untenable, and if it ever moves,
        // the reasoning above needs redoing.
        let opposed = aligned.mapv(|v| -v);
        let flipped = compute_constraint_kkt_diagnostics(&beta, &opposed, &constraints);
        let opposed_scale = opposed.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        assert!(
            !flipped.cone_projection_refused,
            "the projector answered for the negated gradient; it did not refuse"
        );
        let absorbed_fraction = 1.0 - flipped.stationarity / opposed_scale;
        assert!(
            absorbed_fraction > 0.25,
            "a polar-cone gradient still recruits rows on this face -- it does NOT \
             reproduce #2601's stat_rel = 1.0, which is why a flipped sign cannot be \
             the explanation there (stat={:.6e}, ‖g‖∞={:.6e}, absorbed={:.6e})",
            flipped.stationarity,
            opposed_scale,
            absorbed_fraction,
        );
        assert_eq!(
            flipped.n_active, m,
            "the face is a property of β, not of the gradient's sign"
        );
    }
/// #979 CTN plateau regression: the operator-native Lawson-Hanson Moreau
    /// solve must certify a degenerate fully-pinned vertex directly instead of
    /// spending a primal-QP iteration budget on one-row blocker exchanges.
    #[test]
    fn operator_nnls_certifies_pinned_degenerate_vertex_projection_979() {
        // Four generators in R^3 (degenerate: a4 = a1 + a2), all tight at the
        // origin. The stationarity residual is a nonnegative combination, so
        // the projected residual is exactly zero.
        let a = array![
            [1.0_f64, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ];
        let b = array![0.0_f64, 0.0, 0.0, 0.0];
        let set = ConstraintSet::Dense(
            LinearInequalityConstraints::new(a, b).expect("degenerate vertex cone"),
        );
        let beta = array![0.0_f64, 0.0, 0.0];
        let residual = array![3.0_f64, 2.0, 0.0]; // = a1 + 2·a4
        let (projected, active) = project_stationarity_residual_on_constraint_set(
            &residual,
            &beta,
            &set,
            &[0, 1],
        )
        .expect("operator NNLS must solve the degenerate vertex");
        let closure = projected.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        assert!(
            closure <= 1e-9,
            "residual is in the cone; projection must close to zero, got {closure:.3e}"
        );
        assert!(!active.is_empty(), "a supported face must be reported");

        // A component outside the cone must survive the projection exactly.
        let outside = array![1.0_f64, 0.0, -1.0];
        let (projected_outside, _) =
            project_stationarity_residual_on_constraint_set(&outside, &beta, &set, &[])
                .expect("operator NNLS must solve the outside-component case");
        assert_relative_eq!(projected_outside[0], 0.0, epsilon = 1e-9);
        assert_relative_eq!(projected_outside[1], 0.0, epsilon = 1e-9);
        assert_relative_eq!(projected_outside[2], -1.0, epsilon = 1e-9);
    }

/// The projector is a KKT certificate input, so a row that is NOT tight at
    /// `beta` must never enter the generator set: a residual
    /// aligned with a slack row must stay unprojected rather than be absorbed
    /// by a constraint that is not active at the iterate.
    #[test]
    fn operator_nnls_excludes_rows_not_tight_at_beta() {
        let a = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let b = array![0.0_f64, -1.0]; // row 1 has slack 1 at the origin
        let set = ConstraintSet::Dense(
            LinearInequalityConstraints::new(a, b).expect("half-tight system"),
        );
        let beta = array![0.0_f64, 0.0];
        let residual = array![0.0_f64, 1.0];
        let (projected, active) =
            project_stationarity_residual_on_constraint_set(&residual, &beta, &set, &[])
                .expect("operator NNLS must solve the half-tight system");
        assert_relative_eq!(projected[1], 1.0, epsilon = 1e-12);
        assert!(
            !active.contains(&1),
            "slack row 1 must not appear in the certified face"
        );
    }

#[test]
    fn separable_khatri_rao_tangent_projection_matches_dense_oracle() {
        let cone = small_cone();
        let set = ConstraintSet::KhatriRaoCone(cone.clone());
        let dense = cone.to_dense().expect("dense projection oracle");
        let beta = Array1::<f64>::zeros(set.ncols());
        let residual = array![0.4_f64, -0.2, 1.1, -0.7, -0.9, 0.8];

        let (operator_projected, _) =
            project_stationarity_residual_on_constraint_set(&residual, &beta, &set, &[])
                .expect("separable operator projection");
        let (dense_projected, _) =
            project_stationarity_residual_on_constraint_cone(&residual, &dense.a)
                .expect("dense cone projection");

        for index in 0..residual.len() {
            assert_relative_eq!(
                operator_projected[index],
                dense_projected[index],
                epsilon = 1e-8
            );
        }
    }

/// The current #979 production shape has hundreds of thousands of
    /// factored rows over only 24 coefficients. Projection work must scale
    /// with batched row products plus the coefficient-dimensional passive
    /// set, not with one primal-QP transition per row id.
    #[test]
    fn operator_moreau_projection_has_coefficient_sized_support_979() {
        let rows = 24_000;
        let psi = Array2::from_shape_fn((rows, 3), |(row, column)| {
            let axis = (row % 6) / 2;
            if column == axis {
                if row % 2 == 0 { 1.0 } else { -1.0 }
            } else {
                0.0
            }
        });
        let cone =
            KhatriRaoConeConstraints::new(std::sync::Arc::new(psi), vec![0], 1)
                .expect("many-row low-dimensional cone");
        let dense = cone.to_dense().expect("dense parity oracle");
        let set = ConstraintSet::KhatriRaoCone(cone);
        let beta = Array1::<f64>::zeros(3);
        let residual = array![3.0_f64, -2.0, 1.0];

        let (operator_projected, active) =
            project_stationarity_residual_on_constraint_set(&residual, &beta, &set, &[])
                .expect("operator Moreau projection");
        let (_, dense_projected) =
            nonnegative_cone_multipliers(&dense.a, &residual).expect("dense NNLS oracle");

        for index in 0..residual.len() {
            assert_relative_eq!(
                operator_projected[index],
                dense_projected[index],
                epsilon = 1e-10
            );
            assert_relative_eq!(operator_projected[index], 0.0, epsilon = 1e-10);
        }
        assert!(
            active.len() <= residual.len(),
            "a three-dimensional cone projection gathered {} supported rows",
            active.len()
        );
    }

#[test]
    fn operator_tangent_projection_does_not_constrain_interior_rows() {
        let psi = array![[1.0_f64, 0.0], [1.0, 1.0], [1.0, -1.0]];
        let cone = KhatriRaoConeConstraints::new(std::sync::Arc::new(psi), vec![1], 2)
            .expect("interior tangent cone");
        let set = ConstraintSet::KhatriRaoCone(cone);
        // The shaped response row is strictly positive for every observation,
        // so its tangent cone is the complete coefficient space. A projection
        // against the original cone at the origin would incorrectly erase the
        // shaped constant component of this residual.
        let beta = array![0.0_f64, 0.0, 1.0, 0.0];
        let residual = array![0.0_f64, 0.0, 1.0, 0.0];
        let (projected, active) =
            project_stationarity_residual_on_constraint_set(&residual, &beta, &set, &[])
                .expect("interior tangent projection");

        for index in 0..residual.len() {
            assert_relative_eq!(projected[index], residual[index], epsilon = 1e-12);
        }
        assert!(active.is_empty(), "interior rows entered the tangent face");
    }

#[test]
    fn operator_tangent_projection_homogenizes_an_affine_boundary() {
        let set = ConstraintSet::Dense(
            LinearInequalityConstraints::new(array![[1.0_f64, 0.0]], array![2.0])
                .expect("affine half-space"),
        );
        let beta = array![2.0_f64, 0.0];
        let residual = array![1.0_f64, -1.0];
        let (projected, active) =
            project_stationarity_residual_on_constraint_set(&residual, &beta, &set, &[0])
                .expect("affine-boundary tangent projection");

        assert_relative_eq!(projected[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(projected[1], -1.0, epsilon = 1e-12);
        assert_eq!(active, vec![0]);
    }

}
