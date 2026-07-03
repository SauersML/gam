use crate::estimate::EstimationError;
use gam_linalg::faer_ndarray::{FaerArrayView, FaerLinalgError, FaerSvd, array1_to_col_matmut};
use gam_linalg::utils::{StableSolver, array_is_finite, boundary_hit_step_fraction};
use faer::linalg::solvers::{Lblt as FaerLblt, Solve as FaerSolve};
use faer::{Side, Unbind};
use gam_problem::LinearInequalityConstraints;
use ndarray::{Array1, Array2, s};
use serde::{Deserialize, Serialize};
use std::cell::Cell;
use std::collections::HashSet;

/// Primal-feasibility tolerance the inequality-constrained active-set Newton
/// solver guarantees on its returned iterate, measured in the *scaled*
/// constraint-row coordinate system in which `A * beta >= b` is expressed.
///
/// The solver accepts a step when the worst scaled violation
/// `max_i (b_i - a_i^T beta)` is below this threshold (see the acceptance
/// gate in [`solve_linear_constrained_newton_step`] and the KKT diagnostics
/// in [`compute_constraint_kkt_diagnostics`]). Any consumer that re-derives a
/// raw (un-scaled) feasibility tolerance from a returned iterate must scale
/// this value by the per-row normalization that the constraint builder
/// applied; demanding tighter feasibility than this is inconsistent with the
/// solver contract and will spuriously reject valid boundary solutions.
pub const ACTIVE_SET_PRIMAL_FEASIBILITY_TOL: f64 = 1e-8;

/// Stationarity tolerance for the strong-KKT acceptance gate: the projected
/// (working-set) gradient residual ‖∇L − Aᵀλ‖∞, either absolute or relative to
/// `max(1, ‖∇L‖∞)`, must fall below this to certify a constrained stationary
/// point. Matched against `ACTIVE_SET_KKT_COMPLEMENTARITY_TOL` so both KKT
/// residual channels are certified at compatible scales.
const ACTIVE_SET_KKT_STATIONARITY_TOL: f64 = 2e-6;

/// Complementarity-slackness tolerance for the KKT acceptance gate:
/// `max_i |λ_i · slack_i|` must fall below this for the
/// active-inactive partition to be consistent.
const ACTIVE_SET_KKT_COMPLEMENTARITY_TOL: f64 = 1e-6;

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

/// Relative scale on the predicted-decrease test `predicted_delta ≤
/// −ε·(1 + ‖∇L‖∞·‖d‖∞)`: when the working-set Newton step still buys a
/// quadratic-model decrease at this relative margin the step is a usable
/// descent direction even if the KKT residual has not yet tightened.
const ACTIVE_SET_MODEL_DESCENT_REL_TOL: f64 = 1e-10;

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

    let factor = StableSolver::new("active-set newton direction")
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

pub(crate) fn compute_constraint_kkt_diagnostics(
    beta: &Array1<f64>,
    gradient: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
) -> ConstraintKktDiagnostics {
    let m = constraints.a.nrows();
    let active_tolerance = ACTIVE_SET_PRIMAL_FEASIBILITY_TOL;

    // Measure feasibility in the *scaled* (geometric) coordinate system the
    // solver's tolerance is expressed in: normalize each inequality
    // `a_i·β ≥ b_i` by ‖a_i‖ so its slack becomes the signed Euclidean
    // distance from β to the constraint hyperplane. Without this, a row with a
    // large norm — e.g. a B-spline endpoint *derivative* clamp, whose rows
    // carry ‖a_i‖ ≫ 1 — reports a raw slack inflated by ‖a_i‖, so an iterate
    // that is feasible to the solver's scaled `ACTIVE_SET_PRIMAL_FEASIBILITY_TOL`
    // guarantee can still exceed a raw primal gate downstream and be spuriously
    // refused. Per-row normalization makes the diagnostic scale-invariant and
    // consistent with that contract. Dual/complementarity/stationarity are
    // invariant under this positive per-row rescaling (with λ̂_i = ‖a_i‖·λ_i:
    // Âᵀλ̂ = Aᵀλ and λ̂_i·ŝ_i = λ_i·s_i), so only primal feasibility and the
    // active-set threshold change meaning — both toward the geometric distance
    // the tolerance is meant to bound.
    let p = constraints.a.ncols();
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

    let active_idx: Vec<usize> = (0..m).filter(|&i| slack[i] <= active_tolerance).collect();
    let mut lambda = Array1::<f64>::zeros(m);
    let mut working_set_rank_deficient = false;
    if !active_idx.is_empty() {
        let n_active = active_idx.len();
        let mut a_active = Array2::<f64>::zeros((n_active, p));
        for (r, &idx) in active_idx.iter().enumerate() {
            a_active.row_mut(r).assign(&a_scaled.row(idx));
        }
        if let Some((_, lambda_active)) =
            project_stationarity_residual_on_constraint_cone(gradient, &a_active)
        {
            for (r, &idx) in active_idx.iter().enumerate() {
                lambda[idx] = lambda_active[r];
            }
        }
        // Rank-deficiency detection on the (scaled) active rows. Per-row
        // positive scaling is rank-preserving, so this answers the same
        // question the inner solver's `CompressedActiveWorkingSet::
        // is_degenerate_face` does — `rank(A_active) < n_active`. For curvature
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
        gradient_scale: gradient_inf_norm(gradient),
    }
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

    let m = active_a.nrows();
    let residual_scale = residual
        .iter()
        .fold(0.0_f64, |acc, &v| acc.max(v.abs()))
        .max(1.0);
    let row_scale = active_a
        .iter()
        .fold(0.0_f64, |acc, &v| acc.max(v.abs()))
        .max(1.0);
    let tol = 100.0 * f64::EPSILON * (p.max(m).max(1) as f64) * residual_scale * row_scale;

    let mut lambda = Array1::<f64>::zeros(m);
    let mut passive = vec![false; m];
    let mut projected = residual.clone();
    let max_iter = (3 * m * m).max(10);

    for _ in 0..max_iter {
        let dual = active_a.dot(&projected);
        let entering = (0..m)
            .filter(|&idx| !passive[idx] && dual[idx] > tol)
            .max_by(|&left, &right| {
                dual[left]
                    .partial_cmp(&dual[right])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        let Some(entering) = entering else {
            lambda.mapv_inplace(|v| if v > tol { v } else { 0.0 });
            projected = residual - &active_a.t().dot(&lambda);
            if !array_is_finite(&projected) || !array_is_finite(&lambda) {
                return None;
            }
            return Some((projected, lambda));
        };
        passive[entering] = true;

        loop {
            let passive_rows: Vec<usize> = (0..m).filter(|&idx| passive[idx]).collect();
            if passive_rows.is_empty() {
                lambda.fill(0.0);
                projected.assign(residual);
                break;
            }

            let mut a_passive = Array2::<f64>::zeros((passive_rows.len(), p));
            for (pos, &row) in passive_rows.iter().enumerate() {
                a_passive.row_mut(pos).assign(&active_a.row(row));
            }
            let gram = a_passive.dot(&a_passive.t());
            let rhs = a_passive.dot(residual);
            let mut lambda_passive = Array1::<f64>::zeros(passive_rows.len());
            solve_dense_system_via_pseudoinverse(&gram, &rhs, &mut lambda_passive).ok()?;
            if !array_is_finite(&lambda_passive) {
                return None;
            }

            let all_positive = lambda_passive.iter().all(|&v| v > tol);
            if all_positive {
                lambda.fill(0.0);
                for (pos, &row) in passive_rows.iter().enumerate() {
                    lambda[row] = lambda_passive[pos];
                }
                projected = residual - &active_a.t().dot(&lambda);
                break;
            }

            let mut alpha = f64::INFINITY;
            for (pos, &row) in passive_rows.iter().enumerate() {
                let candidate = lambda_passive[pos];
                if candidate <= tol {
                    let current = lambda[row];
                    let denom = current - candidate;
                    if denom > 0.0 {
                        alpha = alpha.min(current / denom);
                    }
                }
            }
            if !alpha.is_finite() {
                alpha = 0.0;
            }
            alpha = alpha.clamp(0.0, 1.0);

            for (pos, &row) in passive_rows.iter().enumerate() {
                lambda[row] += alpha * (lambda_passive[pos] - lambda[row]);
                if lambda[row] <= tol {
                    lambda[row] = 0.0;
                    passive[row] = false;
                }
            }
            if passive.iter().all(|&is_passive| !is_passive) {
                projected.assign(residual);
                break;
            }
        }
    }

    // Exhausting the Lawson-Hanson iterations means we do not have the
    // normal-cone projection required for a KKT certificate.
    None
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
    if constraints.b.iter().all(|v| v.abs() <= 1e-14) {
        return Some(Array1::zeros(p));
    }

    let gram = constraints.a.dot(&constraints.a.t());
    let (u_opt, singular, vt_opt) = gram.svd(true, true).ok()?;
    let (Some(u), Some(vt)) = (u_opt, vt_opt) else {
        return None;
    };
    let max_singular = singular.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let tol = 100.0 * f64::EPSILON * constraints.a.nrows().max(1) as f64 * max_singular.max(1.0);
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
    let slack = constraints.a.dot(&beta) - &constraints.b;
    if slack.iter().all(|v| *v >= -1e-8) {
        Some(beta)
    } else {
        None
    }
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
/// [`project_point_strictly_into_feasible_cone`] and
/// [`solve_quadratic_with_linear_constraints`] are mutually recursive: the
/// quadratic solve's final feasibility contract (#1108) projects an infeasible
/// iterate back onto the cone, and that projection itself solves an inner
/// identity-Hessian QP whose *own* feasibility contract can project again. On a
/// well-conditioned cone the repair converges at depth 0–1 — each level shifts
/// every one-sided row strictly further inward. But on near-anti-parallel rows
/// (the clamped / anchored monotone time-warp constraints an interval-censored
/// survival fit emits, which are only *near* — not exactly — anti-parallel and so
/// slip past the zero-width equality lift below), the inward-shifted QP can keep
/// returning an infeasible candidate, so the `solve ↔ project` recursion never
/// bottoms out and exhausts the worker stack. A cone that cannot be certified
/// feasible within this many successive inward shifts is degenerate; the
/// projection then returns `None`, which the quadratic solve reports as
/// [`EstimationError::ParameterConstraintViolation`] rather than recursing.
const MAX_FEASIBILITY_REPAIR_DEPTH: u32 = 16;

thread_local! {
    /// Current nesting depth of the `solve ↔ project` feasibility-repair cycle on
    /// this thread. Every recursion path (the quadratic solve's repair step and
    /// the projected-gradient fallback alike) routes back through
    /// [`project_point_strictly_into_feasible_cone`], so bounding its re-entrancy
    /// bounds the whole cycle. Per-thread because each solve runs to completion on
    /// a single call stack; independent solves on other worker threads carry their
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
/// [`feasible_point_for_linear_constraints`], which returns the *minimum-norm*
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
/// `None` if the constraints are malformed or the active-set QP cannot certify a
/// feasible solution, so callers can fall back.
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

/// Worst primal-feasibility violation across all rows of `constraints`,
/// measured in the per-row-scaled (geometric) coordinate system documented on
/// [`ACTIVE_SET_PRIMAL_FEASIBILITY_TOL`]. A row's slack is divided by ‖a_i‖
/// so the returned value is the signed Euclidean distance from `beta` to the
/// constraint hyperplane, not the raw dot-product residual. This matches
/// [`compute_constraint_kkt_diagnostics`] and keeps the in-solver acceptance
/// gate, the downstream KKT report, and any post-fit feasibility check on a
/// single scale-invariant metric. A row with ‖a_i‖ = 38 (a typical B-spline
/// endpoint-derivative clamp at k = 12) has a 1e-6 raw violation that is only
/// 2.6e-8 in geometric units — accepting on raw alone is anisotropic in
/// input-space and breaks the published contract.
fn max_linear_constraint_violation(
    beta: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
) -> (f64, usize) {
    let mut worst = 0.0_f64;
    let mut worst_row = 0usize;
    for i in 0..constraints.a.nrows() {
        let norm = constraints.a.row(i).dot(&constraints.a.row(i)).sqrt();
        let inv = if norm > 0.0 { 1.0 / norm } else { 0.0 };
        let slack = (constraints.a.row(i).dot(beta) - constraints.b[i]) * inv;
        let viol = (-slack).max(0.0);
        if viol > worst {
            worst = viol;
            worst_row = i;
        }
    }
    (worst, worst_row)
}

/// Per-row signed scaled slack: `(a_i·beta - b_i) / ‖a_i‖`. Returns zero for
/// degenerate rows with `‖a_i‖ = 0` (those rows carry no geometric content).
/// Used wherever the active-set solver needs to compare per-row feasibility
/// against a scale-invariant tolerance.
#[inline]
fn scaled_constraint_slack(
    beta: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
    i: usize,
) -> f64 {
    let norm = constraints.a.row(i).dot(&constraints.a.row(i)).sqrt();
    let inv = if norm > 0.0 { 1.0 / norm } else { 0.0 };
    (constraints.a.row(i).dot(beta) - constraints.b[i]) * inv
}

pub(crate) fn solve_kkt_direction(
    hessian: &Array2<f64>,
    gradient: &Array1<f64>,
    active_a: &Array2<f64>,
    active_residual: Option<&Array1<f64>>,
) -> Result<(Array1<f64>, Array1<f64>), EstimationError> {
    let p = hessian.nrows();
    let m = active_a.nrows();
    if hessian.ncols() != p || gradient.len() != p || active_a.ncols() != p {
        crate::bail_invalid_estim!("KKT solve dimension mismatch");
    }
    if let Some(residual) = active_residual
        && residual.len() != m
    {
        crate::bail_invalid_estim!(
            "KKT active residual length mismatch: got {}, expected {}",
            residual.len(),
            m
        );
    }
    if m == 0 {
        let mut d = Array1::<f64>::zeros(p);
        solve_newton_direction_dense(hessian, gradient, &mut d)?;
        return Ok((d, Array1::zeros(0)));
    }
    let mut kkt = Array2::<f64>::zeros((p + m, p + m));
    kkt.slice_mut(s![0..p, 0..p]).assign(hessian);
    kkt.slice_mut(s![0..p, p..(p + m)]).assign(&active_a.t());
    kkt.slice_mut(s![p..(p + m), 0..p]).assign(active_a);

    let mut rhs = Array1::<f64>::zeros(p + m);
    for i in 0..p {
        rhs[i] = -gradient[i];
    }
    if let Some(residual) = active_residual {
        for i in 0..m {
            rhs[p + i] = residual[i];
        }
    }
    let rhs_target = rhs.clone();

    let kkt_view = FaerArrayView::new(&kkt);
    let factor = FaerLblt::new(kkt_view.as_ref(), Side::Lower);
    let mut rhs_col = array1_to_col_matmut(&mut rhs);
    factor.solve_in_place(rhs_col.as_mut());
    if !rhs.iter().all(|v| v.is_finite()) {
        solve_dense_system_via_pseudoinverse(&kkt, &rhs_target, &mut rhs)?;
    }
    let d = rhs.slice(s![0..p]).to_owned();
    let lambda = rhs.slice(s![p..(p + m)]).to_owned();
    Ok((d, lambda))
}

#[derive(Clone, Debug)]
pub(crate) struct CompressedActiveWorkingSet {
    pub(crate) constraints: LinearInequalityConstraints,
    pub(crate) groups: Vec<Vec<usize>>,
    multiplier_dependence: Vec<Vec<ActiveRowDependence>>,
    pub(crate) original_active_count: usize,
}

// Visible at `pub` (with private fields) so it can appear in the signature of
// the `pub` cross-crate `rank_reduce_rows_pivoted_qr_with_dependence`, which the
// gam-custom-family rank-reduction tests call directly after the #1521 crate
// carve. The fields stay private: external callers ignore the dependence return.
#[derive(Clone, Copy, Debug)]
pub struct ActiveRowDependence {
    active_pos: usize,
    coeff: f64,
}

impl CompressedActiveWorkingSet {
    fn is_degenerate_face(&self) -> bool {
        self.constraints.a.nrows() < self.original_active_count
            || self.groups.iter().any(|group| group.len() > 1)
    }

    fn reconstructed_active_multipliers(&self, lambda_system: &Array1<f64>) -> Vec<(usize, f64)> {
        let mut multipliers = Vec::new();
        for (group_pos, &lambda_system_value) in lambda_system.iter().enumerate() {
            let lambda_true = -lambda_system_value;
            if let Some(dependencies) = self.multiplier_dependence.get(group_pos) {
                for dependency in dependencies {
                    if dependency.coeff.abs() > f64::EPSILON {
                        multipliers.push((dependency.active_pos, lambda_true / dependency.coeff));
                    }
                }
            }
        }
        multipliers
    }
}

pub(crate) fn compress_active_working_set(
    x: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
    active: &[usize],
) -> Result<CompressedActiveWorkingSet, EstimationError> {
    let p = constraints.a.ncols();
    if x.len() != p {
        crate::bail_invalid_estim!("active working-set compression dimension mismatch");
    }

    let mut a_out = Array2::<f64>::zeros((active.len(), p));
    let mut b_out = Array1::<f64>::zeros(active.len());
    let mut groups_out: Vec<Vec<usize>> = Vec::with_capacity(active.len());
    for (pos, &idx) in active.iter().enumerate() {
        if idx >= constraints.a.nrows() {
            crate::bail_invalid_estim!(
                "active working-set index {} out of bounds for {} constraints",
                idx,
                constraints.a.nrows()
            );
        }
        a_out.row_mut(pos).assign(&constraints.a.row(idx));
        b_out[pos] = constraints.b[idx];
        groups_out.push(vec![pos]);
    }

    let (a_out, b_out, groups_out, multiplier_dependence) =
        rank_reduce_rows_pivoted_qr_with_dependence(a_out, b_out, groups_out);

    Ok(CompressedActiveWorkingSet {
        constraints: LinearInequalityConstraints::new(a_out, b_out)
            .expect("compressed active constraint shape invariant"),
        groups: groups_out,
        multiplier_dependence,
        original_active_count: active.len(),
    })
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

    let mut at_faer = faer::Mat::<f64>::zeros(p, k);
    for i in 0..k {
        for j in 0..p {
            at_faer[(j, i)] = a[[i, j]];
        }
    }

    let qr = at_faer.as_ref().col_piv_qr();
    let r_mat = qr.thin_R();
    let diag_len = r_mat.nrows().min(r_mat.ncols());
    let leading_diag = if diag_len > 0 {
        r_mat[(0, 0)].abs()
    } else {
        0.0
    };

    const RANK_ALPHA: f64 = 100.0;
    let tol = RANK_ALPHA * f64::EPSILON * (k.max(p).max(1) as f64) * leading_diag.max(1.0);

    let rank = (0..diag_len).filter(|&i| r_mat[(i, i)].abs() > tol).count();
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

    let (perm_fwd, _) = qr.P().arrays();
    let kept_orig: Vec<usize> = (0..rank).map(|j| perm_fwd[j].unbound()).collect();
    let dropped_orig: Vec<usize> = (rank..k).map(|j| perm_fwd[j].unbound()).collect();

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

    for &dropped_idx in &dropped_orig {
        let dropped_row = a.row(dropped_idx);
        let dropped_norm = dropped_row.dot(&dropped_row).sqrt();
        let mut best_positive_align = -1.0_f64;
        let mut best_positive_target: Option<(usize, f64)> = None;
        let mut best_abs_align = -1.0_f64;
        let mut best_signed_target = (kept_orig[0], 1.0_f64);
        for &kept_idx in &kept_orig {
            let kept_row = a.row(kept_idx);
            let kept_norm = kept_row.dot(&kept_row).sqrt();
            let dot = kept_row.dot(&dropped_row);
            let align = if kept_norm > 0.0 && dropped_norm > 0.0 {
                dot / (kept_norm * dropped_norm)
            } else {
                dot
            };
            let coeff = if kept_norm > 0.0 {
                dot / (kept_norm * kept_norm)
            } else {
                0.0
            };
            if align.abs() > best_abs_align {
                best_abs_align = align.abs();
                best_signed_target = (kept_idx, coeff);
            }
            if coeff > 0.0 && align > best_positive_align {
                best_positive_align = align;
                best_positive_target = Some((kept_idx, coeff));
            }
        }
        let (best_target, coeff) = best_positive_target.unwrap_or(best_signed_target);
        let &out_idx = orig_to_out
            .get(&best_target)
            .expect("merge target must be a kept row");
        for &active_pos in &groups[dropped_idx] {
            multiplier_dependence[out_idx].push(ActiveRowDependence { active_pos, coeff });
        }
        if coeff > 0.0 {
            groups_out[out_idx].extend_from_slice(&groups[dropped_idx]);
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

pub(crate) fn working_set_kkt_diagnostics_from_multipliers(
    x: &Array1<f64>,
    gradient: &Array1<f64>,
    working_constraints: &LinearInequalityConstraints,
    lambda_active_true: &Array1<f64>,
    n_total_constraints: usize,
) -> Result<ConstraintKktDiagnostics, EstimationError> {
    let p = working_constraints.a.ncols();
    if x.len() != p || gradient.len() != p {
        crate::bail_invalid_estim!("working-set KKT diagnostic dimension mismatch");
    }
    if lambda_active_true.len() != working_constraints.a.nrows() {
        crate::bail_invalid_estim!(
            "working-set KKT multiplier length mismatch: got {}, expected {}",
            lambda_active_true.len(),
            working_constraints.a.nrows()
        );
    }
    // Primal feasibility and complementarity are measured in the per-row-scaled
    // (geometric) coordinate system the public solver contract is expressed in
    // (see [`ACTIVE_SET_PRIMAL_FEASIBILITY_TOL`] and
    // [`compute_constraint_kkt_diagnostics`]). Without scaling, a row with
    // ‖a_i‖ ≫ 1 — e.g. a B-spline endpoint-derivative clamp — reports a raw
    // slack inflated by ‖a_i‖, and the in-solver acceptance gate
    // (`worst <= ACTIVE_SET_PRIMAL_FEASIBILITY_TOL`) becomes anisotropic across
    // rows. Complementarity is the SCALED product `λ̂_i · ŝ_i` with
    // `λ̂_i = ‖a_i‖·λ_i`; that's invariant under the same per-row rescaling, so
    // its semantics are unchanged while the units match the primal column.
    let m = working_constraints.a.nrows();
    let mut slack = Array1::<f64>::zeros(m);
    let mut primal_feasibility: f64 = 0.0;
    for i in 0..m {
        let s_i = scaled_constraint_slack(x, working_constraints, i);
        slack[i] = s_i;
        primal_feasibility = primal_feasibility.max((-s_i).max(0.0));
    }

    let lambda = lambda_active_true.to_owned();

    let mut dual_feasibility: f64 = 0.0;
    let mut complementarity: f64 = 0.0;
    for i in 0..m {
        dual_feasibility = dual_feasibility.max((-lambda[i]).max(0.0));
        // Scale-invariant complementarity `λ̂_i · ŝ_i` with `λ̂_i = ‖a_i‖·λ_i`
        // and `ŝ_i` the already-scaled slack: this product equals the raw
        // `λ_i · (a_iᵀx − b_i)`, invariant under per-row rescaling — matching the
        // documented contract above (`λ̂_i = ‖a_i‖·λ_i`). `lambda_active_true`
        // here is the RAW multiplier, so without the `‖a_i‖` factor this would
        // understate complementarity by `1/‖a_i‖` on high-norm rows (e.g. a
        // B-spline endpoint-derivative clamp, ‖a‖ ≈ 38).
        let norm_i = working_constraints
            .a
            .row(i)
            .dot(&working_constraints.a.row(i))
            .sqrt();
        complementarity = complementarity.max((norm_i * lambda[i] * slack[i]).abs());
    }
    let stationarity = {
        let mut resid = gradient.to_owned();
        resid -= &working_constraints.a.t().dot(&lambda);
        resid.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()))
    };

    Ok(ConstraintKktDiagnostics {
        n_constraints: n_total_constraints,
        n_active: m,
        primal_feasibility,
        dual_feasibility,
        complementarity,
        stationarity,
        active_tolerance: ACTIVE_SET_PRIMAL_FEASIBILITY_TOL,
        // `working_constraints` is the already-rank-reduced compressed
        // working set, so by construction `rank(working_constraints.a) ==
        // n_active`. Whether the *original* (uncompressed) active set was
        // rank-deficient is the caller's responsibility to track when it
        // needs to surface that to a downstream gate; here we report the
        // post-compression view honestly.
        working_set_rank_deficient: false,
        gradient_scale: gradient_inf_norm(gradient),
    })
}

fn canonicalize_active_constraint_ids(
    x: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
    active: &[usize],
) -> Result<Vec<usize>, EstimationError> {
    if active.is_empty() {
        return Ok(Vec::new());
    }
    let compressed_working = compress_active_working_set(x, constraints, active)?;
    let mut canonical = Vec::with_capacity(compressed_working.groups.len());
    for group in &compressed_working.groups {
        if let Some(&active_pos) = group.first() {
            canonical.push(active[active_pos]);
        }
    }
    Ok(canonical)
}

fn fallback_projected_gradient_direction(
    x: &Array1<f64>,
    d_total: &Array1<f64>,
    gradient: &Array1<f64>,
    working_constraints: &LinearInequalityConstraints,
    constraints: &LinearInequalityConstraints,
) -> Result<Option<(Array1<f64>, Vec<usize>)>, EstimationError> {
    let p = gradient.len();
    if x.len() != p || d_total.len() != p || constraints.a.ncols() != p {
        crate::bail_invalid_estim!("projected-gradient fallback dimension mismatch");
    }

    let tangent_direction = if working_constraints.a.nrows() == 0 {
        -gradient
    } else {
        let identity = Array2::<f64>::eye(p);
        let residual = &working_constraints.b - &working_constraints.a.dot(x);
        let (direction, _) =
            solve_kkt_direction(&identity, gradient, &working_constraints.a, Some(&residual))?;
        direction
    };

    if !array_is_finite(&tangent_direction) {
        return Ok(None);
    }

    let step_inf = tangent_direction
        .iter()
        .fold(0.0_f64, |acc, &value| acc.max(value.abs()));
    if step_inf <= 1e-12 {
        // The projected-gradient tangent step has collapsed to ~0: the working
        // set holds the gradient stationary, so no descent direction remains in
        // the working tangent space. Returning `d_total` as-is is ONLY correct
        // when the iterate `x = beta_start + d_total` is itself feasible. When
        // `x` is infeasible (an inactive row was violated by an earlier
        // `alpha`-clipped step and never repaired — the KKT solve only closes
        // residuals on ACTIVE rows), returning `d_total` leaks an infeasible
        // iterate that the downstream `check_linear_feasibility` gate rejects
        // as an "infeasible iterate" (#1108: the interval-censored survival
        // surrogate landed here at scaled-violation 5.7e-3..0.12 while the
        // exact projection of the SAME point reaches ~0 violation). Project the
        // stationary iterate onto the feasible cone and return the corresponding
        // direction so the solve returns a feasible point. The returned result
        // is `beta_start + dir`, and `x = beta_start + d_total`, so to return a
        // feasible `p` the direction is `dir = d_total + (p - x)`.
        let (worst, _) = max_linear_constraint_violation(x, constraints);
        if worst > ACTIVE_SET_PRIMAL_FEASIBILITY_TOL {
            let projected = project_point_strictly_into_feasible_cone(x, constraints)
                .or_else(|| {
                    let identity = Array2::<f64>::eye(p);
                    solve_quadratic_with_linear_constraints(&identity, x, x, constraints, None)
                        .ok()
                        .map(|(beta, _active)| beta)
                })
                .filter(|p_candidate| {
                    max_linear_constraint_violation(p_candidate, constraints).0
                        <= ACTIVE_SET_PRIMAL_FEASIBILITY_TOL
                });
            let Some(projected) = projected else {
                // No feasible repair available — let the caller report honestly
                // rather than returning an infeasible direction.
                return Ok(None);
            };
            let repair = &projected - x;
            let active = canonicalize_active_constraint_ids(&projected, constraints, &[])?;
            return Ok(Some((d_total + &repair, active)));
        }
        let active = canonicalize_active_constraint_ids(x, constraints, &[])?;
        return Ok(Some((d_total.clone(), active)));
    }

    let directional_derivative = gradient.dot(&tangent_direction);
    if !directional_derivative.is_finite() || directional_derivative >= 0.0 {
        return Ok(None);
    }

    let mut alpha = 1.0_f64;
    for i in 0..constraints.a.nrows() {
        // Scale the step-fraction test by 1/‖a_i‖ so it matches the geometric
        // activation tolerance used elsewhere (see in-loop alpha computation in
        // `solve_newton_direction_with_linear_constraints_impl`).
        let norm = constraints.a.row(i).dot(&constraints.a.row(i)).sqrt();
        let inv = if norm > 0.0 { 1.0 / norm } else { 0.0 };
        let slack = (constraints.a.row(i).dot(x) - constraints.b[i]) * inv;
        let ai_d = constraints.a.row(i).dot(&tangent_direction) * inv;
        if let Some(candidate) = boundary_hit_step_fraction(slack, ai_d, alpha) {
            alpha = candidate;
        }
    }
    if !alpha.is_finite() || alpha <= 0.0 {
        return Ok(None);
    }

    let fallback_step = tangent_direction * alpha;
    let new_x = x + &fallback_step;
    // Per-row-scaled feasibility, matching ACTIVE_SET_PRIMAL_FEASIBILITY_TOL.
    let (worst, _) = max_linear_constraint_violation(&new_x, constraints);
    if worst > ACTIVE_SET_PRIMAL_FEASIBILITY_TOL {
        return Ok(None);
    }
    let active = (0..constraints.a.nrows())
        .filter(|&i| scaled_constraint_slack(&new_x, constraints, i) <= 1e-10)
        .collect::<Vec<_>>();
    let active = canonicalize_active_constraint_ids(&new_x, constraints, &active)?;
    Ok(Some((d_total + &fallback_step, active)))
}

fn log_active_set_transition(
    event: &str,
    iteration: usize,
    active_len: usize,
    constraint: Option<usize>,
) {
    log::debug!(
        "[active-set/QP] iter={} event={} active={} constraint={}",
        iteration,
        event,
        active_len,
        constraint
            .map(|idx| idx.to_string())
            .unwrap_or_else(|| "NA".to_string()),
    );
}

/// Record the current working set; returns `false` when this exact set was
/// already visited. Both the entering and leaving rules above are
/// lowest-index (Bland), so a repeat cannot be a true simplex cycle — it is
/// tolerance-band oscillation (a row added by the slack band and dropped by a
/// noise-level negative dual, repeatedly). The caller breaks to the post-loop
/// KKT exit gate, which inspects the iterate's full KKT residuals and either
/// accepts via the projected-gradient fallback or returns an explicit error —
/// never a silent acceptance, and no longer a hard seed-killing error on an
/// otherwise numerically-converged iterate (#1025).
fn record_active_working_set(
    visited: &mut HashSet<Vec<usize>>,
    active: &[usize],
    iteration: usize,
) -> bool {
    let mut key = active.to_vec();
    key.sort_unstable();
    if visited.insert(key.clone()) {
        return true;
    }
    log::debug!(
        "[active-set/QP] iter={iteration} repeated working set ({} rows); \
         deferring to the post-loop KKT exit gate",
        key.len()
    );
    false
}

fn solve_newton_direction_with_linear_constraints_impl(
    hessian: &Array2<f64>,
    gradient: &Array1<f64>,
    beta: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
    direction_out: &mut Array1<f64>,
    active_hint: Option<&mut Vec<usize>>,
    max_iterations: usize,
) -> Result<(), EstimationError> {
    let p = gradient.len();
    if direction_out.len() != p {
        *direction_out = Array1::zeros(p);
    }
    let m = constraints.a.nrows();
    if constraints.a.ncols() != p || constraints.b.len() != m || beta.len() != p {
        crate::bail_invalid_estim!(
            "linear constraint shape mismatch: A={}x{}, b={}, p={}",
            constraints.a.nrows(),
            constraints.a.ncols(),
            constraints.b.len(),
            p
        );
    }

    let tol_active = 1e-10;
    let tol_step = 1e-12;
    let tol_dual = 1e-10;
    let mut x = beta.to_owned();
    let mut d_total = Array1::<f64>::zeros(p);
    let mut g_cur = gradient.to_owned();

    let has_active_hint = active_hint
        .as_ref()
        .map(|hint| !hint.is_empty())
        .unwrap_or(false);
    if !has_active_hint && solve_newton_direction_dense(hessian, gradient, direction_out).is_ok() {
        let candidate = beta + &*direction_out;
        let mut feasible = true;
        for i in 0..m {
            // Scaled (geometric) slack — matches `tol_active`'s semantics and
            // the public solver contract on `ACTIVE_SET_PRIMAL_FEASIBILITY_TOL`.
            let slack = scaled_constraint_slack(&candidate, constraints, i);
            if slack < -tol_active {
                feasible = false;
                break;
            }
        }
        if feasible {
            return Ok(());
        }
    }

    let mut active: Vec<usize> = Vec::new();
    let mut is_active = vec![false; m];
    if let Some(hint) = active_hint.as_ref() {
        for &idx in hint.iter() {
            if idx < m && !is_active[idx] {
                active.push(idx);
                is_active[idx] = true;
                log_active_set_transition("warm-add", 0, active.len(), Some(idx));
            }
        }
    }
    for i in 0..m {
        // Scaled (geometric) slack: a row with ‖a_i‖ ≫ 1 (e.g. a B-spline
        // endpoint-derivative clamp at k = 12, ‖a_i‖ ≈ 38) would otherwise
        // require a raw slack of `tol_active·‖a_i‖` ≈ 4e-9 to activate, which
        // is below LBLT-solve precision on the KKT system and starves the
        // active set of rows that genuinely belong on the boundary.
        let slack = scaled_constraint_slack(&x, constraints, i);
        if slack <= tol_active && !is_active[i] {
            active.push(i);
            is_active[i] = true;
            log_active_set_transition("initial-boundary-add", 0, active.len(), Some(i));
        }
    }
    let mut visited_working_sets: HashSet<Vec<usize>> = HashSet::new();
    record_active_working_set(&mut visited_working_sets, &active, 0);

    for iteration in 0..max_iterations {
        let compressed_working = compress_active_working_set(&x, constraints, &active)?;
        let mut residualw = Array1::<f64>::zeros(compressed_working.constraints.a.nrows());
        for r in 0..compressed_working.constraints.a.nrows() {
            residualw[r] = compressed_working.constraints.b[r]
                - compressed_working.constraints.a.row(r).dot(&x);
        }
        let (d, lambdaw) = solve_kkt_direction(
            hessian,
            &g_cur,
            &compressed_working.constraints.a,
            Some(&residualw),
        )?;
        let step_norm = d.iter().map(|v| v * v).sum::<f64>().sqrt();
        if step_norm <= tol_step {
            // A "stationary" iterate is only a genuine KKT point if it is
            // also primal-feasible on the FULL constraint set. The
            // "blocking-add" loop further down only catches rows that
            // become tight DURING a non-degenerate step, so a tiny step
            // entered with a residual violation on an inactive row would
            // otherwise leak an infeasible direction through this
            // short-circuit unchecked. Grow the active set with the
            // most-violated inactive row and re-solve; `residualw` for that
            // row will be non-zero on the next KKT solve, so the resulting
            // step is non-degenerate.
            let (worst, worst_row) = max_linear_constraint_violation(&x, constraints);
            if worst > ACTIVE_SET_PRIMAL_FEASIBILITY_TOL && !is_active[worst_row] {
                active.push(worst_row);
                is_active[worst_row] = true;
                log_active_set_transition(
                    "stationary-infeasible-add",
                    iteration,
                    active.len(),
                    Some(worst_row),
                );
                if !record_active_working_set(&mut visited_working_sets, &active, iteration) {
                    break;
                }
                continue;
            }
            if worst > ACTIVE_SET_PRIMAL_FEASIBILITY_TOL {
                // The worst-violating row is already in the active set —
                // the KKT step failed to restore its feasibility, typically
                // because a previous iteration's `alpha`-clip prevented the
                // full residual closure. Fall through to the post-loop exit
                // gate, which inspects the iterate's full KKT residuals and
                // either tries the projected-gradient fallback or returns
                // an explicit error. Both are correct outcomes; silently
                // returning the infeasible direction is not.
                break;
            }
            if compressed_working.groups.is_empty() {
                direction_out.assign(&d_total);
                return Ok(());
            }
            let remove_pos = compressed_working
                .reconstructed_active_multipliers(&lambdaw)
                .into_iter()
                .filter(|&(_, lambda_true)| lambda_true < -tol_dual)
                .min_by_key(|(active_pos, _)| (active[*active_pos], *active_pos))
                .map(|(active_pos, _)| active_pos);
            if let Some(active_pos) = remove_pos {
                let idx = active.remove(active_pos);
                is_active[idx] = false;
                log_active_set_transition(
                    "remove-negative-dual",
                    iteration,
                    active.len(),
                    Some(idx),
                );
                if !record_active_working_set(&mut visited_working_sets, &active, iteration) {
                    break;
                }
                continue;
            }
            if let Some(hint) = active_hint {
                hint.clear();
                hint.extend(canonicalize_active_constraint_ids(
                    &x,
                    constraints,
                    &active,
                )?);
            }
            direction_out.assign(&d_total);
            return Ok(());
        }

        let mut alpha = 1.0_f64;
        for i in 0..m {
            if is_active[i] {
                continue;
            }
            // boundary_hit_step_fraction is scale-equivariant in `(slack, ai_d)`:
            // scaling both by 1/‖a_i‖ leaves `step = slack / -ai_d` unchanged,
            // and its directional-tol/finite checks operate on a max-magnitude
            // scale of the same triple. Doing the geometric rescale here keeps
            // the step-fraction's "moving toward the boundary" test in the same
            // scaled coordinates the activation tests use.
            let norm = constraints.a.row(i).dot(&constraints.a.row(i)).sqrt();
            let inv = if norm > 0.0 { 1.0 / norm } else { 0.0 };
            let slack = (constraints.a.row(i).dot(&x) - constraints.b[i]) * inv;
            let ai_d = constraints.a.row(i).dot(&d) * inv;
            if let Some(cand) = boundary_hit_step_fraction(slack, ai_d, alpha) {
                alpha = cand;
            }
        }

        ndarray::Zip::from(&mut x)
            .and(&mut d_total)
            .and(&d)
            .for_each(|x_i, dt_i, &d_i| {
                let alpha_d = alpha * d_i;
                *x_i += alpha_d;
                *dt_i += alpha_d;
            });
        g_cur = gradient + &hessian.dot(&d_total);

        let mut added_new_active = false;
        let mut working_set_repeated = false;
        for i in 0..m {
            if is_active[i] {
                continue;
            }
            let slack = scaled_constraint_slack(&x, constraints, i);
            if slack <= tol_active {
                active.push(i);
                is_active[i] = true;
                added_new_active = true;
                log_active_set_transition("blocking-add", iteration, active.len(), Some(i));
                working_set_repeated =
                    !record_active_working_set(&mut visited_working_sets, &active, iteration);
                break;
            }
        }
        if working_set_repeated {
            break;
        }

        if active.is_empty() && !added_new_active {
            if let Some(hint) = active_hint {
                hint.clear();
            }
            direction_out.assign(&d_total);
            return Ok(());
        }
    }

    let compressed_working = compress_active_working_set(&x, constraints, &active)?;
    let mut residualw = Array1::<f64>::zeros(compressed_working.constraints.a.nrows());
    for r in 0..compressed_working.constraints.a.nrows() {
        residualw[r] =
            compressed_working.constraints.b[r] - compressed_working.constraints.a.row(r).dot(&x);
    }
    let (_, lambdaw) = solve_kkt_direction(
        hessian,
        &g_cur,
        &compressed_working.constraints.a,
        Some(&residualw),
    )?;
    let lambda_true = lambdaw.mapv(|lam_sys| -lam_sys);
    let (worst, row) = max_linear_constraint_violation(&x, constraints);
    let working_kkt = working_set_kkt_diagnostics_from_multipliers(
        &x,
        &g_cur,
        &compressed_working.constraints,
        &lambda_true,
        m,
    )?;
    let kkt = compute_constraint_kkt_diagnostics(&x, &g_cur, constraints);
    let grad_inf = gradient_inf_norm(&g_cur);
    let stationarity_rel = working_kkt.stationarity / grad_inf.max(1.0);
    let step_inf = d_total.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let hd_total = hessian.dot(&d_total);
    let predicted_delta = gradient.dot(&d_total)
        + 0.5
            * d_total
                .iter()
                .zip(hd_total.iter())
                .map(|(a, b)| a * b)
                .sum::<f64>();
    let kkt_strong_ok = (working_kkt.stationarity <= ACTIVE_SET_KKT_STATIONARITY_TOL
        || stationarity_rel <= ACTIVE_SET_KKT_STATIONARITY_TOL)
        && working_kkt.complementarity <= ACTIVE_SET_KKT_COMPLEMENTARITY_TOL;
    let model_descent_ok =
        predicted_delta <= -ACTIVE_SET_MODEL_DESCENT_REL_TOL * (1.0 + grad_inf * step_inf);
    let degenerate_boundary_ok = compressed_working.is_degenerate_face()
        && worst <= ACTIVE_SET_PRIMAL_FEASIBILITY_TOL
        && working_kkt.primal_feasibility <= ACTIVE_SET_PRIMAL_FEASIBILITY_TOL
        && working_kkt.complementarity <= ACTIVE_SET_KKT_COMPLEMENTARITY_TOL
        && (working_kkt.stationarity <= ACTIVE_SET_KKT_DEGENERATE_STATIONARITY_TOL
            || stationarity_rel <= ACTIVE_SET_KKT_STATIONARITY_TOL);
    if worst <= ACTIVE_SET_PRIMAL_FEASIBILITY_TOL
        && ((working_kkt.dual_feasibility <= ACTIVE_SET_KKT_DUAL_FEASIBILITY_TOL
            && (kkt_strong_ok || model_descent_ok))
            || degenerate_boundary_ok)
    {
        if let Some(hint) = active_hint {
            hint.clear();
            hint.extend(canonicalize_active_constraint_ids(
                &x,
                constraints,
                &active,
            )?);
        }
        direction_out.assign(&d_total);
        return Ok(());
    }
    if let Some((fallback_direction, fallback_active)) = fallback_projected_gradient_direction(
        &x,
        &d_total,
        &g_cur,
        &compressed_working.constraints,
        constraints,
    )? {
        if let Some(hint) = active_hint {
            hint.clear();
            hint.extend(fallback_active);
        }
        direction_out.assign(&fallback_direction);
        return Ok(());
    }
    Err(EstimationError::ParameterConstraintViolation(format!(
        "linear-constrained Newton active-set failed to converge; max(Aβ-b violation)={worst:.3e} at row {row}; KKT[primal={:.3e}, dual={:.3e}, comp={:.3e}, stat={:.3e}, active={}/{}]; diagnostic-reconstruction[dual={:.3e}, stat={:.3e}]",
        working_kkt.primal_feasibility,
        working_kkt.dual_feasibility,
        working_kkt.complementarity,
        working_kkt.stationarity,
        working_kkt.n_active,
        working_kkt.n_constraints,
        kkt.dual_feasibility,
        kkt.stationarity
    )))
}

pub(crate) fn solve_newton_direction_with_linear_constraints(
    hessian: &Array2<f64>,
    gradient: &Array1<f64>,
    beta: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
    direction_out: &mut Array1<f64>,
    active_hint: Option<&mut Vec<usize>>,
) -> Result<(), EstimationError> {
    let max_iterations = (gradient.len() + constraints.a.nrows() + 8) * 4;
    solve_newton_direction_with_linear_constraints_impl(
        hessian,
        gradient,
        beta,
        constraints,
        direction_out,
        active_hint,
        max_iterations,
    )
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
    let gradient = hessian.dot(beta_start) - rhs;
    let mut delta = Array1::<f64>::zeros(beta_start.len());
    let mut active_hint = warm_active_set.map_or_else(Vec::new, |active| active.to_vec());
    solve_newton_direction_with_linear_constraints(
        hessian,
        &gradient,
        beta_start,
        constraints,
        &mut delta,
        Some(&mut active_hint),
    )?;
    let candidate = beta_start + &delta;
    // FINAL FEASIBILITY CONTRACT (#1108). The active-set inner step computes its
    // Newton direction only against the ACTIVE working rows and line-searches
    // `alpha` against the inactive rows; a row whose approach rate falls inside
    // `boundary_hit_step_fraction`'s directional tolerance is not clipped, so the
    // returned `candidate` can overshoot a currently-inactive row and land
    // outside the cone by a small-but-gate-failing amount (the interval-censored
    // survival surrogate leaked 5.5e-3..2.2e-2 raw at cycles 3/9 here, accepted
    // into `states`, then rejected by the next cycle's `check_linear_feasibility`
    // — `infeasible iterate`). The solver's PUBLIC CONTRACT is a feasible point;
    // enforce it at the single chokepoint every caller flows through. A genuinely
    // converged feasible solve has `worst ~ 0`, so this is a no-op there and does
    // not perturb a well-conditioned constrained fit. When the step did leak,
    // project the iterate onto the feasible cone (the exact projection the #1108
    // diag proves reaches ~0 violation: the nearest strictly-interior point) and
    // return THAT. If no feasible repair is achievable, surface the active-set
    // error rather than returning an infeasible point.
    let (worst, _) = max_linear_constraint_violation(&candidate, constraints);
    if worst <= ACTIVE_SET_PRIMAL_FEASIBILITY_TOL {
        return Ok((candidate, active_hint));
    }
    let repaired = project_point_strictly_into_feasible_cone(&candidate, constraints).filter(|p| {
        max_linear_constraint_violation(p, constraints).0 <= ACTIVE_SET_PRIMAL_FEASIBILITY_TOL
    });
    match repaired {
        Some(feasible) => {
            let active = canonicalize_active_constraint_ids(&feasible, constraints, &[])?;
            Ok((feasible, active))
        }
        None => Err(EstimationError::ParameterConstraintViolation(format!(
            "constrained quadratic solve returned an infeasible iterate \
             (max scaled violation {worst:.3e}) and no feasible projection could be \
             certified onto the constraint cone",
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ACTIVE_SET_INTERIOR_SEED_MARGIN, LinearInequalityConstraints,
        compute_constraint_kkt_diagnostics, project_point_strictly_into_feasible_cone,
        project_stationarity_residual_on_constraint_cone,
        rank_reduce_rows_pivoted_qr_with_dependence, scaled_constraint_slack,
        solve_newton_direction_with_linear_constraints_impl,
        solve_quadratic_with_linear_constraints,
    };
    use approx::assert_relative_eq;
    use ndarray::{Array1, Array2, array};

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
    fn maxiter_accepts_current_boundary_solution() {
        let hessian = array![[1.0]];
        let gradient = array![-1.0];
        let beta = array![0.0];
        let constraints = LinearInequalityConstraints {
            a: array![[-1.0]],
            b: array![-0.1],
        };
        let mut direction = Array1::zeros(1);
        let mut active_hint = Vec::new();

        solve_newton_direction_with_linear_constraints_impl(
            &hessian,
            &gradient,
            &beta,
            &constraints,
            &mut direction,
            Some(&mut active_hint),
            1,
        )
        .expect("solver should accept the current boundary solution at the iteration limit");

        assert_relative_eq!(direction[0], 0.1, epsilon = 1e-12);
        assert_eq!(active_hint, vec![0]);
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
}
