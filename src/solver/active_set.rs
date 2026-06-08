use crate::estimate::EstimationError;
use crate::faer_ndarray::{FaerArrayView, FaerLinalgError, FaerSvd, array1_to_col_matmut};
use crate::linalg::utils::{StableSolver, array_is_finite, boundary_hit_step_fraction};
use faer::linalg::solvers::{Lblt as FaerLblt, Solve as FaerSolve};
use faer::{Side, Unbind};
use ndarray::{Array1, Array2, s};
use serde::{Deserialize, Serialize};
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

/// Relaxed stationarity tolerance accepted only on a *degenerate boundary
/// face* (linearly-dependent active rows), where the exact projected gradient
/// cannot reach `ACTIVE_SET_KKT_STATIONARITY_TOL`. Still requires primal
/// feasibility, complementarity, and a relative-stationarity backstop.
const ACTIVE_SET_KKT_DEGENERATE_STATIONARITY_TOL: f64 = 1e-3;

/// Relative scale on the predicted-decrease test `predicted_delta ≤
/// −ε·(1 + ‖∇L‖∞·‖d‖∞)`: when the working-set Newton step still buys a
/// quadratic-model decrease at this relative margin the step is a usable
/// descent direction even if the KKT residual has not yet tightened.
const ACTIVE_SET_MODEL_DESCENT_REL_TOL: f64 = 1e-10;

#[derive(Clone, Debug)]
pub struct LinearInequalityConstraints {
    pub a: Array2<f64>,
    pub b: Array1<f64>,
}

impl LinearInequalityConstraints {
    /// Construct with the equal-row-count invariant enforced. The dimensions
    /// `a.nrows() == b.len()` are required by every downstream KKT / active-set
    /// routine; routing every construction site through this constructor
    /// eliminates a class of "rows out of sync" bugs at the type boundary.
    #[inline]
    pub fn new(a: Array2<f64>, b: Array1<f64>) -> Result<Self, String> {
        if a.nrows() != b.len() {
            return Err(format!(
                "LinearInequalityConstraints: row count mismatch (A has {} rows, b has length {})",
                a.nrows(),
                b.len(),
            ));
        }
        Ok(Self { a, b })
    }

    /// Internal helper for sites that have *just* produced `(a, b)` with the
    /// invariant guaranteed (e.g. via a row-by-row push loop). Skips the
    /// runtime length check; callers that aren't in that position must use
    /// [`Self::new`] instead.
    #[inline]
    pub(crate) fn from_paired(a: Array2<f64>, b: Array1<f64>) -> Self {
        assert_eq!(a.nrows(), b.len(), "paired constraint shape invariant");
        Self { a, b }
    }

    /// Build the per-coordinate `β_i ≥ lower_bounds[i]` inequality system.
    /// Non-finite entries are treated as "no bound" and skipped; returns
    /// `None` when every entry is non-finite so callers can short-circuit
    /// the no-constraint case without allocating the empty A/b pair.
    pub fn from_per_coordinate_lower_bounds(lower_bounds: &Array1<f64>) -> Option<Self> {
        let active_rows: Vec<usize> = (0..lower_bounds.len())
            .filter(|&i| lower_bounds[i].is_finite())
            .collect();
        if active_rows.is_empty() {
            return None;
        }
        let p = lower_bounds.len();
        let mut a = Array2::<f64>::zeros((active_rows.len(), p));
        let mut b = Array1::<f64>::zeros(active_rows.len());
        for (r, &idx) in active_rows.iter().enumerate() {
            a[[r, idx]] = 1.0;
            b[r] = lower_bounds[idx];
        }
        Some(Self::from_paired(a, b))
    }
}

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
    }
}

pub(crate) fn project_stationarity_residual_on_constraint_cone(
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

/// Euclidean projection of `point` onto the feasible polyhedron
/// `{β : A·β ≥ b}`, i.e. the solution of `min_β ½‖β − point‖² s.t. A·β ≥ b`.
///
/// This is the principled feasible cold-start seed for a shape-constrained
/// (convex/concave/monotone) smooth: it carries the data-driven curvature of
/// the unconstrained seed `point` and merely nudges it the minimum distance
/// required to enter the constraint cone. It is qualitatively different from
/// [`feasible_point_for_linear_constraints`], which returns the *minimum-norm*
/// feasible point — for a homogeneous cone (`b = 0`, as the second-difference
/// convexity/concavity constraints are) that minimum-norm point is the cone
/// **vertex** `β = 0` (a flat line), where every constraint row is tight. P-IRLS
/// launched from that vertex stalls on a non-stationary face of the cone and
/// the fit's success then depends on whether a warm-start seed happens to land
/// it in the right basin (#873). Projecting the unconstrained seed instead
/// lands a strictly-interior-or-boundary point already near the constrained
/// optimum, so the fit converges identically from a cold or warm cache.
///
/// The projection is the identity-Hessian instance of
/// [`solve_quadratic_with_linear_constraints`]: with `H = I` and `rhs = point`
/// the quadratic `½βᵀβ − pointᵀβ = ½‖β − point‖² − ½‖point‖²` is minimized by
/// the projection. Returns `None` if the constraints are malformed or the
/// active-set QP fails to certify a feasible solution, so callers can fall back.
pub(crate) fn project_point_onto_feasible_cone(
    point: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
) -> Option<Array1<f64>> {
    let p = point.len();
    if constraints.a.ncols() != p
        || constraints.a.nrows() == 0
        || constraints.b.len() != constraints.a.nrows()
    {
        return None;
    }
    let identity = Array2::<f64>::eye(p);
    let (beta, _active) =
        solve_quadratic_with_linear_constraints(&identity, point, point, constraints, None).ok()?;
    if beta.len() != p || beta.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let (worst, _row) = max_linear_constraint_violation(&beta, constraints);
    if worst <= ACTIVE_SET_PRIMAL_FEASIBILITY_TOL {
        Some(beta)
    } else {
        None
    }
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

#[derive(Clone, Copy, Debug)]
pub(crate) struct ActiveRowDependence {
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
        constraints: LinearInequalityConstraints::from_paired(a_out, b_out),
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

pub(crate) fn rank_reduce_rows_pivoted_qr_with_dependence(
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
        complementarity = complementarity.max((lambda[i] * slack[i]).abs());
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

fn record_active_working_set(
    visited: &mut HashSet<Vec<usize>>,
    active: &[usize],
    iteration: usize,
) -> Result<(), EstimationError> {
    let mut key = active.to_vec();
    key.sort_unstable();
    if visited.insert(key.clone()) {
        return Ok(());
    }
    Err(EstimationError::ParameterConstraintViolation(format!(
        "linear-constrained Newton active-set cycled at iteration {iteration}; repeated active set {key:?}"
    )))
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
    record_active_working_set(&mut visited_working_sets, &active, 0)?;

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
                record_active_working_set(&mut visited_working_sets, &active, iteration)?;
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
                record_active_working_set(&mut visited_working_sets, &active, iteration)?;
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
                record_active_working_set(&mut visited_working_sets, &active, iteration)?;
                break;
            }
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
    let grad_inf = g_cur.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
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

pub(crate) fn solve_quadratic_with_linear_constraints(
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
    Ok((beta_start + &delta, active_hint))
}

#[cfg(test)]
mod tests {
    use super::{
        LinearInequalityConstraints, compute_constraint_kkt_diagnostics,
        project_stationarity_residual_on_constraint_cone,
        rank_reduce_rows_pivoted_qr_with_dependence,
        solve_newton_direction_with_linear_constraints_impl,
        solve_quadratic_with_linear_constraints,
    };
    use approx::assert_relative_eq;
    use ndarray::{Array1, array};

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
