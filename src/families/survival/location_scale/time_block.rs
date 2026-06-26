use super::*;
use crate::solver::gauge::Gauge;

pub(crate) fn structural_time_coefficient_constraints(
    design_derivative_exit: &DesignMatrix,
    derivative_offset_exit: &Array1<f64>,
    derivative_guard: f64,
) -> Result<Option<LinearInequalityConstraints>, String> {
    time_derivative_guard_constraints(
        design_derivative_exit,
        derivative_offset_exit,
        derivative_guard,
    )
}

pub(crate) fn time_derivative_guard_constraints(
    design_derivative_exit: &DesignMatrix,
    derivative_offset_exit: &Array1<f64>,
    derivative_guard: f64,
) -> Result<Option<LinearInequalityConstraints>, String> {
    build_time_derivative_guard_constraints(
        design_derivative_exit,
        derivative_offset_exit,
        derivative_guard,
        LOCATION_SCALE_GUARD_POLICY,
    )
    .map_err(map_guard_constraint_failure)
}

/// Render a shared guard-constraint failure into the location-scale error
/// vocabulary, preserving the family's historical wording.
pub(crate) fn map_guard_constraint_failure(failure: GuardConstraintFailure) -> String {
    match failure {
        GuardConstraintFailure::RowOffsetMismatch { rows, offsets } => {
            SurvivalLocationScaleError::InvalidConfiguration {
                reason: format!(
                    "time derivative guard constraints require matching rows/offsets: rows={rows}, offsets={offsets}"
                ),
            }
            .into()
        }
        GuardConstraintFailure::GuardOutOfRange { guard, range } => {
            SurvivalLocationScaleError::ConstraintViolation {
                reason: format!(
                    "time derivative guard must be finite and {range}, got {guard}"
                ),
            }
            .into()
        }
        GuardConstraintFailure::NonFiniteOffset { row, offset } => {
            SurvivalLocationScaleError::ConstraintViolation {
                reason: format!(
                    "time derivative guard constraints require finite derivative offsets; found offset[{row}]={offset}"
                ),
            }
            .into()
        }
        GuardConstraintFailure::NonFiniteDesign { row, col } => {
            SurvivalLocationScaleError::ConstraintViolation {
                reason: format!(
                    "time derivative guard constraints require finite derivative design entries; found row {row}, column {col}"
                ),
            }
            .into()
        }
        GuardConstraintFailure::InfeasibleRow {
            row,
            offset,
            guard,
            no_time_coefficients,
        } => {
            let detail = if no_time_coefficients {
                "with no time coefficients"
            } else {
                "zero derivative design row"
            };
            let reason = if no_time_coefficients {
                format!(
                    "time derivative guard is infeasible at row {row}: offset={offset:.3e} < guard={guard:.3e} {detail}"
                )
            } else {
                format!(
                    "time derivative guard is infeasible at row {row}: {detail} with offset={offset:.3e} < guard={guard:.3e}"
                )
            };
            SurvivalLocationScaleError::ConstraintViolation { reason }.into()
        }
    }
}

pub(crate) fn structural_time_initial_beta_guess(
    design_derivative_exit: &Array2<f64>,
    derivative_offset_exit: &Array1<f64>,
    age_exit: &Array1<f64>,
    derivative_guard: f64,
    coefficient_lower_bounds: Option<&Array1<f64>>,
) -> Option<Array1<f64>> {
    let n = design_derivative_exit.nrows();
    let p = design_derivative_exit.ncols();
    if p == 0 || n == 0 || derivative_offset_exit.len() != n || age_exit.len() != n {
        return None;
    }

    let mut target = Array1::<f64>::zeros(n);
    for i in 0..n {
        let desired = 1.0 / age_exit[i].max(STRUCTURAL_GUESS_AGE_FLOOR);
        target[i] = (desired - derivative_offset_exit[i]).max(0.0);
    }

    let xtx = crate::faer_ndarray::fast_ata(design_derivative_exit);
    let xty = fast_atv(design_derivative_exit, &target);
    let eps =
        STRUCTURAL_GUESS_RIDGE_REL * (0..p).map(|i| xtx[[i, i]]).fold(0.0_f64, f64::max).max(1.0);
    let mut lhs = xtx;
    for i in 0..p {
        lhs[[i, i]] += eps;
    }

    use crate::faer_ndarray::FaerCholesky;
    let chol = lhs.cholesky(faer::Side::Lower).ok()?;
    let mut beta_init = chol.solvevec(&xty);
    if let Some(lower_bounds) = coefficient_lower_bounds
        && let Some(constraints) = lower_bound_constraints(lower_bounds)
    {
        // `beta_init` is the length-`p` ridge solution and `constraints` is
        // derived from the same `p`-column derivative design, so the projection
        // is dimensionally consistent by construction. If a future refactor
        // breaks that invariant, abandon the structural guess rather than
        // propagate a hard error out of this best-effort warm start.
        beta_init = project_onto_linear_constraints(p, &constraints, Some(&beta_init)).ok()?;
    }

    let d_raw_init = fast_av(design_derivative_exit, &beta_init) + derivative_offset_exit;
    if d_raw_init
        .iter()
        .all(|v| v.is_finite() && *v >= derivative_guard)
    {
        Some(beta_init)
    } else {
        None
    }
}

#[derive(Clone, Debug)]
pub(crate) struct TimeIdentifiabilityTransform {
    /// Maps the inner solver's reduced (active) time coefficients back to the
    /// raw I-spline layout through the canonical Gauge-owned affine section:
    /// `β_time_raw = T · θ + a`.
    pub(crate) gauge: Gauge,
}

#[derive(Clone, Debug)]
pub(crate) struct TimeBlockPrepared {
    pub(crate) design_entry: Array2<f64>,
    pub(crate) design_exit: Array2<f64>,
    pub(crate) design_derivative_exit: Array2<f64>,
    pub(crate) coefficient_lower_bounds: Option<Array1<f64>>,
    pub(crate) linear_constraints: Option<LinearInequalityConstraints>,
    pub(crate) penalties: Vec<Array2<f64>>,
    /// Structural null-space dimension of each (possibly reduced) penalty,
    /// aligned with `penalties`. Carries the reduced count when the block has
    /// been collapsed to its identifiable parametric form so the REML log-det
    /// accounting matches the actual `zᵀ S z` rank rather than the raw basis.
    pub(crate) nullspace_dims: Vec<usize>,
    pub(crate) initial_beta: Option<Array1<f64>>,
    pub(crate) transform: TimeIdentifiabilityTransform,
    /// Augmented geometry offsets the caller must use in place of
    /// `spec.time_block.offset_*`. They equal the input offsets passed through
    /// unchanged on the non-reduce / identity / non-clean-split paths, and the
    /// input offsets plus the folded unit-log-t value/derivative contributions
    /// when the warp slope is pinned (issue #892).
    pub(crate) offset_entry: Array1<f64>,
    pub(crate) offset_exit: Array1<f64>,
    pub(crate) derivative_offset_exit: Array1<f64>,
    /// True iff the unit-log-t warp-slope pin fired (issue #892), so the single
    /// surviving free time column `z_c` is ROW-CONSTANT (it carries the location
    /// level). The threshold block must then DROP its own intercept to avoid a
    /// two-constant alias in the joint Hessian. False on every other path,
    /// including the both-columns-free fallback reduce — there the reduced
    /// I-spline columns are strictly monotone (not row-constant), so the
    /// threshold keeps its intercept.
    pub(crate) pinned_free_row_constant: bool,
    /// True iff the rank-1 reduced parametric-AFT regime fired (issue #892), in
    /// which the time warp is removed entirely (zero free columns, `h ≡ 0`) and
    /// the `log t` baseline is instead applied as a per-row LOCATION offset that
    /// rides the existing σ-scaled `q` channel: `u = inv_sigma·(log t − η_t)`.
    /// The caller threads the exact `−log t` (value at entry/exit) and `−1/t`
    /// (time-derivative at exit) into the family so the standardized residual
    /// carries the canonical survreg/lifelines AFT gauge — the event Jacobian
    /// then contributes `log_g = −η_ls − log t = −log σ − log t`, the `−log σ`
    /// term that identifies σ. On every other path this is `false`.
    pub(crate) location_log_time_offset: bool,
}

pub(crate) fn lower_bound_constraints(
    lower_bounds: &Array1<f64>,
) -> Option<LinearInequalityConstraints> {
    LinearInequalityConstraints::from_per_coordinate_lower_bounds(lower_bounds)
}

pub(crate) fn append_linear_constraints(
    first: Option<LinearInequalityConstraints>,
    second: Option<LinearInequalityConstraints>,
) -> Result<Option<LinearInequalityConstraints>, String> {
    match (first, second) {
        (None, None) => Ok(None),
        (Some(constraints), None) | (None, Some(constraints)) => Ok(Some(constraints)),
        (Some(lhs), Some(rhs)) => {
            if lhs.a.ncols() != rhs.a.ncols() {
                return Err(SurvivalLocationScaleError::DimensionMismatch {
                    reason: format!(
                        "time linear constraint width mismatch: left={}, right={}",
                        lhs.a.ncols(),
                        rhs.a.ncols()
                    ),
                }
                .into());
            }
            let rows = lhs.a.nrows() + rhs.a.nrows();
            let cols = lhs.a.ncols();
            let mut a = Array2::<f64>::zeros((rows, cols));
            let mut b = Array1::<f64>::zeros(rows);
            a.slice_mut(s![..lhs.a.nrows(), ..]).assign(&lhs.a);
            a.slice_mut(s![lhs.a.nrows().., ..]).assign(&rhs.a);
            b.slice_mut(s![..lhs.b.len()]).assign(&lhs.b);
            b.slice_mut(s![lhs.b.len()..]).assign(&rhs.b);
            LinearInequalityConstraints::new(a, b).map(Some)
        }
    }
}

pub(crate) fn structural_time_coefficient_lower_bounds(
    design_derivative_exit: &DesignMatrix,
    derivative_offset_exit: &Array1<f64>,
    lower_bound: f64,
) -> Result<Option<Array1<f64>>, String> {
    if design_derivative_exit.nrows() != derivative_offset_exit.len() {
        return Err(SurvivalLocationScaleError::InvalidConfiguration { reason: format!(
            "structural time coefficient bounds require matching rows/offsets: rows={}, offsets={}",
            design_derivative_exit.nrows(),
            derivative_offset_exit.len()
        ) }.into());
    }
    if design_derivative_exit.ncols() == 0 {
        return Ok(None);
    }
    if !lower_bound.is_finite() || lower_bound <= 0.0 {
        return Err(SurvivalLocationScaleError::ConstraintViolation {
            reason: format!(
                "structural time coefficient lower bound must be finite and > 0, got {lower_bound}"
            ),
        }
        .into());
    }

    const DERIVATIVE_TOL: f64 = 1e-12;
    const FEASIBILITY_TOL: f64 = 1e-12;
    // Diagnostics only: entries with magnitude in this open band are reported as
    // "sub-tolerance nonzeros" to explain a missing structural lower bound. The
    // lower edge separates genuine round-off from a hard zero; the upper edge is
    // the derivative-activity tolerance above.
    const SUBTOL_NONZERO_FLOOR: f64 = 1e-30;
    // How many leading columns' max(|·|) to surface in the diagnostic message
    // when no derivative-active column is found.
    const DIAGNOSTIC_COLUMN_PREVIEW: usize = 8;

    let p = design_derivative_exit.ncols();
    let nrows = design_derivative_exit.nrows();
    let mut lower_bounds = Array1::from_elem(p, f64::NEG_INFINITY);
    let mut has_structural_support = false;
    for (row, &offset) in derivative_offset_exit.iter().enumerate() {
        if !offset.is_finite() {
            return Err(SurvivalLocationScaleError::ConstraintViolation { reason: format!(
                "structural time coefficient bounds require finite derivative offsets; found offset[{row}]={offset}"
            ) }.into());
        }
        if lower_bound - offset > FEASIBILITY_TOL {
            return Err(SurvivalLocationScaleError::ConstraintViolation { reason: format!(
                "structural time coefficient bounds require derivative offsets to encode the derivative guard at row {row}: offset={offset:.3e} < guard={lower_bound:.3e}"
            ) }.into());
        }
    }
    // Stream column-by-column so operator-backed (Lazy) designs never have to
    // materialize as a single nrows×ncols dense buffer. `extract_column` is
    // O(n) for dense, O(nnz_j) for sparse, and O(matvec_n) for lazy operators
    // — the operator-form path the strict policy demands.
    let mut col_maxes: Vec<(usize, f64)> = Vec::with_capacity(p.min(DIAGNOSTIC_COLUMN_PREVIEW));
    let mut total_subtol_nonzeros = 0_usize;
    for col in 0..p {
        let column = design_derivative_exit.extract_column(col);
        if column.len() != nrows {
            return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
                "structural time coefficient bounds: extract_column returned {} entries for column {col}, expected {nrows}",
                column.len()
            ) }.into());
        }
        let mut has_positive_support = false;
        let mut col_max = 0.0_f64;
        for (row, &value) in column.iter().enumerate() {
            if !value.is_finite() {
                return Err(SurvivalLocationScaleError::ConstraintViolation { reason: format!(
                    "structural time coefficient bounds require finite derivative design entries; found row {row}, column {col}"
                ) }.into());
            }
            if value < -DERIVATIVE_TOL {
                return Err(SurvivalLocationScaleError::ConstraintViolation { reason: format!(
                    "structural time coefficient bounds require a non-negative derivative basis at row {row}, column {col}; found {value:.3e}"
                ) }.into());
            }
            if value > DERIVATIVE_TOL {
                has_positive_support = true;
            }
            let abs_value = value.abs();
            if abs_value > col_max {
                col_max = abs_value;
            }
            if abs_value > SUBTOL_NONZERO_FLOOR && abs_value <= DERIVATIVE_TOL {
                total_subtol_nonzeros += 1;
            }
        }
        if has_positive_support {
            lower_bounds[col] = 0.0;
            has_structural_support = true;
        }
        if col < DIAGNOSTIC_COLUMN_PREVIEW {
            col_maxes.push((col, col_max));
        }
    }

    if !has_structural_support {
        // No derivative-active column on this candidate's exit-time design.
        //
        // Two distinct regimes reach this branch and only one of them is
        // surprising:
        //
        // 1. `learn_timewiggle = true` (the large-scale survival
        //    marginal-slope path). `main.rs:3846` deliberately routes to
        //    `SurvivalTimeBasisConfig::None`, which produces an `(n, 0)`
        //    empty time-basis: the parametric baseline plus the timewiggle
        //    block carry the entire time structure, and the exit-time
        //    derivative information lives in `derivative_offset_exit`,
        //    not in any basis column. `prepare_survival_time_stack` then
        //    appends `tw.ncols` **exactly zero** tail columns to keep
        //    shapes aligned with the timewiggle-extended coefficient
        //    vector. Those tail zeros correctly carry no exit-derivative
        //    signal, there is nothing to constrain with a structural
        //    lower-bound ridge, and the right answer is `Ok(None)`.
        //
        // 2. `--time-basis ispline` (or bspline) without timewiggle. Here
        //    the basis was *intended* to span exit-time variation; a
        //    degenerate build that produces only sub-tolerance entries
        //    points at a real numerical bug upstream (knot inference,
        //    cell-moment construction, derivative formula, etc.).
        //
        // The two regimes differentiate by whether the design has any
        // entry whose magnitude exceeds 1e-30 but stays at or below
        // `DERIVATIVE_TOL`. Regime 1 leaves the tail columns at exact
        // zero (no entry passes 1e-30); regime 2 leaves residual
        // float-scale entries from the upstream basis builder. We log
        // warn-level only in the surprising regime.
        if total_subtol_nonzeros > 0 {
            log::warn!(
                "structural time coefficient bounds: no derivative-active column on this candidate's exit-time design ({} rows × {} cols, sub-tolerance nonzero entries ({:.0e} < |v| ≤ {:.0e}): {}, first-{} col max(|.|): {:?}); skipping the structural lower-bound ridge — fit may converge to a non-monotone-in-time hazard",
                nrows,
                p,
                SUBTOL_NONZERO_FLOOR,
                DERIVATIVE_TOL,
                total_subtol_nonzeros,
                DIAGNOSTIC_COLUMN_PREVIEW,
                col_maxes,
            );
        }
        return Ok(None);
    }
    Ok(Some(lower_bounds))
}

pub(crate) fn structural_time_coefficient_lower_bounds_with_monotone_time_wiggle(
    design_derivative_exit: &DesignMatrix,
    derivative_offset_exit: &Array1<f64>,
    lower_bound: f64,
    monotone_time_wiggle_ncols: usize,
) -> Result<Option<Array1<f64>>, String> {
    let mut lower_bounds = structural_time_coefficient_lower_bounds(
        design_derivative_exit,
        derivative_offset_exit,
        lower_bound,
    )?;
    let Some(bounds) = lower_bounds.as_mut() else {
        return Ok(None);
    };
    if monotone_time_wiggle_ncols == 0 {
        return Ok(lower_bounds);
    }
    if monotone_time_wiggle_ncols > bounds.len() {
        return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
            "structural time coefficient bounds cannot reserve {monotone_time_wiggle_ncols} monotone wiggle columns from {} coefficients",
            bounds.len()
        ) }.into());
    }

    // Time wiggle columns are appended as zero-derivative tail columns in the
    // linear time block, but they re-enter through a monotone I-spline
    // composition h = h_base + I(h_base) @ beta_w. For that composition,
    // beta_w >= 0 implies dq/dh_base = 1 + I'(h_base) @ beta_w >= 1 because
    // I' is an M-spline and therefore non-negative. The sign of the baseline
    // hazard trend does not change this requirement: negative beta_w is the
    // wrong monotonicity direction for the time wiggle in every case.
    let tail_start = bounds.len() - monotone_time_wiggle_ncols;
    for col in tail_start..bounds.len() {
        if !bounds[col].is_finite() || bounds[col] < 0.0 {
            bounds[col] = 0.0;
        }
    }
    Ok(lower_bounds)
}

/// Project `beta0` (or the origin when `beta0` is `None`) onto the feasible
/// polytope `{x : A x >= b}` via cyclic Dykstra projections.
///
/// The geometry only makes sense when every operand lives in the same
/// `dim`-dimensional space, so the function validates its three independent
/// dimensions up front rather than letting an ndarray broadcast mismatch
/// `unwrap()` into a process-wide panic (see issue #374: a stale, lower-
/// dimensional warm-start hint reached this projection with `beta0.len() !=
/// dim` and the `&beta + &corrections.row(i)` add panicked with
/// `IncompatibleShape`). A length mismatch is a caller contract violation,
/// so it is surfaced as a structured `Result::Err` that the marginal-slope /
/// location-scale pipelines turn into a clean `GamError` instead of a panic
/// crossing the Rust/Python boundary.
pub fn project_onto_linear_constraints(
    dim: usize,
    constraints: &LinearInequalityConstraints,
    beta0: Option<&Array1<f64>>,
) -> Result<Array1<f64>, String> {
    if let Some(b0) = beta0
        && b0.len() != dim
    {
        return Err(SurvivalLocationScaleError::DimensionMismatch {
            reason: format!(
                "project_onto_linear_constraints: beta0 length {} does not match dim {dim}",
                b0.len()
            ),
        }
        .into());
    }
    if constraints.a.nrows() != constraints.b.len() {
        return Err(SurvivalLocationScaleError::DimensionMismatch {
            reason: format!(
                "project_onto_linear_constraints: constraint A has {} rows but b has length {}",
                constraints.a.nrows(),
                constraints.b.len()
            ),
        }
        .into());
    }
    let beta0_vec = beta0.cloned().unwrap_or_else(|| Array1::zeros(dim));
    if constraints.a.ncols() != dim || constraints.a.nrows() == 0 {
        return Ok(beta0_vec);
    }

    // EXACT projection of the seed onto the feasible polytope `{β : Aβ ≥ b}`.
    //
    // The monotone time-derivative guard emits ONE constraint row per
    // observation, so a survival fit on a discrete / tied inspection grid (the
    // interval-censored `SurvInterval(L, R, event)` path) hands this O(n)
    // heavily near-parallel rows — and they are NOT bit-identical (per-row
    // offsets / covariate-dependent derivative designs make tied-time rows
    // distinct but nearly collinear). Dykstra alternating projection over such
    // a system converges only LINEARLY and, at any finite sweep cap, stalls far
    // above tolerance (5e-3..2e-2 at 100 sweeps); the previous implementation
    // then SILENTLY returned that infeasible point, which the downstream
    // active-set QP entry gate (`check_linear_feasibility`, `tol = 1e-8`)
    // rejected as an "infeasible iterate" — aborting the whole interval
    // warm start (#1108).
    //
    // The active-set QP solves the projection `min ½‖β − β0‖²  s.t.  Aβ ≥ b`
    // EXACTLY in a single solve, rank-reducing the dependent guard rows
    // internally (pivoted QR + Bland anti-cycling), so it does not stall on
    // redundancy. Take the STRICTLY-INTERIOR projection first: it returns the
    // nearest point whose every row clears the `1e-6` scaled interior margin,
    // so the seed clears the downstream raw `1e-8` gate with ~100× room. If the
    // margin-shifted system has empty interior, fall back to the exact boundary
    // projection (`H = I`). Only if BOTH exact solves refuse AND a Dykstra
    // safety net cannot reach tolerance do we return a HARD ERROR with the
    // residual — never a silently-accepted infeasible seed.
    let n_rows = constraints.a.nrows();
    // Accept a candidate projection iff it clears the downstream feasibility
    // gate (`check_linear_feasibility`, raw absolute `1e-8`) — the exact
    // contract the consumer enforces. The strict-interior projection clears it
    // by orders of magnitude; the boundary projection sits at the gate, so we
    // accept it only when it genuinely lands inside.
    const DOWNSTREAM_FEASIBILITY_GATE_TOL: f64 = MONOTONE_CONE_FEASIBILITY_GATE_TOL;
    let worst_raw_violation = |b: &Array1<f64>| -> (f64, usize) {
        let mut worst = 0.0_f64;
        let mut worst_row = 0usize;
        for i in 0..n_rows {
            let slack = constraints.a.row(i).dot(b) - constraints.b[i];
            let viol = (-slack).max(0.0);
            if viol > worst {
                worst = viol;
                worst_row = i;
            }
        }
        (worst, worst_row)
    };

    if let Some(interior) = crate::solver::active_set::project_point_strictly_into_feasible_cone(
        &beta0_vec,
        constraints,
    ) && worst_raw_violation(&interior).0 <= DOWNSTREAM_FEASIBILITY_GATE_TOL
    {
        return Ok(interior);
    }
    let identity = Array2::<f64>::eye(dim);
    if let Ok((boundary, _active)) =
        crate::solver::active_set::solve_quadratic_with_linear_constraints(
            &identity,
            &beta0_vec,
            &beta0_vec,
            constraints,
            None,
        )
        && worst_raw_violation(&boundary).0 <= DOWNSTREAM_FEASIBILITY_GATE_TOL
    {
        return Ok(boundary);
    }

    // Dykstra safety net (alternating projection is guaranteed to converge to
    // the exact projection onto a convex intersection of halfspaces). Collapse
    // bit-identical rows first (tied-time duplicates that DO coincide), run to
    // the internal tolerance, then HARD-CHECK feasibility against the
    // downstream gate — no silent infeasible return.
    let mut seen: std::collections::HashSet<Box<[u64]>> =
        std::collections::HashSet::with_capacity(n_rows);
    let mut unique_rows: Vec<usize> = Vec::with_capacity(n_rows);
    for i in 0..n_rows {
        let row_i = constraints.a.row(i);
        if row_i.dot(&row_i) <= DYKSTRA_ROW_DEGENERACY_FLOOR {
            continue;
        }
        let mut key: Vec<u64> = Vec::with_capacity(dim + 1);
        key.extend(row_i.iter().map(|v| v.to_bits()));
        key.push(constraints.b[i].to_bits());
        if seen.insert(key.into_boxed_slice()) {
            unique_rows.push(i);
        }
    }
    let mut beta = beta0_vec;
    let mut corrections = Array2::<f64>::zeros((unique_rows.len(), dim));
    let max_sweeps = DYKSTRA_PROJECTION_MAX_SWEEPS;
    for _ in 0..max_sweeps {
        let mut max_violation = 0.0_f64;
        for (slot, &i) in unique_rows.iter().enumerate() {
            let row = constraints.a.row(i);
            let row_norm_sq = row.dot(&row);
            if row_norm_sq <= DYKSTRA_ROW_DEGENERACY_FLOOR {
                continue;
            }
            let y = &beta + &corrections.row(slot);
            let slack = row.dot(&y) - constraints.b[i];
            max_violation = max_violation.max((-slack).max(0.0));
            if slack >= 0.0 {
                corrections.row_mut(slot).assign(&(&y - &beta));
                continue;
            }
            let step = (constraints.b[i] - row.dot(&y)) / row_norm_sq;
            let projected = &y + &(row.to_owned() * step);
            corrections.row_mut(slot).assign(&(&y - &projected));
            beta.assign(&projected);
        }
        if max_violation <= DYKSTRA_PROJECTION_TOL {
            break;
        }
    }
    let (worst, worst_row) = worst_raw_violation(&beta);
    if worst > DOWNSTREAM_FEASIBILITY_GATE_TOL {
        return Err(SurvivalLocationScaleError::ConstraintViolation {
            reason: format!(
                "project_onto_linear_constraints could not certify a feasible projection of the \
                 seed onto the monotone time-derivative cone: worst raw violation {worst:.3e} at \
                 row {worst_row} ({} unique of {n_rows} guard rows). Both exact active-set \
                 projections (strict-interior and boundary) refused and the Dykstra safety net \
                 did not reach the downstream gate tol={DOWNSTREAM_FEASIBILITY_GATE_TOL:.1e}. This \
                 is a genuine feasibility failure of the constraint system, surfaced rather than \
                 silently returning an infeasible seed.",
                unique_rows.len(),
            ),
        }
        .into());
    }
    Ok(beta)
}

pub(crate) fn validate_linear_constraints(
    label: &str,
    beta: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
) -> Result<(), String> {
    if beta.len() != constraints.a.ncols() {
        return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
            "survival location-scale {label} constraint dimension mismatch: beta={}, constraints={}",
            beta.len(),
            constraints.a.ncols()
        ) }.into());
    }
    if constraints.a.nrows() != constraints.b.len() {
        return Err(SurvivalLocationScaleError::DimensionMismatch {
            reason: format!(
                "survival location-scale {label} constraint row mismatch: A rows={}, b len={}",
                constraints.a.nrows(),
                constraints.b.len()
            ),
        }
        .into());
    }

    let mut worst_row = None;
    let mut worst_slack = 0.0_f64;
    let mut worst_tol = 0.0_f64;
    for row in 0..constraints.a.nrows() {
        let a_row = constraints.a.row(row);
        let slack = a_row.dot(beta) - constraints.b[row];
        let scale = a_row
            .iter()
            .zip(beta.iter())
            .map(|(a, b)| (a * b).abs())
            .sum::<f64>()
            .max(constraints.b[row].abs())
            .max(1.0);
        // Floor the relative tolerance at the absolute downstream feasibility
        // gate (#1569): this post-update check must accept any β the consumer
        // gates (`check_linear_feasibility` / `project_onto_linear_constraints`,
        // both `1e-8`) accept, or it hard-errors a round-off-feasible iterate the
        // rest of the pipeline treats as feasible (the survival-LS final-refit
        // cone-projected β at slack ~-6.6e-9).
        let tol =
            (CONSTRAINT_NONNEGATIVITY_REL_TOL * scale).max(MONOTONE_CONE_FEASIBILITY_GATE_TOL);
        if slack < -tol && (worst_row.is_none() || slack < worst_slack) {
            worst_row = Some(row);
            worst_slack = slack;
            worst_tol = tol;
        }
    }
    if let Some(row) = worst_row {
        return Err(SurvivalLocationScaleError::ConstraintViolation { reason: format!(
            "survival location-scale {label} violates represented linear constraint at row {row}: slack={worst_slack:.3e}, tol={worst_tol:.3e}"
        ) }.into());
    }
    Ok(())
}

/// Orthonormal basis `z` (raw `p` × reduced `r`) of the penalty null space —
/// the affine `{1, log t}` AFT baseline an I-spline 2nd-order difference penalty
/// leaves unpenalized. The penalized (curvature) directions are exactly the
/// non-affine deviation the constant-scale data cannot identify, so the
/// null-space columns are precisely the identifiable parametric subspace.
///
/// The basis is the eigenvectors of the summed time penalty whose eigenvalues
/// sit at the bottom of the spectrum (geometric kernel). `r` is read off the
/// eigenvalue gap with the same relative threshold the I-spline builder uses to
/// report `nullspace_dims`, so this is the structural null-space dimension
/// rather than a hard-coded `2`.
pub(crate) fn time_parametric_null_space_basis(
    penalties: &[Array2<f64>],
    p: usize,
) -> Option<Array2<f64>> {
    if p == 0 || penalties.is_empty() {
        return None;
    }
    let mut total = Array2::<f64>::zeros((p, p));
    for s_mat in penalties {
        if s_mat.nrows() != p || s_mat.ncols() != p {
            return None;
        }
        total += s_mat;
    }
    let (evals, evecs) = total.eigh(faer::Side::Lower).ok()?;
    let max_ev = evals
        .iter()
        .copied()
        .fold(0.0_f64, |a, b| a.max(b.abs()))
        .max(1.0);
    // Mirror the I-spline builder's null-space threshold (survival_construction).
    let threshold = 100.0 * (p as f64) * f64::EPSILON * max_ev;
    // `eigh` returns ascending eigenvalues with eigenvectors as the matching
    // columns of `evecs`; the kernel is the leading low-eigenvalue block.
    let null_cols: Vec<usize> = evals
        .iter()
        .enumerate()
        .filter(|&(_, &e)| e <= threshold)
        .map(|(idx, _)| idx)
        .collect();
    if null_cols.is_empty() || null_cols.len() >= p {
        // No surplus flexibility to remove (or a degenerate all-null penalty):
        // there is nothing to reduce, keep the full basis.
        return None;
    }
    Some(evecs.select(ndarray::Axis(1), &null_cols))
}

/// Does a certified constant-scale AFT (no timewiggle) carry the `log t`
/// baseline in its FULL time-warp design span, so the whole warp can collapse to
/// the σ-scaled log-t LOCATION offset regardless of how the penalty's affine
/// null space spectrally resolves?
///
/// `time_parametric_null_space_basis` re-derives the I-spline penalty's affine
/// null space spectrally. At the DEFAULT high knot count (8 internal knots) the
/// value-space congruence `Lᵀ S_B L` is badly conditioned, so that re-derivation
/// can miss the single structural null direction (`null_cols.is_empty()` →
/// `None`) or absorb extra near-null directions — leaving the rank-1/rank-2/
/// general null-space collapses below unfired and the warp full, penalized, and
/// (with the strong-smoothing ρ seed) flattened to a fraction of its true scale.
/// That flattening multiplicatively shrinks BOTH the location slopes and σ by a
/// common factor and aliases the threshold intercept against the warp constant,
/// collapsing the lowest-leverage interaction to 0 (the lognormal-AFT defect).
///
/// The `reduce_to_parametric` regime is ALREADY certified constant-scale with no
/// timewiggle, and the survival time basis is built over `log t`
/// (survival_construction.rs), so the parametric-AFT baseline IS exactly `log t`.
/// This predicate confirms `log t` is recoverable from the full warp design
/// image (`reduced_warp_logt_baseline_usable` on the identity basis — i.e. the
/// full column span), which it is for any non-degenerate I-spline-over-log-t
/// basis. When it holds the warp collapses to `location_logt_offset_time_block`
/// — `h ≡ 0`, the `log t` baseline carried on the σ-scaled `q` channel, and the
/// `−log σ` event-Jacobian term that IDENTIFIES σ — the same σ-pinning
/// representation the rank-1 gauge produces, robust to the spectral fragility.
pub(crate) fn time_block_collapses_to_logt_baseline(
    constant_scale: bool,
    protected_timewiggle_cols: usize,
    design_exit: &Array2<f64>,
    log_time_exit: ndarray::ArrayView1<f64>,
) -> bool {
    if !constant_scale || protected_timewiggle_cols != 0 {
        return false;
    }
    let p = design_exit.ncols();
    if p == 0 {
        return false;
    }
    let identity = Array2::<f64>::eye(p);
    reduced_warp_logt_baseline_usable(&identity, design_exit, log_time_exit)
}

/// Does the constant-scale-AFT regime actually reduce the time block to its
/// unpenalized affine parametric null space?
///
/// This is the single predicate that both the inner block preparation
/// (`prepare_identified_time_block`) and the OUTER ρ layout
/// (`SurvivalLambdaLayout`) consult so they agree on the reduced time block's
/// smoothing-parameter count. The reduction fires when the regime is
/// constant-scale with no monotone timewiggle reintroducing flexibility AND
/// either the time penalty has a spectrally-resolved affine null space to
/// collapse onto OR the full warp design carries the `log t` baseline
/// (`time_block_collapses_to_logt_baseline`, robust to the spectral fragility of
/// the high-knot-count null-space re-derivation). When it fires the reduced
/// block is genuinely unpenalized, so it carries ZERO smoothing parameters and
/// must contribute no ρ coordinate to the outer REML search — exactly like the
/// constant `log_sigma` and rigid `threshold` blocks (issue #736/#735/#721).
pub(crate) fn time_block_reduces_to_parametric(
    time_penalties: &[Array2<f64>],
    time_ncols: usize,
    constant_scale: bool,
    protected_timewiggle_cols: usize,
    design_exit: &Array2<f64>,
    log_time_exit: ndarray::ArrayView1<f64>,
) -> bool {
    constant_scale
        && protected_timewiggle_cols == 0
        && (time_parametric_null_space_basis(time_penalties, time_ncols).is_some()
            || time_block_collapses_to_logt_baseline(
                constant_scale,
                protected_timewiggle_cols,
                design_exit,
                log_time_exit,
            ))
}

/// Number of time-warp smoothing parameters (outer ρ coordinates) the survival
/// location-scale model exposes. The flexible regime keeps one ρ per time
/// penalty; the reduced constant-scale-AFT regime drops them all because the
/// affine parametric block it collapses to is unpenalized.
pub(crate) fn survival_time_rho_count(
    time_penalties: &[Array2<f64>],
    time_ncols: usize,
    constant_scale: bool,
    protected_timewiggle_cols: usize,
    design_exit: &Array2<f64>,
    log_time_exit: ndarray::ArrayView1<f64>,
) -> usize {
    if time_block_reduces_to_parametric(
        time_penalties,
        time_ncols,
        constant_scale,
        protected_timewiggle_cols,
        design_exit,
        log_time_exit,
    ) {
        0
    } else {
        time_penalties.len()
    }
}

/// Whether this fit is the reduced, fully PARAMETRIC constant-scale AFT regime,
/// in which the location and scale carry no genuine smoothing — only (at most)
/// full-rank parametric shrinkage ridges (`nullspace_dim == 0`, e.g. the
/// linear-term `LinearTermRidge` gam places on a non-intercept covariate such as
/// `age` in `Surv(time,event) ~ age`) — and so must be fit as a plain
/// few-parameter AFT MLE (loglogistic / lognormal), exactly like
/// `survreg`/`lifelines` (issue #736/#735/#721).
///
/// The conditions are:
///   * the time-warp has collapsed to its identifiable affine null space
///     (`survival_time_rho_count == 0`), i.e. constant scale with no protected
///     timewiggle;
///   * there is no link-wiggle and no monotone time-wiggle;
///   * every threshold and every log-σ penalty is a full-rank parametric ridge
///     (`nullspace_dim == 0`) — NEVER a wiggliness/smoothing penalty, whose
///     structural null space (the unpenalized polynomial/affine subspace) is
///     always nonzero.
///
/// In this regime those parametric ridges are dropped from BOTH the inner
/// prepared model and the outer ρ layout (mirroring how the reduced time block
/// drops its projected-to-zero penalties): the affine time-warp plus the
/// location intercept already identify the location/scale, the ridge has no
/// smoothing parameter worth a vacuous outer ρ coordinate (its default λ would
/// merely bias the parametric coefficient away from the `survreg`/`lifelines`
/// MLE), so the fit becomes an unpenalized direct parametric-AFT Newton MLE
/// (`fit_parametric_aft_direct_mle`) with zero outer coordinates — converging in
/// milliseconds instead of stalling the coupled exact-joint REML optimizer on a
/// flat, vacuous ρ surface.
///
/// Certification is conservative: if either block's `nullspace_dims` metadata is
/// absent or length-mismatched (so a null space cannot be certified zero) the
/// regime is NOT recognized and the fit stays on the full coupled path. A
/// genuinely flexible fit (smooth mean `~ s(z)`, smooth scale
/// `noise_formula = s(...)`, a link-wiggle, an active timewiggle, or a varying
/// scale) carries a wiggliness penalty with `nullspace_dim > 0` or a surviving
/// time ρ, so it never matches here.
pub(crate) fn survival_reduced_parametric_aft_regime(
    time_penalties: &[Array2<f64>],
    time_ncols: usize,
    constant_scale: bool,
    protected_timewiggle_cols: usize,
    threshold_nullspace_dims: &[usize],
    threshold_npenalties: usize,
    log_sigma_nullspace_dims: &[usize],
    log_sigma_npenalties: usize,
    has_linkwiggle: bool,
    design_exit: &Array2<f64>,
    log_time_exit: ndarray::ArrayView1<f64>,
) -> bool {
    if has_linkwiggle || protected_timewiggle_cols > 0 {
        return false;
    }
    if survival_time_rho_count(
        time_penalties,
        time_ncols,
        constant_scale,
        protected_timewiggle_cols,
        design_exit,
        log_time_exit,
    ) != 0
    {
        return false;
    }
    block_penalties_all_parametric_ridges(threshold_nullspace_dims, threshold_npenalties)
        && block_penalties_all_parametric_ridges(log_sigma_nullspace_dims, log_sigma_npenalties)
}

/// True iff a block's `npenalties` penalties are all full-rank parametric ridges
/// — every certified structural null-space dimension is zero. A block with no
/// penalties trivially passes. When the `nullspace_dims` metadata is absent or
/// length-mismatched the null space cannot be certified, so this conservatively
/// returns `false` (treat as potentially smoothing).
pub(crate) fn block_penalties_all_parametric_ridges(
    nullspace_dims: &[usize],
    npenalties: usize,
) -> bool {
    if npenalties == 0 {
        return true;
    }
    nullspace_dims.len() == npenalties && nullspace_dims.iter().all(|&d| d == 0)
}

/// Data-scale slope of `design_exit · direction` regressed on `log_time_exit`
/// (centered). `Some(s)` where `s = Σ(logt_c · y) / Σ(logt_c²)`; `None` when
/// `log t` has no spread (all exit times equal) or the time direction is flat
/// (zero slope), so the unit-log-t normalization would be undefined. Dividing the
/// raw direction by `s` yields a design image with unit slope vs log t — the
/// canonical survreg/lifelines AFT gauge.
pub(crate) fn unit_log_time_slope(
    design_exit: &Array2<f64>,
    direction: &Array1<f64>,
    log_time_exit: ndarray::ArrayView1<f64>,
) -> Option<f64> {
    let n = design_exit.nrows();
    if n == 0 || log_time_exit.len() != n {
        return None;
    }
    let y = design_exit.dot(direction);
    let log_mean = log_time_exit.sum() / n as f64;
    let mut sxx = 0.0_f64;
    let mut sxy = 0.0_f64;
    for i in 0..n {
        let xc = log_time_exit[i] - log_mean;
        sxx += xc * xc;
        sxy += xc * y[i];
    }
    let y_scale = y.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs())).max(1.0);
    if !sxx.is_finite() || sxx <= f64::EPSILON {
        return None;
    }
    let slope = sxy / sxx;
    if !slope.is_finite() || slope.abs() <= f64::EPSILON * y_scale {
        return None;
    }
    Some(slope)
}

/// Does the rank-1 reduced parametric-AFT regime apply (issue #892)?
///
/// The real survival time penalty is a 1st-difference penalty, so its null space
/// is DIMENSION 1: a single monotone log-t trend column `z` (p×1). When it does,
/// the time warp is REMOVED entirely (`h ≡ 0`) and the `log t` baseline is
/// carried as a per-row σ-scaled LOCATION offset instead — `u = inv_sigma·(log t
/// − η_t) = (log t − μ)/σ` — so the event Jacobian gains the `−log σ` term that
/// identifies σ (the survreg / lifelines / flexsurv AFT gauge).
///
/// This predicate only certifies the regime is genuinely log-t parametric: the
/// null space is rank-1, sized to the basis, and the single null-space direction
/// has a usable data-scale slope versus log t (so `log t` actually varies and the
/// floored times are finite). It does NOT build any design — the warp is gone and
/// the log-t baseline rides the location channel, threaded from the caller. When
/// it returns `false` the caller falls through to the prior both-columns-free
/// reduce. `log t` values come from the same floor (`SURVIVAL_TIME_FLOOR`) as
/// `checked_log_survival_times`.
pub(crate) fn rank1_reduced_time_warp_applies(
    z: &Array2<f64>,
    design_exit: &Array2<f64>,
    log_time_entry: ndarray::ArrayView1<f64>,
    log_time_exit: ndarray::ArrayView1<f64>,
) -> bool {
    if z.ncols() != 1 {
        return false;
    }
    let n = design_exit.nrows();
    if log_time_entry.len() != n || log_time_exit.len() != n {
        return false;
    }
    if log_time_entry.iter().any(|v| !v.is_finite()) || log_time_exit.iter().any(|v| !v.is_finite())
    {
        return false;
    }
    // The null-space direction must have a usable data-scale slope versus log t —
    // i.e. `log t` genuinely varies across the sample — else the σ-scaled log-t
    // gauge is degenerate and the prior reduce should handle it instead.
    let z_dir = z.column(0).to_owned();
    unit_log_time_slope(design_exit, &z_dir, log_time_exit).is_some()
}

/// Result of pinning the reduced parametric-AFT time-warp slope to the canonical
/// unit-log-t gauge (issue #892). `z_c` (p×1) is the kept-free row-constant
/// direction; `z_t` (p-vector) is the pinned unit-log-t direction owned by the
/// Gauge affine shift and folded into offsets by `Gauge::restrict_design_and_offset`.
pub(crate) struct PinnedTimeWarp {
    pub(crate) z_c: Array2<f64>,
    pub(crate) z_t: Array1<f64>,
}

/// Split the 2-D affine null-space basis `z` (p×2, orthonormal columns) into the
/// row-constant location direction `z_c` (kept free) and the time-varying warp
/// direction `z_t`, normalized so `design_exit · z_t` has unit data-scale slope
/// versus `log t_exit` (the canonical survreg/lifelines AFT gauge). Returns
/// `None` when the split is not clean — no usable row-constant direction
/// (`‖z_c_raw‖` tiny) or a degenerate log-t slope (`|s|` tiny) — so the caller
/// can fall back to the prior both-columns-free reduce behavior without
/// regressing the non-pin case.
pub(crate) fn pin_reduced_time_warp_slope(
    z: &Array2<f64>,
    design_exit: &Array2<f64>,
    log_time_exit: ndarray::ArrayView1<f64>,
) -> Option<PinnedTimeWarp> {
    let p = z.nrows();
    let n = design_exit.nrows();
    if z.ncols() != 2 || n == 0 || log_time_exit.len() != n {
        return None;
    }
    // `G = design_exit · z` (n×2). The row-constant direction `a` minimizes
    // ‖G a − 1_n‖² (least-squares constant fit): a = (GᵀG)⁻¹ Gᵀ 1_n. Solving the
    // 2×2 normal equations in closed form keeps the helper self-contained.
    let g = design_exit.dot(z);
    let m00 = g.column(0).dot(&g.column(0));
    let m01 = g.column(0).dot(&g.column(1));
    let m11 = g.column(1).dot(&g.column(1));
    let ones = Array1::<f64>::ones(n);
    let r0 = g.column(0).dot(&ones);
    let r1 = g.column(1).dot(&ones);
    let det = m00 * m11 - m01 * m01;
    // Scale-relative singularity guard for the 2×2 GramGram: a degenerate G has
    // no distinct constant/time split to exploit.
    let gram_scale = m00.max(m11).max(1.0);
    if !det.is_finite() || det.abs() <= f64::EPSILON * gram_scale * gram_scale {
        return None;
    }
    let a0 = (m11 * r0 - m01 * r1) / det;
    let a1 = (m00 * r1 - m01 * r0) / det;
    // `z_c_raw = z · a` (p-vector): the raw-coefficient direction whose design
    // image `G a` is the best row-constant column. Normalize to a unit-norm
    // basis vector so the reduced free coefficient is well scaled.
    let z_c_raw = z.dot(&Array1::from(vec![a0, a1]));
    let z_c_norm = z_c_raw.dot(&z_c_raw).sqrt();
    if !z_c_norm.is_finite() || z_c_norm <= f64::EPSILON * (p as f64).sqrt() {
        return None;
    }
    let z_c_vec = &z_c_raw / z_c_norm;
    // Time-varying direction: the in-span(z) complement of `a` in the 2-D
    // coefficient plane. With `a = [a0, a1]`, `a_perp = [-a1, a0]` is orthogonal
    // to `a`, so `z_t_raw = z · a_perp` is the part of span(z) carrying the
    // non-constant (log-t trend) warp.
    let a_perp = Array1::from(vec![-a1, a0]);
    let z_t_raw = z.dot(&a_perp);
    // Normalize `z_t` to unit data-scale slope vs log t (the canonical
    // unit-log-t AFT gauge): `design_exit · z_t` rises by exactly 1 per unit of
    // log t, so the derivative design reproduces u'(t) = 1/t.
    let slope = unit_log_time_slope(design_exit, &z_t_raw, log_time_exit)?;
    let z_t = &z_t_raw / slope;
    // p×1 free design columns and the kept-free basis matrix.
    let z_c = z_c_vec.insert_axis(ndarray::Axis(1));
    Some(PinnedTimeWarp { z_c, z_t })
}

/// Build the reduced time block for the canonical σ-scaled log-t AFT gauge
/// (issue #892): the time warp is REMOVED entirely (`h ≡ 0`, zero free columns,
/// no penalties, no monotonicity constraint) and the `log t` baseline is carried
/// as a per-row LOCATION offset on the σ-scaled `q` channel
/// (`location_log_time_offset = true`). This is the clean parametric-AFT
/// representation `u = (log t − μ)/σ` that `survreg`/`lifelines` fit: it has no
/// free time coefficient (so no `1e7` cold-start gradient, no ill-conditioned
/// time curvature), no derivative-guard inequality (so the joint Newton step is
/// never frozen by a binding time row), and no time-warp constant aliased with
/// the threshold intercept.
pub(crate) fn location_logt_offset_time_block(
    design_entry: &Array2<f64>,
    design_exit: &Array2<f64>,
    design_derivative_exit: &Array2<f64>,
    p: usize,
) -> TimeBlockPrepared {
    TimeBlockPrepared {
        design_entry: Array2::<f64>::zeros((design_entry.nrows(), 0)),
        design_exit: Array2::<f64>::zeros((design_exit.nrows(), 0)),
        design_derivative_exit: Array2::<f64>::zeros((design_derivative_exit.nrows(), 0)),
        coefficient_lower_bounds: None,
        // No free time coefficients → no derivative-guard constraint. The warp
        // derivative is `h′ ≡ 0`; monotonicity holds because the q channel
        // contributes `qdot = inv_sigma/t > 0` to `g`.
        linear_constraints: None,
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_beta: Some(Array1::<f64>::zeros(0)),
        transform: TimeIdentifiabilityTransform {
            gauge: Gauge::from_block_transforms(&[Array2::<f64>::zeros((p, 0))]),
        },
        offset_entry: Array1::zeros(design_entry.nrows()),
        offset_exit: Array1::zeros(design_exit.nrows()),
        derivative_offset_exit: Array1::zeros(design_derivative_exit.nrows()),
        pinned_free_row_constant: false,
        location_log_time_offset: true,
    }
}

/// Does the time-penalty null space `z` (raw `p` × reduced `r`) genuinely carry
/// the `log t` AFT baseline, so collapsing the whole warp to the σ-scaled log-t
/// LOCATION offset (`location_logt_offset_time_block`) is an EXACT
/// reparameterisation of a parametric constant-scale AFT?
///
/// The rank-1 gauge (`rank1_reduced_time_warp_applies`) answers this for the
/// single-column null space by checking that direction's image has a usable
/// data-scale log-t slope. This generalises it to any rank: least-squares
/// project `log t` onto the null-space warp image `G = design_exit · z` and ask
/// whether the best-fitting null direction `z · c` has a usable log-t slope. When
/// it does, `{log t}` lies in the span the warp would otherwise fit freely, so
/// the higher null directions are spurious parametric-AFT flexibility and the
/// log-t offset captures the baseline exactly. A degenerate or non-log-t basis
/// fails the check and the caller keeps the (constrained) free-warp fallback.
pub(crate) fn reduced_warp_logt_baseline_usable(
    z: &Array2<f64>,
    design_exit: &Array2<f64>,
    log_time_exit: ndarray::ArrayView1<f64>,
) -> bool {
    use crate::faer_ndarray::FaerCholesky;
    let n = design_exit.nrows();
    let r = z.ncols();
    if n == 0 || r == 0 || log_time_exit.len() != n {
        return false;
    }
    let g = design_exit.dot(z); // (n, r) warp images of the null directions
    let gtg = g.t().dot(&g); // (r, r)
    let gtl = g.t().dot(&log_time_exit); // (r,)
    // Tiny Levenberg ridge so a rank-deficient G (some null directions producing
    // a (near-)zero warp image) still yields a well-posed projection rather than
    // a Cholesky failure that would spuriously reject the collapse.
    let scale = gtg
        .diag()
        .iter()
        .fold(0.0_f64, |acc, &v| acc.max(v.abs()))
        .max(1.0);
    let mut ridged = gtg;
    for i in 0..r {
        ridged[[i, i]] += 1e-10 * scale;
    }
    let Ok(chol) = ridged.cholesky(faer::Side::Lower) else {
        return false;
    };
    let c = chol.solvevec(&gtl);
    if c.iter().any(|v| !v.is_finite()) {
        return false;
    }
    let direction = z.dot(&c);
    unit_log_time_slope(design_exit, &direction, log_time_exit).is_some()
}

pub(crate) fn prepare_identified_time_block(
    input: &TimeBlockInput,
    derivative_guard: f64,
    monotone_time_wiggle_ncols: usize,
    reduce_to_parametric: bool,
    log_time_entry: ndarray::ArrayView1<f64>,
    log_time_exit: ndarray::ArrayView1<f64>,
) -> Result<TimeBlockPrepared, String> {
    let p = input.design_exit.ncols();
    if !input.time_monotonicity.is_coordinate_cone() {
        return Err(SurvivalLocationScaleError::InvalidConfiguration { reason: format!(
            "time_block requires a coordinate-cone monotonicity strategy by construction; got {:?}",
            input.time_monotonicity
        ) }.into());
    }
    // Materialize to dense at the location-scale boundary — the hot path
    // uses dense matrix operations (scale_dense_rows, weighted_crossprod_dense).
    let design_entry = input.design_entry.to_dense();
    let design_exit = input.design_exit.to_dense();
    let design_derivative_exit = input.design_derivative_exit.to_dense();

    // Constant-scale AFT regime: reduce the unidentified flexible I-spline
    // time-warp to its identifiable affine parametric form before any
    // constraint is generated (issue #736/#735/#721).
    //
    // `z` (raw `p` × reduced `r`) is an orthonormal basis of the time penalty's
    // null space — the affine `1 + log t` AFT baseline. Projecting the three
    // exit/entry/derivative designs onto `X z` leaves a clean `r`-column
    // parametric block whose every direction is data-identified, so the coupled
    // exact-joint inner solve has no free penalized-stationarity null space to
    // choke on. The penalty `zᵀ S z` collapses to (numerically) zero on the
    // null space — the correct, unpenalized parametric block, exactly like the
    // constant-scale `log_sigma` block — and the regime's pinned `ρ = 10` time
    // seed (`build_survival_two_block_exact_joint_setup`) certifies its flat
    // box-bound KKT immediately rather than crawling the old 12-column ridge.
    //
    // Monotonicity is preserved structurally: the per-row derivative guard
    // `(X' z) β_r + offset ≥ guard` is built from the *reduced* derivative
    // design, so `h(t)` stays non-decreasing pointwise at every observed time.
    // (The coordinate-cone per-column `γ ≥ 0` lower bounds are dropped — they
    // are a property of individual I-spline columns, not of the affine
    // generators that span their null space — and the row-wise guard takes over
    // that role exactly.)
    if reduce_to_parametric && let Some(z) = time_parametric_null_space_basis(&input.penalties, p) {
        let r = z.ncols();
        // Canonical log-t gauge (issue #892). In the reduced constant-scale
        // parametric-AFT regime the I-spline time-warp collapses onto its log-t
        // affine null space (the basis is over `log t`, survival_construction.rs).
        // The full multi-column I-spline warp carries a numerically unidentified
        // ridge, so the unconstrained direct-MLE Newton picks an arbitrary scale
        // and miscalibrates the absolute survival curve.
        //
        // RANK-1 case (the one that actually fires for real fits): the survival
        // time penalty is a 1st-difference penalty, so its null space is
        // DIMENSION 1 — a single monotone log-t trend column. Pin the warp SHAPE
        // to exactly `log t` (built straight from the event times, NOT the
        // I-spline's curved image of it) but keep its SCALE `θ` a single FREE
        // coefficient: `h(t) = θ · log t`. The standardized residual is
        // `u = h − η_loc/σ` with the warp UN-scaled by σ, so a lognormal/loglogistic
        // AFT `(log t − μ)/σ` needs the warp to carry slope `1/σ` versus log t;
        // the MLE drives `θ → 1/σ` and σ recovers to truth (folding `θ ≡ 1` instead
        // would lock the residual log-t slope at 1 and over-determine σ). `θ` is
        // identified — no flat ridge — by the event Jacobian's `log|h′| = log θ −
        // log t` term, so the collapse to one log-t column is well posed. The
        // single free column is the (non-constant) log-t warp, so the threshold
        // keeps its intercept and `pinned_free_row_constant` stays false.
        if r == 1
            && z.nrows() == p
            && rank1_reduced_time_warp_applies(&z, &design_exit, log_time_entry, log_time_exit)
        {
            // σ-scaled log-t AFT gauge (issue #892). REMOVE the time warp entirely
            // (`h ≡ 0`, zero free time columns) and instead carry the `log t`
            // baseline as a per-row LOCATION offset on the existing σ-scaled `q`
            // channel: `u = inv_sigma·(log t − η_t) = (log t − μ)/σ`. The caller
            // threads the exact `−log t` value (entry/exit) and `−1/t` derivative
            // into the family (`location_log_time_offset = true` below), so the
            // standardized residual matches survreg/lifelines and the event
            // Jacobian gains `log_g = −η_ls − log t = −log σ − log t` — the `−log σ`
            // term that IDENTIFIES σ.
            //
            // Why not a free warp scale θ (the prior attempt): with the warp
            // un-scaled by σ the pair `(η_t, σ)` co-scaled freely (only `η_t/σ`
            // identified, no `−log σ` term), so every parameter shrank by a common
            // factor. Routing `log t` through the σ-scaled `q` channel supplies the
            // missing `−log σ` Jacobian and pins σ. The warp is gone, so the time
            // block is empty (no free columns, no penalties, no constraints); all
            // the σ-coupling rides the existing `q`-derivative/Hessian stack, with
            // no new time×log_sigma cross-terms.
            return Ok(location_logt_offset_time_block(
                &design_entry,
                &design_exit,
                &design_derivative_exit,
                p,
            ));
        }
        // RANK-2 case (2nd-difference penalty `{1, log t}`): kept for correctness
        // where it occurs (golden unit test), though real fits use rank-1 above.
        if r == 2
            && z.nrows() == p
            && let Some(pinned) = pin_reduced_time_warp_slope(&z, &design_exit, log_time_exit)
        {
            let PinnedTimeWarp { z_c, z_t } = pinned;
            let gauge = Gauge::from_block_transform_with_shift(z_c.clone(), z_t);
            // Augmented offsets carry the Gauge-owned pinned unit-log-t warp out
            // of the free design. `Gauge::restrict_design_and_offset` applies
            // `X · a` using the same affine shift that final beta lifting uses.
            let (reduced_entry, offset_entry) =
                gauge.restrict_design_and_offset(&design_entry, &input.offset_entry);
            let (reduced_exit, offset_exit) =
                gauge.restrict_design_and_offset(&design_exit, &input.offset_exit);
            let (reduced_derivative_exit, derivative_offset_exit) = gauge
                .restrict_design_and_offset(&design_derivative_exit, &input.derivative_offset_exit);
            let reduced_derivative_design =
                DesignMatrix::Dense(DenseDesignMatrix::from(reduced_derivative_exit.clone()));
            // Pointwise monotonicity uses the AUGMENTED derivative offset so the
            // guard `(X' z_c) β_c + offset' ≥ guard` accounts for the pinned
            // warp's own (positive, unit-log-t) derivative.
            let linear_constraints = time_derivative_guard_constraints(
                &reduced_derivative_design,
                &derivative_offset_exit,
                derivative_guard,
            )?;
            // Project the caller seed onto the single free constant direction.
            // The pinned warp lives entirely in the offset, so the reduced seed
            // only needs the `z_c` component.
            let initial_beta = match (linear_constraints.as_ref(), input.initial_beta.as_ref()) {
                (Some(constraints), Some(beta0)) => Some(project_onto_linear_constraints(
                    1,
                    constraints,
                    Some(&z_c.t().dot(beta0)),
                )?),
                (_, Some(beta0)) => Some(z_c.t().dot(beta0)),
                _ => None,
            };
            return Ok(TimeBlockPrepared {
                design_entry: reduced_entry,
                design_exit: reduced_exit,
                design_derivative_exit: reduced_derivative_exit,
                coefficient_lower_bounds: None,
                linear_constraints,
                penalties: Vec::new(),
                nullspace_dims: Vec::new(),
                initial_beta,
                transform: TimeIdentifiabilityTransform { gauge },
                offset_entry,
                offset_exit,
                derivative_offset_exit,
                // Pin fired: the free `z_c` column is row-constant, so the
                // threshold intercept must be dropped to break the alias (#892).
                pinned_free_row_constant: true,
                // Rank-2 path keeps a free warp; no location-channel log-t offset.
                location_log_time_offset: false,
            });
        }
        // GENERAL case (r ≥ 3, or r ∈ {1,2} where the clean rank-1 / rank-2 pin
        // gauges did not match): for a fully PARAMETRIC constant-scale AFT the
        // baseline transform is exactly `log t`, so the I-spline null space's
        // higher affine directions are spurious flexibility. When the null space
        // genuinely carries the `log t` baseline, collapse the ENTIRE warp to the
        // canonical σ-scaled log-t LOCATION offset — the same clean representation
        // the rank-1 gauge uses — instead of keeping a free monotone warp with a
        // derivative-guard inequality.
        //
        // The prior free-warp fallback (kept below as a last resort) made the
        // reduced time block carry `r` free columns AND a per-row monotonicity
        // constraint. For this regime that cold-started (β = 0) at a degenerate
        // warp with a ~1e7 gradient and a derivative-guard row active at the
        // boundary; the direct-MLE joint Newton's single global step length was
        // then capped to 0 by that one binding time row, FREEZING every block —
        // including the fully unconstrained location covariate — at its cold-start
        // 0 (gam#1110: log-logistic AFT `age` pinned to exactly 0, RMSE 0.267 vs
        // lifelines 0.024). Collapsing to the log-t offset removes the free warp,
        // the constraint, and the ill-conditioning together, recovering the clean
        // survreg/lifelines-style parametric AFT MLE.
        if reduced_warp_logt_baseline_usable(&z, &design_exit, log_time_exit) {
            return Ok(location_logt_offset_time_block(
                &design_entry,
                &design_exit,
                &design_derivative_exit,
                p,
            ));
        }
        let reduced_entry = design_entry.dot(&z);
        let reduced_exit = design_exit.dot(&z);
        let reduced_derivative_exit = design_derivative_exit.dot(&z);
        // `z` spans the penalty null space, so every `zᵀ S z` is (numerically)
        // the zero `r×r` matrix: the reduced affine block has NO curvature left
        // to penalize. An unpenalized parametric block has no smoothing
        // parameter to select, so we drop the projected-to-zero penalties (and
        // their null-space-dimension bookkeeping) entirely rather than carry a
        // list of zero matrices. This is what makes the reduced time block
        // contribute ZERO ρ coordinates to the outer REML search — identical to
        // the constant `log_sigma` and rigid `threshold` blocks — so the outer
        // optimizer no longer crawls a flat, irrelevant time-smoothing ridge
        // (issue #736/#735/#721). The parametric design and the row-wise
        // monotonicity guard below carry all the time-warp structure; the
        // dropped penalties were exactly zero and contributed nothing.
        let reduced_penalties: Vec<Array2<f64>> = Vec::new();
        let reduced_nullspace_dims: Vec<usize> = Vec::new();
        let reduced_derivative_design =
            DesignMatrix::Dense(DenseDesignMatrix::from(reduced_derivative_exit.clone()));
        // Pointwise monotonicity in the reduced affine space: enforce the
        // derivative guard directly via row constraints on `X' z`. There is no
        // coordinate-cone lower-bound ridge here because the affine generators
        // are not individually sign-definite I-spline columns.
        let linear_constraints = time_derivative_guard_constraints(
            &reduced_derivative_design,
            &input.derivative_offset_exit,
            derivative_guard,
        )?;
        let initial_beta = match (linear_constraints.as_ref(), input.initial_beta.as_ref()) {
            (Some(constraints), Some(beta0)) => Some(project_onto_linear_constraints(
                r,
                constraints,
                Some(&z.t().dot(beta0)),
            )?),
            (_, Some(beta0)) => Some(z.t().dot(beta0)),
            _ => None,
        };
        return Ok(TimeBlockPrepared {
            design_entry: reduced_entry,
            design_exit: reduced_exit,
            design_derivative_exit: reduced_derivative_exit,
            coefficient_lower_bounds: None,
            linear_constraints,
            penalties: reduced_penalties,
            nullspace_dims: reduced_nullspace_dims,
            initial_beta,
            // Non-clean split (r != 2 or degenerate constant/time split): keep
            // both affine columns free with no pinned warp, offsets passthrough.
            transform: TimeIdentifiabilityTransform {
                gauge: Gauge::from_block_transforms(&[z]),
            },
            offset_entry: input.offset_entry.clone(),
            offset_exit: input.offset_exit.clone(),
            derivative_offset_exit: input.derivative_offset_exit.clone(),
            // Fallback reduce: the reduced I-spline columns are strictly
            // monotone (not row-constant), so the threshold keeps its intercept.
            pinned_free_row_constant: false,
            // Fallback reduce keeps a free warp; no location-channel log-t offset.
            location_log_time_offset: false,
        });
    }

    // Robust σ-scaled log-t collapse (lognormal/Gaussian AFT warp-scale defect).
    // The certified constant-scale parametric-AFT regime may reach here with the
    // null-space branch above UNFIRED: at the DEFAULT high knot count (8 internal
    // knots) the value-space congruence `Lᵀ S_B L` is ill-conditioned, so
    // `time_parametric_null_space_basis`'s spectral re-derivation can return
    // `None` (the single structural affine null direction pushed above the
    // `100·p·eps·max_ev` threshold) even though the construction-certified
    // `nullspace_dims` is 1. Without this collapse the warp stays full,
    // penalized, and — under the strong-smoothing time-ρ seed — flattened, which
    // multiplicatively shrinks every location slope and σ by a common factor and
    // aliases the threshold intercept against the warp constant (the lowest-
    // leverage interaction then sticks at its cold-start 0).
    //
    // This is the SAME σ-pinning representation the rank-1 gauge produces: the
    // survival time basis is built over `log t`, so when the full warp design
    // carries the `log t` baseline (`time_block_collapses_to_logt_baseline`) the
    // whole warp collapses to `h ≡ 0` with the `log t` baseline on the σ-scaled
    // `q` channel, supplying the `−log σ` event-Jacobian term that IDENTIFIES σ
    // (`location_logt_offset_time_block`). The OUTER ρ layout agrees because
    // `survival_time_rho_count` ORs the same predicate, so both see `k_time == 0`.
    if reduce_to_parametric
        && time_block_collapses_to_logt_baseline(true, 0, &design_exit, log_time_exit)
    {
        return Ok(location_logt_offset_time_block(
            &design_entry,
            &design_exit,
            &design_derivative_exit,
            p,
        ));
    }

    let penalties = input.penalties.clone();
    let coefficient_lower_bounds = structural_time_coefficient_lower_bounds_with_monotone_time_wiggle(
        &input.design_derivative_exit,
        &input.derivative_offset_exit,
        derivative_guard,
        monotone_time_wiggle_ncols,
    )?
    .ok_or_else(|| {
        "structural time block requires derivative offsets to encode the derivative guard and a non-negative derivative basis"
            .to_string()
    })?;
    let coefficient_constraints = lower_bound_constraints(&coefficient_lower_bounds);
    let derivative_constraints = time_derivative_guard_constraints(
        &input.design_derivative_exit,
        &input.derivative_offset_exit,
        derivative_guard,
    )?;
    let linear_constraints =
        append_linear_constraints(coefficient_constraints.clone(), derivative_constraints)?;
    let initial_beta = match (linear_constraints.as_ref(), input.initial_beta.as_ref()) {
        (Some(constraints), Some(beta0)) => {
            let mut clipped = beta0.clone();
            for (value, &lower) in clipped.iter_mut().zip(coefficient_lower_bounds.iter()) {
                if lower.is_finite() && *value < lower {
                    *value = lower;
                }
            }
            if validate_linear_constraints("time initial beta", &clipped, constraints).is_ok() {
                Some(clipped)
            } else {
                Some(project_onto_linear_constraints(
                    p,
                    constraints,
                    Some(beta0),
                )?)
            }
        }
        (_, Some(beta0)) => Some(beta0.clone()),
        _ => None,
    };

    Ok(TimeBlockPrepared {
        design_entry,
        design_exit,
        design_derivative_exit,
        coefficient_lower_bounds: Some(coefficient_lower_bounds),
        linear_constraints,
        penalties,
        nullspace_dims: input.nullspace_dims.clone(),
        initial_beta,
        // Identity (non-reduce) path: the raw time block passes through
        // unchanged, so the lift is `z = I`, no pinned warp, offsets verbatim.
        transform: TimeIdentifiabilityTransform {
            gauge: Gauge::identity(&[p]),
        },
        offset_entry: input.offset_entry.clone(),
        offset_exit: input.offset_exit.clone(),
        derivative_offset_exit: input.derivative_offset_exit.clone(),
        // Full flexible I-spline: no pin, threshold keeps its intercept.
        pinned_free_row_constant: false,
        // Flexible regime keeps the full warp; no location-channel log-t offset.
        location_log_time_offset: false,
    })
}
