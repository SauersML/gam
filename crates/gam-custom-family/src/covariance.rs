//! Joint covariance/geometry and the stationarity/KKT-residual machinery in the
//! flattened joint coefficient space: matrix-free path selection, joint penalty
//! application + preconditioner, flat-beta state sync, projected-stationarity and
//! KKT-residual-for-IFT computations, and the joint covariance/geometry assembly.

use super::*;

pub const JOINT_MATRIX_FREE_MIN_DIM: usize = 512;

pub(crate) const JOINT_MATRIX_FREE_MIN_ROWS: usize = 50_000;

pub(crate) const JOINT_MATRIX_FREE_MIN_DIM_AT_LARGE_N: usize = 128;

pub(crate) const JOINT_MATRIX_FREE_MIN_LINEAR_WORK: usize = 4_000_000;

pub(crate) const JOINT_TRACE_STABILITY_RIDGE: f64 = 1e-10;

pub(crate) const JOINT_PCG_MAX_ITER_MULTIPLIER: usize = 4;

pub fn joint_exact_analytic_outer_hessian_available() -> bool {
    true
}

pub(crate) fn joint_observation_count(states: &[ParameterBlockState]) -> usize {
    states
        .iter()
        .map(|state| state.eta.len())
        .max()
        .unwrap_or(0)
}

/// Whether the unified evaluator will pick the matrix-free joint Hessian path
/// for a problem of size `(total_p, total_n)`. Exposed at crate scope so
/// families with matrix-free operators can branch their `coefficient_hessian_cost`
/// estimate on the same predicate the evaluator will use at fit time.
///
/// For large-scale row counts with only tens of coefficients, exact
/// materialization is bounded by `total_p` Hessian-vector products and then a
/// tiny dense factorization. That is cheaper and more predictable than PCG when
/// each matrix-free product streams all rows through expensive FLEX marginal-
/// slope kernels and the initial joint Hessian is ill-conditioned. Keep the
/// matrix-free route for genuinely wide joint systems, where `total_p` dense
/// products and factorization dominate.
pub fn use_joint_matrix_free_path(total_p: usize, total_n: usize) -> bool {
    total_p >= JOINT_MATRIX_FREE_MIN_DIM
        || (total_n >= JOINT_MATRIX_FREE_MIN_ROWS
            && total_p >= JOINT_MATRIX_FREE_MIN_DIM_AT_LARGE_N)
        || (total_p >= JOINT_MATRIX_FREE_MIN_DIM_AT_LARGE_N
            && total_n.saturating_mul(total_p) >= JOINT_MATRIX_FREE_MIN_LINEAR_WORK)
}

pub(crate) fn apply_joint_block_penalty(
    ranges: &[(usize, usize)],
    s_lambdas: &[Array2<f64>],
    vector: &Array1<f64>,
    diagonal_ridge: f64,
    joint_full_width: Option<&gam_problem::JointPenaltyBundle>,
) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(vector.len());
    apply_joint_block_penalty_into(
        ranges,
        s_lambdas,
        vector,
        diagonal_ridge,
        &mut out,
        joint_full_width,
    );
    out
}

/// In-place variant of [`apply_joint_block_penalty`]. Caller supplies the
/// output buffer to eliminate per-call allocation.
///
/// Uses `fast_av_view_into` to write directly into the per-block slice of
/// `out`, avoiding the per-block intermediate `Array1` from `fast_av`. At
/// large scale this is invoked inside the PCG matvec closure (called
/// once per CG iter, hundreds-to-thousands of times per outer iter per
/// the perf-scout report).
pub(crate) fn apply_joint_block_penalty_into(
    ranges: &[(usize, usize)],
    s_lambdas: &[Array2<f64>],
    vector: &Array1<f64>,
    diagonal_ridge: f64,
    out: &mut Array1<f64>,
    joint_full_width: Option<&gam_problem::JointPenaltyBundle>,
) {
    assert_eq!(out.len(), vector.len());
    assert!(s_lambdas.len() <= ranges.len());
    out.fill(0.0);

    if s_lambdas.len() <= 1 {
        for (b, s_lambda) in s_lambdas.iter().enumerate() {
            let (start, end) = ranges[b];
            let block = vector.slice(s![start..end]);
            let mut out_slice = out.slice_mut(s![start..end]);
            gam_linalg::faer_ndarray::fast_av_view_into(s_lambda, &block, out_slice.view_mut());
        }
        if diagonal_ridge > 0.0 {
            out.scaled_add(diagonal_ridge, vector);
        }
        if let Some(bundle) = joint_full_width
            && !bundle.is_empty()
        {
            bundle.add_apply_into(vector.view(), out);
        }
        return;
    }

    if out.as_slice_mut().is_none() {
        for (b, s_lambda) in s_lambdas.iter().enumerate() {
            let (start, end) = ranges[b];
            let block = vector.slice(s![start..end]);
            let mut out_slice = out.slice_mut(s![start..end]);
            gam_linalg::faer_ndarray::fast_av_view_into(s_lambda, &block, out_slice.view_mut());
        }
        if diagonal_ridge > 0.0 {
            out.scaled_add(diagonal_ridge, vector);
        }
        if let Some(bundle) = joint_full_width
            && !bundle.is_empty()
        {
            bundle.add_apply_into(vector.view(), out);
        }
        return;
    }

    {
        let out_values = out
            .as_slice_mut()
            .expect("joint penalty output should be contiguous");
        let mut out_blocks = Vec::with_capacity(s_lambdas.len());
        let mut remaining = out_values;
        let mut cursor = 0usize;
        for &(start, end) in ranges.iter().take(s_lambdas.len()) {
            assert!(start >= cursor);
            assert!(end >= start);
            let (_, after_gap) = remaining.split_at_mut(start - cursor);
            let (out_block, after_block) = after_gap.split_at_mut(end - start);
            out_blocks.push(out_block);
            remaining = after_block;
            cursor = end;
        }

        use rayon::prelude::*;

        out_blocks
            .into_par_iter()
            .enumerate()
            .for_each(|(b, out_block)| {
                let (start, end) = ranges[b];
                let block = vector.slice(s![start..end]);
                let out_view = ArrayViewMut1::from(out_block);
                gam_linalg::faer_ndarray::fast_av_view_into(&s_lambdas[b], &block, out_view);
            });
    }

    if diagonal_ridge > 0.0 {
        if let (Some(out_values), Some(vector_values)) = (out.as_slice_mut(), vector.as_slice()) {
            use rayon::prelude::*;

            out_values
                .par_iter_mut()
                .zip(vector_values.par_iter())
                .for_each(|(out_value, vector_value)| {
                    *out_value += diagonal_ridge * *vector_value;
                });
        } else {
            out.scaled_add(diagonal_ridge, vector);
        }
    }

    if let Some(bundle) = joint_full_width
        && !bundle.is_empty()
    {
        bundle.add_apply_into(vector.view(), out);
    }
}

/// Penalty-aware Jacobi preconditioner used by every matrix-free PCG path
/// in the inner coefficient solve.
///
/// Builds `|diag(H)| + Σ_k gershgorin(S_k(λ)) + ridge`, clamped at 1e-10, where
/// `gershgorin(S)[i] = Σ_j |S[i,j]|` is the absolute row-sum (Gershgorin
/// radius) of each penalty block. This strictly dominates `diag(S)` for any
/// penalty with off-diagonal mass — the high-order difference / thin-plate
/// smooths (the cubic-Duchon `[mass, tension, stiffness]` triple, orders
/// [1,2,3] in `WigglePenaltyConfig::cubic_triple_operator_default`) are
/// strongly off-diagonal-dominant, so `S[i,i]` alone understates the
/// operator's true row scale by orders of magnitude there.
///
/// The absolute likelihood diagonal is essential for exact-Newton families:
/// their observed Hessian may be indefinite away from the mode.  A negative
/// diagonal is real curvature scale, not an absent direction.  Flooring it to
/// `1e-10` makes the trust metric nearly singular, inflates the corresponding
/// whitened eigenvalue, and can misclassify a resolvable direction as numerical
/// null space.  `|diag(H)|` is the standard positive Jacobi scale for an
/// indefinite operator; for Fisher/PIRLS Hessians (whose diagonal is already
/// non-negative) it is exactly unchanged.
///
/// Why the row-sum and not just the diagonal: a plain Jacobi (diagonal-only)
/// preconditioner collapses to `diag(S_λ)` exactly in the saturated-softmax
/// regime, where the data Fisher weight `W = diag(p) − ppᵀ → 0` near the
/// simplex boundary and the data part of `diag(H)` vanishes. When the penalty
/// is off-diagonal-dominant, `diag(S_λ)` is a poor spectral match for
/// `H + S_λ`, leaving PCG with a large effective condition number and only
/// geometric (linear) convergence — the multinomial-penguins grind in #715.
/// The Gershgorin row-sum diagonal tracks the operator's per-coordinate scale
/// (`|S| 𝟙` bounds `S`'s action), tightening the preconditioned spectrum and
/// cutting CG iterations sharply in that regime. It is `≥ diag(S)` entrywise
/// for SPD `S`, so it stays strictly positive and SPD: it changes only the
/// PCG trajectory, never the converged Newton step or the KKT certificate
/// (PCG converges to the same `(H + S_λ)⁻¹ rhs` under any SPD preconditioner).
/// Design docs sometimes call this the "triple-operator penalty
/// preconditioner"; in code it is the single, unified preconditioner shared by
/// all PCG callsites.
///
/// Callers in the PIRLS inner Newton PCG path feed the result as the diagonal
/// rescale every CG iteration: PCG applies `M^{-1}` to residuals directly.
/// Do not square-root or trace-normalize these entries, and do not apply a
/// second preconditioner-side rescale to the returned Newton step.
pub(crate) fn positive_joint_diagonal_entry(value: f64) -> f64 {
    if value.is_finite() && value > 1.0e-10 {
        value
    } else {
        1.0e-10
    }
}

pub(crate) fn joint_penalty_preconditioner_diag(
    base_diagonal: &Array1<f64>,
    ranges: &[(usize, usize)],
    s_lambdas: &[Array2<f64>],
    diagonal_ridge: f64,
    joint_full_width: Option<&gam_problem::JointPenaltyBundle>,
) -> Array1<f64> {
    assert!(s_lambdas.len() <= ranges.len());
    // This diagonal is both the PCG preconditioner and the trust-region metric,
    // so it must be positive while preserving the magnitude of negative
    // observed curvature.  Do the absolute-value conversion once at this
    // shared boundary before adding the positive penalty scales.
    let mut diag = base_diagonal.mapv(f64::abs);
    for (b, s_lambda) in s_lambdas.iter().enumerate() {
        let (start, end) = ranges[b];
        assert_eq!(s_lambda.nrows(), end - start);
        assert_eq!(s_lambda.ncols(), end - start);
        // Gershgorin radius: the absolute row-sum `Σ_j |S[i,j]|` of the penalty
        // block, not just its diagonal `S[i,i]`. For an off-diagonal-dominant
        // smooth penalty (high-order difference / thin-plate) this tracks the
        // operator's true per-coordinate scale, where `S[i,i]` understates it.
        // For SPD `S` the row-sum is `≥ |S[i,i]| = S[i,i]`, so the result still
        // strictly dominates the plain-diagonal preconditioner and stays SPD.
        for (local_idx, global_idx) in (start..end).enumerate() {
            let row_abs_sum: f64 = s_lambda
                .row(local_idx)
                .iter()
                .map(|value| value.abs())
                .sum();
            diag[global_idx] += row_abs_sum;
        }
    }
    if diagonal_ridge > 0.0 {
        for value in &mut diag {
            *value += diagonal_ridge;
        }
    }
    if let Some(bundle) = joint_full_width
        && !bundle.is_empty()
    {
        bundle.add_diag(&mut diag);
    }
    diag.mapv(positive_joint_diagonal_entry)
}

pub(crate) fn log_joint_pcg_diagnostics(
    cycle: usize,
    total_p: usize,
    total_n: usize,
    preconditioner_diag: &Array1<f64>,
    info: &gam_linalg::utils::PcgSolveInfo,
) {
    let (diag_min, diag_max) = preconditioner_diag.iter().fold(
        (f64::INFINITY, 0.0_f64),
        |(min_value, max_value), &value| {
            if value.is_finite() {
                (min_value.min(value), max_value.max(value))
            } else {
                (min_value, max_value)
            }
        },
    );
    let diag_ratio = if diag_min.is_finite() && diag_min > 0.0 && diag_max.is_finite() {
        Some(diag_max / diag_min)
    } else {
        None
    };
    log::info!(
        "[PIRLS/blockwise joint-Newton/PCG] cycle={} p={} n={} iters={} rel_res={:.3e} res0={:.3e} res_final={:.3e} res_ratio={:.3e} ritz_cond~{} jacobi_diag_ratio~{}",
        cycle,
        total_p,
        total_n,
        info.iterations,
        info.relative_residual_norm,
        info.initial_residual_norm,
        info.final_residual_norm,
        info.residual_reduction,
        info.condition_estimate
            .map(|value| format!("{value:.3e}"))
            .unwrap_or_else(|| "NA".to_string()),
        diag_ratio
            .map(|value| format!("{value:.3e}"))
            .unwrap_or_else(|| "NA".to_string()),
    );
}

pub(crate) fn add_joint_penalty_to_matrix(
    matrix: &mut Array2<f64>,
    ranges: &[(usize, usize)],
    s_lambdas: &[Array2<f64>],
    diagonal_ridge: f64,
    joint_full_width: Option<&gam_problem::JointPenaltyBundle>,
) {
    for (b, s_lambda) in s_lambdas.iter().enumerate() {
        let (start, end) = ranges[b];
        let mut block = matrix.slice_mut(s![start..end, start..end]);
        block += s_lambda;
    }
    if diagonal_ridge > 0.0 {
        for d in 0..matrix.nrows() {
            matrix[[d, d]] += diagonal_ridge;
        }
    }
    if let Some(bundle) = joint_full_width
        && !bundle.is_empty()
    {
        bundle.add_to_matrix(matrix);
    }
}

pub(crate) fn flatten_state_betas(
    states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
) -> Array1<f64> {
    let total = specs.iter().map(|s| s.design.ncols()).sum::<usize>();
    let mut beta = Array1::<f64>::zeros(total);
    let ranges = block_param_ranges(specs);
    for (b, (start, end)) in ranges.into_iter().enumerate() {
        beta.slice_mut(ndarray::s![start..end])
            .assign(&states[b].beta);
    }
    beta
}

pub(crate) fn set_states_from_flat_beta(
    states: &mut [ParameterBlockState],
    specs: &[ParameterBlockSpec],
    beta_flat: &Array1<f64>,
) -> Result<(), String> {
    let ranges = block_param_ranges(specs);
    let total = ranges.last().map(|(_, e)| *e).unwrap_or(0);
    if beta_flat.len() != total {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: format!(
                "flat beta length mismatch: got {}, expected {}",
                beta_flat.len(),
                total
            ),
        }
        .into());
    }
    for (b, (start, end)) in ranges.into_iter().enumerate() {
        states[b]
            .beta
            .assign(&beta_flat.slice(ndarray::s![start..end]).to_owned());
    }
    Ok(())
}

pub(crate) fn synchronized_states_from_flat_beta<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &[ParameterBlockState],
    beta_flat: &Array1<f64>,
) -> Result<Vec<ParameterBlockState>, String> {
    let mut synced = states.to_vec();
    set_states_from_flat_beta(&mut synced, specs, beta_flat)?;
    refresh_all_block_etas(family, specs, &mut synced)?;
    Ok(synced)
}

/// Inf-norm of the penalized stationarity residual with valid KKT multipliers
/// projected out at active linear constraints.
///
/// For a linearly constrained convex quadratic with constraints `Aβ ≥ b`,
/// the KKT conditions at β̂ read
///
///   S·β̂ − ∇ℓ(β̂) = A_activeᵀ λ
///   Aβ̂ − b ≥ 0
///   λ ≥ 0
///   λᵢ(Aᵢβ̂ − bᵢ) = 0
///
/// The residual component represented by nonnegative active multipliers is
/// therefore not a convergence defect. This helper removes that normal-cone
/// component before taking the inf-norm. Axis-aligned lower bounds are just a
/// special case; coupled derivative-guard rows must use the same KKT geometry.
///
/// `known_active_rows`, when provided, seeds the working set with the QP
/// solver's authoritative active rows. Trust-region damping and finite
/// precision can leave the committed β with row slacks slightly above the slack
/// tolerance even though the QP identified the row as binding; slack-based
/// detection alone then misses the row and leaves its Lagrange-multiplier mass
/// in the projected residual. Seeding from the QP's active set is exact; the
/// non-negative-multiplier iteration below then removes any seeded row whose
/// least-squares multiplier turns out to be strictly negative, so the union
/// of (QP active) ∪ (slack-detected) never declares false convergence.
pub(crate) fn projected_stationarity_inf_norm(
    residual: &Array1<f64>,
    beta: &Array1<f64>,
    constraints: Option<&LinearInequalityConstraints>,
    known_active_rows: Option<&[usize]>,
) -> f64 {
    assert_eq!(residual.len(), beta.len());
    let raw_inf = residual.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let Some(constraints) = constraints else {
        return raw_inf;
    };
    projected_linear_constraint_stationarity_inf_norm(
        residual,
        beta,
        constraints,
        known_active_rows,
    )
    .unwrap_or(raw_inf)
}

pub(crate) fn projected_linear_constraint_stationarity_inf_norm(
    residual: &Array1<f64>,
    beta: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
    known_active_rows: Option<&[usize]>,
) -> Option<f64> {
    let projected = projected_linear_constraint_stationarity_vector(
        residual,
        beta,
        constraints,
        known_active_rows,
    )?;
    let primal_violation = linear_constraint_primal_violation(beta, constraints)?;
    Some(
        projected
            .iter()
            .fold(0.0_f64, |acc, &v| acc.max(v.abs()))
            .max(primal_violation),
    )
}

pub(crate) fn linear_constraint_primal_violation(
    beta: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
) -> Option<f64> {
    if constraints.a.ncols() != beta.len() || constraints.a.nrows() != constraints.b.len() {
        return None;
    }
    let mut primal_violation = 0.0_f64;
    for row in 0..constraints.a.nrows() {
        if constraints.b[row] == f64::NEG_INFINITY {
            continue;
        }
        if !constraints.b[row].is_finite() {
            return None;
        }
        let value = constraints.a.row(row).dot(beta);
        let slack = value - constraints.b[row];
        if !slack.is_finite() {
            return None;
        }
        primal_violation = primal_violation.max((-slack).max(0.0));
    }
    Some(primal_violation)
}

pub fn projected_linear_constraint_stationarity_vector(
    residual: &Array1<f64>,
    beta: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
    known_active_rows: Option<&[usize]>,
) -> Option<Array1<f64>> {
    let p = beta.len();
    if residual.len() != p
        || constraints.a.ncols() != p
        || constraints.a.nrows() != constraints.b.len()
    {
        return None;
    }
    let n_rows = constraints.a.nrows();
    // Union the slack-detected active rows with the optional QP-supplied
    // hint. Using a boolean membership table preserves a canonical row order
    // (matching the constraint matrix) so the rank-reduction below is
    // deterministic across calls.
    let mut in_active = vec![false; n_rows];
    if let Some(hint) = known_active_rows {
        for &row in hint {
            if row < n_rows && constraints.b[row].is_finite() {
                in_active[row] = true;
            }
        }
    }
    for row in 0..n_rows {
        if constraints.b[row] == f64::NEG_INFINITY {
            continue;
        }
        if !constraints.b[row].is_finite() {
            return None;
        }
        let a_row = constraints.a.row(row);
        let value = a_row.dot(beta);
        let slack = value - constraints.b[row];
        if !slack.is_finite() {
            return None;
        }
        if in_active[row] {
            continue;
        }
        // Active-row inclusion band for the stationarity-residual cone projection.
        // A constraint binding at the constrained optimum carries a Lagrange
        // multiplier whose mass IS the stationarity residual (`r = A_activeᵀ λ`,
        // λ >= 0); to project it out, every genuinely tight row must be a candidate.
        // The constrained QP only reports rows it drove tight during a
        // non-degenerate step, so monotone derivative-guard rows tight at the
        // optimum but never explicitly stepped sit just above the old `1e-6·scale`
        // band, get excluded, and leave the multiplier unresolved — tripping the
        // `active_set_incomplete` refusal on an exactly constrained-stationary
        // iterate (gam#797 survival time block). Widen the band so every near-tight
        // row is a CANDIDATE; over-inclusion is safe because the downstream NNLS
        // (`project_stationarity_residual_on_constraint_cone`) assigns λ = 0 to any
        // candidate carrying no multiplier mass, so a non-binding row cannot
        // spuriously shrink the residual.
        let scale = value.abs().max(constraints.b[row].abs()).max(1.0);
        let beta_inf = beta
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max)
            .max(1.0);
        let row_norm1 = a_row.iter().map(|v| v.abs()).sum::<f64>().max(1.0);
        // A row that is mathematically binding can appear a small positive
        // distance inside the feasible cone after repeated dense/spectral
        // Newton projections on a flat baseline-hazard valley: the objective is
        // insensitive along that direction, so round-off in the derivative-basis
        // coordinates dominates the true slack.  The active-set QP reports only
        // rows it explicitly pivoted on, so the KKT residual projection must also
        // recover these numerically-pinned rows from primal slack.  Use a
        // coefficient-space slack band, scaled by the row norm and coefficient
        // magnitude, not just by `Aβ` (which is exactly zero for monotone
        // derivative constraints with `b=0`).  Over-inclusion is safe because the
        // downstream nonnegative cone projection assigns λ=0 to rows that do not
        // carry multiplier mass; under-inclusion leaves a genuine multiplier in
        // the residual and falsely reports `active_set_incomplete` (#1793/#1040).
        let coordinate_slack_tol = 5e-3 * row_norm1 * beta_inf + 1e-8;
        let active_tol = (1e-3 * scale + 1e-8).max(coordinate_slack_tol);
        if slack <= active_tol {
            in_active[row] = true;
        }
    }
    let active_rows: Vec<usize> = (0..n_rows).filter(|&row| in_active[row]).collect();
    if active_rows.is_empty() {
        return Some(residual.clone());
    }

    let mut a_active = Array2::<f64>::zeros((active_rows.len(), p));
    for (pos, &row) in active_rows.iter().enumerate() {
        a_active.row_mut(pos).assign(&constraints.a.row(row));
    }
    project_stationarity_residual_on_constraint_cone(residual, &a_active)
        .map(|(projected, _)| projected)
}

pub(crate) fn exact_newton_joint_stationarity_inf_norm<F: CustomFamily + ?Sized>(
    family: &F,
    specs: &[ParameterBlockSpec],
    eval: &FamilyEvaluation,
    states: &[ParameterBlockState],
    s_lambdas: &[Array2<f64>],
    ridge: f64,
    ridge_policy: RidgePolicy,
    block_active_sets: Option<&[Option<Vec<usize>>]>,
) -> Result<Option<f64>, String> {
    if eval.blockworking_sets.len() != states.len() || states.len() != s_lambdas.len() {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: "exact-newton joint stationarity check: block dimension mismatch".to_string(),
        }
        .into());
    }
    if specs.len() != states.len() {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: "exact-newton joint stationarity check: spec/state count mismatch".to_string(),
        }
        .into());
    }
    if let Some(sets) = block_active_sets
        && sets.len() != states.len()
    {
        return Err(CustomFamilyError::DimensionMismatch { reason: format!(
            "exact-newton joint stationarity check: active-set count mismatch, got {}, expected {}",
            sets.len(),
            states.len()
        ) }.into());
    }

    let block_constraints = collect_block_linear_constraints(family, states, specs)?;
    let mut inf_norm = 0.0_f64;
    for b in 0..states.len() {
        let gradient = match &eval.blockworking_sets[b] {
            // For exact-Newton families the block score is ∇ log L with respect
            // to that block, while the penalized negative objective is
            //
            //   Q(beta, rho) = -log L(beta) + 0.5 beta^T P_mode(rho) beta,
            //
            // where `P_mode` includes the rho-independent stabilization ridge
            // exactly when that ridge participates in the quadratic objective.
            //
            // The coupled first-order condition is therefore
            //
            //   ∇Q = -∇ log L + P beta = 0.
            //
            // So the exact penalized stationarity residual for block b is
            //
            //   r_b = P_mode,b * beta_b - gradient_b.
            //
            // For blocks with simple lower-bound constraints (e.g. I-spline
            // monotone time coefficients, monotone wiggle coefficients) the
            // residual on an active-bound coordinate is the KKT multiplier
            // λ_j ≥ 0 rather than a convergence defect; the projection in
            // `projected_stationarity_inf_norm` drops those entries so the
            // inf-norm measures only the free-set residual that must be
            // driven to zero. Using only coordinate step size or an
            // unprojected norm can declare convergence too early OR fail to
            // ever declare convergence at a constrained optimum.
            BlockWorkingSet::ExactNewton { gradient, .. } => gradient,
            _ => return Ok(None),
        };
        let mut residual = s_lambdas[b].dot(&states[b].beta) - gradient;
        if ridge_policy.include_quadratic_penalty && ridge > 0.0 {
            residual += &states[b].beta.mapv(|v| ridge * v);
        }
        let block_active_hint = block_active_sets
            .and_then(|sets| sets.get(b))
            .and_then(|opt| opt.as_deref());
        let block_inf = projected_stationarity_inf_norm(
            &residual,
            &states[b].beta,
            block_constraints[b].as_ref(),
            block_active_hint,
        );
        inf_norm = inf_norm.max(block_inf);
    }
    Ok(Some(inf_norm))
}

pub(crate) fn exact_newton_joint_gradient_from_eval(
    eval: &FamilyEvaluation,
    specs: &[ParameterBlockSpec],
    states: &[ParameterBlockState],
) -> Result<Option<Array1<f64>>, String> {
    if eval.blockworking_sets.len() != specs.len() {
        return Err(format!(
            "exact-newton joint gradient extraction: family returned {} block working sets, expected {}",
            eval.blockworking_sets.len(),
            specs.len()
        ));
    }
    if states.len() != specs.len() {
        return Err(CustomFamilyError::DimensionMismatch { reason: format!(
            "exact-newton joint gradient extraction: state count {} does not match spec count {}",
            states.len(),
            specs.len()
        ) }.into());
    }
    let total_p = specs.iter().map(|spec| spec.design.ncols()).sum::<usize>();
    let mut gradient = Array1::<f64>::zeros(total_p);
    let mut offset = 0usize;
    for ((spec, work), state) in specs
        .iter()
        .zip(eval.blockworking_sets.iter())
        .zip(states.iter())
    {
        let width = spec.design.ncols();
        match work {
            BlockWorkingSet::ExactNewton {
                gradient: block_gradient,
                ..
            } => {
                if block_gradient.len() != width {
                    return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                        "exact-newton joint gradient extraction: block gradient length mismatch, got {}, expected {}",
                        block_gradient.len(),
                        width
                    ) }.into());
                }
                gradient
                    .slice_mut(ndarray::s![offset..offset + width])
                    .assign(block_gradient);
            }
            BlockWorkingSet::Diagonal {
                working_response,
                working_weights,
            } => {
                // Recover the per-block log-likelihood score from the IRLS
                // working set.  By construction of the IRLS pseudo-response
                //
                //     z_i = η_i + (∂ℓ/∂η_i) / w_i,
                //
                // so the row score is `w_i (z_i − η_i)` and the
                // coefficient-space score is
                //
                //     ∇_β_b log L = X_b^T (w ⊙ (z − η)).
                //
                // Without this branch the joint-Newton path is unable to
                // assemble its RHS for families that emit Diagonal working
                // sets alongside an exact joint Hessian (e.g. Gaussian
                // location-scale): the inner fit returns non-converged, and
                // the outer evaluator falls into the nonconverged-result
                // branch and reports a zero outer gradient.
                let n = working_response.len();
                if working_weights.len() != n || state.eta.len() != n || spec.design.nrows() != n {
                    return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                        "exact-newton joint gradient extraction: diagonal working-set length mismatch (z={}, w={}, η={}, X_rows={})",
                        working_response.len(),
                        working_weights.len(),
                        state.eta.len(),
                        spec.design.nrows()
                    ) }.into());
                }
                let mut weighted = Array1::<f64>::zeros(n);
                for i in 0..n {
                    weighted[i] = working_weights[i] * (working_response[i] - state.eta[i]);
                }
                let block_gradient =
                    <DesignMatrix as LinearOperator>::apply_transpose(&spec.design, &weighted);
                if block_gradient.len() != width {
                    return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                        "exact-newton joint gradient extraction: diagonal block transpose length mismatch, got {}, expected {}",
                        block_gradient.len(),
                        width
                    ) }.into());
                }
                gradient
                    .slice_mut(ndarray::s![offset..offset + width])
                    .assign(&block_gradient);
            }
        }
        offset += width;
    }
    Ok(Some(gradient))
}

pub(crate) fn exact_newton_joint_stationarity_inf_norm_from_gradient(
    gradient: &Array1<f64>,
    states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    s_lambdas: &[Array2<f64>],
    ridge: f64,
    ridge_policy: RidgePolicy,
    block_constraints: &[Option<LinearInequalityConstraints>],
    block_active_sets: Option<&[Option<Vec<usize>>]>,
    // gam#979: per-coordinate simple lower bounds (`f64::NEG_INFINITY` where
    // unbounded, length = total joint p), from `extract_simple_lower_bounds` on
    // the joint constraints. Used to project out the KKT multipliers of ACTIVE
    // simple lower bounds — the box-bound analog of the linear-constraint
    // projection that `projected_stationarity_inf_norm` already does. Without it
    // the stationarity test on a `solve_quadratic_with_simple_lower_bounds`-
    // constrained block (survival monotone baseline hazard, monotone smooths)
    // reads the raw bound-multiplier mass (e.g. the 626 on `time_surface`) and
    // mis-refuses a genuinely-optimal constrained iterate. `None` ⇒ no box path
    // (byte-identical to the pre-fix / linear-constraint behaviour).
    joint_lower_bounds: Option<&Array1<f64>>,
) -> Result<f64, String> {
    if states.len() != specs.len() || states.len() != s_lambdas.len() {
        return Err(
            "exact-newton joint stationarity check from gradient: block dimension mismatch"
                .to_string(),
        );
    }
    if block_constraints.len() != states.len() {
        return Err(CustomFamilyError::DimensionMismatch { reason: format!(
            "exact-newton joint stationarity check from gradient: constraint count mismatch, got {}, expected {}",
            block_constraints.len(),
            states.len()
        ) }.into());
    }
    if let Some(sets) = block_active_sets
        && sets.len() != states.len()
    {
        return Err(CustomFamilyError::DimensionMismatch { reason: format!(
            "exact-newton joint stationarity check from gradient: active-set count mismatch, got {}, expected {}",
            sets.len(),
            states.len()
        ) }.into());
    }
    let total_p = specs.iter().map(|spec| spec.design.ncols()).sum::<usize>();
    if gradient.len() != total_p {
        return Err(CustomFamilyError::DimensionMismatch { reason: format!(
            "exact-newton joint stationarity check from gradient: joint gradient length mismatch, got {}, expected {}",
            gradient.len(),
            total_p
        ) }.into());
    }

    // Same KKT projection as `exact_newton_joint_stationarity_inf_norm`:
    // multipliers at active lower bounds are not convergence defects, so we
    // measure only the free-set residual. See `projected_stationarity_inf_norm`
    // for the tolerance choice and its parallel with `projected_gradient_norm`
    // in `pirls.rs`.
    //
    // The optional `block_active_sets` arrives from the joint-Newton inner
    // loop's `cached_active_sets` and carries the QP solver's authoritative
    // active rows per block. Threading it through is what makes the
    // stationarity test correctly fire at the constrained optimum: a damped
    // constrained step may commit β with row slacks slightly above the slack
    // tolerance even though the QP identified the rows as binding, and
    // slack-based detection alone then misses the rows and leaves the
    // Lagrange-multiplier mass in the residual.
    let mut inf_norm = 0.0_f64;
    let mut offset = 0usize;
    for b in 0..states.len() {
        let width = specs[b].design.ncols();
        let mut residual =
            s_lambdas[b].dot(&states[b].beta) - gradient.slice(ndarray::s![offset..offset + width]);
        if ridge_policy.include_quadratic_penalty && ridge > 0.0 {
            residual += &states[b].beta.mapv(|v| ridge * v);
        }
        // gam#979 box-bound (simple lower bound) KKT residual. `residual` here is
        // the objective gradient `r = Sβ − ∇ℓ`. The correct stationarity measure
        // for `β_j ≥ L_j` is the PROJECTED GRADIENT
        //     pg_j = β_j − Π_{[L_j,∞)}(β_j − r_j) = β_j − max(L_j, β_j − r_j),
        // which (a) is byte-identical to `r_j` on any coordinate whose gradient
        // step stays feasible (interior, or a bound that wants to be left), and
        // (b) collapses a VALID lower-bound multiplier to the mere distance to
        // the bound: at `β_j = L_j + ε` with `r_j ≥ 0` (gradient pushing INTO the
        // bound), `pg_j = min(r_j, ε) = ε → 0` as the (possibly damped) iterate
        // reaches `L_j`. This is why the certificate no longer mis-reads the huge
        // pushing-into-bound multiplier (the 626 on the survival monotone-hazard
        // `time_surface` coeff pinned near its ≥0 bound) as a stationarity defect
        // — the failure mode noted just above (slack-based detection missing a
        // damped binding row at `β = L + ε` with ε over tol). The SIGN CHECK is
        // INTRINSIC, not a separate slack test: a coordinate at its bound whose
        // `r_j < 0` (wants to INCREASE β_j, i.e. LEAVE the bound) has `β_j − r_j >
        // β_j ≥ L_j`, so `pg_j = r_j` is UNCHANGED and the cert still (correctly)
        // refuses a non-optimal point. Blocks with no simple lower bound skip this
        // and are byte-identical to before.
        if let Some(lowers) = joint_lower_bounds {
            for j in 0..width {
                let lower = lowers[offset + j];
                if !lower.is_finite() {
                    continue;
                }
                let beta_j = states[b].beta[j];
                residual[j] = beta_j - (beta_j - residual[j]).max(lower);
            }
        }
        let block_active_hint = block_active_sets
            .and_then(|sets| sets.get(b))
            .and_then(|opt| opt.as_deref());
        let block_inf = projected_stationarity_inf_norm(
            &residual,
            &states[b].beta,
            block_constraints[b].as_ref(),
            block_active_hint,
        );
        inf_norm = inf_norm.max(block_inf);
        offset += width;
    }
    Ok(inf_norm)
}

pub(crate) fn exact_newton_joint_stationarity_vector_from_gradient(
    gradient: &Array1<f64>,
    states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    s_lambdas: &[Array2<f64>],
    ridge: f64,
    ridge_policy: RidgePolicy,
) -> Result<Array1<f64>, String> {
    if states.len() != specs.len() || states.len() != s_lambdas.len() {
        return Err(
            "exact-newton joint stationarity vector from gradient: block dimension mismatch"
                .to_string(),
        );
    }
    let total_p = specs.iter().map(|spec| spec.design.ncols()).sum::<usize>();
    if gradient.len() != total_p {
        return Err(CustomFamilyError::DimensionMismatch { reason: format!(
            "exact-newton joint stationarity vector from gradient: joint gradient length mismatch, got {}, expected {}",
            gradient.len(),
            total_p
        ) }.into());
    }

    let mut residual = Array1::<f64>::zeros(total_p);
    let mut offset = 0usize;
    for b in 0..states.len() {
        let width = specs[b].design.ncols();
        let start = offset;
        let end = offset + width;
        let mut block = s_lambdas[b].dot(&states[b].beta) - gradient.slice(ndarray::s![start..end]);
        if ridge_policy.include_quadratic_penalty && ridge > 0.0 {
            block += &states[b].beta.mapv(|v| ridge * v);
        }
        residual.slice_mut(ndarray::s![start..end]).assign(&block);
        offset = end;
    }
    Ok(residual)
}

/// Compute `Σ_t λ_t (M⊗S_t) · β` — the full-width joint penalty's contribution
/// to the penalized stationarity condition — from the active `BlockwiseFitOptions`
/// joint-penalty bundle and the current block betas (stacked class-major).
///
/// Returns `None` when the options carry no joint penalty (every per-block-only
/// family), so the KKT-residual path stays byte-identical there. gam#1587/#561:
/// the multinomial centered penalty lives ONLY here, so without this term the
/// inner KKT residual omits the penalty entirely.
pub(crate) fn joint_penalty_stationarity_score(
    options: &BlockwiseFitOptions,
    specs: &[ParameterBlockSpec],
    states: &[ParameterBlockState],
) -> Option<Array1<f64>> {
    let bundle = options.joint_penalties.as_deref()?;
    if bundle.is_empty() {
        return None;
    }
    let total_p: usize = specs.iter().map(|spec| spec.design.ncols()).sum();
    let mut beta = Array1::<f64>::zeros(total_p);
    let mut offset = 0usize;
    for (spec, state) in specs.iter().zip(states.iter()) {
        let width = spec.design.ncols();
        if state.beta.len() == width && offset + width <= total_p {
            beta.slice_mut(ndarray::s![offset..offset + width])
                .assign(&state.beta);
        }
        offset += width;
    }
    let mut score = Array1::<f64>::zeros(total_p);
    bundle.add_apply_into(beta.view(), &mut score);
    Some(score)
}

pub(crate) fn exact_newton_joint_projected_stationarity_vector_from_gradient(
    gradient: &Array1<f64>,
    states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    s_lambdas: &[Array2<f64>],
    ridge: f64,
    ridge_policy: RidgePolicy,
    block_constraints: &[Option<LinearInequalityConstraints>],
    block_active_sets: Option<&[Option<Vec<usize>>]>,
    // gam#1587/#561: `Σ_t λ_t (M⊗S_t) · β` — the full-width joint penalty's
    // contribution to the penalized stationarity condition, in stacked
    // (class-major) coordinates over the whole `total_p` vector. Families whose
    // smoothing rides entirely on a JOINT penalty (multinomial: per-block
    // `s_lambdas` are empty) would otherwise report a KKT residual of
    // `−gradient` — which at the penalized optimum equals `Sλ_joint·β̂ ≠ 0` — a
    // large PHANTOM residual that (a) stops the inner solve from certifying on
    // the raw residual (it falls back to the decrement certificate) and (b)
    // drives a spurious IFT/KKT cost correction whose ρ-derivative desyncs the
    // outer REML gradient. Adding this term makes the residual the true
    // `∇penalized(β̂)`. `None` (no joint penalty) keeps every per-block-only
    // family byte-identical.
    joint_penalty_score: Option<&Array1<f64>>,
) -> Result<Array1<f64>, String> {
    if states.len() != specs.len()
        || states.len() != s_lambdas.len()
        || states.len() != block_constraints.len()
    {
        return Err(
            "exact-newton projected stationarity vector from gradient: block dimension mismatch"
                .to_string(),
        );
    }
    if let Some(sets) = block_active_sets
        && sets.len() != states.len()
    {
        return Err(CustomFamilyError::DimensionMismatch { reason: format!(
            "exact-newton projected stationarity vector from gradient: active-set count mismatch, got {}, expected {}",
            sets.len(),
            states.len()
        ) }.into());
    }
    let total_p = specs.iter().map(|spec| spec.design.ncols()).sum::<usize>();
    if gradient.len() != total_p {
        return Err(CustomFamilyError::DimensionMismatch { reason: format!(
            "exact-newton projected stationarity vector from gradient: joint gradient length mismatch, got {}, expected {}",
            gradient.len(),
            total_p
        ) }.into());
    }
    if let Some(js) = joint_penalty_score
        && js.len() != total_p
    {
        return Err(CustomFamilyError::DimensionMismatch { reason: format!(
            "exact-newton projected stationarity vector from gradient: joint penalty score length mismatch, got {}, expected {}",
            js.len(),
            total_p
        ) }.into());
    }

    let mut residual = Array1::<f64>::zeros(total_p);
    let mut offset = 0usize;
    for b in 0..states.len() {
        let width = specs[b].design.ncols();
        let start = offset;
        let end = offset + width;
        let mut block = s_lambdas[b].dot(&states[b].beta) - gradient.slice(ndarray::s![start..end]);
        if let Some(js) = joint_penalty_score {
            block += &js.slice(ndarray::s![start..end]);
        }
        if ridge_policy.include_quadratic_penalty && ridge > 0.0 {
            block += &states[b].beta.mapv(|v| ridge * v);
        }
        if let Some(constraints) = block_constraints[b].as_ref() {
            let block_active_hint = block_active_sets
                .and_then(|sets| sets.get(b))
                .and_then(|opt| opt.as_deref());
            match projected_linear_constraint_stationarity_vector(
                &block,
                &states[b].beta,
                constraints,
                block_active_hint,
            ) {
                Some(projected) => block = projected,
                None => {
                    // Cone projection can only SHRINK the residual (it removes
                    // nonnegative multiplier mass on active rows), so a failed
                    // projection degrades to the conservative unprojected
                    // residual — the convergence test gets harder, never
                    // easier — instead of rejecting the whole seed (#1025:
                    // 'failed to project block 0' killed an otherwise-healthy
                    // competing-risks seed outright).
                    log::warn!(
                        "exact-newton projected stationarity vector: cone projection failed \
                         for block {b}; using the conservative unprojected residual"
                    );
                }
            }
        }
        residual.slice_mut(ndarray::s![start..end]).assign(&block);
        offset = end;
    }
    Ok(residual)
}

/// Build the free-space-projected KKT residual for the IFT correction.
///
/// The active set passed via `block_active_sets` is consumed by the inner
/// projection so the returned vector lies in `range(I − P_normal_cone)`. The
/// [`gam_solve::model_types::ProjectedKktResidual`] return type makes
/// that invariant visible at every call site — callers cannot forget to
/// project, and `reml/unified.rs` cannot accidentally accept an unprojected
/// vector.
pub(crate) fn exact_newton_joint_kkt_residual_for_ift<F: CustomFamily + ?Sized>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &[ParameterBlockState],
    s_lambdas: &[Array2<f64>],
    ridge: f64,
    ridge_policy: RidgePolicy,
    block_active_sets: Option<&[Option<Vec<usize>>]>,
    joint_penalty_score: Option<&Array1<f64>>,
) -> Result<Option<ProjectedKktResidual>, String> {
    let eval = family.evaluate(states)?;
    let Some(gradient) = exact_newton_joint_gradient_from_eval(&eval, specs, states)? else {
        return Ok(None);
    };
    let block_constraints = collect_block_linear_constraints(family, states, specs)?;
    exact_newton_joint_projected_kkt_residual_for_ift_from_gradient(
        &gradient,
        specs,
        states,
        s_lambdas,
        ridge,
        ridge_policy,
        &block_constraints,
        block_active_sets,
        joint_penalty_score,
    )
}

pub(crate) fn exact_newton_joint_kkt_residual_for_ift_from_cached_gradient<
    F: CustomFamily + ?Sized,
>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &[ParameterBlockState],
    s_lambdas: &[Array2<f64>],
    ridge: f64,
    ridge_policy: RidgePolicy,
    block_active_sets: Option<&[Option<Vec<usize>>]>,
    cached_gradient: Option<&Array1<f64>>,
    joint_penalty_score: Option<&Array1<f64>>,
) -> Result<Option<ProjectedKktResidual>, String> {
    if let Some(gradient) = cached_gradient {
        let block_constraints = collect_block_linear_constraints(family, states, specs)?;
        return exact_newton_joint_projected_kkt_residual_for_ift_from_gradient(
            gradient,
            specs,
            states,
            s_lambdas,
            ridge,
            ridge_policy,
            &block_constraints,
            block_active_sets,
            joint_penalty_score,
        );
    }
    exact_newton_joint_kkt_residual_for_ift(
        family,
        specs,
        states,
        s_lambdas,
        ridge,
        ridge_policy,
        block_active_sets,
        joint_penalty_score,
    )
}

pub(crate) fn exact_newton_joint_projected_kkt_residual_for_ift_from_gradient(
    gradient: &Array1<f64>,
    specs: &[ParameterBlockSpec],
    states: &[ParameterBlockState],
    s_lambdas: &[Array2<f64>],
    ridge: f64,
    ridge_policy: RidgePolicy,
    block_constraints: &[Option<LinearInequalityConstraints>],
    block_active_sets: Option<&[Option<Vec<usize>>]>,
    joint_penalty_score: Option<&Array1<f64>>,
) -> Result<Option<ProjectedKktResidual>, String> {
    let residual = exact_newton_joint_projected_stationarity_vector_from_gradient(
        gradient,
        states,
        specs,
        s_lambdas,
        ridge,
        ridge_policy,
        block_constraints,
        block_active_sets,
        joint_penalty_score,
    )?;
    if residual.iter().all(|v| v.is_finite()) {
        Ok(Some(ProjectedKktResidual::from_active_projected(residual)))
    } else {
        // Surface this clearly: a non-finite projected residual reaches the
        // unified evaluator as `kkt_residual = None`, which then makes the
        // envelope-consistency tripwire fire with "no projected residual"
        // as the suspected cause. Emit the count and magnitude so the
        // failure is diagnosable from a single log line.
        let nan_count = residual.iter().filter(|v| v.is_nan()).count();
        let inf_count = residual.iter().filter(|v| v.is_infinite()).count();
        let finite_max = residual
            .iter()
            .filter(|v| v.is_finite())
            .copied()
            .map(f64::abs)
            .fold(0.0_f64, f64::max);
        log::warn!(
            "[exact-newton kkt-residual projection] dropping projected KKT residual to None: \
             len={} nan_count={} inf_count={} finite_max={:.3e}. The unified evaluator will \
             treat this convergent path as if no residual were available, which silently \
             disables the IFT correction and can trip the envelope-gradient consistency check \
             on near-singular H. Investigate which block produced the non-finite entry.",
            residual.len(),
            nan_count,
            inf_count,
            finite_max,
        );
        Ok(None)
    }
}

pub(crate) fn compute_joint_covariance<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &[ParameterBlockState],
    per_block_log_lambdas: &[Array1<f64>],
    options: &BlockwiseFitOptions,
) -> Result<Array2<f64>, String> {
    let ranges = block_param_ranges(specs);
    let total = ranges.last().map(|(_, e)| *e).unwrap_or(0);
    let Some(mut h) = exact_newton_joint_hessian_symmetrized(
        family,
        states,
        specs,
        total,
        "joint exact-newton Hessian shape mismatch in covariance",
    )?
    else {
        return Err(
            "joint covariance requires an exact analytic Hessian; objective perturbation is forbidden"
                .to_string(),
        );
    };
    for (b, spec) in specs.iter().enumerate() {
        let (start, end) = ranges[b];
        let lambdas = per_block_log_lambdas[b].mapv(f64::exp);
        let mut s_lambda = Array2::<f64>::zeros((end - start, end - start));
        for (k, s) in spec.penalties.iter().enumerate() {
            s.add_scaled_to(lambdas[k], &mut s_lambda);
        }
        h.slice_mut(ndarray::s![start..end, start..end])
            .scaled_add(1.0, &s_lambda);
    }
    // gam#1587/#561: families whose smoothing is carried by a full-width JOINT
    // penalty (the multinomial centered `Σ_t λ_t (M ⊗ S_t)` metric) leave their
    // per-block penalty lists empty — every block above contributes nothing — so
    // the exported posterior precision MUST also add the joint contribution
    // `Σ_t exp(ρ_t) S_t` at the SAME selected `ρ_t` the inner solve used. Without
    // this the reported covariance is the UNPENALIZED `H_lik⁻¹` (standard errors
    // silently too wide) and the EDF trace `tr(H⁻¹ S_λ)` reads zero (no smoothing
    // spent), even though `fit_custom_family` threads the joint bundle through
    // `options` for exactly this reason. A no-op for every per-block-only family
    // (`joint_penalties` is `None`).
    if let Some(bundle) = options.joint_penalties.as_deref()
        && !bundle.is_empty()
    {
        bundle.add_to_matrix(&mut h);
    }
    symmetrize_dense_in_place(&mut h);
    // #748 + audit-41: a Laplace posterior covariance exists only when the
    // posterior precision `H + S_λ` is strictly positive definite AT THE
    // CONVERGED OPTIMUM. Indefinite means the mode is not a maximum;
    // singular means the posterior is IMPROPER along every flat direction
    // (unbounded variance). Neither can be repaired at reporting time: a
    // δ-ridge inverse `(H + S_λ + δI)⁻¹` reports an arbitrary ridge-bounded
    // variance, and a positive-eigenspace pseudo-inverse reports exactly
    // ZERO variance for the very direction the model does not identify
    // (H = diag(1,0) → Σ = diag(1,0)). Both fabrications bias every
    // downstream standard error, so both regimes surface as errors with the
    // spectrum diagnostics; the remedy is at the model (constraint, penalty,
    // canonicalisation), not in the covariance report.
    let p = h.nrows();
    let (evals, evecs) = FaerEigh::eigh(&h, Side::Lower)
        .map_err(|e| format!("joint posterior-precision eigendecomposition failed: {e}"))?;
    let max_abs_eval = evals.iter().fold(0.0_f64, |acc, &ev| acc.max(ev.abs()));
    let eps_np = f64::EPSILON * (p as f64) * (p as f64);
    let tol = (10.0 * eps_np * max_abs_eval).max(100.0 * f64::EPSILON);
    if let Some(&min_eval) = evals
        .iter()
        .filter(|&&ev| ev < -tol)
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    {
        let below = evals.iter().filter(|&&ev| ev < -tol).count();
        return Err(format!(
            "joint posterior precision H + S_λ is non-PD at the converged optimum \
             ({below} eigenvalue(s) below -tol, min(λ)={min_eval:.6e}, \
             max|λ|={max_abs_eval:.6e}, tol={tol:.6e}); the mode is not a strict posterior \
             maximum, so the reported covariance would be meaningless — fit-quality failure \
             surfaced instead of δ-ridge masking (gam#748)"
        ));
    }
    let flat = evals.iter().filter(|&&ev| ev <= tol).count();
    if flat > 0 {
        return Err(format!(
            "joint posterior precision H + S_λ is singular at the converged optimum: {flat} \
             flat direction(s) (max|λ|={max_abs_eval:.6e}, tol={tol:.6e}). The posterior is \
             improper along a flat direction — its variance is unbounded, so no finite \
             covariance entry is honest there (a pseudo-inverse would claim zero variance, a \
             δ-ridge an arbitrary finite one). Identify the direction via a constraint, \
             penalty, or canonicalisation instead"
        ));
    }
    // Strictly SPD: exact inverse through the eigenbasis, Σ = V diag(1/λ) Vᵀ.
    let mut cov = Array2::<f64>::zeros((p, p));
    for (k, &ev) in evals.iter().enumerate() {
        let inv_ev = 1.0 / ev;
        for i in 0..p {
            let vi = evecs[[i, k]];
            for j in 0..p {
                cov[[i, j]] += inv_ev * vi * evecs[[j, k]];
            }
        }
    }
    symmetrize_dense_in_place(&mut cov);
    Ok(cov)
}

pub(crate) fn compute_joint_covariance_required<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &[ParameterBlockState],
    per_block_log_lambdas: &[Array1<f64>],
    options: &BlockwiseFitOptions,
) -> Result<Option<Array2<f64>>, CustomFamilyError> {
    if !options.compute_covariance {
        return Ok(None);
    }
    compute_joint_covariance(family, specs, states, per_block_log_lambdas, options)
        .map(Some)
        .map_err(|e| CustomFamilyError::InvalidInput {
            context: "compute_joint_covariance_required",
            reason: format!("joint covariance computation failed: {e}"),
        })
}

/// Compute joint working-set geometry at convergence for ALO diagnostics.
pub(crate) fn compute_joint_geometry<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &[ParameterBlockState],
    per_block_log_lambdas: &[Array1<f64>],
    options: &BlockwiseFitOptions,
) -> Result<Option<FitGeometry>, String> {
    if specs.len() != per_block_log_lambdas.len() {
        return Ok(None);
    }
    if specs.len() == 1 {
        let eval = family.evaluate(states).ok();
        let Some(eval) = eval else {
            return Ok(None);
        };
        let spec = &specs[0];
        let lambdas = per_block_log_lambdas[0].mapv(f64::exp);
        // The penalized joint Hessian `H_pen = H_lik + Σ_k λ_k S_k` is the exact
        // mgcv quantity the trace edf `p − Σ_k λ_k·tr(H_pen⁻¹ S_k)` consumes. Two
        // single-block working-set shapes reach here:
        //
        // * `Diagonal` — IRLS/GLM families expose only the diagonal working
        //   weights, so the likelihood curvature is reconstructed as the
        //   Gauss–Newton gram `XᵀWX`.
        // * `ExactNewton` — coefficient-space exact-curvature families (CTN
        //   transformation-normal, …) already carry the dense negative
        //   log-likelihood Hessian `−∇²log L = H_lik` directly. Materialize it
        //   and add the penalties, so these families report inference / total
        //   edf instead of dropping geometry (and therefore inference) for the
        //   whole fit (#720).
        let (mut h, working_weights, working_response) = match eval.blockworking_sets.as_slice() {
            [
                BlockWorkingSet::Diagonal {
                    working_response,
                    working_weights,
                },
            ] => {
                let Some(h) = spec
                    .design
                    .xt_diag_x_signed_op(SignedWeightsView::from_array(working_weights))
                    .ok()
                else {
                    return Ok(None);
                };
                (h, working_weights.clone(), working_response.clone())
            }
            [BlockWorkingSet::ExactNewton { hessian, .. }] => {
                let h = hessian.to_dense();
                if h.nrows() != spec.design.ncols() || h.ncols() != spec.design.ncols() {
                    return Ok(None);
                }
                // The exact-Newton block carries no IRLS pseudo-data; the
                // trace edf reads only the penalized Hessian, and the
                // downstream IRLS covariance path is unused for these
                // families (they report dispersion = 1). Match the joint
                // multi-block branch's zero-length convention.
                let working_len = states.first().map(|state| state.eta.len()).unwrap_or(0);
                (h, Array1::zeros(working_len), Array1::zeros(working_len))
            }
            _ => return Ok(None),
        };
        for (k, s) in spec.penalties.iter().enumerate() {
            let s_dense = s.as_dense_cow();
            h.scaled_add(lambdas[k], &*s_dense);
        }
        // gam#1587/#561: add the full-width JOINT penalty (the multinomial
        // centered `Σ_t λ_t (M ⊗ S_t)`) at the selected `ρ_t` so the exported
        // geometry's penalized Hessian matches the inner-converged operator and
        // the trace EDF `tr(H⁻¹ S_λ)` is non-zero. No-op for per-block-only
        // families. (Single-block joint penalties are unusual but handled for
        // symmetry with the multi-block branch.)
        if let Some(bundle) = options.joint_penalties.as_deref()
            && !bundle.is_empty()
            && h.nrows() == bundle.specs.first().map(|s| s.dim()).unwrap_or(h.nrows())
        {
            bundle.add_to_matrix(&mut h);
        }
        // Exact-Newton families may return a Hessian assembled from directional
        // callbacks whose off-diagonal entries differ by floating-point order
        // or, for pseudo-Laplace tests, by a deliberately non-symmetric input
        // that is accepted only after symmetrization. Export the same symmetric
        // penalized Hessian used by the determinant/covariance path instead of
        // letting result assembly reject an otherwise valid fit geometry.
        symmetrize_dense_in_place(&mut h);
        return Ok(Some(FitGeometry {
            penalized_hessian: h.into(),
            working_weights,
            working_response,
        }));
    }

    let requires_explicit_joint_hessian = specs.iter().enumerate().any(|(idx, spec)| {
        custom_family_block_role(&spec.name, idx, specs.len()) == gam_problem::BlockRole::LinkWiggle
    });
    let total_p: usize = specs.iter().map(|spec| spec.design.ncols()).sum();
    let Some(mut h) = exact_newton_joint_hessian_symmetrized(
        family,
        states,
        specs,
        total_p,
        "compute_joint_geometry",
    )?
    else {
        if requires_explicit_joint_hessian {
            return Err(
                "link-wiggle fits require an exact explicit joint Hessian for posterior sampling"
                    .to_string(),
            );
        }
        return Ok(None);
    };
    let ranges = block_param_ranges(specs);
    for (block_idx, spec) in specs.iter().enumerate() {
        let Some(block_log_lambdas) = per_block_log_lambdas.get(block_idx) else {
            return Ok(None);
        };
        let lambdas = block_log_lambdas.mapv(f64::exp);
        if lambdas.len() != spec.penalties.len() {
            return Ok(None);
        }
        let (start, end) = ranges[block_idx];
        let block_dim = end - start;
        for (penalty_idx, penalty) in spec.penalties.iter().enumerate() {
            let scale = lambdas[penalty_idx];
            if scale == 0.0 {
                continue;
            }
            let dense = penalty.as_dense_cow();
            if dense.nrows() == block_dim && dense.ncols() == block_dim {
                h.slice_mut(ndarray::s![start..end, start..end])
                    .scaled_add(scale, &*dense);
            } else if dense.nrows() == total_p && dense.ncols() == total_p {
                h.scaled_add(scale, &*dense);
            } else {
                return Ok(None);
            }
        }
    }
    // gam#1587/#561: add the full-width JOINT penalty `Σ_t exp(ρ_t) S_t` at the
    // selected `ρ_t`. The multinomial centered metric carries ALL of a fit's
    // smoothing here (its per-block penalty lists are empty), so without this the
    // exported geometry is the unpenalized likelihood Hessian and the trace EDF
    // reads as the full coefficient count (no smoothing spent). No-op for the
    // per-block-only families (`joint_penalties` is `None`).
    if let Some(bundle) = options.joint_penalties.as_deref()
        && !bundle.is_empty()
    {
        bundle.add_to_matrix(&mut h);
    }
    let working_len = states.first().map(|state| state.eta.len()).unwrap_or(0);
    Ok(Some(FitGeometry {
        penalized_hessian: h.into(),
        working_weights: Array1::zeros(working_len),
        working_response: Array1::zeros(working_len),
    }))
}

pub(crate) fn joint_penalty_subspace_trace_parts(
    h_joint_unpen: &JointHessianSource,
    ranges: &[(usize, usize)],
    s_lambdas: &[Array2<f64>],
    total: usize,
    hessian_diagonal_ridge: f64,
    // Pre-scaled outer-REML Jeffreys curvature (already multiplied by
    // `rho_curvature_scale` to live in the same scaled space as `s_lambdas`).
    // Folded into `M = H + Sλ (+ H_Φ)` so the projected logdet AND its trace
    // kernel `(H+Sλ+H_Φ)⁺` match the Jeffreys-augmented operator the LAML score
    // runs on. `None` ⇒ byte-identical released projected logdet.
    scaled_jeffreys_hphi: Option<&Array2<f64>>,
    // gam#1587/#561: the full-width centered joint penalty `Σ_t λ_t (M⊗S_t)`,
    // already scaled into the same space as `s_lambdas`. For the multinomial
    // family ALL smoothing rides on this joint penalty (the per-block
    // `s_lambdas` are empty), so without folding it into both the structural-
    // null rank gate AND the materialized `M = H + Sλ` the projected logdet
    // collapses to `(0.0, None)` — the cost then drops `½log|H_pen|` entirely
    // (correction `= −hop.logdet()`) while the analytic gradient keeps its
    // `½tr(H⁻¹∂H)` derivative, desyncing value and gradient. `None` ⇒ no joint
    // penalty (every per-block-only family) keeps this byte-identical.
    joint_penalty: Option<&Array2<f64>>,
) -> Result<(f64, Option<PenaltySubspaceTrace>), String> {
    if total == 0 {
        return Ok((0.0, None));
    }

    // Structural-null gate: with no positive penalty eigenvalue there is no
    // `log|Sλ|₊` term in the LAML ratio, hence no Hessian-side correction to
    // pair with it — the caller keeps the operator's own logdet untouched.
    // (The kernel itself no longer uses the Sλ eigenvectors: since #901 it is
    // the full spectral `M⁺`, built from M's own eigendecomposition below.)
    let mut s_lambda = Array2::<f64>::zeros((total, total));
    add_joint_penalty_to_matrix(&mut s_lambda, ranges, s_lambdas, 0.0, None);
    if let Some(joint) = joint_penalty {
        s_lambda += joint;
    }
    let s_evals = s_lambda
        .eigh(Side::Lower)
        .map_err(|e| format!("joint penalty subspace eigendecomposition failed: {e}"))?
        .0;
    let s_threshold = positive_eigenvalue_threshold(s_evals.as_slice().unwrap());
    let rank = (0..total).filter(|&j| s_evals[j] > s_threshold).count();
    if rank == 0 {
        return Ok((0.0, None));
    }

    // ── REML log|H + Sλ|₊ and its trace kernel over the FULL identifiable
    //    subspace range(H + Sλ) ──────────────────────────────────────────────
    //
    // The REML penalty-determinant term is `½ log|H + Sλ|₊`, and its ρ-gradient
    // is the trace `½ tr((H + Sλ)⁻¹ ∂Sλ/∂ρ)`. BOTH must be taken over
    // range(H + Sλ) — the full identifiable subspace — not over range(Sλ).
    //
    // The previous code projected onto range(Sλ): it computed
    // `log|U_Sᵀ(H+Sλ)U_S| = log|M_rr|` and the kernel `M_rr⁻¹`. That DROPS the
    // determinant of the penalty-null block `M_kk = U_kᵀ H U_k` (on ker(Sλ), Sλ
    // vanishes, so this is pure likelihood curvature) and the Schur coupling
    // between the two. `M_kk` is the unpenalized polynomial trend; on a
    // near-collinear design (admixture-cline PCs at small n) its curvature is
    // large and GROWS as the smooth part is shrunk. Omitting it from
    // `log|H+Sλ|` while `½ log|Sλ|₊` is correctly taken over range(Sλ) makes
    // the ρ-derivative of the REML criterion inconsistent in the marginal
    // block: the outer optimizer drives that block's λ → ∞ chasing a
    // flat-increasing profile (gh#752), the coupled inner joint-Newton can no
    // longer certify stationarity on the now-ill-conditioned trend, and the
    // envelope-theorem outer gradient — valid only at a stationary β̂ — diverges
    // on the coupled (logslope) block while the objective stalls, so ARC never
    // reaches a KKT point.
    //
    // The correct generalized determinant (mgcv's treatment) takes both terms
    // over range(H + Sλ): identical to the ordinary log-det / inverse when
    // H + Sλ is non-singular (the well-posed case), and dropping only the truly
    // unidentified directions ker(H) ∩ ker(Sλ) when it is singular — exactly the
    // directions `½ log|Sλ|₊` also omits, keeping value and gradient consistent.
    //
    // To preserve value/gradient consistency the trace kernel must be the
    // FULL pseudo-inverse `M⁺ = (H+Sλ)⁺` itself, carried in spectral form
    // `(U_M, diag(1/σ_a))` over the kept eigenpairs (#901; supersedes the
    // intermediate #752 realization that reduced `M⁺` to its range(Sλ)
    // block). For penalty-supported drifts `∂Sλ/∂ρ` the two coincide:
    //   tr(M⁺ ∂Sλ) = tr(U_Sᵀ M⁺ U_S · U_Sᵀ ∂Sλ U_S) = ∂_ρ log|H+Sλ|₊.
    // But the joint adaptive/ψ hyper-coordinates trace drifts with
    // null(Sλ) support (basis κ-derivatives, the GLM cubic correction
    // `D_β H[v]` through the intercept column), for which the range(Sλ)
    // reduction silently discards the leaked component while the FD of
    // `log|M|₊` keeps it. `tr(M⁺ Ḣ)` is the exact pseudo-logdet derivative
    // for EVERY drift on a constant-rank stratum (first-order eigenvector
    // motion cancels), so one spectral object serves the whole θ-vector.
    // Value and kernel come from the same eigendecomposition of the same
    // materialized `M` so they cannot drift apart.
    //
    // The #752 fix requires the full identifiable-subspace determinant. There
    // is no lower-dimensional fallback that preserves that objective: the old
    // range(Sλ) reduction is exactly the bug, because it drops the penalty-null
    // likelihood determinant. If the dense path is over budget, fail loudly so
    // the caller can choose a different Hessian representation instead of
    // optimizing a different REML surface.
    ensure_exact_joint_hessian_dense_budget(total, "joint penalty subspace logdet")?;
    let m_dense =
        materialize_joint_hessian_source(h_joint_unpen, total, "joint penalty subspace logdet")?;
    let mut m = m_dense;
    add_joint_penalty_to_matrix(&mut m, ranges, s_lambdas, hessian_diagonal_ridge, None);
    if let Some(joint) = joint_penalty {
        m += joint;
    }
    if let Some(hphi) = scaled_jeffreys_hphi {
        m += hphi;
    }
    symmetrize_dense_in_place(&mut m);
    let (m_evals, m_evecs) = m.eigh(Side::Lower).map_err(|e| {
        format!("joint penalty subspace full Hessian eigendecomposition failed: {e}")
    })?;
    let m_threshold = positive_eigenvalue_threshold(m_evals.as_slice().unwrap());
    let logdet = exact_pseudo_logdet(m_evals.as_slice().unwrap(), m_threshold);
    // Full Moore–Penrose pseudo-inverse `M⁺` (drop ker(H+Sλ)) in spectral
    // form: kept eigenvectors as the kernel basis, diag(1/σ) as the reduced
    // kernel. In this basis `h_proj_inverse = (U_Mᵀ M U_M)⁻¹ = diag(1/σ)`
    // exactly, so every `PenaltySubspaceTrace` consumer evaluates the one
    // true `tr(M⁺ ·)` / `M⁺`-bilinear — exact for penalty-supported AND
    // null(Sλ)-leaking drifts alike (#901).
    let kept: Vec<usize> = (0..total)
        .filter(|&eig_idx| m_evals[eig_idx] > m_threshold)
        .collect();
    if kept.is_empty() {
        return Ok((0.0, None));
    }
    let r_kept = kept.len();
    let mut u_m = Array2::<f64>::zeros((total, r_kept));
    let mut h_proj_inverse = Array2::<f64>::zeros((r_kept, r_kept));
    for (out_col, &src_col) in kept.iter().enumerate() {
        for row in 0..total {
            u_m[[row, out_col]] = m_evecs[[row, src_col]];
        }
        h_proj_inverse[[out_col, out_col]] = 1.0 / m_evals[src_col];
    }

    Ok((
        logdet,
        Some(PenaltySubspaceTrace {
            u_s: u_m,
            h_proj_inverse,
        }),
    ))
}
