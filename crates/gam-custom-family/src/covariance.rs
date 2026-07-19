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
/// `known_active_rows`, when provided, is the QP solver's authoritative active
/// face. Trust-region damping and finite
/// precision can leave the committed β with row slacks slightly above the slack
/// tolerance even though the QP identified the row as binding; slack-based
/// detection alone then misses the row and leaves its Lagrange-multiplier mass
/// in the projected residual. Conversely, unioning the authoritative face with
/// every slack-tight row destroys the factored-cone contract: one zero CTN
/// coefficient row makes all `n` observation rows tight although the QP needs
/// only a small working face, so the residual checker materializes thousands of
/// redundant rows and sends them through an `O(m²)` NNLS iteration bound. Slack
/// discovery is therefore used only when the caller has no QP face provenance.
/// The non-negative-multiplier projection still rejects every supplied row with
/// the wrong multiplier sign.
pub(crate) fn projected_stationarity_inf_norm(
    residual: &Array1<f64>,
    beta: &Array1<f64>,
    constraints: Option<&ConstraintSet>,
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
    constraints: &ConstraintSet,
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
    constraints: &ConstraintSet,
) -> Option<f64> {
    if constraints.ncols() != beta.len() {
        return None;
    }
    let values = constraints.values(beta.view()).ok()?;
    let mut primal_violation = 0.0_f64;
    for row in 0..constraints.nrows() {
        let bound = constraints.bound(row).ok()?;
        if bound == f64::NEG_INFINITY {
            continue;
        }
        if !bound.is_finite() {
            return None;
        }
        let slack = values[row] - bound;
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
    constraints: &ConstraintSet,
    known_active_rows: Option<&[usize]>,
) -> Option<Array1<f64>> {
    let p = beta.len();
    if residual.len() != p || constraints.ncols() != p {
        return None;
    }
    if let Some(hint) = known_active_rows {
        // The QP face is the sparse seed, not necessarily the complete normal
        // cone: an omitted factored row may be tight and cut off the negative
        // projected residual at zero step. Use the same separation oracle as
        // the operator QP cycle escape so step, convergence certificate,
        // covariance, and return all certify against one cone geometry.
        return gam_solve::active_set::project_stationarity_residual_on_constraint_set(
            residual,
            beta,
            constraints,
            hint,
        )
        .map(|(projected, _active)| projected);
    }
    let n_rows = constraints.nrows();
    let values = constraints.values(beta.view()).ok()?;
    // With no QP provenance, discover candidates from slack. Using a boolean
    // membership table preserves canonical row order.
    let mut in_active = vec![false; n_rows];
    for row in 0..n_rows {
            let bound = constraints.bound(row).ok()?;
            if bound == f64::NEG_INFINITY {
                continue;
            }
            if !bound.is_finite() {
                return None;
            }
            let value = values[row];
            let slack = value - bound;
            if !slack.is_finite() {
                return None;
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
            let scale = value.abs().max(bound.abs()).max(1.0);
            let beta_inf = beta
                .iter()
                .map(|v| v.abs())
                .fold(0.0_f64, f64::max)
                .max(1.0);
            // ℓ¹ row norm bounded below by the Euclidean norm the carrier exposes;
            // for the factored cone the Euclidean norm is exact and the ℓ¹ norm is
            // within √p of it, so the slack band keeps its magnitude semantics.
            let row_norm1 = constraints.row_norm(row).ok()?.max(1.0);
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

    let gathered = constraints.gather_rows(&active_rows).ok()?;
    project_stationarity_residual_on_constraint_cone(residual, &gathered.a)
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
        if ridge_policy.accounts_for_objective() && ridge > 0.0 {
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
    block_constraints: &[Option<ConstraintSet>],
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
        if ridge_policy.accounts_for_objective() && ridge > 0.0 {
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
        if ridge_policy.accounts_for_objective() && ridge > 0.0 {
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
    block_constraints: &[Option<ConstraintSet>],
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
        if ridge_policy.accounts_for_objective() && ridge > 0.0 {
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
    block_constraints: &[Option<ConstraintSet>],
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

/// Add the exact per-block and joint penalties to an owned returned-beta
/// likelihood Hessian. This coefficient-space precision is shared by
/// covariance, EDF, and `FitGeometry`; optional row evidence remains a
/// separate field and is never inferred from this matrix.
pub(crate) fn penalized_hessian_from_owned_mode(
    specs: &[ParameterBlockSpec],
    per_block_log_lambdas: &[Array1<f64>],
    options: &BlockwiseFitOptions,
    unpenalized_hessian: &Array2<f64>,
) -> Result<Array2<f64>, String> {
    let ranges = block_param_ranges(specs);
    let total = ranges.last().map(|(_, e)| *e).unwrap_or(0);
    if unpenalized_hessian.dim() != (total, total)
        || unpenalized_hessian.iter().any(|value| !value.is_finite())
    {
        return Err(format!(
            "owned returned-beta Hessian must be finite with shape {total}x{total}, got {}x{}",
            unpenalized_hessian.nrows(),
            unpenalized_hessian.ncols(),
        ));
    }
    if per_block_log_lambdas.len() != specs.len() {
        return Err(format!(
            "owned returned-beta penalty layout has {} blocks, expected {}",
            per_block_log_lambdas.len(),
            specs.len(),
        ));
    }
    let mut h = unpenalized_hessian.clone();
    for (b, spec) in specs.iter().enumerate() {
        let (start, end) = ranges[b];
        let lambdas = exact_lambdas_from_log_strengths(
            &per_block_log_lambdas[b],
            &format!("owned returned-beta block {b} log strength"),
        )?;
        if lambdas.len() != spec.penalties.len() {
            return Err(format!(
                "owned returned-beta block {b} has {} smoothing strengths, expected {}",
                lambdas.len(),
                spec.penalties.len(),
            ));
        }
        let mut s_lambda = Array2::<f64>::zeros((end - start, end - start));
        for (k, s) in spec.penalties.iter().enumerate() {
            s.add_scaled_to(lambdas[k], &mut s_lambda);
        }
        h.slice_mut(ndarray::s![start..end, start..end])
            .scaled_add(1.0, &s_lambda);
    }
    if let Some(bundle) = options.joint_penalties.as_deref()
        && !bundle.is_empty()
    {
        bundle.add_to_matrix(&mut h);
    }
    symmetrize_dense_in_place(&mut h);
    Ok(h)
}

/// Materialize the unpenalized coefficient Hessian owned by a certified
/// terminal mode without re-evaluating the likelihood.
///
/// A coupled likelihood has exactly one admissible authority: its retained
/// joint workspace.  For a likelihood that explicitly declares its blocks
/// uncoupled, the terminal per-block working sets are an equally exact
/// authority and assemble a block-diagonal joint Hessian.  Keeping those two
/// cases explicit prevents final result assembly from either calling a
/// stateful family a second time or silently dropping cross-block curvature.
pub(crate) fn materialize_owned_terminal_unpenalized_hessian<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &[ParameterBlockState],
    workspace: Option<&Arc<dyn ExactNewtonJointHessianWorkspace>>,
    working_sets: Option<&[BlockWorkingSet]>,
    context: &str,
) -> Result<Array2<f64>, String> {
    let ranges = block_param_ranges(specs);
    let total = ranges.last().map(|(_, end)| *end).unwrap_or(0);
    if let Some(workspace) = workspace {
        let source = exact_newton_joint_hessian_source_from_workspace(
            workspace,
            total,
            MaterializationIntent::LogdetFactorization,
            context,
        )?
        .ok_or_else(|| {
            format!(
                "{context}: the certified terminal workspace did not expose its exact returned-beta Hessian"
            )
        })?;
        return materialize_joint_hessian_source(&source, total, context);
    }

    let working_sets = working_sets.ok_or_else(|| {
        format!(
            "{context}: the certified terminal mode retained neither a joint Hessian workspace nor per-block working sets"
        )
    })?;
    if working_sets.len() != specs.len() {
        return Err(format!(
            "{context}: the certified terminal mode retained {} working sets for {} parameter blocks",
            working_sets.len(),
            specs.len(),
        ));
    }
    if states.len() != specs.len() {
        return Err(format!(
            "{context}: the certified terminal mode retained {} block states for {} parameter blocks",
            states.len(),
            specs.len(),
        ));
    }
    if specs.len() > 1 && !family.likelihood_blocks_uncoupled() {
        // A coupled likelihood's joint Hessian carries cross-block curvature that
        // the block-diagonal per-block working sets omit, so those working sets
        // are not an admissible source here. Exact-Newton families whose mode
        // curvature certificate ran retain a joint workspace (handled above), but
        // that certificate is deliberately skipped for Jeffreys-armed families
        // (their definiteness is certified on the joint Jeffreys subspace
        // instead), so a Jeffreys-armed coupled family — every dispersion /
        // location-scale GLM (gamma, NB, beta, tweedie) — reaches this branch
        // with only working sets. Recompute the exact joint likelihood Hessian at
        // the FROZEN converged mode: a deterministic re-evaluation at fixed beta
        // that cannot move the mode or perturb a stateful augmentation, and the
        // exact same source `compute_joint_covariance_required` consumes. This
        // restores terminal curvature ownership for coupled Jeffreys families
        // that #979 `da5fd654b` + #2298 `ab6752762` together left unable to
        // assemble a terminal Hessian.
        if let Some(hessian) =
            exact_newton_joint_hessian_symmetrized(family, states, specs, total, context)?
        {
            return Ok(hessian);
        }
        return Err(format!(
            "{context}: a coupled {}-block likelihood cannot derive its joint Hessian from block working sets, and the family exposes no exact joint Hessian to recompute at the certified mode",
            specs.len(),
        ));
    }

    let mut hessian = Array2::<f64>::zeros((total, total));
    for (block_idx, ((spec, state), work)) in specs
        .iter()
        .zip(states.iter())
        .zip(working_sets.iter())
        .enumerate()
    {
        let (start, end) = ranges[block_idx];
        let width = end - start;
        if state.beta.len() != width {
            return Err(format!(
                "{context}: block {block_idx} terminal beta has length {}, expected {width}",
                state.beta.len(),
            ));
        }
        let block_hessian = match work {
            BlockWorkingSet::Diagonal {
                working_response,
                working_weights,
            } => {
                let expected_rows = spec.solver_design().nrows();
                if working_response.len() != expected_rows
                    || working_weights.len() != expected_rows
                    || state.eta.len() != expected_rows
                {
                    return Err(format!(
                        "{context}: block {block_idx} diagonal terminal evidence has response/weight/eta lengths {}/{}/{}, expected {expected_rows}",
                        working_response.len(),
                        working_weights.len(),
                        state.eta.len(),
                    ));
                }
                with_block_geometry(family, states, spec, block_idx, |design, _| {
                    let weights = certify_finite_working_weights(working_weights)?;
                    let (xtwx, _) = weighted_normal_equations(design, weights, None)?;
                    Ok(xtwx)
                })?
            }
            BlockWorkingSet::ExactNewton { hessian, .. } => {
                if hessian.nrows() != width || hessian.ncols() != width {
                    return Err(format!(
                        "{context}: block {block_idx} exact terminal Hessian has shape {}x{}, expected {width}x{width}",
                        hessian.nrows(),
                        hessian.ncols(),
                    ));
                }
                hessian.to_dense()
            }
        };
        if block_hessian.iter().any(|value| !value.is_finite()) {
            return Err(format!(
                "{context}: block {block_idx} terminal Hessian contains non-finite values"
            ));
        }
        hessian
            .slice_mut(ndarray::s![start..end, start..end])
            .assign(&block_hessian);
    }
    symmetrize_dense_in_place(&mut hessian);
    Ok(hessian)
}

pub(crate) fn compute_joint_covariance<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &[ParameterBlockState],
    per_block_log_lambdas: &[Array1<f64>],
    options: &BlockwiseFitOptions,
    preferred_unpenalized_hessian: Option<&Array2<f64>>,
) -> Result<Array2<f64>, String> {
    let total = specs.iter().map(|spec| spec.design.ncols()).sum();
    let unpenalized_hessian = if let Some(hessian) = preferred_unpenalized_hessian {
        hessian.clone()
    } else {
        let Some(hessian) = exact_newton_joint_hessian_symmetrized(
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
        hessian
    };
    let h = penalized_hessian_from_owned_mode(
        specs,
        per_block_log_lambdas,
        options,
        &unpenalized_hessian,
    )?;
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
    preferred_unpenalized_hessian: Option<&Array2<f64>>,
) -> Result<Option<Array2<f64>>, CustomFamilyError> {
    if !options.compute_covariance {
        return Ok(None);
    }
    match compute_joint_covariance(
        family,
        specs,
        states,
        per_block_log_lambdas,
        options,
        preferred_unpenalized_hessian,
    ) {
        Ok(covariance) => Ok(Some(covariance)),
        // A converged fit with a PSD penalized Hessian is a VALID fit even when
        // the covariance cannot be factorized; escalating that into a whole-fit
        // failure would discard usable coefficients and point predictions. When
        // the consumer opted into best-effort covariance, downgrade to a typed
        // absence (covariance `None`, reason logged) so the fit is still minted
        // and inference simply reports itself unavailable (#2299).
        Err(e) if options.covariance_best_effort => {
            log::warn!(
                "[custom-family covariance] joint covariance unavailable for a converged fit; minting the fit without it: {e}"
            );
            Ok(None)
        }
        Err(e) => Err(CustomFamilyError::InvalidInput {
            context: "compute_joint_covariance_required",
            reason: format!("joint covariance computation failed: {e}"),
        }),
    }
}

/// Compute terminal coefficient geometry, with optional single-diagonal row
/// evidence for consumers such as ALO.
pub(crate) fn compute_joint_geometry<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &[ParameterBlockState],
    per_block_log_lambdas: &[Array1<f64>],
    options: &BlockwiseFitOptions,
    preferred_unpenalized_hessian: Option<&Array2<f64>>,
    preferred_working_sets: Option<&[BlockWorkingSet]>,
) -> Result<FitGeometry, String> {
    if specs.len() != per_block_log_lambdas.len() {
        return Err(format!(
            "terminal geometry has {} parameter blocks but {} per-block smoothing vectors",
            specs.len(),
            per_block_log_lambdas.len(),
        ));
    }
    if let Some(working_sets) = preferred_working_sets
        && working_sets.len() != specs.len()
    {
        return Err(format!(
            "terminal geometry has {} parameter blocks but {} owned working sets",
            specs.len(),
            working_sets.len(),
        ));
    }

    let total = specs.iter().map(|spec| spec.design.ncols()).sum::<usize>();
    let unpenalized_hessian = if let Some(hessian) = preferred_unpenalized_hessian {
        hessian.clone()
    } else {
        exact_newton_joint_hessian_symmetrized(
            family,
            states,
            specs,
            total,
            "terminal coefficient geometry Hessian shape mismatch",
        )?
        .ok_or_else(|| {
            "terminal coefficient geometry requires an exact analytic Hessian; objective perturbation and placeholder geometry are forbidden"
                .to_string()
        })?
    };
    let penalized_hessian = penalized_hessian_from_owned_mode(
        specs,
        per_block_log_lambdas,
        options,
        &unpenalized_hessian,
    )?;

    // A single diagonal working set is the only live row-wise contract. A
    // multi-block fit has several distinct row measures, and Exact-Newton
    // curvature lives in coefficient space; both therefore retain `None`
    // rather than fabricated empty/zero vectors or an unused stacked variant.
    let working = if specs.len() == 1 {
        let evaluated;
        let working_sets = match preferred_working_sets {
            Some(working_sets) => working_sets,
            None => {
                evaluated = family.evaluate(states)?;
                evaluated.blockworking_sets.as_slice()
            }
        };
        match working_sets {
            [BlockWorkingSet::Diagonal {
                working_response,
                working_weights,
            }] => {
                let expected_rows = specs[0].design.nrows();
                if working_weights.len() != expected_rows
                    || working_response.len() != expected_rows
                {
                    return Err(format!(
                        "single-diagonal terminal working geometry has weights/response lengths {}/{}, expected {expected_rows}",
                        working_weights.len(),
                        working_response.len(),
                    ));
                }
                Some(WorkingGeometry {
                    weights: working_weights.clone(),
                    response: working_response.clone(),
                })
            }
            [BlockWorkingSet::ExactNewton { .. }] => None,
            _ => {
                return Err(format!(
                    "single-block terminal geometry requires exactly one owned working set, got {}",
                    working_sets.len(),
                ));
            }
        }
    } else {
        None
    };

    let block_widths = specs
        .iter()
        .map(|spec| spec.design.ncols())
        .collect::<Vec<_>>();
    Ok(FitGeometry {
        coefficient_gauge: gam_problem::gauge::Gauge::identity(&block_widths),
        penalized_hessian: penalized_hessian.into(),
        working,
    })
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

/// First-order ρ-uncertainty inflation of the joint coefficient covariance for
/// custom-family fits (#2346): the correction term `C = A · V_ρ · Aᵀ` with
/// `A = V_cond · U` and `U[:, o] = (∂S_λ/∂ρ_o) · β̂` — each outer smoothing
/// coordinate's penalty derivative applied to the fitted coefficients. By
/// first-order IFT `J_o = ∂β̂/∂ρ_o = −V_cond · U[:, o]`, so
/// `Σ_{o,t} J_o · V_ρ[o,t] · J_tᵀ = C` and `V_c = V_cond + C` is the Vc-style
/// corrected covariance the standard lane ships, with the same typed
/// `FirstOrderIdentifiedSubspace` provenance.
///
/// Rail-aware (#2337 Thm 2.3): outer coordinates in `excluded_outer` (box
/// rails and typed AsymptoteRail coordinates) have no finite ρ-variance and
/// are excluded from the inflation. The interior sub-Hessian must be strictly
/// PD; a non-PD interior returns `Ok(None)` — a typed absence, not an error,
/// because the deep-smoothing regime legitimately reaches it. Returns the
/// correction together with the identified interior rank.
pub(crate) fn joint_smoothing_correction(
    v_cond: &Array2<f64>,
    specs: &[ParameterBlockSpec],
    layout: &crate::penalty_labels::PenaltyLabelLayout,
    rho_outer: &Array1<f64>,
    block_states: &[ParameterBlockState],
    outer_hessian: &Array2<f64>,
    excluded_outer: &[usize],
) -> Result<Option<(Array2<f64>, usize)>, String> {
    let p_total: usize = specs.iter().map(|spec| spec.design.ncols()).sum();
    let k_outer = rho_outer.len();
    if v_cond.dim() != (p_total, p_total) {
        return Err(format!(
            "joint smoothing correction: V_cond shape {:?} ≠ ({p_total}, {p_total})",
            v_cond.dim()
        ));
    }
    if outer_hessian.dim() != (k_outer, k_outer) {
        return Err(format!(
            "joint smoothing correction: outer Hessian shape {:?} ≠ ({k_outer}, {k_outer})",
            outer_hessian.dim()
        ));
    }
    if block_states.len() != specs.len() {
        return Err(format!(
            "joint smoothing correction: {} block states vs {} specs",
            block_states.len(),
            specs.len()
        ));
    }

    // β̂ stacked in block order — the reduced coefficient frame V_cond lives in.
    let mut beta_flat = Array1::<f64>::zeros(p_total);
    let mut offsets = Vec::with_capacity(specs.len() + 1);
    let mut at = 0usize;
    for (spec, state) in specs.iter().zip(block_states) {
        let width = spec.design.ncols();
        if state.beta.len() != width {
            return Err(format!(
                "joint smoothing correction: block '{}' beta length {} ≠ design width {width}",
                spec.name,
                state.beta.len()
            ));
        }
        offsets.push(at);
        beta_flat
            .slice_mut(ndarray::s![at..at + width])
            .assign(&state.beta);
        at += width;
    }
    offsets.push(at);

    // U[:, o] = Σ_{slots tied to outer o} λ_slot · S_slot · β̂. Per-block
    // penalties act on their block slice; joint specs act on the full stacked
    // space. Fixed (untied) physical slots carry no ρ coordinate — no
    // ρ-uncertainty flows through them.
    let mut u_mat = Array2::<f64>::zeros((p_total, k_outer));
    let mut physical = 0usize;
    for (block_idx, spec) in specs.iter().enumerate() {
        let base = offsets[block_idx];
        let width = spec.design.ncols();
        for penalty in &spec.penalties {
            let outer = layout.physical_to_outer.get(physical).copied().flatten();
            physical += 1;
            let Some(outer) = outer else {
                continue;
            };
            let lambda = rho_outer[outer].exp();
            if lambda == 0.0 {
                continue;
            }
            let s_dense = penalty.to_dense();
            if s_dense.dim() != (width, width) {
                return Err(format!(
                    "joint smoothing correction: block '{}' penalty shape {:?} ≠ ({width}, {width})",
                    spec.name,
                    s_dense.dim()
                ));
            }
            let s_beta = s_dense.dot(&beta_flat.slice(ndarray::s![base..base + width]));
            for i in 0..width {
                u_mat[[base + i, outer]] += lambda * s_beta[i];
            }
        }
    }
    for (joint_idx, spec) in layout.joint_specs.iter().enumerate() {
        let outer = layout.joint_to_outer[joint_idx];
        let lambda = rho_outer[outer].exp();
        if lambda == 0.0 {
            continue;
        }
        if spec.matrix.dim() != (p_total, p_total) {
            return Err(format!(
                "joint smoothing correction: joint penalty '{}' shape {:?} ≠ ({p_total}, {p_total})",
                spec.label.as_deref().unwrap_or("<unlabeled>"),
                spec.matrix.dim()
            ));
        }
        let m_beta = spec.matrix.dot(&beta_flat);
        for i in 0..p_total {
            u_mat[[i, outer]] += lambda * m_beta[i];
        }
    }

    // Interior V_ρ: strict SPD inverse of the non-excluded outer sub-block.
    let included: Vec<usize> = (0..k_outer)
        .filter(|o| !excluded_outer.contains(o))
        .collect();
    if included.is_empty() {
        return Ok(None);
    }
    let ki = included.len();
    let mut h_sub = Array2::<f64>::zeros((ki, ki));
    for (i, &oi) in included.iter().enumerate() {
        for (j, &oj) in included.iter().enumerate() {
            h_sub[[i, j]] = outer_hessian[[oi, oj]];
        }
    }
    let (evals, evecs) = FaerEigh::eigh(&h_sub, Side::Lower).map_err(|e| {
        format!("joint smoothing correction: outer Hessian eigendecomposition failed: {e}")
    })?;
    let max_abs = evals.iter().fold(0.0_f64, |acc, &ev| acc.max(ev.abs()));
    let tol = (100.0 * f64::EPSILON * (ki as f64) * max_abs).max(100.0 * f64::EPSILON);
    if evals.iter().any(|&ev| ev <= tol) {
        return Ok(None);
    }
    let mut v_rho = Array2::<f64>::zeros((ki, ki));
    for (idx, &ev) in evals.iter().enumerate() {
        let inv = 1.0 / ev;
        for i in 0..ki {
            let vi = evecs[[i, idx]];
            for j in 0..ki {
                v_rho[[i, j]] += inv * vi * evecs[[j, idx]];
            }
        }
    }

    // C = (V·U_inc) · V_ρ · (V·U_inc)ᵀ — symmetric PSD by construction.
    let mut u_inc = Array2::<f64>::zeros((p_total, ki));
    for (col, &o) in included.iter().enumerate() {
        u_inc.column_mut(col).assign(&u_mat.column(o));
    }
    let a_mat = v_cond.dot(&u_inc);
    let mut correction = a_mat.dot(&v_rho).dot(&a_mat.t());
    symmetrize_dense_in_place(&mut correction);
    Ok(Some((correction, ki)))
}

#[cfg(test)]
mod best_effort_covariance_tests {
    //! Pins the #2299 `covariance_best_effort` downgrade at its production seam.
    //! A converged fit whose joint posterior precision `M = H + S_λ` is singular
    //! cannot produce a Laplace covariance; `compute_joint_covariance_required`
    //! must return that as a typed absence (`Ok(None)`) when the consumer opted
    //! into best-effort, and as an error otherwise. This is exercised with a
    //! genuinely singular `M` (not a marginal knife-edge) so the assertion is
    //! deterministic and load-independent -- the marginal fit that reaches this
    //! path from Python is nondeterministic (rayon-fold-order-sensitive at the
    //! tolerance) and is guarded upstream by the gauge + REML anyway (#2299).
    use super::*;
    use ndarray::array;

    #[derive(Clone)]
    struct TrivialFamily;

    impl CustomFamily for TrivialFamily {
        fn evaluate(&self, _: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
            Ok(FamilyEvaluation {
                log_likelihood: 0.0,
                blockworking_sets: vec![],
            })
        }
    }

    /// One 3-coefficient block whose unpenalized Hessian and penalty share the
    /// null direction `e3`, so `M = H + S_λ` (at λ = e^0 = 1) is
    /// `[[5, 0.2, 0], [0.2, 11, 0], [0, 0, 0]]` -- singular along `e3`, i.e. the
    /// posterior is improper and no finite covariance exists.
    fn singular_joint_fixture() -> (
        Vec<ParameterBlockSpec>,
        Vec<ParameterBlockState>,
        Vec<Array1<f64>>,
        Array2<f64>,
    ) {
        let s = array![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]];
        let unpenalized = array![[4.0, 0.2, 0.0], [0.2, 9.0, 0.0], [0.0, 0.0, 0.0]];
        let beta = array![1.0, -1.0, 3.0];
        let spec = ParameterBlockSpec {
            name: "degenerate".to_string(),
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                Array2::zeros((1, 3)),
            )),
            offset: Array1::zeros(1),
            penalties: vec![PenaltyMatrix::Dense(s)],
            nullspace_dims: vec![1],
            initial_log_lambdas: array![0.0],
            initial_beta: Some(beta.clone()),
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        };
        let state = ParameterBlockState {
            beta,
            eta: Array1::zeros(1),
        };
        (vec![spec], vec![state], vec![array![0.0]], unpenalized)
    }

    #[test]
    fn best_effort_downgrades_singular_covariance_to_typed_absence() {
        let (specs, states, per_block, unpenalized) = singular_joint_fixture();
        let options = BlockwiseFitOptions {
            compute_covariance: true,
            covariance_best_effort: true,
            ..BlockwiseFitOptions::default()
        };
        let result = compute_joint_covariance_required(
            &TrivialFamily,
            &specs,
            &states,
            &per_block,
            &options,
            Some(&unpenalized),
        );
        assert!(
            matches!(result, Ok(None)),
            "best-effort must downgrade an unfactorizable covariance to Ok(None); got {result:?}"
        );
    }

    #[test]
    fn covariance_required_without_best_effort_errors_on_singular() {
        let (specs, states, per_block, unpenalized) = singular_joint_fixture();
        let options = BlockwiseFitOptions {
            compute_covariance: true,
            covariance_best_effort: false,
            ..BlockwiseFitOptions::default()
        };
        let result = compute_joint_covariance_required(
            &TrivialFamily,
            &specs,
            &states,
            &per_block,
            &options,
            Some(&unpenalized),
        );
        assert!(
            result.is_err(),
            "without best-effort a singular joint covariance must surface as an error; got {result:?}"
        );
    }

    /// A comfortably positive-definite joint precision must yield a finite
    /// covariance (the ordinary success path), so the singular-case tests above
    /// are pinning the degenerate branch and not a blanket refusal.
    #[test]
    fn well_conditioned_covariance_is_computed() {
        // M = H + S_lambda = diag(5, 11, 4): strictly PD, invertible.
        let s = array![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]];
        let unpenalized = array![[4.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 3.0]];
        let beta = array![0.5, -0.5, 0.25];
        let spec = ParameterBlockSpec {
            name: "identified".to_string(),
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                Array2::zeros((1, 3)),
            )),
            offset: Array1::zeros(1),
            penalties: vec![PenaltyMatrix::Dense(s)],
            nullspace_dims: vec![0],
            initial_log_lambdas: array![0.0],
            initial_beta: Some(beta.clone()),
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        };
        let states = vec![ParameterBlockState {
            beta,
            eta: Array1::zeros(1),
        }];
        let options = BlockwiseFitOptions {
            compute_covariance: true,
            covariance_best_effort: true,
            ..BlockwiseFitOptions::default()
        };
        let result = compute_joint_covariance_required(
            &TrivialFamily,
            &vec![spec],
            &states,
            &vec![array![0.0]],
            &options,
            Some(&unpenalized),
        );
        match result {
            Ok(Some(cov)) => {
                assert_eq!(cov.dim(), (3, 3));
                assert!(cov.iter().all(|v| v.is_finite()));
                // Σ = M⁻¹ = diag(1/5, 1/11, 1/4).
                assert!((cov[[0, 0]] - 0.2).abs() < 1e-9);
                assert!((cov[[2, 2]] - 0.25).abs() < 1e-9);
            }
            other => panic!("a PD joint precision must yield a finite covariance; got {other:?}"),
        }
    }

    #[test]
    fn covariance_disabled_returns_none_before_any_factorization() {
        let (specs, states, per_block, unpenalized) = singular_joint_fixture();
        let options = BlockwiseFitOptions {
            compute_covariance: false,
            covariance_best_effort: true,
            ..BlockwiseFitOptions::default()
        };
        let result = compute_joint_covariance_required(
            &TrivialFamily,
            &specs,
            &states,
            &per_block,
            &options,
            Some(&unpenalized),
        );
        assert!(matches!(result, Ok(None)));
    }
}
