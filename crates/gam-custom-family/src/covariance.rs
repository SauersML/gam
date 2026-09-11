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
) -> Result<(), CustomFamilyError> {
    let ranges = block_param_ranges(specs);
    let total = ranges.last().map(|(_, e)| *e).unwrap_or(0);
    if beta_flat.len() != total {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: format!(
                "flat beta length mismatch: got {}, expected {}",
                beta_flat.len(),
                total
            ),
        });
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
) -> Result<Vec<ParameterBlockState>, CustomFamilyError> {
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
        // QP provenance selects the strict point-local tangent-face contract.
        // The operator-native Moreau solve discovers its complete multiplier
        // support deterministically; the historical warm row ids no longer
        // alter generator selection or the projected stationarity vector.
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
) -> Result<Option<f64>, CustomFamilyError> {
    if eval.blockworking_sets.len() != states.len() || states.len() != s_lambdas.len() {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: "exact-newton joint stationarity check: block dimension mismatch".to_string(),
        });
    }
    if specs.len() != states.len() {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: "exact-newton joint stationarity check: spec/state count mismatch".to_string(),
        });
    }
    if let Some(sets) = block_active_sets
        && sets.len() != states.len()
    {
        return Err(CustomFamilyError::DimensionMismatch { reason: format!(
            "exact-newton joint stationarity check: active-set count mismatch, got {}, expected {}",
            sets.len(),
            states.len()
        ) });
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
) -> Result<Option<Array1<f64>>, CustomFamilyError> {
    if eval.blockworking_sets.len() != specs.len() {
        return Err(CustomFamilyError::DimensionMismatch { reason: format!(
            "exact-newton joint gradient extraction: family returned {} block working sets, expected {}",
            eval.blockworking_sets.len(),
            specs.len()
        ) });
    }
    if states.len() != specs.len() {
        return Err(CustomFamilyError::DimensionMismatch { reason: format!(
            "exact-newton joint gradient extraction: state count {} does not match spec count {}",
            states.len(),
            specs.len()
        ) });
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
                    ) });
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
                    ) });
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
                    ) });
                }
                gradient
                    .slice_mut(ndarray::s![offset..offset + width])
                    .assign(&block_gradient);
            }
            BlockWorkingSet::NaturalDiagonal { score, .. } => {
                let n = score.len();
                if state.eta.len() != n || spec.solver_design().nrows() != n {
                    return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                        "exact-newton joint gradient extraction: natural-diagonal length mismatch (score={}, η={}, X_rows={})",
                        score.len(),
                        state.eta.len(),
                        spec.solver_design().nrows(),
                    ) });
                }
                let block_gradient = spec.solver_design().transpose_vector_multiply(score);
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
    // gam#2612: `Σ_t λ_t (M⊗S_t)·β` — the full-width joint penalty's contribution,
    // exactly as
    // [`exact_newton_joint_projected_stationarity_vector_from_gradient`] already
    // takes it. Without it this gate prices `S_perblock·β − ∇ℓ`, and a family
    // whose smoothing rides ENTIRELY on the joint bundle has per-block
    // `s_lambdas` identically zero, so the gate measures `−∇ℓ` — which at the
    // penalized optimum IS `S_joint·β̂`. Measured on the penguins multinomial:
    // the refusing residual and `‖S_joint·β̂‖∞` agree to seven significant
    // figures (`3.356046e-1`), and the gate floors there from cycle 4 to 1199
    // while the objective decrement sits three orders INSIDE its tolerance. The
    // certificate was refusing a converged point because it was not measuring a
    // residual at all. Same failure the `∇Φ` fold above this call site already
    // fixes for the Jeffreys term. `None` ⇒ no joint penalty, byte-identical for
    // every per-block-only family.
    joint_penalty_score: Option<&Array1<f64>>,
) -> Result<f64, CustomFamilyError> {
    if states.len() != specs.len() || states.len() != s_lambdas.len() {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: "exact-newton joint stationarity check from gradient: block dimension mismatch"
                .to_string(),
        });
    }
    if block_constraints.len() != states.len() {
        return Err(CustomFamilyError::DimensionMismatch { reason: format!(
            "exact-newton joint stationarity check from gradient: constraint count mismatch, got {}, expected {}",
            block_constraints.len(),
            states.len()
        ) });
    }
    if let Some(sets) = block_active_sets
        && sets.len() != states.len()
    {
        return Err(CustomFamilyError::DimensionMismatch { reason: format!(
            "exact-newton joint stationarity check from gradient: active-set count mismatch, got {}, expected {}",
            sets.len(),
            states.len()
        ) });
    }
    let total_p = specs.iter().map(|spec| spec.design.ncols()).sum::<usize>();
    if gradient.len() != total_p {
        return Err(CustomFamilyError::DimensionMismatch { reason: format!(
            "exact-newton joint stationarity check from gradient: joint gradient length mismatch, got {}, expected {}",
            gradient.len(),
            total_p
        ) });
    }
    if let Some(js) = joint_penalty_score
        && js.len() != total_p
    {
        return Err(CustomFamilyError::DimensionMismatch { reason: format!(
            "exact-newton joint stationarity check from gradient: joint penalty score length mismatch, got {}, expected {}",
            js.len(),
            total_p
        ) });
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
        if let Some(js) = joint_penalty_score {
            residual += &js.slice(ndarray::s![offset..offset + width]);
        }
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
) -> Result<Array1<f64>, CustomFamilyError> {
    if states.len() != specs.len() || states.len() != s_lambdas.len() {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: "exact-newton joint stationarity vector from gradient: block dimension mismatch"
                .to_string(),
        });
    }
    let total_p = specs.iter().map(|spec| spec.design.ncols()).sum::<usize>();
    if gradient.len() != total_p {
        return Err(CustomFamilyError::DimensionMismatch { reason: format!(
            "exact-newton joint stationarity vector from gradient: joint gradient length mismatch, got {}, expected {}",
            gradient.len(),
            total_p
        ) });
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
) -> Result<Array1<f64>, CustomFamilyError> {
    if states.len() != specs.len()
        || states.len() != s_lambdas.len()
        || states.len() != block_constraints.len()
    {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: "exact-newton projected stationarity vector from gradient: block dimension mismatch"
                .to_string(),
        });
    }
    if let Some(sets) = block_active_sets
        && sets.len() != states.len()
    {
        return Err(CustomFamilyError::DimensionMismatch { reason: format!(
            "exact-newton projected stationarity vector from gradient: active-set count mismatch, got {}, expected {}",
            sets.len(),
            states.len()
        ) });
    }
    let total_p = specs.iter().map(|spec| spec.design.ncols()).sum::<usize>();
    if gradient.len() != total_p {
        return Err(CustomFamilyError::DimensionMismatch { reason: format!(
            "exact-newton projected stationarity vector from gradient: joint gradient length mismatch, got {}, expected {}",
            gradient.len(),
            total_p
        ) });
    }
    if let Some(js) = joint_penalty_score
        && js.len() != total_p
    {
        return Err(CustomFamilyError::DimensionMismatch { reason: format!(
            "exact-newton projected stationarity vector from gradient: joint penalty score length mismatch, got {}, expected {}",
            js.len(),
            total_p
        ) });
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
) -> Result<Option<ProjectedKktResidual>, CustomFamilyError> {
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
) -> Result<Option<ProjectedKktResidual>, CustomFamilyError> {
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
) -> Result<Option<ProjectedKktResidual>, CustomFamilyError> {
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
) -> Result<Array2<f64>, CustomFamilyError> {
    let ranges = block_param_ranges(specs);
    let total = ranges.last().map(|(_, e)| *e).unwrap_or(0);
    if unpenalized_hessian.dim() != (total, total)
        || unpenalized_hessian.iter().any(|value| !value.is_finite())
    {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: format!(
                "owned returned-beta Hessian must be finite with shape {total}x{total}, got {}x{}",
                unpenalized_hessian.nrows(),
                unpenalized_hessian.ncols(),
            ),
        });
    }
    if per_block_log_lambdas.len() != specs.len() {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: format!(
                "owned returned-beta penalty layout has {} blocks, expected {}",
                per_block_log_lambdas.len(),
                specs.len(),
            ),
        });
    }
    let mut h = unpenalized_hessian.clone();
    for (b, spec) in specs.iter().enumerate() {
        let (start, end) = ranges[b];
        let lambdas = exact_lambdas_from_log_strengths(
            &per_block_log_lambdas[b],
            &format!("owned returned-beta block {b} log strength"),
        )?;
        if lambdas.len() != spec.penalties.len() {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "owned returned-beta block {b} has {} smoothing strengths, expected {}",
                    lambdas.len(),
                    spec.penalties.len(),
                ),
            });
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
/// A coupled likelihood has exactly one admissible KIND of authority: an exact
/// JOINT Hessian, with its cross-block curvature intact.  Two things can supply
/// one — the retained joint workspace, and a deterministic re-evaluation of the
/// exact joint likelihood Hessian at the frozen converged mode — and they are
/// tried in that order.  For a likelihood that explicitly declares its blocks
/// uncoupled, the terminal per-block working sets are an equally exact
/// authority and assemble a block-diagonal joint Hessian.  Keeping those cases
/// explicit prevents final result assembly from either calling a stateful
/// family a second time or silently dropping cross-block curvature.
///
/// What is NOT admissible, and stays inadmissible here, is substituting the
/// block-diagonal working-set assembly for a COUPLED family: that trades a loud
/// refusal for a quietly understated precision in every covariance, EDF and
/// `FitGeometry` consumer downstream.
pub(crate) fn materialize_owned_terminal_unpenalized_hessian<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &[ParameterBlockState],
    workspace: Option<&Arc<dyn ExactNewtonJointHessianWorkspace>>,
    working_sets: Option<&[BlockWorkingSet]>,
    context: &str,
) -> Result<Array2<f64>, CustomFamilyError> {
    let ranges = block_param_ranges(specs);
    let total = ranges.last().map(|(_, end)| *end).unwrap_or(0);
    // #2580.  `Ok(None)` from the workspace means "this workspace exposes no
    // curvature by ANY route" — its preferred one, `hessian_dense`, and the
    // operator fallback all came back empty.  That is a statement about the
    // WORKSPACE, not about whether an exact authority exists for this mode, and
    // converting it straight into an error here vetoed the route two branches
    // below: the exact joint likelihood Hessian recomputed at the FROZEN
    // converged mode from the block states alone.
    //
    // That route is not the block-diagonal working-set substitute the
    // single-authority contract forbids for a coupled likelihood — it is a full
    // joint Hessian with its cross-block curvature intact, and its own comment
    // names the location-scale survival AFT path as the shape it exists for.
    // The forbidden substitution is still forbidden: a coupled family that
    // cannot recompute still refuses rather than assembling block-diagonal
    // working sets.
    //
    // What the workspace could not supply is carried forward instead of
    // discarded, so a refusal further down names every route that was consulted
    // rather than only the last one (#2465).
    let mut workspace_refusal: Option<String> = None;
    if let Some(workspace) = workspace {
        match exact_newton_joint_hessian_source_from_workspace(
            workspace,
            total,
            MaterializationIntent::LogdetFactorization,
            context,
        )? {
            Some(source) => return materialize_joint_hessian_source(&source, total, context),
            None => {
                workspace_refusal = Some(format!(
                    "the certified terminal workspace exposed its exact returned-beta Hessian by                      no route (LogdetFactorization preference = {:?}; dense, diagonal and                      operator all absent)",
                    workspace.hessian_source_preference_for_intent(
                        MaterializationIntent::LogdetFactorization,
                    )
                ));
            }
        }
    }

    if states.len() != specs.len() {
        return Err(CustomFamilyError::trial_point(format!(
            "{context}: the certified terminal mode retained {} block states for {} parameter blocks",
            states.len(),
            specs.len(),
        )));
    }
    // The coupled-Jeffreys recompute is attempted BEFORE requiring per-block
    // working sets, because it derives the joint Hessian from the frozen block
    // STATES alone and never consumes a working set (#2373). A coupled family
    // that exposes an analytic joint gradient runs the joint-Newton accept path
    // with `eval == None`, so its converged inner result retains NEITHER a joint
    // workspace NOR terminal working sets — and the location-scale survival AFT
    // family is exactly that shape. Gating this recompute behind the working-set
    // unwrap (as before) made every such fit fail with "retained neither a joint
    // Hessian workspace nor per-block working sets", even though the recompute
    // it needs consumes only states. Working sets remain required for the
    // uncoupled block-diagonal assembly below.
    if specs.len() > 1 && !family.likelihood_blocks_uncoupled() {
        // A coupled likelihood's joint Hessian carries cross-block curvature that
        // the block-diagonal per-block working sets omit, so those working sets
        // are not an admissible source here. Exact-Newton families whose mode
        // curvature certificate ran retain a joint workspace (handled above), but
        // that certificate is deliberately skipped for Jeffreys-armed families
        // (their definiteness is certified on the joint Jeffreys subspace
        // instead), so a Jeffreys-armed coupled family — every dispersion /
        // location-scale GLM (gamma, NB, beta, tweedie) and the location-scale
        // survival AFT path — reaches this branch. Recompute the exact joint
        // likelihood Hessian at the FROZEN converged mode: a deterministic
        // re-evaluation at fixed beta that cannot move the mode or perturb a
        // stateful augmentation, and the exact same source
        // `compute_joint_posterior` consumes. This restores terminal
        // curvature ownership for coupled Jeffreys families that #979
        // `da5fd654b` + #2298 `ab6752762` together left unable to assemble a
        // terminal Hessian.
        if let Some(hessian) =
            exact_newton_joint_hessian_symmetrized(family, states, specs, total, context)?
        {
            return Ok(hessian);
        }
        return Err(CustomFamilyError::trial_point(format!(
            "{context}: a coupled {}-block likelihood cannot derive its joint Hessian from block working sets, and the family exposes no exact joint Hessian to recompute at the certified mode{}",
            specs.len(),
            workspace_refusal
                .as_deref()
                .map(|refusal| format!("; {refusal}"))
                .unwrap_or_default(),
        )));
    }

    let working_sets = working_sets.ok_or_else(|| {
        format!(
            "{context}: the certified terminal mode retained neither a usable joint Hessian workspace nor per-block working sets{}",
            workspace_refusal
                .as_deref()
                .map(|refusal| format!(" ({refusal})"))
                .unwrap_or_default(),
        )
    })?;
    if working_sets.len() != specs.len() {
        return Err(CustomFamilyError::trial_point(format!(
            "{context}: the certified terminal mode retained {} working sets for {} parameter blocks",
            working_sets.len(),
            specs.len(),
        )));
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
            return Err(CustomFamilyError::trial_point(format!(
                "{context}: block {block_idx} terminal beta has length {}, expected {width}",
                state.beta.len(),
            )));
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
                    return Err(CustomFamilyError::trial_point(format!(
                        "{context}: block {block_idx} diagonal terminal evidence has response/weight/eta lengths {}/{}/{}, expected {expected_rows}",
                        working_response.len(),
                        working_weights.len(),
                        state.eta.len(),
                    )));
                }
                with_block_geometry(family, states, spec, block_idx, |design, _| {
                    let weights = certify_finite_working_weights(working_weights)?;
                    let (xtwx, _) = weighted_normal_equations(design, weights, None)?;
                    Ok(xtwx)
                })?
            }
            BlockWorkingSet::NaturalDiagonal {
                score,
                observed_curvature,
            } => {
                let expected_rows = spec.solver_design().nrows();
                if score.len() != expected_rows
                    || observed_curvature.len() != expected_rows
                    || state.eta.len() != expected_rows
                {
                    return Err(CustomFamilyError::trial_point(format!(
                        "{context}: block {block_idx} natural-diagonal terminal evidence has score/curvature/eta lengths {}/{}/{}, expected {expected_rows}",
                        score.len(),
                        observed_curvature.len(),
                        state.eta.len(),
                    )));
                }
                with_block_geometry(family, states, spec, block_idx, |design, _| {
                    let curvature = certify_finite_working_weights(observed_curvature)?;
                    let (xtwx, _) = weighted_normal_equations(design, curvature, None)?;
                    Ok(xtwx)
                })?
            }
            BlockWorkingSet::ExactNewton { hessian, .. } => {
                if hessian.nrows() != width || hessian.ncols() != width {
                    return Err(CustomFamilyError::trial_point(format!(
                        "{context}: block {block_idx} exact terminal Hessian has shape {}x{}, expected {width}x{width}",
                        hessian.nrows(),
                        hessian.ncols(),
                    )));
                }
                hessian.to_dense()
            }
        };
        if block_hessian.iter().any(|value| !value.is_finite()) {
            return Err(CustomFamilyError::trial_point(format!(
                "{context}: block {block_idx} terminal Hessian contains non-finite values"
            )));
        }
        hessian
            .slice_mut(ndarray::s![start..end, start..end])
            .assign(&block_hessian);
    }
    symmetrize_dense_in_place(&mut hessian);
    Ok(hessian)
}

/// Positive-definite gate + exact inverse of a converged posterior precision.
///
/// Inequality constraints restrict support; they do not remove coefficient
/// directions. Their Laplace posterior therefore needs a proper ambient
/// Gaussian before it can be truncated. An indefinite or singular ambient
/// precision cannot be rescued by projecting onto the optimizer's active face:
/// that projection changes an inequality into an equality and manufactures
/// zero variance in every constraint-normal direction.
fn spd_covariance_from_precision(
    precision: &Array2<f64>,
    face: &str,
) -> Result<Array2<f64>, CustomFamilyError> {
    // Small positive curvature is not a structural null: its units depend on
    // the coefficient chart. Use the existing strict, unjittered Cholesky
    // and inverse backward-error certificate, not a spectral rank cutoff.
    gam_linalg::utils::certified_spd_inverse(precision, face)
        .map(|certified| certified.into_inverse())
        .map_err(|error| CustomFamilyError::trial_point(format!(
            "joint posterior precision is non-PD at the converged optimum or its inverse could not be certified on the {face}: {error}"
        )))
}

/// `Debug` is derived because the assembly is named in test panic messages that
/// report which variant a refusal produced; every field already derives it.
#[derive(Debug)]
pub(crate) struct JointPosteriorAssembly {
    pub(crate) covariance_conditional: Option<Array2<f64>>,
    pub(crate) geometry: FitGeometry,
    pub(crate) reported_beta: Option<Array1<f64>>,
}

fn terminal_score_from_working_sets(
    working_sets: &[BlockWorkingSet],
    specs: &[ParameterBlockSpec],
    states: &[ParameterBlockState],
) -> Result<Array1<f64>, CustomFamilyError> {
    if working_sets.len() != specs.len() || states.len() != specs.len() {
        return Err(CustomFamilyError::trial_point(format!(
            "terminal score ownership mismatch: working sets={}, specs={}, states={}",
            working_sets.len(),
            specs.len(),
            states.len(),
        )));
    }
    let total: usize = specs.iter().map(|spec| spec.design.ncols()).sum();
    let mut score = Array1::<f64>::zeros(total);
    let mut offset = 0usize;
    for (block_idx, ((work, spec), state)) in working_sets
        .iter()
        .zip(specs.iter())
        .zip(states.iter())
        .enumerate()
    {
        let width = spec.design.ncols();
        let block_score = match work {
            BlockWorkingSet::ExactNewton { gradient, .. } => {
                if gradient.len() != width {
                    return Err(CustomFamilyError::trial_point(format!(
                        "terminal score block {block_idx} has gradient length {}, expected {width}",
                        gradient.len(),
                    )));
                }
                gradient.clone()
            }
            BlockWorkingSet::Diagonal {
                working_response,
                working_weights,
            } => {
                let design = spec.solver_design();
                let n = design.nrows();
                if working_response.len() != n || working_weights.len() != n || state.eta.len() != n
                {
                    return Err(CustomFamilyError::trial_point(format!(
                        "terminal score block {block_idx} has z/w/eta lengths {}/{}/{}, expected {n}",
                        working_response.len(),
                        working_weights.len(),
                        state.eta.len(),
                    )));
                }
                let weighted_score = Array1::from_iter(
                    working_weights
                        .iter()
                        .zip(working_response.iter())
                        .zip(state.eta.iter())
                        .map(|((&weight, &response), &eta)| weight * (response - eta)),
                );
                <DesignMatrix as LinearOperator>::apply_transpose(design, &weighted_score)
            }
            BlockWorkingSet::NaturalDiagonal {
                score: natural_score,
                ..
            } => {
                let design = spec.solver_design();
                if natural_score.len() != design.nrows() || state.eta.len() != design.nrows() {
                    return Err(CustomFamilyError::trial_point(format!(
                        "terminal score block {block_idx} has natural-score/eta lengths {}/{}, expected {}",
                        natural_score.len(),
                        state.eta.len(),
                        design.nrows(),
                    )));
                }
                <DesignMatrix as LinearOperator>::apply_transpose(design, natural_score)
            }
        };
        score
            .slice_mut(ndarray::s![offset..offset + width])
            .assign(&block_score);
        offset += width;
    }
    Ok(score)
}

/// Check a candidate joint score against the block layout it must index.
fn validated_joint_score(
    score: Array1<f64>,
    specs: &[ParameterBlockSpec],
    source: &str,
) -> Result<Array1<f64>, CustomFamilyError> {
    let total = specs.iter().map(|spec| spec.design.ncols()).sum::<usize>();
    if score.len() != total || score.iter().any(|value| !value.is_finite()) {
        return Err(CustomFamilyError::trial_point(format!(
            "terminal {source} has length {} with finite={}, expected {total}",
            score.len(),
            score.iter().all(|value| value.is_finite()),
        )));
    }
    Ok(score)
}

/// The exact `∇ℓ(β̂)` at the certified mode, from whichever evidence the inner
/// solve retained.
///
/// The three sources are the same quantity in the same layout — block-major,
/// `spec.design.ncols()` wide, each block's log-likelihood score. They are
/// tried in the order that leaves every previously-succeeding assembly on the
/// source it already used: working sets, then the joint workspace, then the
/// score the inner solve carried out. The carried score therefore runs only
/// where this function used to return an error.
///
/// It is what makes the lookup total. A family with an analytic joint gradient
/// produces no `FamilyEvaluation` at all, so `load_joint_gradient_evaluation`
/// hands back a gradient and no evaluation and there are no working sets; and
/// the joint workspace is retained only when the inner solve requested one,
/// which a `use_joint_newton` family without a workspace source never does.
/// Those two facts together left the score unreachable for exactly the
/// families that compute it most directly — a converged, certified
/// constrained mode discarded for want of a vector the solve had already
/// computed (gam#2474).
fn terminal_likelihood_score(
    workspace: Option<&Arc<dyn ExactNewtonJointHessianWorkspace>>,
    working_sets: Option<&[BlockWorkingSet]>,
    retained_score: Option<&TerminalLikelihoodScore>,
    specs: &[ParameterBlockSpec],
    states: &[ParameterBlockState],
) -> Result<Array1<f64>, CustomFamilyError> {
    if let Some(working_sets) = working_sets {
        return terminal_score_from_working_sets(working_sets, specs, states);
    }
    if let Some(evaluation) = workspace
        .map(|workspace| workspace.joint_gradient_evaluation())
        .transpose()?
        .flatten()
    {
        return validated_joint_score(evaluation.gradient, specs, "workspace gradient");
    }
    let retained = retained_score.ok_or_else(|| {
        CustomFamilyError::trial_point(
            "constrained posterior requires the exact terminal likelihood score, but the certified \
             mode retained no working sets, no carried joint score, and no joint workspace exposing \
             a gradient evaluation",
        )
    })?;
    if !retained.evaluated_at(states) {
        return Err(CustomFamilyError::trial_point(
            "constrained posterior requires the exact terminal likelihood score, but the \
             retained joint score was evaluated at a different coefficient vector than the \
             mode being assembled",
        ));
    }
    validated_joint_score(
        retained.score.clone(),
        specs,
        "retained joint likelihood score",
    )
}

/// Assemble the one terminal posterior identity consumed by reporting,
/// prediction, sampling, EDF, and saved-model replay.
///
/// The accepted Hessian, Jeffreys augmentation, likelihood score, penalty
/// score, inequality system, ambient centre, truncated moments, and optional
/// dense covariance are computed once in the same active coefficient frame.
/// This prevents the former split path from certifying one operator for
/// geometry while inverting a different active-face operator for covariance.
pub(crate) fn compute_joint_posterior<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &[ParameterBlockState],
    per_block_log_lambdas: &[Array1<f64>],
    options: &BlockwiseFitOptions,
    preferred_unpenalized_hessian: Option<&Array2<f64>>,
    preferred_working_sets: Option<&[BlockWorkingSet]>,
    preferred_workspace: Option<&Arc<dyn ExactNewtonJointHessianWorkspace>>,
    preferred_likelihood_score: Option<&TerminalLikelihoodScore>,
) -> Result<JointPosteriorAssembly, CustomFamilyError> {
    if specs.len() != per_block_log_lambdas.len() {
        return Err(CustomFamilyError::trial_point(format!(
            "terminal posterior has {} parameter blocks but {} per-block smoothing vectors",
            specs.len(),
            per_block_log_lambdas.len(),
        )));
    }
    if let Some(working_sets) = preferred_working_sets
        && working_sets.len() != specs.len()
    {
        return Err(CustomFamilyError::trial_point(format!(
            "terminal posterior has {} parameter blocks but {} owned working sets",
            specs.len(),
            working_sets.len(),
        )));
    }

    let total = specs.iter().map(|spec| spec.design.ncols()).sum();
    let unpenalized_hessian = preferred_unpenalized_hessian
        .ok_or_else(|| {
            "terminal posterior requires the exact Hessian owned by the certified mode; \
             re-evaluating a possibly stateful family at assembly is forbidden"
                .to_string()
        })?
        .clone();
    let mut precision = penalized_hessian_from_owned_mode(
        specs,
        per_block_log_lambdas,
        options,
        &unpenalized_hessian,
    )?;
    let mode = flatten_state_betas(states, specs);
    let penalty_score = (&precision - &unpenalized_hessian).dot(&mode);
    let mut jeffreys_gradient = Array1::<f64>::zeros(total);
    if family.joint_jeffreys_term_required() {
        let jeffreys_ranges = block_param_ranges(specs);
        if let Some(z_joint) =
            crate::jeffreys::build_joint_jeffreys_subspace(family, specs, &jeffreys_ranges)?
            && let Some((_, gradient, hphi, completion)) =
                crate::jeffreys::custom_family_joint_jeffreys_term_with_exact_completion(
                    family,
                    states,
                    specs,
                    &jeffreys_ranges,
                    &z_joint,
                )?
        {
            if gradient.len() != total
                || hphi.dim() != (total, total)
                || completion.dim() != (total, total)
            {
                return Err(CustomFamilyError::trial_point(format!(
                    "terminal Jeffreys geometry has gradient/divided-difference/completion shapes {}/{:?}/{:?}, expected {total}/({total}, {total})/({total}, {total})",
                    gradient.len(),
                    hphi.dim(),
                    completion.dim(),
                )));
            }
            jeffreys_gradient = gradient;
            precision += &hphi;
            precision += &completion;
            symmetrize_dense_in_place(&mut precision);
        }
    }

    // A single diagonal working set is the only live row-wise contract. A
    // multi-block fit has several distinct row measures, and Exact-Newton
    // curvature lives in coefficient space.
    let working = if specs.len() == 1 {
        match preferred_working_sets {
            None => None,
            Some(
                [
                    BlockWorkingSet::Diagonal {
                        working_response,
                        working_weights,
                    },
                ],
            ) => Some(WorkingGeometry {
                weights: working_weights.clone(),
                response: working_response.clone(),
            }),
            Some([BlockWorkingSet::ExactNewton { .. }]) => None,
            Some([BlockWorkingSet::NaturalDiagonal { .. }]) => None,
            Some(working_sets) => {
                return Err(CustomFamilyError::trial_point(format!(
                    "single-block terminal geometry requires exactly one owned working set, got {}",
                    working_sets.len(),
                )));
            }
        }
    } else {
        None
    };

    let p = precision.nrows();
    let block_constraints = collect_block_linear_constraints(family, states, specs)?;
    let ranges = block_param_ranges(specs);
    let joint_constraints =
        crate::blockwise_solve::assemble_joint_linear_constraints(&block_constraints, &ranges, p)?;

    let block_widths = specs
        .iter()
        .map(|spec| spec.design.ncols())
        .collect::<Vec<_>>();
    let (covariance_conditional, constrained_posterior, reported_beta) = match joint_constraints {
        None => {
            let covariance = options
                .compute_covariance
                .then(|| {
                    spd_covariance_from_precision(
                        &precision,
                        "full posterior precision H + S_λ + H_Φ + H_completion",
                    )
                })
                .transpose()?;
            (covariance, None, None)
        }
        Some(constraints) => {
            let constraints = constraints.to_dense()?;
            // #2442: this route reaches the truncated posterior only through
            // `Σ = (H + S_λ + H_Φ + H_completion)⁻¹`, so it needs a PROPER
            // ambient Gaussian.
            // A constrained mode is not obliged to supply one. Constrained
            // optimality requires `dᵀHd > 0` only along the FEASIBLE cone —
            // copositivity — and the Gaussian location-scale observed
            // information is structurally indefinite off it (its per-row
            // block `[[κ, 2rκ],[2rκ, 2r²κ]]` has determinant `−2r²κ² < 0`,
            // the case #2387 documents). There the cone-truncated posterior
            // still EXISTS and is proper; this decomposition simply cannot
            // reach its moments.
            //
            // So decline the COVARIANCE CHANNEL, not the fit. The
            // optimization converged and its coefficients are honest; a
            // derived quantity being unreachable by one particular route is
            // not a fit-quality failure, and promoting it to one deletes a
            // converged model.
            //
            // Do NOT substitute the active-face reduction here. It is the
            // `λ → ∞` endpoint of the very formula below and reports exactly
            // zero variance in every constraint-normal direction — the #748
            // fabrication this path exists to remove. A visible decline
            // beats a silent wrong number.
            let ambient = match spd_covariance_from_precision(
                &precision,
                "ambient constrained-posterior precision H + S_λ + H_Φ + H_completion",
            ) {
                Ok(ambient) => ambient,
                Err(reason) => {
                    // #2442/#2529: the decline above is decided on the AMBIENT
                    // precision, but the quantity that decides whether there is
                    // anything to compute is a different one — whether `H` is
                    // strictly copositive on the recession cone `{Ad ≥ 0}`. Ask
                    // it here, exactly, so the decline names what it was decided
                    // against instead of only reporting the absence of a
                    // covariance. Two outcomes are genuinely different:
                    // an UNREACHABLE proper posterior (this route's limitation,
                    // #2529's quadrature) and an IMPROPER one (no posterior to
                    // report, whatever the route).
                    let properness = match gam_solve::cone_reduction::cone_properness_certificate(
                        precision.view(),
                        constraints.a.view(),
                        f64::EPSILON.sqrt(),
                    ) {
                        Ok(certificate) => {
                            gam_solve::constrained_posterior::ConePropernessEvidence::Certificate(
                                certificate,
                            )
                        }
                        Err(error) => {
                            gam_solve::constrained_posterior::ConePropernessEvidence::CertificationFailed {
                                reason: error,
                            }
                        }
                    };
                    let cone_verdict = properness.summary();
                    if properness.is_proper() == Some(false) {
                        return Err(CustomFamilyError::trial_point(format!(
                            "constrained fit converged at a point whose cone-truncated posterior \
                             is provably IMPROPER, so no posterior covariance exists to report: \
                             {cone_verdict}. The ambient gate saw only ({reason})"
                        )));
                    }
                    log::warn!(
                        "[custom-family covariance] constrained fit converged, but its \
                         ambient posterior precision is not positive definite, so the \
                         inequality-truncated covariance is unreachable by this route \
                         ({reason}); the cone itself was certified separately and \
                         {cone_verdict}; retaining the converged constrained MODE under a \
                         typed posterior-moment decline (#2635)"
                    );
                    let constrained =
                        gam_solve::constrained_posterior::ConstrainedPosteriorGeometry::with_decline(
                            constraints,
                            mode,
                            gam_solve::constrained_posterior::ConePosteriorMomentDecline {
                                // Display boundary (gam#2689): gam-solve's
                                // decline carries the reason as text.
                                ambient_precision_failure: reason.to_string(),
                                properness,
                            },
                        );
                    constrained.validate_for_dimension(p)?;
                    return Ok(JointPosteriorAssembly {
                        covariance_conditional: None,
                        geometry: FitGeometry {
                            coefficient_gauge: gam_problem::gauge::Gauge::identity(&block_widths),
                            penalized_hessian: precision.into(),
                            constrained_posterior: Some(constrained),
                            working,
                        },
                        // The only available coefficient location is the mode.
                        // Its typed decline prevents every posterior-mean model
                        // assembler and predictor from consuming it as a mean.
                        reported_beta: None,
                    });
                }
            };
            let likelihood_score = terminal_likelihood_score(
                preferred_workspace,
                preferred_working_sets,
                preferred_likelihood_score,
                specs,
                states,
            )?;
            let penalized_gradient = &penalty_score - &likelihood_score - &jeffreys_gradient;
            let unconstrained_center = &mode - &ambient.dot(&penalized_gradient);
            let correction =
                gam_solve::constrained_posterior::constrained_posterior_correction_from_covariance(
                    &ambient,
                    &unconstrained_center,
                    &constraints,
                )?;
            let constrained =
                gam_solve::constrained_posterior::ConstrainedPosteriorGeometry::with_moments(
                    constraints,
                    mode,
                    unconstrained_center,
                    correction,
                );
            constrained.validate_for_dimension(p)?;
            let reported = constrained.posterior_mean()?;
            let covariance = if options.compute_covariance {
                Some(
                    constrained
                        .correction()?
                        .map(|value| value.apply_to_covariance(&ambient))
                        .unwrap_or(ambient),
                )
            } else {
                None
            };
            (covariance, Some(constrained), Some(reported))
        }
    };

    Ok(JointPosteriorAssembly {
        covariance_conditional,
        geometry: FitGeometry {
            coefficient_gauge: gam_problem::gauge::Gauge::identity(&block_widths),
            penalized_hessian: precision.into(),
            constrained_posterior,
            working,
        },
        reported_beta,
    })
}

pub(crate) fn install_reported_posterior_mean<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &mut [ParameterBlockState],
    reported_beta: Option<&Array1<f64>>,
) -> Result<(), CustomFamilyError> {
    let Some(reported_beta) = reported_beta else {
        return Ok(());
    };
    set_states_from_flat_beta(states, specs, reported_beta)?;
    refresh_all_block_etas(family, specs, states)
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
) -> Result<(f64, Option<PenaltySubspaceTrace>), CustomFamilyError> {
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
        .map_err(|e| CustomFamilyError::trial_point(format!("joint penalty subspace eigendecomposition failed: {e}")))?
        .0;
    let s_threshold = positive_eigenvalue_threshold(
        s_evals
            .as_slice()
            .expect("eigh returns an owned standard-layout eigenvalue vector"),
    );
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
    // on the coupled (slope) block while the objective stalls, so ARC never
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
    let m_slice = m_evals
        .as_slice()
        .expect("eigh returns an owned standard-layout eigenvalue vector");
    let m_threshold = positive_eigenvalue_threshold(m_slice);
    let logdet = exact_pseudo_logdet(m_slice, m_threshold);
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
            // Filled by the caller, which is the only place that holds the
            // operator's own `logdet()` this pseudo-determinant replaces.
            logdet_correction: 0.0,
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
///
/// When EVERY outer coordinate is excluded the answer is not an absence: with
/// no free ρ direction `Var(ρ)` is the zero-dimensional zero matrix, so the
/// correction is exactly `0` at identified rank `0` and `V_c = V_cond`. That
/// is returned as a value, so an exactly-known zero correction is never
/// confused with an undefined one.
pub(crate) fn joint_smoothing_correction(
    v_cond: &Array2<f64>,
    specs: &[ParameterBlockSpec],
    layout: &crate::penalty_labels::PenaltyLabelLayout,
    rho_outer: &Array1<f64>,
    block_states: &[ParameterBlockState],
    outer_hessian: &Array2<f64>,
    excluded_outer: &[usize],
) -> Result<Option<(Array2<f64>, usize)>, CustomFamilyError> {
    let p_total: usize = specs.iter().map(|spec| spec.design.ncols()).sum();
    let k_outer = rho_outer.len();
    if v_cond.dim() != (p_total, p_total) {
        return Err(CustomFamilyError::trial_point(format!(
            "joint smoothing correction: V_cond shape {:?} ≠ ({p_total}, {p_total})",
            v_cond.dim()
        )));
    }
    if outer_hessian.dim() != (k_outer, k_outer) {
        return Err(CustomFamilyError::trial_point(format!(
            "joint smoothing correction: outer Hessian shape {:?} ≠ ({k_outer}, {k_outer})",
            outer_hessian.dim()
        )));
    }
    if block_states.len() != specs.len() {
        return Err(CustomFamilyError::trial_point(format!(
            "joint smoothing correction: {} block states vs {} specs",
            block_states.len(),
            specs.len()
        )));
    }

    // β̂ stacked in block order — the reduced coefficient frame V_cond lives in.
    let mut beta_flat = Array1::<f64>::zeros(p_total);
    let mut offsets = Vec::with_capacity(specs.len() + 1);
    let mut at = 0usize;
    for (spec, state) in specs.iter().zip(block_states) {
        let width = spec.design.ncols();
        if state.beta.len() != width {
            return Err(CustomFamilyError::trial_point(format!(
                "joint smoothing correction: block '{}' beta length {} ≠ design width {width}",
                spec.name,
                state.beta.len()
            )));
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
                return Err(CustomFamilyError::trial_point(format!(
                    "joint smoothing correction: block '{}' penalty shape {:?} ≠ ({width}, {width})",
                    spec.name,
                    s_dense.dim()
                )));
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
            return Err(CustomFamilyError::trial_point(format!(
                "joint smoothing correction: joint penalty '{}' shape {:?} ≠ ({p_total}, {p_total})",
                spec.label.as_deref().unwrap_or("<unlabeled>"),
                spec.matrix.dim()
            )));
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
        // Every outer coordinate is railed: no free rho direction survives, so
        // Var(rho) is the zero-dimensional zero matrix and the inflation
        // C = A Var(rho) A^T is EXACTLY the p x p zero matrix at identified
        // interior rank 0. This is an exact value, not a missing one — the same
        // identity `gam-predict/src/lib.rs:325-336` relies on when `lambdas` is
        // empty ("Vp = Vb exactly ... an identity, not a fallback to a weaker
        // uncertainty definition"). Returning `Ok(None)` here made it
        // indistinguishable from the genuinely-undefined non-PD-interior case
        // below and left the fit reporting no corrected covariance at all.
        return Ok(Some((Array2::<f64>::zeros((p_total, p_total)), 0)));
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
mod required_covariance_tests {
    //! Pins the #2299 posterior-completeness invariant at its production seam.
    //! A converged mode whose joint posterior precision `M = H + S_λ` is
    //! singular cannot define the required posterior mean. Covariance
    //! factorization must therefore refuse fit assembly, never mint a
    //! mode-only artifact. The fixture is genuinely singular (not a marginal
    //! knife-edge), so the assertion is deterministic and load-independent.
    use super::*;
    use ndarray::array;

    #[test]
    fn posterior_inverse_preserves_small_positive_curvature() {
        let precision = array![[1.0e-12, 2.0e-7], [2.0e-7, 1.0]];
        let covariance = spd_covariance_from_precision(&precision, "small score units").unwrap();
        let expected = array![[1.0e12, -2.0e5], [-2.0e5, 1.0]] / 0.96;
        for (&actual, &expected) in covariance.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() <= 1e-12 * expected.abs());
        }
        for precision in [array![[1., 1.], [1., 1.]], array![[1., 2.], [2., 1.]]] {
            assert!(spd_covariance_from_precision(&precision, "invalid precision").is_err());
        }
    }

    #[derive(Clone)]
    struct TrivialFamily;

    impl CustomFamily for TrivialFamily {
        fn evaluate(&self, states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
            if states.len() != 1 {
                return Err(format!(
                    "TrivialFamily is a single-block fixture; got {} blocks",
                    states.len()
                ));
            }
            Ok(FamilyEvaluation {
                log_likelihood: 0.0,
                blockworking_sets: vec![],
            })
        }
    }

    /// One-dimensional Jeffreys fixture whose information is locally
    /// `I(beta) = 0.5 - 2 beta²` at `beta = 0`. Its first derivative vanishes,
    /// so the divided-difference `H_Phi` is exactly zero, while the omitted
    /// second-directional completion is
    /// `-0.5 * I'' / I = 4`. This isolates the terminal-covariance seam: a
    /// precision assembled from only `H + H_Phi` is 1, while the true
    /// inner-objective precision is 5.
    #[derive(Clone)]
    struct CompletionOnlyJeffreysFamily;

    impl CustomFamily for CompletionOnlyJeffreysFamily {
        fn evaluate(&self, states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
            if states.len() != 1 {
                return Err(format!(
                    "CompletionOnlyJeffreysFamily is a single-block fixture; got {} blocks",
                    states.len()
                ));
            }
            Ok(FamilyEvaluation {
                log_likelihood: -0.5 * states[0].beta[0].powi(2),
                blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                    gradient: array![-states[0].beta[0]],
                    hessian: SymmetricMatrix::Dense(array![[1.0]]),
                }],
            })
        }

        fn joint_jeffreys_term_required(&self) -> bool {
            true
        }

        fn joint_jeffreys_information_with_specs(
            &self,
            states: &[ParameterBlockState],
            specs: &[ParameterBlockSpec],
        ) -> Result<Option<Array2<f64>>, String> {
            assert_eq!(states.len(), 1);
            assert_eq!(specs.len(), 1);
            Ok(Some(array![[0.5]]))
        }

        fn joint_jeffreys_information_directional_derivative_all_axes_with_specs(
            &self,
            states: &[ParameterBlockState],
            specs: &[ParameterBlockSpec],
        ) -> Result<Option<Vec<Array2<f64>>>, String> {
            assert_eq!(states.len(), 1);
            assert_eq!(specs.len(), 1);
            Ok(Some(vec![array![[0.0]]]))
        }

        fn joint_jeffreys_information_contracted_trace_hessian_with_specs(
            &self,
            states: &[ParameterBlockState],
            specs: &[ParameterBlockSpec],
            weight: &Array2<f64>,
        ) -> Result<Option<Array2<f64>>, String> {
            assert_eq!(states.len(), 1);
            assert_eq!(specs.len(), 1);
            Ok(Some(array![[-4.0 * weight[[0, 0]]]]))
        }

        fn joint_jeffreys_information_contracted_trace_hessian_available(&self) -> bool {
            true
        }
    }

    #[test]
    fn terminal_covariance_includes_exact_jeffreys_completion_2612() {
        let spec = ParameterBlockSpec {
            name: "completion-only-jeffreys".to_string(),
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                Array2::zeros((1, 1)),
            )),
            offset: Array1::zeros(1),
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(array![0.0]),
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        };
        let states = vec![ParameterBlockState {
            beta: array![0.0],
            eta: array![0.0],
        }];
        let posterior = compute_joint_posterior(
            &CompletionOnlyJeffreysFamily,
            &[spec],
            &states,
            &[Array1::zeros(0)],
            &BlockwiseFitOptions {
                compute_covariance: true,
                ..BlockwiseFitOptions::default()
            },
            Some(&array![[1.0]]),
            None,
            None,
            None,
        )
        .expect("completion-augmented posterior");
        assert_eq!(
            posterior.geometry.penalized_hessian.as_array(),
            &array![[5.0]],
            "terminal geometry must expose H + H_Phi + H_completion"
        );
        let covariance = posterior
            .covariance_conditional
            .expect("requested covariance");
        assert!(
            (covariance[[0, 0]] - 0.2).abs() <= 1.0e-12,
            "inverse true precision should be 1/5, got {}",
            covariance[[0, 0]]
        );
    }

    #[derive(Clone)]
    struct LowerBoundedQuadratic;

    impl CustomFamily for LowerBoundedQuadratic {
        fn evaluate(&self, states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
            let beta = states[0].beta[0];
            Ok(FamilyEvaluation {
                log_likelihood: -0.5 * (beta + 1.0).powi(2),
                blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                    gradient: array![-1.0 - beta],
                    hessian: SymmetricMatrix::Dense(array![[1.0]]),
                }],
            })
        }

        fn block_linear_constraints(
            &self,
            states: &[ParameterBlockState],
            block_idx: usize,
            spec: &ParameterBlockSpec,
        ) -> Result<Option<ConstraintSet>, String> {
            assert_eq!(block_idx, 0);
            assert_eq!(
                states[block_idx].beta.len(),
                1,
                "block `{}` must carry the single bounded coefficient this constraint row addresses",
                spec.name
            );
            Ok(Some(ConstraintSet::Dense(LinearInequalityConstraints {
                a: array![[1.0]],
                b: array![0.0],
            })))
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

    /// One 2-coefficient block with a lower bound on the first coordinate and an
    /// unpenalized Hessian whose ambient form is INDEFINITE. `H + S_λ` is
    /// `[[1, 0], [0, -2]]`: strictly positive along the constrained coordinate
    /// and negative along the free one, which is the shape a constrained
    /// optimum takes when curvature is blocked by an active bound (#2387's
    /// Gaussian location-scale case, whose per-row information has determinant
    /// `−2r²κ² < 0`).
    fn indefinite_ambient_constrained_fixture() -> (
        Vec<ParameterBlockSpec>,
        Vec<ParameterBlockState>,
        Vec<Array1<f64>>,
        Array2<f64>,
    ) {
        // `[[1, 3], [3, 1]]` has eigenvalues `-2` and `4`, so the AMBIENT
        // precision is indefinite and the SPD gate must refuse it — and it is
        // strictly copositive on the nonnegative orthant, so the posterior
        // truncated to `{beta >= 0}` is PROPER. Both halves are load-bearing:
        // the test's whole claim is that a proper-but-unreachable law costs the
        // covariance channel and nothing else, and a fixture whose law does not
        // exist cannot exhibit that.
        //
        // TWO constraint rows are not a stylistic choice. With one inequality
        // the recession cone is a half-space, which contains a full line
        // through the origin, so it contains `d` or `-d` for EVERY direction —
        // an indefinite `H` is then improper on it no matter which coordinate
        // the row touches. Properness with an indefinite ambient needs a cone
        // salient enough to exclude the negative direction together with its
        // negation, and that takes at least two rows.
        let unpenalized = array![[1.0, 3.0], [3.0, 1.0]];
        let beta = array![0.25, 0.5];
        let spec = ParameterBlockSpec {
            name: "indefinite-ambient".to_string(),
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                Array2::zeros((1, 2)),
            )),
            offset: Array1::zeros(1),
            penalties: vec![PenaltyMatrix::Dense(Array2::zeros((2, 2)))],
            nullspace_dims: vec![2],
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

    #[derive(Clone)]
    struct TwoCoefficientLowerBounded;

    impl CustomFamily for TwoCoefficientLowerBounded {
        fn evaluate(&self, states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
            let beta = states[0].beta.clone();
            Ok(FamilyEvaluation {
                log_likelihood: 0.0,
                blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                    gradient: array![-beta[0] - 3.0 * beta[1], -3.0 * beta[0] - beta[1]],
                    hessian: SymmetricMatrix::Dense(array![[1.0, 3.0], [3.0, 1.0]]),
                }],
            })
        }

        fn block_linear_constraints(
            &self,
            states: &[ParameterBlockState],
            block_idx: usize,
            spec: &ParameterBlockSpec,
        ) -> Result<Option<ConstraintSet>, String> {
            assert_eq!(block_idx, 0);
            assert_eq!(
                states[block_idx].beta.len(),
                2,
                "block `{}` must carry both coefficients the constraint row spans",
                spec.name
            );
            Ok(Some(ConstraintSet::Dense(LinearInequalityConstraints {
                a: array![[1.0, 0.0], [0.0, 1.0]],
                b: array![0.0, 0.0],
            })))
        }
    }

    /// #2442, and the executable form of a boundary that a comment could not
    /// hold: an indefinite AMBIENT precision must cost the covariance channel
    /// and nothing else.
    ///
    /// Two ways to break this, and the test fails on both.
    ///
    /// * Propagate the SPD refusal (a bare `?` on the ambient inverse) and the
    ///   whole fit disappears. The optimization converged; a derived quantity
    ///   being unreachable by one route is not a fit-quality failure, and
    ///   #2387 already showed what refusing here costs — one Gaussian
    ///   location-scale wiggle configuration stopped assembling.
    /// * Substitute the active-face reduction and a covariance comes back that
    ///   reports exactly zero variance in every constraint-normal direction.
    ///   That is the `λ → ∞` endpoint of the truncated formula and the #748
    ///   fabrication this path exists to remove.
    ///
    /// The honest answer is neither: report the fit, decline the covariance,
    /// and leave the estimand gap visible until the cone-truncated moments can
    /// be reached without inverting an indefinite `H`.
    ///
    /// The fixture was replaced rather than tuned. It used to be
    /// `H = diag(1, -2)` under the single row `beta_0 >= 0`, whose cone-truncated
    /// posterior is IMPROPER — `d = (0, 1)` is feasible with `d'Hd = -2` — so the
    /// `expect` message below was asserting the opposite of what its own fixture
    /// exhibited, and the test passed only because a declined covariance channel
    /// looked the same either way. That fixture now has its own test, asserting
    /// the refusal, directly below. See
    /// `a_cone_improper_posterior_is_refused_by_name_rather_than_declined`.
    #[test]
    fn indefinite_ambient_precision_declines_the_covariance_and_keeps_the_fit() {
        let (specs, states, per_block, unpenalized) = indefinite_ambient_constrained_fixture();
        let options = BlockwiseFitOptions {
            compute_covariance: true,
            ..BlockwiseFitOptions::default()
        };
        let assembly = compute_joint_posterior(
            &TwoCoefficientLowerBounded,
            &specs,
            &states,
            &per_block,
            &options,
            Some(&unpenalized),
            None,
            None,
            None,
        )
        .expect(
            "a converged constrained fit whose ambient precision is indefinite must still \
             assemble: the posterior is proper on the feasible cone and only this ROUTE to \
             its moments is unavailable",
        );
        assert!(
            assembly.covariance_conditional.is_none(),
            "the covariance channel must be declined, not filled with the active-face \
             answer: a zero-variance constraint-normal direction is the #748 fabrication, \
             got {:?}",
            assembly.covariance_conditional,
        );
        let constrained = assembly
            .geometry
            .constrained_posterior
            .as_ref()
            .expect("the fitted cone identity must survive the moment decline");
        let decline = constrained
            .decline()
            .expect("an ambient-indefinite proper cone must carry a typed moment decline");
        assert_eq!(
            decline.properness.is_proper(),
            Some(true),
            "the decline must preserve the live proof that the cone posterior exists",
        );
        assert_eq!(
            constrained.constraints.a,
            Array2::<f64>::eye(2),
            "the declaration that makes the indefinite posterior proper must not be erased",
        );
        assert!(
            assembly.reported_beta.is_none(),
            "the posterior mean is a function of the same unreachable ambient covariance, so \
             the caller must keep the converged mode rather than receive a fabricated centre",
        );
        assert_eq!(
            assembly.geometry.penalized_hessian.as_array(),
            &array![[1.0, 3.0], [3.0, 1.0]],
            "the full-space precision is still the honest curvature of the fit and must be \
             reported unchanged",
        );
    }

    /// The fixture the test above used to carry, kept because it is a
    /// counterexample rather than a variant: `H = diag(1, -2)` with the single
    /// row `beta_0 >= 0`. The recession cone is the half-space `{d_0 >= 0}`,
    /// which contains the whole `d_0 = 0` line, so `d = (0, 1)` is feasible with
    /// `d'Hd = -2`. The cone-truncated posterior is IMPROPER — its mass diverges
    /// along a direction the constraint does not touch.
    ///
    /// It sat under an assertion that the posterior "is proper on the feasible
    /// cone and only this ROUTE to its moments is unavailable". That reading is
    /// false here by one line of arithmetic, and nothing in the old code could
    /// tell the two apart: both an unreachable proper law and a nonexistent one
    /// came back as a declined covariance channel. So this fixture must produce
    /// the OTHER answer, and it must produce it for a stated reason.
    ///
    /// Declining the channel here would report a fit whose uncertainty is
    /// unbounded inside its own feasible set, which is the same class of
    /// fabrication as the zero-variance active-face answer the decline exists to
    /// avoid — in the opposite direction.
    #[test]
    fn a_cone_improper_posterior_is_refused_by_name_rather_than_declined() {
        let (specs, states, per_block, unpenalized) = cone_improper_constrained_fixture();
        let options = BlockwiseFitOptions {
            compute_covariance: true,
            ..BlockwiseFitOptions::default()
        };
        let message = compute_joint_posterior(
            &OneCoefficientLowerBoundedIndefinite,
            &specs,
            &states,
            &per_block,
            &options,
            Some(&unpenalized),
            None,
            None,
            None,
        )
        .expect_err("a provably improper cone-truncated posterior has no covariance to decline");
        assert!(
            message.to_string().contains("IMPROPER"),
            "the refusal must carry the verdict, got: {message}"
        );
        assert!(
            message.to_string().contains("In(ZᵀHZ)"),
            "the refusal must name the quantity that decided it — the inertia on null(A),              which is where this fixture's negative direction lives — got: {message}"
        );
        // The ambient gate's own numbers must survive into the message too: they
        // are what triggered the branch, and a refusal that replaced them with
        // the cone verdict would lose the reason the route was abandoned.
        assert!(
            message.to_string().contains("non-PD at the converged optimum"),
            "got: {message}"
        );
    }

    fn cone_improper_constrained_fixture() -> (
        Vec<ParameterBlockSpec>,
        Vec<ParameterBlockState>,
        Vec<Array1<f64>>,
        Array2<f64>,
    ) {
        let unpenalized = array![[1.0, 0.0], [0.0, -2.0]];
        let beta = array![0.0, 0.5];
        let spec = ParameterBlockSpec {
            name: "cone-improper".to_string(),
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                Array2::zeros((1, 2)),
            )),
            offset: Array1::zeros(1),
            penalties: vec![PenaltyMatrix::Dense(Array2::zeros((2, 2)))],
            nullspace_dims: vec![2],
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

    #[derive(Clone)]
    struct OneCoefficientLowerBoundedIndefinite;

    impl CustomFamily for OneCoefficientLowerBoundedIndefinite {
        fn evaluate(&self, states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
            let beta = states[0].beta.clone();
            Ok(FamilyEvaluation {
                log_likelihood: 0.0,
                blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                    gradient: array![-beta[0], 2.0 * beta[1]],
                    hessian: SymmetricMatrix::Dense(array![[1.0, 0.0], [0.0, -2.0]]),
                }],
            })
        }

        fn block_linear_constraints(
            &self,
            states: &[ParameterBlockState],
            block_idx: usize,
            spec: &ParameterBlockSpec,
        ) -> Result<Option<ConstraintSet>, String> {
            assert_eq!(block_idx, 0);
            assert_eq!(
                states[block_idx].beta.len(),
                2,
                "block `{}` must carry both coefficients the constraint row spans",
                spec.name
            );
            Ok(Some(ConstraintSet::Dense(LinearInequalityConstraints {
                a: array![[1.0, 0.0]],
                b: array![0.0],
            })))
        }
    }

    #[test]
    fn required_covariance_errors_on_singular_posterior_precision() {
        let (specs, states, per_block, unpenalized) = singular_joint_fixture();
        let options = BlockwiseFitOptions {
            compute_covariance: true,
            ..BlockwiseFitOptions::default()
        };
        let result = compute_joint_posterior(
            &TrivialFamily,
            &specs,
            &states,
            &per_block,
            &options,
            Some(&unpenalized),
            None,
            None,
            None,
        );
        assert!(
            result.is_err(),
            "a singular joint posterior must refuse fit assembly; got {result:?}"
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
            ..BlockwiseFitOptions::default()
        };
        let result = compute_joint_posterior(
            &TrivialFamily,
            &vec![spec],
            &states,
            &vec![array![0.0]],
            &options,
            Some(&unpenalized),
            None,
            None,
            None,
        );
        match result {
            Ok(JointPosteriorAssembly {
                covariance_conditional: Some(cov),
                ..
            }) => {
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
            ..BlockwiseFitOptions::default()
        };
        let result = compute_joint_posterior(
            &TrivialFamily,
            &specs,
            &states,
            &per_block,
            &options,
            Some(&unpenalized),
            None,
            None,
            None,
        );
        assert!(matches!(
            result,
            Ok(JointPosteriorAssembly {
                covariance_conditional: None,
                ..
            })
        ));
    }

    #[test]
    fn inequality_reports_the_truncated_mean_and_nonzero_normal_variance() {
        let spec = ParameterBlockSpec {
            name: "lower-bounded".to_string(),
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                Array2::zeros((1, 1)),
            )),
            offset: Array1::zeros(1),
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: array![],
            initial_beta: Some(array![0.0]),
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        };
        let states = vec![ParameterBlockState {
            beta: array![0.0],
            eta: array![0.0],
        }];
        let working_sets = vec![BlockWorkingSet::ExactNewton {
            gradient: array![-1.0],
            hessian: SymmetricMatrix::Dense(array![[1.0]]),
        }];
        let posterior = compute_joint_posterior(
            &LowerBoundedQuadratic,
            &[spec],
            &states,
            &[Array1::<f64>::zeros(0)],
            &BlockwiseFitOptions {
                compute_covariance: true,
                ..BlockwiseFitOptions::default()
            },
            Some(&array![[1.0]]),
            Some(working_sets.as_slice()),
            None,
            None,
        )
        .expect("proper lower-truncated Gaussian posterior");
        let constrained = posterior
            .geometry
            .constrained_posterior
            .as_ref()
            .expect("exact inequality geometry");
        assert_eq!(constrained.mode, array![0.0]);
        assert_eq!(
            constrained
                .unconstrained_center()
                .expect("available constrained posterior centre"),
            array![-1.0]
        );
        let reported = posterior.reported_beta.expect("posterior mean");
        assert!(
            reported[0] > 0.0,
            "posterior mean must lie strictly inside the half-line, got {}",
            reported[0],
        );
        let variance = posterior
            .covariance_conditional
            .expect("requested truncated covariance")[[0, 0]];
        assert!(
            variance > 0.0 && variance < 1.0,
            "finite inequality multiplier must leave variance strictly between the active-face \
             zero and the ambient variance one, got {variance}",
        );
    }

    /// The fixture of the two preceding assertions, differing only in which
    /// piece of retained evidence carries the score.
    fn lower_bounded_unit_fixture() -> (ParameterBlockSpec, Vec<ParameterBlockState>) {
        let spec = ParameterBlockSpec {
            name: "lower-bounded".to_string(),
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                Array2::zeros((1, 1)),
            )),
            offset: Array1::zeros(1),
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: array![],
            initial_beta: Some(array![0.0]),
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        };
        let states = vec![ParameterBlockState {
            beta: array![0.0],
            eta: array![0.0],
        }];
        (spec, states)
    }

    fn lower_bounded_posterior(
        spec: ParameterBlockSpec,
        states: &[ParameterBlockState],
        working_sets: Option<&[BlockWorkingSet]>,
        score: Option<&TerminalLikelihoodScore>,
    ) -> Result<JointPosteriorAssembly, CustomFamilyError> {
        compute_joint_posterior(
            &LowerBoundedQuadratic,
            &[spec],
            states,
            &[Array1::<f64>::zeros(0)],
            &BlockwiseFitOptions {
                compute_covariance: true,
                ..BlockwiseFitOptions::default()
            },
            Some(&array![[1.0]]),
            working_sets,
            None,
            score,
        )
    }

    /// gam#2474: a family with an analytic joint gradient produces no
    /// `FamilyEvaluation`, so it retains no working sets; and a
    /// Jeffreys-augmented family skips the returned-mode curvature
    /// certificate, so it retains no joint workspace either. The score the
    /// inner solve already loaded is the only evidence left, and the
    /// constrained posterior must assemble from it — identically to the
    /// working-set route, which is the same numbers by construction.
    #[test]
    fn retained_joint_score_alone_assembles_the_constrained_posterior() {
        let (spec, states) = lower_bounded_unit_fixture();
        let working_sets = vec![BlockWorkingSet::ExactNewton {
            gradient: array![-1.0],
            hessian: SymmetricMatrix::Dense(array![[1.0]]),
        }];
        let from_working_sets =
            lower_bounded_posterior(spec.clone(), &states, Some(working_sets.as_slice()), None)
                .expect("working-set route");

        let retained = TerminalLikelihoodScore {
            beta: array![0.0],
            score: array![-1.0],
        };
        let from_score = lower_bounded_posterior(spec, &states, None, Some(&retained))
            .expect("the retained joint score is sufficient evidence for the truncated posterior");

        let center_ws = from_working_sets
            .geometry
            .constrained_posterior
            .as_ref()
            .expect("exact inequality geometry")
            .unconstrained_center()
            .expect("available constrained posterior centre")
            .clone();
        let center_score = from_score
            .geometry
            .constrained_posterior
            .as_ref()
            .expect("exact inequality geometry")
            .unconstrained_center()
            .expect("available constrained posterior centre")
            .clone();
        assert_eq!(
            center_ws, center_score,
            "the two routes read the same score, so the truncation center must be identical",
        );
        assert_eq!(
            from_working_sets.reported_beta, from_score.reported_beta,
            "posterior mean must not depend on which retained evidence carried the score",
        );
    }

    /// The carried score is only the terminal score at the coefficient vector
    /// it was evaluated at. A mismatch is a solver-ordering defect, so it is
    /// refused by name rather than silently used at the wrong operating point.
    #[test]
    fn retained_joint_score_from_a_different_operating_point_is_refused() {
        let (spec, states) = lower_bounded_unit_fixture();
        let stale = TerminalLikelihoodScore {
            beta: array![0.25],
            score: array![-1.0],
        };
        let error = lower_bounded_posterior(spec, &states, None, Some(&stale))
            .expect_err("a score from another beta cannot certify this mode");
        assert!(
            error.to_string().contains("evaluated at a different coefficient vector"),
            "the refusal must name the operating-point mismatch, got: {error}",
        );
    }
}
