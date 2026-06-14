
fn aggregate_labeled_hessian(
    hessian: &Array2<f64>,
    layout: &PenaltyLabelLayout,
) -> Result<Array2<f64>, String> {
    if hessian.nrows() != layout.physical_count() || hessian.ncols() != layout.physical_count() {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: format!(
                "physical Hessian shape mismatch: got {}x{}, expected {}x{}",
                hessian.nrows(),
                hessian.ncols(),
                layout.physical_count(),
                layout.physical_count()
            ),
        }
        .into());
    }
    let mut out = Array2::<f64>::zeros((layout.initial_rho.len(), layout.initial_rho.len()));
    for (i, oi) in layout.physical_to_outer.iter().enumerate() {
        let Some(oi) = *oi else { continue };
        for (j, oj) in layout.physical_to_outer.iter().enumerate() {
            if let Some(oj) = *oj {
                out[[oi, oj]] += hessian[[i, j]];
            }
        }
    }
    Ok(out)
}


/// Adapter over the shared [`rho_prior_eval`](crate::solver::estimate::reml::rho_prior_eval)
/// engine using the custom-family invalid-prior policy
/// (`HardError`): the prior math is shared with the REML/LAML runtime, and a
/// malformed prior surfaces as a structured [`CustomFamilyError`] rather than
/// being folded into the objective.
fn rho_prior_cost_gradient_hessian(
    prior: &crate::types::RhoPrior,
    rho: &Array1<f64>,
) -> Result<(f64, Array1<f64>, Option<Array2<f64>>), String> {
    use crate::solver::estimate::reml::rho_prior_eval::{InvalidPriorPolicy, RhoPriorError};
    match crate::solver::estimate::reml::rho_prior_eval::evaluate(
        prior,
        rho,
        InvalidPriorPolicy::HardError,
    ) {
        Ok(eval) => Ok((eval.cost, eval.gradient, eval.hessian)),
        Err(RhoPriorError::DimensionMismatch { reason }) => {
            Err(CustomFamilyError::DimensionMismatch { reason }.into())
        }
        Err(RhoPriorError::ConstraintViolation { reason }) => {
            Err(CustomFamilyError::ConstraintViolation { reason }.into())
        }
    }
}


fn add_labeled_rho_prior_to_outer_eval(
    mut result: OuterObjectiveEvalResult,
    rho: &Array1<f64>,
    rho_prior: &crate::types::RhoPrior,
    eval_mode: EvalMode,
) -> Result<OuterObjectiveEvalResult, String> {
    // For tied physical penalties, the likelihood/LAML contribution is first
    // evaluated in the expanded physical coordinates and then pulled back to
    // the user-facing labeled coordinates.  The configured prior lives on the
    // labeled precision itself, so it is added once after that pullback:
    //
    //   V_label(rho) = V_base(E rho) + pi(rho),
    //   ∇V_label     = E' ∇V_base(E rho) + ∇pi(rho),
    //   ∇²V_label    = E' ∇²V_base(E rho) E + ∇²pi(rho),
    //
    // where E maps each physical penalty piece to its outer label.  This is
    // the same change-of-variables identity used for overlapping/nested group
    // penalties; the prior is not repeated for each physical child component.
    if matches!(rho_prior, crate::types::RhoPrior::Flat) {
        return Ok(result);
    }
    let (cost, gradient, hessian) = rho_prior_cost_gradient_hessian(rho_prior, rho)?;
    result.objective += cost;
    if eval_mode != EvalMode::ValueOnly {
        if result.gradient.len() != gradient.len() {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "rho prior gradient length mismatch: got {}, expected {}",
                    gradient.len(),
                    result.gradient.len()
                ),
            }
            .into());
        }
        result.gradient += &gradient;
    }
    if eval_mode == EvalMode::ValueGradientHessian
        && let Some(prior_hessian) = hessian
    {
        result.outer_hessian.add_rho_block_dense(&prior_hessian)?;
    }
    Ok(result)
}


fn physical_warm_start_for_labeled(
    warm_start: Option<&ConstrainedWarmStart>,
    physical_rho: &Array1<f64>,
    layout: &PenaltyLabelLayout,
) -> Option<ConstrainedWarmStart> {
    if !layout.has_tied_coordinates() {
        return None;
    }
    warm_start.map(|seed| {
        let mut physical_seed = seed.clone();
        physical_seed.rho = physical_rho.clone();
        physical_seed
    })
}


fn pullback_labeled_outer_eval(
    mut result: OuterObjectiveEvalResult,
    rho: &Array1<f64>,
    layout: &PenaltyLabelLayout,
    rho_prior: &crate::types::RhoPrior,
    eval_mode: EvalMode,
) -> Result<OuterObjectiveEvalResult, String> {
    if eval_mode == EvalMode::ValueOnly {
        result.gradient = Array1::<f64>::zeros(layout.initial_rho.len());
    } else {
        result.gradient = aggregate_labeled_gradient(&result.gradient, layout)?;
    }
    if eval_mode == EvalMode::ValueGradientHessian {
        result.outer_hessian = match result.outer_hessian {
            crate::solver::outer_strategy::HessianResult::Analytic(hessian) => {
                crate::solver::outer_strategy::HessianResult::Analytic(aggregate_labeled_hessian(
                    &hessian, layout,
                )?)
            }
            crate::solver::outer_strategy::HessianResult::Operator(operator) => {
                crate::solver::outer_strategy::HessianResult::Operator(Arc::new(
                    LabeledOuterHessianOperator::new(operator, layout),
                ))
            }
            crate::solver::outer_strategy::HessianResult::Unavailable => {
                crate::solver::outer_strategy::HessianResult::Unavailable
            }
        };
    }
    result.warm_start.rho = rho.clone();
    add_labeled_rho_prior_to_outer_eval(result, rho, rho_prior, eval_mode)
}


fn outerobjectivegradienthessian_labeled<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    layout: &PenaltyLabelLayout,
    rho: &Array1<f64>,
    warm_start: Option<&ConstrainedWarmStart>,
    rho_prior: &crate::types::RhoPrior,
    eval_mode: EvalMode,
) -> Result<OuterObjectiveEvalResult, String> {
    let physical_rho = expand_labeled_log_lambdas(rho, layout)?;
    let physical_warm_start = physical_warm_start_for_labeled(warm_start, &physical_rho, layout);
    let base = outerobjectivegradienthessian_internal(
        family,
        specs,
        options,
        &layout.penalty_counts,
        &physical_rho,
        physical_warm_start.as_ref().or(warm_start),
        crate::types::RhoPrior::Flat,
        eval_mode,
    )?;
    pullback_labeled_outer_eval(base, rho, layout, rho_prior, eval_mode)
}


fn custom_family_seed_screening_proxy_labeled<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    layout: &PenaltyLabelLayout,
    rho: &Array1<f64>,
    warm_start: Option<&ConstrainedWarmStart>,
    rho_prior: &crate::types::RhoPrior,
) -> Result<(f64, ConstrainedWarmStart, bool), String> {
    let physical_rho = expand_labeled_log_lambdas(rho, layout)?;
    let per_block = split_log_lambdas(&physical_rho, &layout.penalty_counts)?;
    let physical_warm_start = physical_warm_start_for_labeled(warm_start, &physical_rho, layout);
    // Seed screening only RANKS candidate seeds by their penalized objective; it
    // is capped and never produces the final fit. Mark the inner solve as a
    // screening solve so it skips the O(p · per-axis-Hdot) full Jeffreys/Firth
    // curvature loop and keeps only the cheap value-only Jeffreys term in the
    // score (gam#729/#808). For a K-block coupled family (Dirichlet/multinomial)
    // each per-axis directional derivative is O(K²·n·p), so paying the full term
    // for every cascade candidate over the joint width is the wrong cost class
    // and made the coupled fit non-completing in screening alone. The real fit
    // (after a seed is selected) runs with `seed_screening = false`, so the
    // load-bearing Firth curvature is fully present where it matters.
    let screening_options = BlockwiseFitOptions {
        seed_screening: true,
        ..options.clone()
    };
    let mut inner = inner_blockwise_fit(
        family,
        specs,
        &per_block,
        &screening_options,
        physical_warm_start.as_ref().or(warm_start),
    )?;
    refresh_all_block_etas(family, specs, &mut inner.block_states)?;
    let prior_terms = rho_prior_cost_gradient_hessian(rho_prior, rho)?;
    let score = inner_penalized_objective(
        &inner,
        include_exact_newton_logdet_h(family, options),
        include_exact_newton_logdet_s(family, options),
        "custom-family labeled seed-screening proxy",
    )? + prior_terms.0;
    let warm = ConstrainedWarmStart {
        rho: rho.clone(),
        block_beta: inner
            .block_states
            .iter()
            .map(|state| state.beta.clone())
            .collect(),
        active_sets: inner.active_sets.clone(),
        cached_inner: Some(cached_inner_mode_from_result(&inner)),
    };
    Ok((score, warm, inner.converged))
}


fn split_log_lambdas(
    flat: &Array1<f64>,
    penalty_counts: &[usize],
) -> Result<Vec<Array1<f64>>, String> {
    let expected: usize = penalty_counts.iter().sum();
    if flat.len() != expected {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: format!(
                "log-lambda length mismatch: got {}, expected {expected}",
                flat.len()
            ),
        }
        .into());
    }
    let mut out = Vec::with_capacity(penalty_counts.len());
    let mut at = 0usize;
    for &k in penalty_counts {
        out.push(flat.slice(ndarray::s![at..at + k]).to_owned());
        at += k;
    }
    Ok(out)
}


fn buildblock_states<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
) -> Result<Vec<ParameterBlockState>, String> {
    let mut states = Vec::with_capacity(specs.len());
    for (b, spec) in specs.iter().enumerate() {
        let p = spec.design.ncols();
        let beta = spec
            .initial_beta
            .clone()
            .unwrap_or_else(|| Array1::<f64>::zeros(p));
        let eta = with_block_geometry(family, &states, spec, b, |x, off| {
            let mut eta = x.matrixvectormultiply(&beta);
            eta += off;
            Ok(eta)
        })?;
        states.push(ParameterBlockState { beta, eta });
    }
    // After every block state is populated, pass each β through
    // `post_update_block_beta` so the invariant "every `states[b].beta`
    // in `inner_blockwise_fit` is feasible" holds from the first eval
    // call onward — matching the same projection the warm-start seed
    // path at 5932 already applies.  Defers projection to this second
    // pass because some family overrides (e.g.
    // `SurvivalMarginalSlopeFamily::post_update_block_beta`) read
    // `block_states[block_idx]` during projection, and `block_idx == b`
    // is only populated once the first pass has pushed all states.
    //
    // Without this, a caller that supplies `initial_beta = Some(infeasible)`
    // — or leaves it `None` for a family whose zero vector violates the
    // family's bounds — feeds an infeasible β into
    // `exact_newton_joint_hessian` / `evaluate` before the first
    // line-search trial, silently corrupting the fit or tripping
    // `max_feasible_step_size` guards on iteration 1.  The warm-start
    // path (5925-5938) projects on entry for exactly this reason; this
    // extends the invariant to the cold-start path too.
    for b in 0..specs.len() {
        let raw = states[b].beta.clone();
        let projected = family.post_update_block_beta(&states, b, &specs[b], raw)?;
        states[b].beta.assign(&projected);
    }
    // Note: the caller (`inner_blockwise_fit`) calls `refresh_all_block_etas`
    // immediately after this returns, so η is recomputed against the
    // projected β before any family evaluation runs.  We don't duplicate
    // the refresh here.
    Ok(states)
}


fn refresh_all_block_etas<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &mut [ParameterBlockState],
) -> Result<(), String> {
    if family.block_geometry_is_dynamic() {
        for b in 0..specs.len() {
            refresh_single_block_eta(family, specs, states, b)?;
        }
        return Ok(());
    }

    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    let refreshed_etas: Vec<Array1<f64>> = (0..specs.len())
        .into_par_iter()
        .map(|b| {
            specs[b]
                .solver_design()
                .matrixvectormultiply(&states[b].beta)
                + specs[b].solver_offset()
        })
        .collect();

    for (state, eta) in states.iter_mut().zip(refreshed_etas) {
        state.eta = eta;
    }
    Ok(())
}


fn refresh_single_block_eta<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &mut [ParameterBlockState],
    block_idx: usize,
) -> Result<(), String> {
    let spec = &specs[block_idx];
    let beta = states[block_idx].beta.clone();
    states[block_idx].eta = with_block_geometry(family, states, spec, block_idx, |x, off| {
        Ok(x.matrixvectormultiply(&beta) + off)
    })?;
    Ok(())
}


#[inline]
fn capped_inner_max_cycles(options: &BlockwiseFitOptions, base_cycles: usize) -> usize {
    let mut cap = base_cycles;
    if let Some(screening) = options.screening_max_inner_iterations.as_ref() {
        let screening_cap = screening.load(Ordering::Relaxed);
        if screening_cap > 0 {
            cap = cap.min(screening_cap);
        }
    }
    if let Some(outer) = options.outer_inner_max_iterations.as_ref() {
        let outer_cap = outer.load(Ordering::Relaxed);
        // `0` is the `SEED_SCREENING_UNCAPPED` sentinel: "no cap — use the full
        // `pirls_config.max_iterations`". The outer bridges store it into this
        // atomic for the line-search COST probe so the deciding cost is the true
        // converged-inner envelope objective the analytic gradient differentiates
        // (gam#787/#808). Honoring it requires the SAME `> 0` guard the screening
        // branch above uses; an unconditional `cap.min(0)` would collapse the
        // probe to a single inner cycle (`.max(1)`), guaranteeing a non-converged
        // inner solve and a spurious `∞` cost — re-introducing the frozen-|g|
        // outer stall the uncap was meant to remove.
        if outer_cap > 0 {
            cap = cap.min(outer_cap);
        }
    }
    cap.max(1)
}


fn weighted_normal_equations(
    x: &DesignMatrix,
    w: &Array1<f64>,
    y_star: Option<&Array1<f64>>,
) -> Result<(Array2<f64>, Option<Array1<f64>>), String> {
    let n = x.nrows();
    if w.len() != n {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: "weighted normal-equation dimension mismatch".to_string(),
        }
        .into());
    }
    if let Some(y) = y_star
        && y.len() != n
    {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: "weighted RHS dimension mismatch".to_string(),
        }
        .into());
    }

    let xtwx = x.xt_diag_x_signed_op(SignedWeightsView::from_array(w))?;
    let xtwy = if let Some(y) = y_star {
        Some(x.compute_xtwy(w, y)?)
    } else {
        None
    };
    Ok((xtwx, xtwy))
}


/// Smallest diagonal shift that makes the penalized joint Hessian
/// Cholesky-factorable (i.e. positive definite at the solver floor), or `None`
/// when the matrix is already PD and needs no shift.
///
/// PERF (gam#729/#826): the stabilizing shift is recomputed every inner Newton
/// cycle. For a coupled K-block family (Dirichlet/multinomial) the joint Hessian
/// is structurally near-singular along the cross-block gauge / sum-to-zero null
/// space, so a shift fires on (almost) every cycle. The previous implementation
/// ran a full dense self-adjoint eigendecomposition (`O(p³)`, all eigenpairs)
/// just to read `min_eval` — the dominant per-cycle cost on the coupled inner
/// solve. We only need a PD CERTIFICATE plus the smallest lifting ridge, which a
/// Cholesky probe gives far more cheaply: a plain Cholesky succeeds in one shot
/// on a well-conditioned cycle (no shift), and a geometric ridge escalation
/// finds the lifting shift in a handful of `O(p³/3)` Cholesky attempts on the
/// near-singular cycles — strictly cheaper than the full eigh and short-circuiting
/// on the first PD factorization. The resulting shift makes `H_pen + δI` PD,
/// which is exactly what the downstream solve requires.
fn exact_newton_stabilizing_shift(lhs_dense: &Array2<f64>, ridge_floor: f64) -> Option<f64> {
    let floor = effective_solverridge(ridge_floor);
    // Fast path: already PD at zero shift ⇒ no stabilization needed. One Cholesky
    // (O(p³/3)), the common case on a well-conditioned cycle.
    if lhs_dense.cholesky(Side::Lower).is_ok() {
        return None;
    }
    // Near-singular / indefinite. We need a positive diagonal shift `δ` that makes
    // `H + δI` PD. A full eigendecomposition (the previous implementation) reads
    // the exact `λ_min` but costs `O(p³)` for ALL eigenpairs EVERY inner cycle;
    // for a coupled K-block family the shift fires almost every cycle, so that
    // dominated the inner solve (gam#729/#826). A Cholesky-escalation search is
    // even worse on a hard-near-singular block (many `O(p³)` Cholesky retries).
    //
    // Use the Gershgorin lower bound on `λ_min` instead — a single `O(p²)` pass,
    // no iteration: every eigenvalue lies in some disc
    // `[H_ii − R_i, H_ii + R_i]` with `R_i = Σ_{j≠i} |H_ij|`, so
    // `λ_min ≥ min_i (H_ii − R_i) =: g`. Shifting by `δ = floor − g` (when `g`
    // is at/below the floor) guarantees `λ_min(H + δI) = λ_min + δ ≥ floor > 0`,
    // i.e. `H + δI` is PD. The bound is conservative (δ may be larger than the
    // exact eigh shift), but it is self-vanishing in the well-conditioned regime
    // (handled by the Cholesky fast path above) and the downstream solve only
    // requires PD, not the tightest possible shift — and the trust region governs
    // step size regardless. `O(p²)` per cycle instead of `O(p³)`.
    let p = lhs_dense.nrows();
    let mut gershgorin_min = f64::INFINITY;
    for i in 0..p {
        let diag = lhs_dense[[i, i]];
        let mut radius = 0.0_f64;
        for j in 0..p {
            if j != i {
                radius += lhs_dense[[i, j]].abs();
            }
        }
        gershgorin_min = gershgorin_min.min(diag - radius);
    }
    if !gershgorin_min.is_finite() {
        let diag_max = (0..p)
            .map(|d| lhs_dense[[d, d]].abs())
            .fold(0.0_f64, f64::max);
        return Some(floor.max(diag_max * 1e-6).max(1e-6));
    }
    if gershgorin_min >= floor {
        // Gershgorin certifies PD-at-floor but the no-shift Cholesky failed
        // (round-off on a barely-PD matrix): a floor-sized shift suffices.
        return Some(floor);
    }
    Some(floor - gershgorin_min)
}


fn stabilize_exact_newton_lhs_in_place<F: CustomFamily + ?Sized>(
    family: &F,
    lhs_dense: &mut Array2<f64>,
    ridge_floor: f64,
) {
    if use_exact_newton_strict_spd(family) {
        return;
    }
    if let Some(shift) = exact_newton_stabilizing_shift(lhs_dense, ridge_floor) {
        for d in 0..lhs_dense.nrows() {
            lhs_dense[[d, d]] += shift;
        }
    }
}


fn shift_linear_constraints_to_delta(
    constraints: &LinearInequalityConstraints,
    beta: &Array1<f64>,
) -> Result<LinearInequalityConstraints, String> {
    if constraints.a.ncols() != beta.len() || constraints.a.nrows() != constraints.b.len() {
        return Err(CustomFamilyError::ConstraintViolation {
            reason: "linear constraints: shape mismatch".to_string(),
        }
        .into());
    }
    Ok(LinearInequalityConstraints {
        a: constraints.a.clone(),
        b: &constraints.b - &constraints.a.dot(beta),
    })
}


fn collect_block_linear_constraints<F: CustomFamily + ?Sized>(
    family: &F,
    states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
) -> Result<Vec<Option<LinearInequalityConstraints>>, String> {
    let mut constraints = Vec::with_capacity(specs.len());
    for (block_idx, spec) in specs.iter().enumerate() {
        constraints.push(family.block_linear_constraints(states, block_idx, spec)?);
    }
    Ok(constraints)
}


fn reject_constrained_post_update_repair(
    block_idx: usize,
    spec: &ParameterBlockSpec,
    raw_beta: &Array1<f64>,
    updated_beta: &Array1<f64>,
    constraints: Option<&LinearInequalityConstraints>,
) -> Result<(), String> {
    let Some(constraints) = constraints else {
        return Ok(());
    };
    if raw_beta.len() != updated_beta.len() {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: format!(
                "post-update beta length changed for constrained block '{}' (idx {block_idx}): raw={}, updated={}",
                spec.name,
                raw_beta.len(),
                updated_beta.len(),
            ),
        }
        .into());
    }
    if raw_beta.len() != constraints.a.ncols() {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: format!(
                "post-update constrained block '{}' (idx {block_idx}) width mismatch: beta={}, constraints={}",
                spec.name,
                raw_beta.len(),
                constraints.a.ncols(),
            ),
        }
        .into());
    }
    let max_change = raw_beta
        .iter()
        .zip(updated_beta.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max);
    let raw_scale = raw_beta.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let updated_scale = updated_beta.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let tol = 1e-10 * (1.0 + raw_scale.max(updated_scale));
    if max_change > tol {
        return Err(CustomFamilyError::ConstraintViolation {
            reason: format!(
                "post-update hook materially changed constrained block '{}' (idx {block_idx}): \
                 max |β_post - β_qp|={max_change:.3e} > tol={tol:.3e}; \
                 constraints must be represented analytically in block_linear_constraints, not repaired after the Newton/QP solve",
                spec.name,
            ),
        }
        .into());
    }
    Ok(())
}


fn assemble_joint_linear_constraints(
    block_constraints: &[Option<LinearInequalityConstraints>],
    ranges: &[(usize, usize)],
    total_p: usize,
) -> Result<Option<LinearInequalityConstraints>, String> {
    if block_constraints.len() != ranges.len() {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: format!(
                "joint linear constraint assembly mismatch: {} blocks but {} ranges",
                block_constraints.len(),
                ranges.len()
            ),
        }
        .into());
    }
    let total_rows = block_constraints
        .iter()
        .map(|constraints| constraints.as_ref().map_or(0, |c| c.a.nrows()))
        .sum::<usize>();
    if total_rows == 0 {
        return Ok(None);
    }
    let mut a = Array2::<f64>::zeros((total_rows, total_p));
    let mut b = Array1::<f64>::zeros(total_rows);
    let mut row_offset = 0usize;
    for (block_idx, constraints_opt) in block_constraints.iter().enumerate() {
        let Some(constraints) = constraints_opt else {
            continue;
        };
        let (start, end) = ranges[block_idx];
        let block_p = end - start;
        if constraints.a.ncols() != block_p || constraints.a.nrows() != constraints.b.len() {
            return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                "joint linear constraint assembly mismatch for block {block_idx}: A is {}x{}, b is {}, block width is {}",
                constraints.a.nrows(),
                constraints.a.ncols(),
                constraints.b.len(),
                block_p
            ) }.into());
        }
        let rows = constraints.a.nrows();
        a.slice_mut(s![row_offset..(row_offset + rows), start..end])
            .assign(&constraints.a);
        b.slice_mut(s![row_offset..(row_offset + rows)])
            .assign(&constraints.b);
        row_offset += rows;
    }
    Ok(Some(LinearInequalityConstraints { a, b }))
}


fn flatten_joint_active_set(
    block_active_sets: &[Option<Vec<usize>>],
    block_constraints: &[Option<LinearInequalityConstraints>],
) -> Option<Vec<usize>> {
    if block_active_sets.len() != block_constraints.len() {
        return None;
    }
    let mut offset = 0usize;
    let mut joint_active = Vec::new();
    for (active_opt, constraints_opt) in block_active_sets.iter().zip(block_constraints.iter()) {
        let rows = constraints_opt
            .as_ref()
            .map_or(0, |constraints| constraints.a.nrows());
        if let Some(active) = active_opt {
            joint_active.extend(
                active
                    .iter()
                    .copied()
                    .filter(|&idx| idx < rows)
                    .map(|idx| offset + idx),
            );
        }
        offset += rows;
    }
    if joint_active.is_empty() {
        None
    } else {
        Some(joint_active)
    }
}


fn scatter_joint_active_set(
    joint_active: &[usize],
    block_constraints: &[Option<LinearInequalityConstraints>],
) -> Vec<Option<Vec<usize>>> {
    let mut per_block = Vec::with_capacity(block_constraints.len());
    let mut offset = 0usize;
    for constraints_opt in block_constraints {
        let rows = constraints_opt
            .as_ref()
            .map_or(0, |constraints| constraints.a.nrows());
        if rows == 0 {
            per_block.push(None);
            continue;
        }
        let mut local = joint_active
            .iter()
            .copied()
            .filter(|&idx| idx >= offset && idx < offset + rows)
            .map(|idx| idx - offset)
            .collect::<Vec<_>>();
        offset += rows;
        if local.is_empty() {
            per_block.push(None);
            continue;
        }
        local.sort_unstable();
        local.dedup();
        per_block.push(Some(local));
    }
    per_block
}


/// Assemble the **active rows** of the joint linear inequality constraint
/// matrix into a single `(k_active × total_p)` block, suitable for the
/// unified evaluator's constraint-aware kernel.
///
/// Inputs:
/// * `block_constraints`: per-block dense `LinearInequalityConstraints`
///   (the family's full inequality system per block, output of
///   `collect_block_linear_constraints`).
/// * `block_active_sets`: per-block indices of rows currently active
///   (output of the joint Newton's QP solver / `cached_active_sets`).
/// * `ranges`: per-block column ranges within the joint β.
/// * `total_p`: sum of block widths.
///
/// Returns `None` when no block has any active constraints — the caller
/// can then skip the constraint-aware kernel entirely.
fn assemble_active_constraint_block(
    block_constraints: &[Option<LinearInequalityConstraints>],
    block_active_sets: &[Option<Vec<usize>>],
    ranges: &[(usize, usize)],
    total_p: usize,
) -> Option<crate::solver::estimate::reml::unified::ActiveLinearConstraintBlock> {
    if block_constraints.len() != ranges.len() || block_active_sets.len() != ranges.len() {
        return None;
    }
    let mut active_per_block: Vec<(usize, &[usize], &LinearInequalityConstraints)> = Vec::new();
    let mut total_active = 0usize;
    for (b, (range, (constraints_opt, active_opt))) in ranges
        .iter()
        .zip(block_constraints.iter().zip(block_active_sets.iter()))
        .enumerate()
    {
        let Some(constraints) = constraints_opt else {
            continue;
        };
        let Some(active) = active_opt else {
            continue;
        };
        if active.is_empty() {
            continue;
        }
        if constraints.a.ncols() != range.1 - range.0 {
            return None;
        }
        if !active.iter().all(|&r| r < constraints.a.nrows()) {
            return None;
        }
        total_active += active.len();
        active_per_block.push((b, active.as_slice(), constraints));
    }
    if total_active == 0 {
        return None;
    }
    let mut a = ndarray::Array2::<f64>::zeros((total_active, total_p));
    let mut out_row = 0usize;
    for (b_idx, active, constraints) in active_per_block {
        let (start, end) = ranges[b_idx];
        let block_p = end - start;
        for &local_row in active {
            for col in 0..block_p {
                a[[out_row, start + col]] = constraints.a[[local_row, col]];
            }
            out_row += 1;
        }
    }
    Some(crate::solver::estimate::reml::unified::ActiveLinearConstraintBlock { a })
}


struct SimpleLowerBounds {
    lower_bounds: Array1<f64>,
    row_to_coeff: Vec<usize>,
    coeff_to_row: Vec<Option<usize>>,
}


fn extract_simple_lower_bounds(
    constraints: &LinearInequalityConstraints,
    p: usize,
) -> Result<Option<SimpleLowerBounds>, String> {
    if constraints.a.ncols() != p || constraints.a.nrows() != constraints.b.len() {
        return Err(CustomFamilyError::ConstraintViolation {
            reason: "linear constraints: shape mismatch".to_string(),
        }
        .into());
    }
    let mut lower_bounds = Array1::from_elem(p, f64::NEG_INFINITY);
    let mut coeff_to_row = vec![None; p];
    let mut row_to_coeff = Vec::with_capacity(constraints.a.nrows());
    for row in 0..constraints.a.nrows() {
        let mut coeff_idx = None;
        let mut coeff_value = 0.0;
        for col in 0..p {
            let value = constraints.a[[row, col]];
            if value.abs() <= 1e-12 {
                continue;
            }
            if coeff_idx.is_some() {
                return Ok(None);
            }
            coeff_idx = Some(col);
            coeff_value = value;
        }
        let Some(col) = coeff_idx else {
            return Ok(None);
        };
        if coeff_value <= 0.0 {
            return Ok(None);
        }
        let bound = constraints.b[row] / coeff_value;
        if bound > lower_bounds[col] {
            lower_bounds[col] = bound;
            coeff_to_row[col] = Some(row);
        }
        row_to_coeff.push(col);
    }
    Ok(Some(SimpleLowerBounds {
        lower_bounds,
        row_to_coeff,
        coeff_to_row,
    }))
}


fn lower_bound_active_rows_to_coeffs(
    bounds: &SimpleLowerBounds,
    active_rows: Option<&[usize]>,
) -> Vec<usize> {
    let Some(active_rows) = active_rows else {
        return Vec::new();
    };
    let mut active_coeffs = active_rows
        .iter()
        .copied()
        .filter_map(|row| bounds.row_to_coeff.get(row).copied())
        .collect::<Vec<_>>();
    active_coeffs.sort_unstable();
    active_coeffs.dedup();
    active_coeffs
}


fn lower_bound_active_coeffs_to_rows(
    bounds: &SimpleLowerBounds,
    active_coeffs: &[usize],
) -> Vec<usize> {
    let mut active_rows = active_coeffs
        .iter()
        .copied()
        .filter_map(|coeff| bounds.coeff_to_row.get(coeff).and_then(|row| *row))
        .collect::<Vec<_>>();
    active_rows.sort_unstable();
    active_rows.dedup();
    active_rows
}


fn lower_bound_active_coeffs_from_solution(
    bounds: &SimpleLowerBounds,
    beta: &Array1<f64>,
) -> Vec<usize> {
    let mut active_coeffs = Vec::new();
    for coeff in 0..beta.len() {
        let lower = bounds.lower_bounds[coeff];
        if !lower.is_finite() {
            continue;
        }
        let scale = beta[coeff].abs().max(lower.abs()).max(1.0);
        let tol = 1e-6 * scale + 1e-10;
        if beta[coeff] <= lower + tol {
            active_coeffs.push(coeff);
        }
    }
    active_coeffs
}


fn project_to_lower_bounds(beta: &mut Array1<f64>, lower_bounds: &Array1<f64>) {
    for i in 0..beta.len() {
        let lower = lower_bounds[i];
        if lower.is_finite() && beta[i] < lower {
            beta[i] = lower;
        }
    }
}


fn solve_quadratic_with_simple_lower_bounds(
    lhs: &Array2<f64>,
    rhs: &Array1<f64>,
    beta_start: &Array1<f64>,
    bounds: &SimpleLowerBounds,
    active_rows: Option<&[usize]>,
) -> Result<(Array1<f64>, Vec<usize>), String> {
    let gradient = lhs.dot(beta_start) - rhs;
    let mut delta = Array1::zeros(beta_start.len());
    let mut active_coeffs = lower_bound_active_rows_to_coeffs(bounds, active_rows);
    solve_newton_directionwith_lower_bounds(
        lhs,
        &gradient,
        beta_start,
        &bounds.lower_bounds,
        &mut delta,
        Some(&mut active_coeffs),
    )
    .map_err(|e| format!("lower-bound Newton solve failed: {e}"))?;
    let mut beta_new = beta_start + &delta;
    project_to_lower_bounds(&mut beta_new, &bounds.lower_bounds);
    active_coeffs = lower_bound_active_coeffs_from_solution(bounds, &beta_new);
    let active = lower_bound_active_coeffs_to_rows(bounds, &active_coeffs);
    Ok((beta_new, active))
}


fn normalize_active_set(mut active_set: Vec<usize>) -> Option<Vec<usize>> {
    active_set.sort_unstable();
    active_set.dedup();
    if active_set.is_empty() {
        None
    } else {
        Some(active_set)
    }
}


fn normalize_active_sets(active_sets: Vec<Option<Vec<usize>>>) -> Vec<Option<Vec<usize>>> {
    active_sets
        .into_iter()
        .map(|active_set| active_set.and_then(normalize_active_set))
        .collect()
}


struct BlockUpdateContext<'a> {
    family: &'a dyn CustomFamily,
    states: &'a [ParameterBlockState],
    spec: &'a ParameterBlockSpec,
    block_idx: usize,
    s_lambda: &'a Array2<f64>,
    options: &'a BlockwiseFitOptions,
    linear_constraints: Option<&'a LinearInequalityConstraints>,
    cached_active_set: Option<&'a [usize]>,
}


struct BlockUpdateResult {
    beta_new_raw: Array1<f64>,
    active_set: Option<Vec<usize>>,
}


#[inline]
fn floor_positiveworking_weights(working_weights: &Array1<f64>, minweight: f64) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(working_weights.len());
    ndarray::Zip::from(&mut out)
        .and(working_weights)
        .par_for_each(|o, &wi| *o = if wi <= 0.0 { 0.0 } else { wi.max(minweight) });
    out
}


trait ParameterBlockUpdater {
    fn compute_update_step(
        &self,
        ctx: &BlockUpdateContext<'_>,
    ) -> Result<BlockUpdateResult, String>;
}


struct DiagonalBlockUpdater<'a> {
    working_response: &'a Array1<f64>,
    working_weights: &'a Array1<f64>,
}


impl ParameterBlockUpdater for DiagonalBlockUpdater<'_> {
    fn compute_update_step(
        &self,
        ctx: &BlockUpdateContext<'_>,
    ) -> Result<BlockUpdateResult, String> {
        if self.working_response.len() != ctx.spec.design.nrows()
            || self.working_weights.len() != ctx.spec.design.nrows()
        {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "family diagonal working-set size mismatch on block {} ({})",
                    ctx.block_idx, ctx.spec.name
                ),
            }
            .into());
        }

        // Zero-weight observations are semantically excluded and must stay inactive.
        let w_clamped = floor_positiveworking_weights(self.working_weights, ctx.options.minweight);

        if let Some(constraints) = ctx.linear_constraints {
            check_linear_feasibility(&ctx.states[ctx.block_idx].beta, constraints, 1e-8).map_err(
                |e| {
                    format!(
                        "block {} ({}) constrained diagonal solve: {e}",
                        ctx.block_idx, ctx.spec.name
                    )
                },
            )?;
            with_block_geometry(ctx.family, ctx.states, ctx.spec, ctx.block_idx, |x, off| {
                let mut y_star = self.working_response.clone();
                y_star -= off;
                let (mut lhs, rhs_opt) = weighted_normal_equations(x, &w_clamped, Some(&y_star))?;
                let rhs = rhs_opt.ok_or_else(|| {
                    "missing weighted RHS in constrained diagonal solve".to_string()
                })?;
                lhs += ctx.s_lambda;
                let lower_bounds = extract_simple_lower_bounds(constraints, lhs.ncols())?;
                let (beta_constrained, active_set) = if let Some(bounds) = lower_bounds.as_ref() {
                    solve_quadratic_with_simple_lower_bounds(
                        &lhs,
                        &rhs,
                        &ctx.states[ctx.block_idx].beta,
                        bounds,
                        ctx.cached_active_set,
                    )
                } else {
                    solve_quadratic_with_linear_constraints(
                        &lhs,
                        &rhs,
                        &ctx.states[ctx.block_idx].beta,
                        constraints,
                        ctx.cached_active_set,
                    )
                    .map_err(|e| e.to_string())
                }
                .map_err(|e| {
                    format!(
                        "block {} ({}) constrained diagonal solve failed: {e}",
                        ctx.block_idx, ctx.spec.name
                    )
                })?;
                Ok(BlockUpdateResult {
                    beta_new_raw: beta_constrained,
                    active_set: normalize_active_set(active_set),
                })
            })
        } else {
            with_block_geometry(ctx.family, ctx.states, ctx.spec, ctx.block_idx, |x, off| {
                // Fuse offset subtraction into the weighted RHS: wy[i] = w[i] * (z[i] - off[i]).
                // This avoids an O(n) working_response clone.
                let n = self.working_response.len();
                let wy = Array1::from_shape_fn(n, |i| {
                    (self.working_response[i] - off[i]) * w_clamped[i].max(0.0)
                });
                let xtwy = x.transpose_vector_multiply(&wy);
                let beta = x
                    .solve_systemwith_policy(
                        &w_clamped,
                        &xtwy,
                        Some(ctx.s_lambda),
                        ctx.options.ridge_floor,
                        ctx.options.ridge_policy,
                    )
                    .map_err(|_| "block solve failed after ridge retries".to_string())?;
                Ok(BlockUpdateResult {
                    beta_new_raw: beta,
                    active_set: None,
                })
            })
        }
    }
}


struct ExactNewtonBlockUpdater<'a> {
    gradient: &'a Array1<f64>,
    hessian: &'a SymmetricMatrix,
}


impl ParameterBlockUpdater for ExactNewtonBlockUpdater<'_> {
    fn compute_update_step(
        &self,
        ctx: &BlockUpdateContext<'_>,
    ) -> Result<BlockUpdateResult, String> {
        let p = ctx.spec.design.ncols();
        if self.gradient.len() != p {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "block {} exact-newton gradient length mismatch: got {}, expected {p}",
                    ctx.block_idx,
                    self.gradient.len()
                ),
            }
            .into());
        }
        if self.hessian.nrows() != p || self.hessian.ncols() != p {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "block {} exact-newton Hessian shape mismatch: got {}x{}, expected {}x{}",
                    ctx.block_idx,
                    self.hessian.nrows(),
                    self.hessian.ncols(),
                    p,
                    p
                ),
            }
            .into());
        }

        let lhs = self.hessian.add_dense(ctx.s_lambda)?;
        // Solve in delta-space for both constrained and unconstrained blocks.
        // That keeps the linear system consistent even when we add a
        // numerical ridge to stabilize an indefinite exact-Newton Hessian.
        let rhs_step = self.gradient - &ctx.s_lambda.dot(&ctx.states[ctx.block_idx].beta);
        let mut lhs_dense = lhs.to_dense();
        stabilize_exact_newton_lhs_in_place(ctx.family, &mut lhs_dense, ctx.options.ridge_floor);

        if let Some(constraints) = ctx.linear_constraints {
            check_linear_feasibility(&ctx.states[ctx.block_idx].beta, constraints, 1e-8).map_err(
                |e| {
                    format!(
                        "block {} ({}) constrained exact-newton solve: {e}",
                        ctx.block_idx, ctx.spec.name
                    )
                },
            )?;
            let lower_bounds = extract_simple_lower_bounds(constraints, p).map_err(|e| {
                format!(
                    "block {} ({}) constrained exact-newton solve: {e}",
                    ctx.block_idx, ctx.spec.name
                )
            })?;
            let (beta_new_raw, active_set) = if let Some(bounds) = lower_bounds.as_ref() {
                let rhs_beta = &lhs_dense.dot(&ctx.states[ctx.block_idx].beta) + &rhs_step;
                solve_quadratic_with_simple_lower_bounds(
                    &lhs_dense,
                    &rhs_beta,
                    &ctx.states[ctx.block_idx].beta,
                    bounds,
                    ctx.cached_active_set,
                )
            } else {
                let delta_constraints =
                    shift_linear_constraints_to_delta(constraints, &ctx.states[ctx.block_idx].beta)
                        .map_err(|e| {
                            format!(
                                "block {} ({}) constrained exact-newton solve: {e}",
                                ctx.block_idx, ctx.spec.name
                            )
                        })?;
                let delta_start = Array1::zeros(p);
                let (delta, active_set) = solve_quadratic_with_linear_constraints(
                    &lhs_dense,
                    &rhs_step,
                    &delta_start,
                    &delta_constraints,
                    ctx.cached_active_set,
                )
                .map_err(|e| e.to_string())?;
                Ok((&ctx.states[ctx.block_idx].beta + &delta, active_set))
            }
            .map_err(|e| {
                format!(
                    "block {} ({}) constrained exact-newton solve failed: {e}",
                    ctx.block_idx, ctx.spec.name
                )
            })?;
            Ok(BlockUpdateResult {
                beta_new_raw,
                active_set: normalize_active_set(active_set),
            })
        } else {
            // Solve for the Newton step, not the next beta directly.
            //
            // For the penalized negative objective
            //
            //   Q(beta) = -log L(beta) + 0.5 beta^T S beta,
            //
            // the exact block gradient and Hessian are
            //
            //   grad_Q = S beta - gradient,
            //   hess_Q = hessian + S.
            //
            // The Newton step must therefore satisfy
            //
            //   hess_Q * delta = -grad_Q = gradient - S beta.
            //
            // This form stays correct even when the linear solver adds a
            // numerical ridge to the left-hand side to stabilize an indefinite
            // or nearly singular block. Solving directly for `beta_new` with a
            // ridged matrix would require an extra `ridge * beta` term on the
            // right-hand side; without it the step is distorted, which can trap
            // exact-Newton block updates on nonconvex blocks such as survival
            // `log_sigma`.
            let delta = if use_exact_newton_strict_spd(ctx.family) {
                // Strict-mode Newton step uses the LM δ-ridge continuation:
                // a single near-zero eigenvalue from numerical noise in
                // H_β should not bounce the entire seed evaluation. The
                // bare strict_solve_spd contract is preserved (still used
                // by other paths and the existing test
                // `pseudo_laplace_path_skips_eigendecomposition_avoiding_nan_crash`);
                // here we pay an O(p³) extra Cholesky attempt when needed
                // to keep adaptive optimization moving.
                let (step, lm_stats) =
                    strict_solve_spd_with_lm_continuation(&lhs_dense, &rhs_step)?;
                if lm_stats.escalations > 0 {
                    log::debug!(
                        "[strict-spd-lm] block={} ({}): δ-ridge continuation succeeded \
                         after {} escalation(s) at δ={:.3e}",
                        ctx.block_idx,
                        ctx.spec.name,
                        lm_stats.escalations,
                        lm_stats.delta_used,
                    );
                }
                step
            } else {
                // Non-strict (RidgedQuadraticReml) families share the strict
                // path's LM δ-ridge continuation. For a nonconvex block whose
                // likelihood Hessian H_β is INDEFINITE away from the optimum —
                // e.g. the squared-coefficient SCOP transformation-normal tensor
                // over a smooth covariate, where the I(y)⊗b(x) columns are
                // strongly collinear — the previous `solve_spd_systemwith_policy`
                // (ridge-retry + pinv-positive-part) returns a valid but
                // poorly-scaled descent step that crawls and hits the inner
                // cycle cap. The eigenvalue-floored LM continuation produces a
                // well-scaled Newton step on exactly those indefinite /
                // ill-conditioned systems. It is a STRICT SUPERSET of the plain
                // solve: when H_β + S is SPD and well-conditioned it reduces to
                // the same Cholesky step (zero escalations), only escalating the
                // floor when the system is genuinely indefinite — so
                // well-behaved families see no behaviour change. Internal to the
                // solve; β is recovered in the raw basis, so dimensionality /
                // identifiability are untouched.
                let step = match strict_solve_spd_with_lm_continuation(&lhs_dense, &rhs_step) {
                    Ok((step, lm_stats)) => {
                        if lm_stats.escalations > 0 {
                            log::debug!(
                                "[joint-Newton/lm] block={} ({}): non-strict δ-ridge continuation \
                                 succeeded after {} escalation(s) at δ={:.3e}",
                                ctx.block_idx,
                                ctx.spec.name,
                                lm_stats.escalations,
                                lm_stats.delta_used,
                            );
                        }
                        step
                    }
                    // Final guard: only if the LM continuation itself fails to
                    // produce a finite step do we fall back to the diagonal-
                    // scaled steepest-descent direction (always finite when the
                    // gradient is finite).
                    Err(_) => (0..lhs_dense.nrows())
                        .map(|i| {
                            let d = lhs_dense[[i, i]].abs().max(1e-8);
                            rhs_step[i] / d
                        })
                        .collect(),
                };
                step
            };
            let beta = &ctx.states[ctx.block_idx].beta + &delta;
            Ok(BlockUpdateResult {
                beta_new_raw: beta,
                active_set: None,
            })
        }
    }
}


impl BlockWorkingSet {
    fn updater(&self) -> Box<dyn ParameterBlockUpdater + '_> {
        match self {
            BlockWorkingSet::Diagonal {
                working_response,
                working_weights,
            } => Box::new(DiagonalBlockUpdater {
                working_response,
                working_weights,
            }),
            BlockWorkingSet::ExactNewton { gradient, hessian } => {
                Box::new(ExactNewtonBlockUpdater { gradient, hessian })
            }
        }
    }
}


fn check_linear_feasibility(
    beta: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
    tol: f64,
) -> Result<(), String> {
    if constraints.a.ncols() != beta.len() || constraints.a.nrows() != constraints.b.len() {
        return Err(CustomFamilyError::ConstraintViolation {
            reason: "linear constraints: shape mismatch".to_string(),
        }
        .into());
    }
    let slack = constraints.a.dot(beta) - &constraints.b;
    let mut worst = 0.0_f64;
    let mut worst_idx = 0usize;
    for (i, &s) in slack.iter().enumerate() {
        let v = (-s).max(0.0);
        if v > worst {
            worst = v;
            worst_idx = i;
        }
    }
    if worst > tol {
        return Err(CustomFamilyError::ConstraintViolation {
            reason: format!(
                "infeasible iterate: max(Aβ-b violation)={worst:.3e} at constraint row {worst_idx}"
            ),
        }
        .into());
    }
    Ok(())
}


#[inline]
fn effective_solverridge(ridge_floor: f64) -> f64 {
    ridge_floor.max(1e-15)
}


fn block_quadratic_penalty(
    beta: &Array1<f64>,
    s_lambda: &Array2<f64>,
    ridge: f64,
    ridge_policy: RidgePolicy,
) -> f64 {
    let mut value = 0.5 * beta.dot(&s_lambda.dot(beta));
    if ridge_policy.include_quadratic_penalty {
        value += 0.5 * ridge * beta.dot(beta);
    }
    value
}


fn block_penalized_hessian_vector(
    spec: &ParameterBlockSpec,
    work: &BlockWorkingSet,
    s_lambda: &Array2<f64>,
    direction: &Array1<f64>,
    ridge: f64,
    ridge_policy: RidgePolicy,
) -> Array1<f64> {
    let mut hpen = match work {
        BlockWorkingSet::ExactNewton { hessian, .. } => hessian.dot(direction),
        BlockWorkingSet::Diagonal {
            working_weights, ..
        } => {
            let solver_design = spec.solver_design();
            let x_direction = solver_design.matrixvectormultiply(direction);
            let wx_direction = &x_direction * working_weights;
            solver_design.transpose_vector_multiply(&wx_direction)
        }
    };
    hpen += &s_lambda.dot(direction);
    if ridge_policy.include_quadratic_penalty && ridge > 0.0 {
        hpen.scaled_add(ridge, direction);
    }
    hpen
}


fn symmetric_matrix_diagonal(matrix: &SymmetricMatrix) -> Array1<f64> {
    match matrix {
        SymmetricMatrix::Dense(mat) => mat.diag().to_owned(),
        SymmetricMatrix::Sparse(mat) => {
            let mut out = Array1::<f64>::zeros(mat.ncols());
            let (symbolic, values) = mat.parts();
            let col_ptr = symbolic.col_ptr();
            let row_idx = symbolic.row_idx();
            for col in 0..mat.ncols() {
                for idx in col_ptr[col]..col_ptr[col + 1] {
                    if row_idx[idx] == col {
                        out[col] += values[idx];
                    }
                }
            }
            out
        }
    }
}


fn block_penalized_metric_diagonal(
    spec: &ParameterBlockSpec,
    work: &BlockWorkingSet,
    s_lambda: &Array2<f64>,
    ridge: f64,
    ridge_policy: RidgePolicy,
) -> Result<Array1<f64>, String> {
    let mut diagonal = match work {
        BlockWorkingSet::ExactNewton { hessian, .. } => symmetric_matrix_diagonal(hessian),
        BlockWorkingSet::Diagonal {
            working_weights, ..
        } => spec.design.diag_gram(working_weights)?,
    };
    if diagonal.len() != s_lambda.nrows() || s_lambda.nrows() != s_lambda.ncols() {
        return Err(format!(
            "block penalized metric diagonal shape mismatch: diag={}, S={}x{}",
            diagonal.len(),
            s_lambda.nrows(),
            s_lambda.ncols()
        ));
    }
    for j in 0..diagonal.len() {
        diagonal[j] += s_lambda[[j, j]];
        if ridge_policy.include_quadratic_penalty && ridge > 0.0 {
            diagonal[j] += ridge;
        }
        diagonal[j] = positive_joint_diagonal_entry(diagonal[j]);
    }
    Ok(diagonal)
}


fn block_penalized_metric_norm(
    spec: &ParameterBlockSpec,
    work: &BlockWorkingSet,
    s_lambda: &Array2<f64>,
    direction: &Array1<f64>,
    ridge: f64,
    ridge_policy: RidgePolicy,
) -> Result<f64, String> {
    let diagonal = block_penalized_metric_diagonal(spec, work, s_lambda, ridge, ridge_policy)?;
    if diagonal.len() != direction.len() {
        return Err(format!(
            "block penalized metric direction length mismatch: direction={}, diag={}",
            direction.len(),
            diagonal.len()
        ));
    }
    Ok(joint_trust_region_metric_step_norm(direction, &diagonal))
}


fn truncate_block_step_to_metric_radius(
    spec: &ParameterBlockSpec,
    work: &BlockWorkingSet,
    s_lambda: &Array2<f64>,
    delta: Array1<f64>,
    radius: f64,
    ridge: f64,
    ridge_policy: RidgePolicy,
) -> Result<(Array1<f64>, f64), String> {
    let norm = block_penalized_metric_norm(spec, work, s_lambda, &delta, ridge, ridge_policy)?;
    if norm.is_finite() && norm > radius && radius > 0.0 {
        Ok((&delta * (radius / norm), radius))
    } else {
        Ok((delta, norm))
    }
}


const TOTAL_QUADRATIC_PENALTY_PAR_MIN_BLOCKS: usize = 4;

// Avoid Rayon overhead for a few tiny blocks; this approximates the dense
// mat-vec work in βᵀSβ before splitting independent block penalties.
const TOTAL_QUADRATIC_PENALTY_PAR_MIN_DENSE_WORK: usize = 16_384;


fn total_quadratic_penalty_parallel_worthwhile(
    states: &[ParameterBlockState],
    s_lambdas: &[Array2<f64>],
) -> bool {
    let n_blocks = states.len().min(s_lambdas.len());
    if n_blocks < TOTAL_QUADRATIC_PENALTY_PAR_MIN_BLOCKS || rayon::current_num_threads() <= 1 {
        return false;
    }

    states
        .iter()
        .zip(s_lambdas.iter())
        .map(|(state, s_lambda)| {
            let p = state.beta.len().min(s_lambda.ncols());
            p.saturating_mul(s_lambda.nrows())
        })
        .try_fold(0usize, |acc, work| {
            let next = acc.saturating_add(work);
            (next < TOTAL_QUADRATIC_PENALTY_PAR_MIN_DENSE_WORK).then_some(next)
        })
        .is_none()
}


fn total_quadratic_penalty(
    states: &[ParameterBlockState],
    s_lambdas: &[Array2<f64>],
    ridge: f64,
    ridge_policy: RidgePolicy,
    joint_full_width: Option<&crate::families::joint_penalty::JointPenaltyBundle>,
    specs: Option<&[ParameterBlockSpec]>,
) -> f64 {
    let per_block: f64 = if total_quadratic_penalty_parallel_worthwhile(states, s_lambdas) {
        use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

        states
            .par_iter()
            .zip(s_lambdas.par_iter())
            .map(|(state, s_lambda)| {
                block_quadratic_penalty(&state.beta, s_lambda, ridge, ridge_policy)
            })
            .reduce(|| 0.0, |left, right| left + right)
    } else {
        states
            .iter()
            .zip(s_lambdas.iter())
            .map(|(state, s_lambda)| {
                block_quadratic_penalty(&state.beta, s_lambda, ridge, ridge_policy)
            })
            .sum()
    };
    let joint = match (joint_full_width, specs) {
        (Some(bundle), Some(specs)) if !bundle.is_empty() => {
            let beta_flat = flatten_state_betas(states, specs);
            bundle.quadratic(beta_flat.view())
        }
        _ => 0.0,
    };
    per_block + joint
}


/// Locate the first non-finite entry in a Hessian and report it as a
/// canonical "smooth-regularized logdet boundary" error. The same
/// message is used at every site that refuses to factor or iterate on
/// a non-finite Hessian — the logdet computation itself, and the
/// inner-fit entry where exact-Newton block Hessians arrive from the
/// family. A single canonical phrasing means callers and tests
/// recognise this as one mathematical event regardless of where it
/// was caught: a NaN entry is a contract violation against the
/// family's analytic second derivative, full stop.
fn smooth_regularized_logdet_hessian_finite_check(
    matrix: &Array2<f64>,
    block: Option<usize>,
) -> Result<(), String> {
    let Some((row, col, value)) = matrix
        .indexed_iter()
        .find_map(|((row, col), &value)| (!value.is_finite()).then_some((row, col, value)))
    else {
        return Ok(());
    };
    let block_context = match block {
        Some(b) => format!(" for block {b}"),
        None => String::new(),
    };
    Err(CustomFamilyError::NumericalFailure { reason: format!(
        "smooth-regularized logdet Hessian contains non-finite entry at ({row}, {col}): {value}{block_context}"
    ) }.into())
}


/// Validate that every exact-Newton block working set in a family
/// evaluation has a finite Hessian. Returns Err on the first
/// non-finite entry using the canonical smooth-regularized logdet
/// boundary message, with the offending block index appended for
/// diagnostics.
///
/// Exact-Newton Hessians are part of the mathematical contract: they
/// are the family's analytic second derivative of the log-likelihood,
/// so any non-finite entry means that derivative is invalid math.
/// Catching it at the family-evaluation boundary lets the inner
/// solver refuse to iterate on a poisoned Hessian, instead of
/// silently "converging" because the gradient happens to be zero or
/// the bad entries get hidden behind a downstream eigendecomposition
/// fallback that the outer optimizer's flags may or may not invoke.
fn validate_block_hessians_finite(eval: &FamilyEvaluation) -> Result<(), String> {
    for (b, ws) in eval.blockworking_sets.iter().enumerate() {
        let BlockWorkingSet::ExactNewton { hessian, .. } = ws else {
            continue;
        };
        match hessian {
            SymmetricMatrix::Dense(matrix) => {
                smooth_regularized_logdet_hessian_finite_check(matrix, Some(b))?;
            }
            SymmetricMatrix::Sparse(matrix) => {
                let (symbolic, values) = matrix.parts();
                let col_ptr = symbolic.col_ptr();
                let row_idx = symbolic.row_idx();
                for col in 0..matrix.ncols() {
                    let start = col_ptr[col];
                    let end = col_ptr[col + 1];
                    for idx in start..end {
                        let row = row_idx[idx];
                        let value = values[idx];
                        if !value.is_finite() {
                            return Err(CustomFamilyError::NumericalFailure { reason: format!(
                                "smooth-regularized logdet Hessian contains non-finite entry at ({row}, {col}): {value} for block {b}"
                            ) }.into());
                        }
                    }
                }
            }
        }
    }
    Ok(())
}


fn stable_logdet_with_ridge_policy(
    matrix: &Array2<f64>,
    ridge_floor: f64,
    ridge_policy: RidgePolicy,
) -> Result<f64, String> {
    let mut a = matrix.clone();
    symmetrize_dense_in_place(&mut a);
    let p = a.nrows();
    let ridge = if ridge_policy.include_penalty_logdet {
        effective_solverridge(ridge_floor)
    } else {
        0.0
    };
    for i in 0..p {
        a[[i, i]] += ridge;
    }

    match resolved_ridge_determinant_mode(ridge_policy, p) {
        RidgeDeterminantMode::Full => {
            let chol = a.cholesky(Side::Lower).map_err(|_| {
                "cholesky failed while computing full ridge-aware logdet".to_string()
            })?;
            Ok(2.0 * chol.diag().mapv(f64::ln).sum())
        }
        RidgeDeterminantMode::Auto => Err(
            "internal: resolved_ridge_determinant_mode must resolve Auto to a concrete mode"
                .to_string(),
        ),
        RidgeDeterminantMode::PositivePart => {
            smooth_regularized_logdet_hessian_finite_check(&a, None)?;
            // Smooth-regularized logdet objective, aligned with the gradient
            // operator (`DenseSpectralOperator` in `Smooth` mode):
            //
            //   log |A|_reg = Σ_j log r_ε(σ_j),   r_ε(σ) = ½(σ + √(σ² + 4ε²))
            //
            // Every eigenvalue contributes; none are silently dropped.  The
            // regularizer r_ε is C∞, strictly positive for all real σ, and
            // numerically agrees with plain log σ when σ ≫ ε.  Negative
            // eigenvalues contribute ≈ log(ε²/|σ|) (quadratic damping) so
            // indefinite Hessians produce a finite, differentiable cost
            // rather than a discontinuous positive-part pseudo-determinant.
            //
            // This matches exactly what the downstream
            // `trace_logdet_gradient = Σ φ'(σ) u^T (dH/dρ) u` computes as the
            // analytic gradient — eliminating the cost/gradient mismatch
            // that previously broke BFGS line search on indefinite outer
            // Hessians.
            //
            match crate::faer_ndarray::FaerEigh::eigh(&a, Side::Lower) {
                Ok((evals, _)) => {
                    let eval_vec: Vec<f64> = evals
                        .as_slice()
                        .map(|sl| sl.to_vec())
                        .unwrap_or_else(|| evals.iter().copied().collect());
                    let eps = spectral_epsilon(&eval_vec)
                        .max(ridge.max(CUSTOM_FAMILY_CONDITION_RELATIVE_FLOOR));
                    let n_negative = eval_vec.iter().filter(|&&ev| ev < -eps).count();
                    if n_negative > 0 {
                        // Diagnostic only: indefiniteness is now handled
                        // correctly by the smooth regularizer, not ignored.
                        log::debug!(
                            "[SmoothRegularizedLogdet] Hessian has {n_negative} \
                             eigenvalue(s) below -eps={eps:.2e}; r_ε damps them \
                             smoothly instead of dropping them."
                        );
                    }
                    let logdet: f64 = eval_vec
                        .iter()
                        .map(|&sigma| spectral_regularize(sigma, eps).ln())
                        .sum();
                    Ok(logdet)
                }
                Err(eigh_err) => Err(CustomFamilyError::BasisDecompositionFailed {
                    reason: format!(
                        "smooth-regularized logdet eigendecomposition failed: {eigh_err}"
                    ),
                }
                .into()),
            }
        }
    }
}


/// Try Cholesky with an escalating diagonal ridge.
///
/// On attempt `k` (zero-indexed) the diagonal of `matrix` is boosted by
/// `initial_boost * growth^k`. The first successful Cholesky for which
/// `on_success` returns `Some(r)` short-circuits and yields `Some((r, boost,
/// attempt))`; otherwise (Cholesky failure or `on_success` rejection) the
/// ridge is grown and retried up to `max_attempts` times. Returns `None`
/// when every attempt is exhausted.
///
/// Callers that need a no-ridge probe should perform it explicitly before
/// invoking this helper; the helper itself always adds `initial_boost` on
/// the first attempt (which may itself be zero if the caller passes 0.0).
fn try_cholesky_with_escalating_ridge<R>(
    matrix: &Array2<f64>,
    initial_boost: f64,
    max_attempts: usize,
    growth: f64,
    mut on_success: impl FnMut(&crate::faer_ndarray::FaerCholeskyFactor, usize, f64) -> Option<R>,
) -> Option<(R, f64, usize)> {
    let p = matrix.nrows();
    let mut boost = initial_boost;
    for attempt in 0..max_attempts {
        let mut candidate = matrix.clone();
        if boost != 0.0 {
            for i in 0..p {
                candidate[[i, i]] += boost;
            }
        }
        if let Ok(chol) = candidate.cholesky(Side::Lower)
            && let Some(r) = on_success(&chol, attempt, boost)
        {
            return Some((r, boost, attempt));
        }
        boost *= growth;
    }
    None
}


/// Fallback for penalty pseudo-logdet when eigendecomposition fails.
///
/// Penalty matrices are PSD by construction (weighted sum of PSD penalties),
/// so the ridged matrix should be SPD.  Uses escalating-ridge Cholesky via
/// the shared `try_cholesky_with_escalating_ridge` helper.
fn penalty_logdet_cholesky_fallback(
    s_ridged: &Array2<f64>,
    existing_ridge: f64,
    block: usize,
    p: usize,
    eigh_err: &str,
) -> Result<f64, String> {
    let diag_scale = s_ridged
        .diag()
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0_f64, f64::max)
        .max(1.0);

    const MAX_ATTEMPTS: usize = 6;
    let initial_boost = diag_scale * 1e-8;

    let outcome = try_cholesky_with_escalating_ridge(
        s_ridged,
        initial_boost,
        MAX_ATTEMPTS,
        10.0,
        |chol, attempt, boost| {
            let logdet = 2.0 * chol.diag().mapv(f64::ln).sum();
            if logdet.is_finite() {
                log::warn!(
                    "[PenaltyLogdetFallback] eigendecomposition failed for block {block} \
                     ({eigh_err}); using Cholesky with boosted ridge={:.2e} \
                     (attempt {}/{MAX_ATTEMPTS}, existing_ridge={:.2e}, p={p})",
                    boost + existing_ridge,
                    attempt + 1,
                    existing_ridge,
                );
                Some(logdet)
            } else {
                None
            }
        },
    );

    if let Some((logdet, _, _)) = outcome {
        return Ok(logdet);
    }

    // Mirror the original message: report the ridge that *would* have been
    // applied on the (MAX_ATTEMPTS+1)-th attempt, i.e. initial_boost * 10^MAX_ATTEMPTS.
    let final_boost = initial_boost * 10.0_f64.powi(MAX_ATTEMPTS as i32);
    Err(CustomFamilyError::BasisDecompositionFailed {
        reason: format!(
            "penalty logdet eigendecomposition failed for block {block} ({eigh_err}) and \
         Cholesky fallback also failed after {MAX_ATTEMPTS} attempts \
         (final ridge={:.2e}, p={p})",
            final_boost + existing_ridge,
        ),
    }
    .into())
}


fn resolved_ridge_determinant_mode(ridge_policy: RidgePolicy, dim: usize) -> RidgeDeterminantMode {
    assert!(
        dim.checked_add(1).is_some(),
        "ridge determinant dimension overflow"
    );
    match ridge_policy.determinant_mode {
        RidgeDeterminantMode::Auto => RidgeDeterminantMode::Full,
        mode => mode,
    }
}


fn inverse_spdwith_retry(
    matrix: &Array2<f64>,
    baseridge: f64,
    max_retry: usize,
) -> Result<Array2<f64>, String> {
    let mut sym = matrix.clone();
    symmetrize_dense_in_place(&mut sym);

    let invert_via_chol = |chol: &crate::faer_ndarray::FaerCholeskyFactor, _: usize, _: f64| {
        let mut ident = Array2::<f64>::eye(sym.nrows());
        chol.solve_mat_in_place(&mut ident);
        symmetrize_dense_in_place(&mut ident);
        Some(ident)
    };

    // Attempt 0 in the original schedule uses ridge=0 (no diagonal addition).
    // Express this as a single-attempt call with initial_boost=0.
    if let Some((inv, _, _)) =
        try_cholesky_with_escalating_ridge(&sym, 0.0, 1, 1.0, invert_via_chol)
    {
        return Ok(inv);
    }

    // Subsequent attempts use ridge = baseridge * 10^(k-1) for k = 1..=max_retry,
    // which is `max_retry` total attempts with initial_boost=baseridge, growth=10.
    if max_retry > 0
        && let Some((inv, _, _)) =
            try_cholesky_with_escalating_ridge(&sym, baseridge, max_retry, 10.0, invert_via_chol)
    {
        return Ok(inv);
    }

    Err(CustomFamilyError::BasisDecompositionFailed {
        reason: "failed to invert SPD system after Cholesky ridge retries".to_string(),
    }
    .into())
}


pub(crate) fn symmetrize_dense_in_place(matrix: &mut Array2<f64>) {
    crate::linalg::matrix::symmetrize_in_place(matrix);
}


fn validate_flat_direction_length(
    direction: &Array1<f64>,
    expected: usize,
    context: &str,
) -> Result<(), String> {
    if direction.len() != expected {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: format!(
                "{context}: direction length mismatch: got {}, expected {expected}",
                direction.len()
            ),
        }
        .into());
    }
    Ok::<(), _>(())
}


/// Does a joint Hessian carry genuine cross-block (off-diagonal) coupling?
///
/// The trait's default `exact_newton_joint_hessian` assembles a strictly
/// block-diagonal matrix (per-block `Xᵀ W X` on the diagonal, zeros off-block).
/// A family that overrides it with the true coupled curvature of a multi-block
/// likelihood (GAMLSS μ-σ, Beta-logit `α`/`β`, Dirichlet K-block via the shared
/// concentration sum, …) necessarily fills in nonzero off-diagonal blocks. This
/// is the only structural signal — independent of any hand-set marker — that
/// distinguishes a trusted coupled joint Hessian from the block-diagonal
/// default. The block boundaries come from the per-block β widths.
fn joint_hessian_has_cross_block_coupling(
    hessian: &Array2<f64>,
    block_states: &[ParameterBlockState],
) -> bool {
    let total = block_states
        .iter()
        .map(|state| state.beta.len())
        .sum::<usize>();
    if hessian.nrows() != total || hessian.ncols() != total {
        // Shape disagreement is handled (loudly) by the symmetrizer/consumers;
        // here we only answer the coupling question and must not claim coupling
        // for a malformed matrix.
        return false;
    }
    let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(block_states.len());
    let mut start = 0usize;
    for state in block_states {
        let end = start + state.beta.len();
        ranges.push((start, end));
        start = end;
    }
    for (a, (ra_start, ra_end)) in ranges.iter().copied().enumerate() {
        for (rb_start, rb_end) in ranges.iter().copied().skip(a + 1) {
            for i in ra_start..ra_end {
                for j in rb_start..rb_end {
                    if hessian[[i, j]] != 0.0 || hessian[[j, i]] != 0.0 {
                        return true;
                    }
                }
            }
        }
    }
    false
}


fn exact_newton_joint_hessian_from_exact_blocks<F: CustomFamily + ?Sized>(
    family: &F,
    block_states: &[ParameterBlockState],
) -> Result<Option<Array2<f64>>, String> {
    let evaluation = family.evaluate(block_states)?;
    if evaluation.blockworking_sets.len() != block_states.len() {
        return Err(format!(
            "exact_newton_joint_hessian default: working-set count {} != block count {}",
            evaluation.blockworking_sets.len(),
            block_states.len()
        ));
    }
    if evaluation
        .blockworking_sets
        .iter()
        .any(|working_set| !matches!(working_set, BlockWorkingSet::ExactNewton { .. }))
    {
        return Ok(None);
    }

    let total = block_states
        .iter()
        .map(|state| state.beta.len())
        .sum::<usize>();
    let mut joint = Array2::<f64>::zeros((total, total));
    let mut start = 0usize;
    for (block_idx, (state, working_set)) in block_states
        .iter()
        .zip(evaluation.blockworking_sets.iter())
        .enumerate()
    {
        let p_block = state.beta.len();
        let end = start + p_block;
        let BlockWorkingSet::ExactNewton { hessian, .. } = working_set else {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "exact_newton_joint_hessian default: block {block_idx} working set is not ExactNewton after filter"
                ),
            }
            .into());
        };
        let dense = hessian.to_dense();
        if dense.nrows() != p_block || dense.ncols() != p_block {
            return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                "exact_newton_joint_hessian default: block {block_idx} Hessian shape {}x{} != expected {p_block}x{p_block}",
                dense.nrows(),
                dense.ncols()
            ) }.into());
        }
        joint.slice_mut(s![start..end, start..end]).assign(&dense);
        start = end;
    }
    Ok(Some(joint))
}


fn exact_newton_joint_hessian_from_working_sets<F: CustomFamily + ?Sized>(
    family: &F,
    block_states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
) -> Result<Option<Array2<f64>>, String> {
    if block_states.len() != specs.len() {
        return Err(format!(
            "exact_newton_joint_hessian_with_specs default: block state count {} != spec count {}",
            block_states.len(),
            specs.len()
        ));
    }
    let evaluation = family.evaluate(block_states)?;
    if evaluation.blockworking_sets.len() != block_states.len() {
        return Err(format!(
            "exact_newton_joint_hessian_with_specs default: working-set count {} != block count {}",
            evaluation.blockworking_sets.len(),
            block_states.len()
        ));
    }

    let total = specs.iter().map(|spec| spec.design.ncols()).sum::<usize>();
    let mut joint = Array2::<f64>::zeros((total, total));
    let mut start = 0usize;
    for (block_idx, ((state, spec), working_set)) in block_states
        .iter()
        .zip(specs.iter())
        .zip(evaluation.blockworking_sets.iter())
        .enumerate()
    {
        let p_block = spec.design.ncols();
        if state.beta.len() != p_block {
            return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                "exact_newton_joint_hessian_with_specs default: block {block_idx} beta length {} != design cols {p_block}",
                state.beta.len()
            ) }.into());
        }
        let end = start + p_block;
        let dense = match working_set {
            BlockWorkingSet::ExactNewton { hessian, .. } => hessian.to_dense(),
            BlockWorkingSet::Diagonal {
                working_weights, ..
            } => spec
                .design
                .xt_diag_x_signed_op(SignedWeightsView::from_array(working_weights))?,
        };
        if dense.nrows() != p_block || dense.ncols() != p_block {
            return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                "exact_newton_joint_hessian_with_specs default: block {block_idx} Hessian shape {}x{} != expected {p_block}x{p_block}",
                dense.nrows(),
                dense.ncols()
            ) }.into());
        }
        joint.slice_mut(s![start..end, start..end]).assign(&dense);
        start = end;
    }
    Ok(Some(joint))
}


fn exact_newton_joint_hessian_directional_derivative_from_blocks<F: CustomFamily + ?Sized>(
    family: &F,
    block_states: &[ParameterBlockState],
    d_beta_flat: &Array1<f64>,
) -> Result<Option<Array2<f64>>, String> {
    let total = block_states
        .iter()
        .map(|state| state.beta.len())
        .sum::<usize>();
    validate_flat_direction_length(
        d_beta_flat,
        total,
        "exact_newton_joint_hessian_directional_derivative default",
    )?;
    if !family.exact_newton_joint_hessian_beta_dependent() {
        return Ok(Some(Array2::zeros((total, total))));
    }

    let mut joint = Array2::<f64>::zeros((total, total));
    let mut start = 0usize;
    for (block_idx, state) in block_states.iter().enumerate() {
        let p_block = state.beta.len();
        let end = start + p_block;
        let d_beta_block = d_beta_flat.slice(s![start..end]).to_owned();
        let Some(local) = family.exact_newton_hessian_directional_derivative(
            block_states,
            block_idx,
            &d_beta_block,
        )?
        else {
            return Ok(None);
        };
        if local.nrows() != p_block || local.ncols() != p_block {
            return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                "exact_newton_joint_hessian_directional_derivative default: block {block_idx} dH shape {}x{} != expected {p_block}x{p_block}",
                local.nrows(),
                local.ncols()
            ) }.into());
        }
        joint.slice_mut(s![start..end, start..end]).assign(&local);
        start = end;
    }
    Ok(Some(joint))
}


/// Block-diagonal aggregator for the joint second directional derivative.
///
/// Mirrors `exact_newton_joint_hessian_directional_derivative_from_blocks`:
/// for a beta-independent joint Hessian the answer is identically zero;
/// otherwise we ask each block for `D²H_b[u_b, v_b]` via
/// `exact_newton_hessian_second_directional_derivative` and place those
/// per-block contributions on the joint diagonal.
///
/// The previous default returned `Some(zeros)` for beta-independent and
/// `None` (no aggregation at all) for beta-dependent families, silently
/// dropping the per-block `d²H` overrides that families like
/// `OneBlockQuarticExactFamily` provide for the outer Hessian's drift
/// contribution.  Aggregating here mirrors the first-derivative path so
/// outer REML receives the curvature term whenever the per-block
/// `exact_newton_hessian_second_directional_derivative` is implemented.
fn exact_newton_joint_hessiansecond_directional_derivative_from_blocks<F: CustomFamily + ?Sized>(
    family: &F,
    block_states: &[ParameterBlockState],
    d_beta_u_flat: &Array1<f64>,
    d_betav_flat: &Array1<f64>,
) -> Result<Option<Array2<f64>>, String> {
    let total = block_states
        .iter()
        .map(|state| state.beta.len())
        .sum::<usize>();
    validate_flat_direction_length(d_beta_u_flat, total, "joint exact-newton d2H u")?;
    validate_flat_direction_length(d_betav_flat, total, "joint exact-newton d2H v")?;
    if !family.exact_newton_joint_hessian_beta_dependent() {
        return Ok(Some(Array2::zeros((total, total))));
    }

    let mut joint = Array2::<f64>::zeros((total, total));
    let mut start = 0usize;
    for (block_idx, state) in block_states.iter().enumerate() {
        let p_block = state.beta.len();
        let end = start + p_block;
        let u_block = d_beta_u_flat.slice(s![start..end]).to_owned();
        let v_block = d_betav_flat.slice(s![start..end]).to_owned();
        let Some(local) = family.exact_newton_hessian_second_directional_derivative(
            block_states,
            block_idx,
            &u_block,
            &v_block,
        )?
        else {
            return Ok(None);
        };
        if local.nrows() != p_block || local.ncols() != p_block {
            return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                "exact_newton_joint_hessiansecond_directional_derivative default: block {block_idx} d2H shape {}x{} != expected {p_block}x{p_block}",
                local.nrows(),
                local.ncols()
            ) }.into());
        }
        joint.slice_mut(s![start..end, start..end]).assign(&local);
        start = end;
    }
    Ok(Some(joint))
}


fn exact_newton_joint_hessian_directional_derivative_from_working_sets<F: CustomFamily + ?Sized>(
    family: &F,
    block_states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    d_beta_flat: &Array1<f64>,
) -> Result<Option<Array2<f64>>, String> {
    if block_states.len() != specs.len() {
        return Err(format!(
            "exact_newton_joint_hessian_directional_derivative_with_specs default: block state count {} != spec count {}",
            block_states.len(),
            specs.len()
        ));
    }
    let total = specs.iter().map(|spec| spec.design.ncols()).sum::<usize>();
    validate_flat_direction_length(
        d_beta_flat,
        total,
        "exact_newton_joint_hessian_directional_derivative_with_specs default",
    )?;
    if !family.exact_newton_joint_hessian_beta_dependent() {
        return Ok(Some(Array2::zeros((total, total))));
    }

    let evaluation = family.evaluate(block_states)?;
    if evaluation.blockworking_sets.len() != block_states.len() {
        return Err(format!(
            "exact_newton_joint_hessian_directional_derivative_with_specs default: working-set count {} != block count {}",
            evaluation.blockworking_sets.len(),
            block_states.len()
        ));
    }

    let mut joint = Array2::<f64>::zeros((total, total));
    let mut start = 0usize;
    for (block_idx, ((state, spec), working_set)) in block_states
        .iter()
        .zip(specs.iter())
        .zip(evaluation.blockworking_sets.iter())
        .enumerate()
    {
        let p_block = spec.design.ncols();
        let end = start + p_block;
        let d_beta_block = d_beta_flat.slice(s![start..end]).to_owned();
        let local = match working_set {
            BlockWorkingSet::ExactNewton { .. } => family
                .exact_newton_hessian_directional_derivative(
                    block_states,
                    block_idx,
                    &d_beta_block,
                )?,
            BlockWorkingSet::Diagonal {
                working_weights, ..
            } => {
                let solver_design = spec.solver_design();
                let mut d_eta = solver_design.apply(&d_beta_block);
                let mut geometry_correction = Array2::<f64>::zeros((p_block, p_block));
                if let Some(geometry) = family.block_geometry_directional_derivative(
                    block_states,
                    block_idx,
                    spec,
                    &d_beta_block,
                )? {
                    if geometry.d_offset.len() != d_eta.len() {
                        return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                            "exact_newton_joint_hessian_directional_derivative_with_specs default: block {block_idx} geometry offset derivative length {} != eta length {}",
                            geometry.d_offset.len(),
                            d_eta.len()
                        ) }.into());
                    }
                    d_eta += &geometry.d_offset;
                    if let Some(d_design) = geometry.d_design {
                        if d_design.nrows() != solver_design.nrows() || d_design.ncols() != p_block
                        {
                            return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                                "exact_newton_joint_hessian_directional_derivative_with_specs default: block {block_idx} d_design shape {}x{} != expected {}x{}",
                                d_design.nrows(),
                                d_design.ncols(),
                                solver_design.nrows(),
                                p_block
                            ) }.into());
                        }
                        d_eta += &d_design.dot(&state.beta);

                        let x_dense = solver_design.to_dense();
                        let mut weighted_x = x_dense.clone();
                        let mut weighted_dx = d_design.clone();
                        ndarray::Zip::from(weighted_x.rows_mut())
                            .and(weighted_dx.rows_mut())
                            .and(working_weights.view())
                            .for_each(|mut wx_row, mut wdx_row, &wi| {
                                wx_row.mapv_inplace(|value| value * wi);
                                wdx_row.mapv_inplace(|value| value * wi);
                            });
                        geometry_correction += &fast_atb(&d_design, &weighted_x);
                        geometry_correction += &fast_atb(&x_dense, &weighted_dx);
                    }
                }
                family
                    .diagonalworking_weights_directional_derivative(
                        block_states,
                        block_idx,
                        &d_eta,
                    )?
                    .map(|dw| {
                        let mut local = solver_design
                            .xt_diag_x_signed_op(SignedWeightsView::from_array(&dw))?;
                        local += &geometry_correction;
                        Ok::<Array2<f64>, String>(local)
                    })
                    .transpose()?
            }
        };
        let Some(local) = local else {
            return Ok(None);
        };
        if local.nrows() != p_block || local.ncols() != p_block {
            return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                "exact_newton_joint_hessian_directional_derivative_with_specs default: block {block_idx} dH shape {}x{} != expected {p_block}x{p_block}",
                local.nrows(),
                local.ncols()
            ) }.into());
        }
        joint.slice_mut(s![start..end, start..end]).assign(&local);
        start = end;
    }
    Ok(Some(joint))
}


fn exact_newton_joint_hessian_symmetrized<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    total: usize,
    context: &str,
) -> Result<Option<Array2<f64>>, String> {
    let Some(mut h) = family.exact_newton_joint_hessian_with_specs(states, specs)? else {
        return Ok(None);
    };
    if h.nrows() != total || h.ncols() != total {
        return Err(format!(
            "{context}: got {}x{}, expected {}x{}",
            h.nrows(),
            h.ncols(),
            total,
            total
        ));
    }
    symmetrize_dense_in_place(&mut h);
    Ok(Some(h))
}


/// Scale-aware exact joint curvature payload for the outer REML evaluator.
pub struct ExactNewtonOuterCurvature {
    pub hessian: Array2<f64>,
    pub rho_curvature_scale: f64,
    pub hessian_logdet_correction: f64,
}


enum JointHessianSource {
    Dense(Array2<f64>),
    Operator {
        apply: Arc<dyn Fn(&Array1<f64>) -> Result<Array1<f64>, String> + Send + Sync>,
        /// Write-into matvec used by the inner-Newton PCG hot path so the
        /// matvec result no longer allocates an `Array1<f64>` per CG iter.
        /// At large scale (~6400 inner CG iters per outer iter, p~200) this
        /// removes thousands of small Vec<f64> allocations from the tightest
        /// loop. Wired from `workspace.hessian_matvec_into`.
        apply_into: Arc<dyn Fn(&Array1<f64>, &mut Array1<f64>) -> Result<(), String> + Send + Sync>,
        /// Batched multi-RHS apply: `out = H · V` for `(total, n_rhs)` `V`.
        /// Wired from `workspace.hessian_apply_mat`, which for the BMS tiled
        /// row-primary Hessian sweeps each row tile once and applies its `Hᵢ`
        /// to every column. Column-basis dense reconstruction below uses this
        /// to materialise the operator in one batched sweep (`H = H · I`)
        /// rather than `total` single-vector HVPs, each of which re-reads every
        /// row tile. Numerically identical to looping `apply_into`.
        apply_mat: Arc<dyn Fn(&Array2<f64>, &mut Array2<f64>) -> Result<(), String> + Send + Sync>,
        diagonal: Array1<f64>,
        /// Forced dense materialization that bypasses the workspace's
        /// `hessian_dense` amortization gate. Returns `Some` when the
        /// workspace can build dense via a structural direct path (e.g.
        /// CTN's `scop_gradient_and_negative_hessian`), `None` when the
        /// caller should fall back to column-basis HVP through `apply`.
        dense_forced: Arc<dyn Fn() -> Result<Option<Array2<f64>>, String> + Send + Sync>,
    },
}


const EXACT_JOINT_HESSIAN_DENSE_MAX_BYTES: usize = 512 * 1024 * 1024;


fn exact_joint_hessian_dense_bytes(total: usize) -> Result<usize, String> {
    total
        .checked_mul(total)
        .and_then(|n| n.checked_mul(std::mem::size_of::<f64>()))
        .ok_or_else(|| format!("joint Hessian dense byte count overflow for dim={total}"))
}


fn ensure_exact_joint_hessian_dense_budget(total: usize, context: &str) -> Result<(), String> {
    let bytes = exact_joint_hessian_dense_bytes(total)?;
    if bytes > EXACT_JOINT_HESSIAN_DENSE_MAX_BYTES {
        return Err(CustomFamilyError::UnsupportedConfiguration {
            reason: format!(
                "{context}: exact dense joint Hessian requires {:.2} GiB for dim={total}, \
             exceeding the {:.2} GiB cap; refusing approximate determinant algebra",
                bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                EXACT_JOINT_HESSIAN_DENSE_MAX_BYTES as f64 / (1024.0 * 1024.0 * 1024.0),
            ),
        }
        .into());
    }
    Ok(())
}


struct JointHessianBundle<'a> {
    source: JointHessianSource,
    beta_flat: Array1<f64>,
    compute_dh: Box<DriftDerivFn<'a>>,
    compute_dh_many: Option<Box<DriftDerivManyFn<'a>>>,
    compute_d2h: Box<DriftSecondDerivFn<'a>>,
    /// Optional batched second-derivative callback. The unified evaluator's
    /// outer-Hessian ρ-ρ pair loop forwards the K(K+1)/2 (v_k, v_l) pairs
    /// here in one call when set, so families that fuse the per-row D²H walk
    /// (e.g. survival marginal-slope scanning n rows once per outer eval)
    /// amortise the row-walk across all pairs instead of paying it per pair.
    compute_d2h_many: Option<Box<DriftSecondDerivManyFn<'a>>>,
    owned_compute_dh:
        Option<Arc<dyn Fn(&Array1<f64>) -> Result<Option<DriftDerivResult>, String> + Send + Sync>>,
    owned_compute_dh_many: Option<
        Arc<dyn Fn(&[Array1<f64>]) -> Result<Vec<Option<DriftDerivResult>>, String> + Send + Sync>,
    >,
    owned_compute_d2h: Option<
        Arc<
            dyn Fn(&Array1<f64>, &Array1<f64>) -> Result<Option<DriftDerivResult>, String>
                + Send
                + Sync,
        >,
    >,
    /// Owned twin of `compute_d2h_many`. Threaded through to
    /// `OwnedJointDerivProvider` so the unified evaluator can share the
    /// callback across rayon worker threads when the outer Hessian routes
    /// through the parallel pair dispatch.
    owned_compute_d2h_many: Option<
        Arc<
            dyn Fn(&[(Array1<f64>, Array1<f64>)]) -> Result<Vec<Option<DriftDerivResult>>, String>
                + Send
                + Sync,
        >,
    >,
    rho_curvature_scale: f64,
    hessian_logdet_correction: f64,
}


type DriftDerivFn<'a> =
    dyn Fn(&Array1<f64>) -> Result<Option<DriftDerivResult>, String> + Send + Sync + 'a;

type DriftDerivManyFn<'a> =
    dyn Fn(&[Array1<f64>]) -> Result<Vec<Option<DriftDerivResult>>, String> + Send + Sync + 'a;

type DriftSecondDerivFn<'a> = dyn Fn(&Array1<f64>, &Array1<f64>) -> Result<Option<DriftDerivResult>, String>
    + Send
    + Sync
    + 'a;

type DriftSecondDerivManyFn<'a> = dyn Fn(&[(Array1<f64>, Array1<f64>)]) -> Result<Vec<Option<DriftDerivResult>>, String>
    + Send
    + Sync
    + 'a;


/// Cheap, deterministic non-finite-curvature probe for the joint Hessian
/// source (gam#1088). A `NaN`/`Inf` in the penalized Hessian `H_pen = H +
/// S(λ)` makes its spectrum (`λ_max`, `λ_min`, `cond`) degenerate to `NaN`,
/// so the KKT certificate — which thresholds on that spectrum — can *never*
/// be satisfied, the spectral step solve returns garbage, and the projected
/// residual neither converges nor early-exits through the finite-comparison
/// guards. Left undetected the coupled joint-Newton loop then grinds the full
/// `inner_loop_hard_ceiling` (1200 cycles) on *every* outer ρ-eval / seed,
/// which is the multi-hour benchmark timeout (link-wiggle & location-scale
/// paths). The penalty `S(λ)` is finite by construction, so a non-finite
/// `H_pen` originates entirely in the family curvature `H` we probe here.
///
/// For the `Dense` variant the matrix is already materialized, so we scan it
/// directly. For the matrix-free `Operator` variant the full operator is never
/// formed; we scan its `diagonal`, which carries the per-row assembled
/// curvature (`XᵀWX`) — exactly where a collapsed/`0÷0` row weight injects the
/// `NaN`, corrupting the whole row block including its diagonal entry. Returns
/// `true` when every probed entry is finite.
fn joint_hessian_source_curvature_is_finite(source: &JointHessianSource) -> bool {
    match source {
        JointHessianSource::Dense(h_joint) => h_joint.iter().all(|v| v.is_finite()),
        JointHessianSource::Operator { diagonal, .. } => diagonal.iter().all(|v| v.is_finite()),
    }
}


fn materialize_joint_hessian_source(
    source: &JointHessianSource,
    total: usize,
    context: &str,
) -> Result<Array2<f64>, String> {
    match source {
        JointHessianSource::Dense(matrix) => Ok(matrix.clone()),
        JointHessianSource::Operator {
            apply_mat,
            dense_forced,
            ..
        } => {
            ensure_exact_joint_hessian_dense_budget(total, context)?;
            // Preferred path: the workspace exposes a structural direct-dense
            // build (e.g. SCOP's `scop_gradient_and_negative_hessian`). That
            // is `Θ(n·p²)` like column-basis HVP would be, but the constant
            // factor is much better because the structural build sweeps rows
            // once and uses BLAS-3 for the chain-rule pullback. Falling back
            // to column-basis HVP would re-walk all `n` rows once per column.
            if let Some(mut matrix) = dense_forced()? {
                if matrix.nrows() != total || matrix.ncols() != total {
                    return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                        "{context}: dense_forced shape mismatch: got {}x{}, expected {total}x{total}",
                        matrix.nrows(),
                        matrix.ncols()
                    ) }.into());
                }
                if matrix.iter().any(|value| !value.is_finite()) {
                    return Err(CustomFamilyError::NumericalFailure {
                        reason: format!("{context}: dense_forced returned non-finite values"),
                    }
                    .into());
                }
                symmetrize_dense_in_place(&mut matrix);
                return Ok(matrix);
            }
            // Column-basis reconstruction `H = H · I`. Driving it through the
            // batched multi-RHS apply lets a tiled/streamed operator sweep each
            // row tile exactly once for all `total` columns instead of once per
            // column (`total` full sweeps). The result is, column for column,
            // identical to applying the operator to each unit basis vector.
            let identity = Array2::<f64>::eye(total);
            let mut matrix = Array2::<f64>::zeros((total, total));
            apply_mat(&identity, &mut matrix)?;
            if matrix.iter().any(|value| !value.is_finite()) {
                return Err(CustomFamilyError::NumericalFailure {
                    reason: format!("{context}: operator matvec returned non-finite values"),
                }
                .into());
            }
            symmetrize_dense_in_place(&mut matrix);
            Ok(matrix)
        }
    }
}


fn exact_newton_joint_hessian_source_from_workspace(
    workspace: &Arc<dyn ExactNewtonJointHessianWorkspace>,
    total: usize,
    intent: MaterializationIntent,
    context: &str,
) -> Result<Option<JointHessianSource>, String> {
    if workspace.hessian_source_preference_for_intent(intent)
        == JointHessianSourcePreference::Operator
    {
        return exact_newton_joint_hessian_operator_source_from_workspace(
            workspace, total, intent, context,
        );
    }

    if let Some(mut hessian) = workspace.hessian_dense()? {
        if hessian.nrows() != total || hessian.ncols() != total {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "{context}: dense Hessian shape mismatch: got {}x{}, expected {total}x{total}",
                    hessian.nrows(),
                    hessian.ncols()
                ),
            }
            .into());
        }
        if hessian.iter().any(|value| !value.is_finite()) {
            return Err(CustomFamilyError::NumericalFailure {
                reason: format!("{context}: dense Hessian contains non-finite values"),
            }
            .into());
        }
        symmetrize_dense_in_place(&mut hessian);
        return Ok(Some(JointHessianSource::Dense(hessian)));
    }

    exact_newton_joint_hessian_operator_source_from_workspace(workspace, total, intent, context)
}


fn exact_newton_joint_hessian_operator_source_from_workspace(
    workspace: &Arc<dyn ExactNewtonJointHessianWorkspace>,
    total: usize,
    intent: MaterializationIntent,
    context: &str,
) -> Result<Option<JointHessianSource>, String> {
    let Some(diagonal) = workspace.hessian_diagonal()? else {
        if workspace.hessian_source_preference_for_intent(intent)
            == JointHessianSourcePreference::Operator
        {
            return Err(CustomFamilyError::UnsupportedConfiguration {
                reason: format!(
                    "{context}: operator-preferred Hessian workspace did not provide a diagonal"
                ),
            }
            .into());
        }
        return Ok(None);
    };
    if diagonal.len() != total {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: format!(
                "{context}: operator diagonal length mismatch: got {}, expected {}",
                diagonal.len(),
                total
            ),
        }
        .into());
    }
    if diagonal.iter().any(|value| !value.is_finite()) {
        return Err(CustomFamilyError::NumericalFailure {
            reason: format!("{context}: operator diagonal contains non-finite values"),
        }
        .into());
    }

    if !workspace.hessian_matvec_available() {
        if workspace.hessian_source_preference_for_intent(intent)
            == JointHessianSourcePreference::Operator
        {
            return Err(CustomFamilyError::UnsupportedConfiguration {
                reason: format!(
                    "{context}: operator-preferred Hessian workspace did not provide HVPs"
                ),
            }
            .into());
        }
        return Ok(None);
    }

    let workspace_apply = Arc::clone(workspace);
    let workspace_apply_into = Arc::clone(workspace);
    let workspace_apply_mat = Arc::clone(workspace);
    let workspace_dense_forced = Arc::clone(workspace);
    let context_apply: Arc<str> = Arc::from(context);
    let context_apply_into = Arc::clone(&context_apply);
    let context_apply_mat = Arc::clone(&context_apply);
    let context_dense_forced = Arc::clone(&context_apply);
    Ok(Some(JointHessianSource::Operator {
        apply: Arc::new(move |v: &Array1<f64>| {
            if v.len() != total {
                return Err(CustomFamilyError::DimensionMismatch {
                    reason: format!(
                        "{}: operator input length mismatch: got {}, expected {total}",
                        &*context_apply,
                        v.len()
                    ),
                }
                .into());
            }
            let Some(out) = workspace_apply.hessian_matvec(v)? else {
                return Err(CustomFamilyError::UnsupportedConfiguration {
                    reason: "joint exact-newton operator matvec unavailable".to_string(),
                }
                .into());
            };
            if out.len() != total {
                return Err(CustomFamilyError::DimensionMismatch {
                    reason: format!(
                        "{}: operator matvec length mismatch: got {}, expected {total}",
                        &*context_apply,
                        out.len()
                    ),
                }
                .into());
            }
            if out.iter().any(|value| !value.is_finite()) {
                return Err(CustomFamilyError::NumericalFailure {
                    reason: format!(
                        "{}: operator matvec returned non-finite values",
                        &*context_apply
                    ),
                }
                .into());
            }
            Ok(out)
        }),
        apply_into: Arc::new(move |v: &Array1<f64>, out: &mut Array1<f64>| {
            if v.len() != total || out.len() != total {
                return Err(CustomFamilyError::DimensionMismatch {
                    reason: format!(
                        "{}: operator input/output length mismatch: v={} out={} expected={total}",
                        &*context_apply_into,
                        v.len(),
                        out.len()
                    ),
                }
                .into());
            }
            if !workspace_apply_into.hessian_matvec_into(v, out)? {
                return Err(CustomFamilyError::UnsupportedConfiguration {
                    reason: "joint exact-newton operator matvec unavailable".to_string(),
                }
                .into());
            }
            if out.iter().any(|value| !value.is_finite()) {
                return Err(CustomFamilyError::NumericalFailure {
                    reason: format!(
                        "{}: operator matvec returned non-finite values",
                        &*context_apply_into
                    ),
                }
                .into());
            }
            Ok(())
        }),
        apply_mat: Arc::new(move |v_cols: &Array2<f64>, out: &mut Array2<f64>| {
            if v_cols.nrows() != total || out.nrows() != total {
                return Err(CustomFamilyError::DimensionMismatch {
                    reason: format!(
                        "{}: operator batched apply row mismatch: v_cols={}x{} out={}x{} expected rows={total}",
                        &*context_apply_mat,
                        v_cols.nrows(),
                        v_cols.ncols(),
                        out.nrows(),
                        out.ncols()
                    ),
                }
                .into());
            }
            if v_cols.ncols() != out.ncols() {
                return Err(CustomFamilyError::DimensionMismatch {
                    reason: format!(
                        "{}: operator batched apply column mismatch: v_cols has {} columns, out has {}",
                        &*context_apply_mat,
                        v_cols.ncols(),
                        out.ncols()
                    ),
                }
                .into());
            }
            if !workspace_apply_mat.hessian_apply_mat(v_cols, out)? {
                return Err(CustomFamilyError::UnsupportedConfiguration {
                    reason: "joint exact-newton operator batched apply unavailable".to_string(),
                }
                .into());
            }
            if out.iter().any(|value| !value.is_finite()) {
                return Err(CustomFamilyError::NumericalFailure {
                    reason: format!(
                        "{}: operator batched apply returned non-finite values",
                        &*context_apply_mat
                    ),
                }
                .into());
            }
            Ok(())
        }),
        diagonal,
        dense_forced: Arc::new(move || -> Result<Option<Array2<f64>>, String> {
            match workspace_dense_forced.hessian_dense_forced()? {
                Some(mut matrix) => {
                    if matrix.nrows() != total || matrix.ncols() != total {
                        return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                            "{}: hessian_dense_forced shape mismatch: got {}x{}, expected {total}x{total}",
                            &*context_dense_forced,
                            matrix.nrows(),
                            matrix.ncols()
                        ) }.into());
                    }
                    if matrix.iter().any(|value| !value.is_finite()) {
                        return Err(CustomFamilyError::NumericalFailure {
                            reason: format!(
                                "{}: hessian_dense_forced returned non-finite values",
                                &*context_dense_forced
                            ),
                        }
                        .into());
                    }
                    symmetrize_dense_in_place(&mut matrix);
                    Ok(Some(matrix))
                }
                None => Ok(None),
            }
        }),
    }))
}


fn symmetrized_square_matrix(
    mut matrix: Array2<f64>,
    expected: usize,
    context: &str,
) -> Result<Array2<f64>, String> {
    if matrix.nrows() != expected || matrix.ncols() != expected {
        return Err(format!(
            "{context}: got {}x{}, expected {}x{}",
            matrix.nrows(),
            matrix.ncols(),
            expected,
            expected
        ));
    }
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(CustomFamilyError::NumericalFailure {
            reason: format!("{context}: matrix contains non-finite values"),
        }
        .into());
    }
    symmetrize_dense_in_place(&mut matrix);
    Ok(matrix)
}


/// Try exact Newton joint Hessian first, then surrogate. Returns `None` if
/// neither path provides a joint Hessian. When successful, returns the joint
/// Hessian source, flat beta, and boxed closures for computing directional
/// derivatives dH[v] and d²H[u,v].
///
/// This eliminates the previously duplicated exact-Newton and surrogate
/// code blocks in `outerobjectivegradienthessian_internal`.
fn build_joint_hessian_closures<'a, F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &'a F,
    block_states: &'a [ParameterBlockState],
    specs: &'a [ParameterBlockSpec],
    total: usize,
    options: &BlockwiseFitOptions,
    preferred_workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
) -> Result<Option<JointHessianBundle<'a>>, String> {
    // Path 1: exact Newton joint Hessian (preferred).
    let beta_flat = flatten_state_betas(block_states, specs);
    let synced = Arc::new(synchronized_states_from_flat_beta(
        family,
        specs,
        block_states,
        &beta_flat,
    )?);
    let hessian_workspace = match preferred_workspace {
        Some(workspace) => Some(workspace),
        None => family.exact_newton_joint_hessian_workspace_with_options(
            synced.as_ref(),
            specs,
            options,
        )?,
    };
    // Outer-eval entry: prime any per-row jet caches the workspace will hand
    // to the directional-derivative path. Runs at top-level rayon (we are
    // outside the ext-coord `par_iter` here), so the cache build's own
    // `par_iter` enjoys full thread-pool parallelism. PIRLS-side workspace
    // construction skips this priming because PIRLS never invokes
    // `directional_derivative_operator`.
    if let Some(workspace) = hessian_workspace.as_ref() {
        workspace.warm_up_outer_caches()?;
    }
    if let Some(curvature) = family.exact_newton_outer_curvature(block_states)? {
        let h_joint_unpen = JointHessianSource::Dense(symmetrized_square_matrix(
            curvature.hessian,
            total,
            "joint exact-newton Hessian shape mismatch in outer gradient (rescaled)",
        )?);
        let compute_dh = Box::new(exact_newton_dh_closure(
            family,
            Arc::clone(&synced),
            specs,
            total,
            true,
            1.0,
            hessian_workspace.clone(),
        ));
        let compute_dh_many = None;
        let compute_d2h = Box::new(exact_newton_d2h_closure(
            family,
            Arc::clone(&synced),
            specs,
            total,
            true,
            1.0,
            hessian_workspace.clone(),
        ));
        let owned_compute_dh = exact_newton_dh_closure_owned(
            family.clone(),
            Arc::clone(&synced),
            specs.to_vec(),
            total,
            true,
            1.0,
            hessian_workspace.clone(),
        );
        let owned_compute_dh_many = None;
        let owned_compute_d2h = exact_newton_d2h_closure_owned(
            family.clone(),
            Arc::clone(&synced),
            specs.to_vec(),
            total,
            true,
            1.0,
            hessian_workspace.clone(),
        );
        return Ok(Some(JointHessianBundle {
            source: h_joint_unpen,
            beta_flat,
            compute_dh,
            compute_dh_many,
            compute_d2h,
            compute_d2h_many: None,
            owned_compute_dh: Some(owned_compute_dh),
            owned_compute_dh_many,
            owned_compute_d2h: Some(owned_compute_d2h),
            owned_compute_d2h_many: None,
            rho_curvature_scale: curvature.rho_curvature_scale,
            hessian_logdet_correction: curvature.hessian_logdet_correction,
        }));
    }
    let exact_joint_source = if let Some(workspace) = hessian_workspace.as_ref() {
        exact_newton_joint_hessian_source_from_workspace(
            workspace,
            total,
            MaterializationIntent::OuterGradient,
            "joint exact-newton operator mismatch in outer gradient",
        )?
    } else {
        None
    };
    let exact_joint_source = match exact_joint_source {
        Some(source) => Some(source),
        None => exact_newton_joint_hessian_symmetrized(
            family,
            block_states,
            specs,
            total,
            "joint exact-newton Hessian shape mismatch in outer gradient",
        )
        .map(|source| source.map(JointHessianSource::Dense))?,
    };
    if let Some(h_joint_unpen) = exact_joint_source {
        let compute_dh = Box::new(exact_newton_dh_closure(
            family,
            Arc::clone(&synced),
            specs,
            total,
            false,
            1.0,
            hessian_workspace.clone(),
        ));
        let compute_dh_many = exact_newton_dh_many_closure(1.0, hessian_workspace.clone());
        let compute_d2h = Box::new(exact_newton_d2h_closure(
            family,
            Arc::clone(&synced),
            specs,
            total,
            false,
            1.0,
            hessian_workspace.clone(),
        ));
        let owned_compute_dh = exact_newton_dh_closure_owned(
            family.clone(),
            Arc::clone(&synced),
            specs.to_vec(),
            total,
            false,
            1.0,
            hessian_workspace.clone(),
        );
        let owned_compute_dh_many =
            exact_newton_dh_many_closure_owned(1.0, hessian_workspace.clone());
        let owned_compute_d2h = exact_newton_d2h_closure_owned(
            family.clone(),
            Arc::clone(&synced),
            specs.to_vec(),
            total,
            false,
            1.0,
            hessian_workspace.clone(),
        );
        let compute_d2h_many = exact_newton_d2h_many_closure(1.0, hessian_workspace.clone());
        let owned_compute_d2h_many =
            exact_newton_d2h_many_closure_owned(1.0, hessian_workspace.clone());
        return Ok(Some(JointHessianBundle {
            source: h_joint_unpen,
            beta_flat,
            compute_dh,
            compute_dh_many,
            compute_d2h,
            compute_d2h_many,
            owned_compute_dh: Some(owned_compute_dh),
            owned_compute_dh_many,
            owned_compute_d2h: Some(owned_compute_d2h),
            owned_compute_d2h_many,
            rho_curvature_scale: 1.0,
            hessian_logdet_correction: 0.0,
        }));
    }

    // Path 2: surrogate joint Hessian (fallback).
    if let Some(h_joint_unpen) = family
        .joint_outer_hyper_surrogate_hessian_with_specs(block_states, specs)?
        .map(|h| {
            symmetrized_square_matrix(
                h,
                total,
                "joint outer-hyper surrogate Hessian shape mismatch",
            )
        })
        .transpose()?
    {
        let beta_flat = flatten_state_betas(block_states, specs);

        let compute_dh = Box::new(
            move |v_k: &Array1<f64>| -> Result<Option<DriftDerivResult>, String> {
                let h_rho = family
                    .joint_outer_hyper_surrogate_hessian_directional_derivative_with_specs(
                        block_states,
                        specs,
                        v_k,
                    )?;
                match h_rho {
                    Some(h) => Ok(Some(DriftDerivResult::Dense(symmetrized_square_matrix(
                        h,
                        total,
                        "joint surrogate dH shape mismatch",
                    )?))),
                    None => Err(CustomFamilyError::UnsupportedConfiguration {
                        reason: "joint surrogate dH unavailable for analytic outer gradient"
                            .to_string(),
                    }
                    .into()),
                }
            },
        );
        let compute_d2h = Box::new(
            move |u: &Array1<f64>, v: &Array1<f64>| -> Result<Option<DriftDerivResult>, String> {
                match family
                    .joint_outer_hyper_surrogate_hessian_second_directional_derivative_with_specs(
                        block_states,
                        specs,
                        u,
                        v,
                    )? {
                    Some(m) => Ok(Some(DriftDerivResult::Dense(symmetrized_square_matrix(
                        m,
                        total,
                        "joint surrogate d2H shape mismatch",
                    )?))),
                    None => Ok(None),
                }
            },
        );
        let family_owned = family.clone();
        let states_owned = block_states.to_vec();
        let specs_owned = specs.to_vec();
        let owned_compute_dh = Arc::new(
            move |v_k: &Array1<f64>| -> Result<Option<DriftDerivResult>, String> {
                match family_owned
                    .joint_outer_hyper_surrogate_hessian_directional_derivative_with_specs(
                        &states_owned,
                        &specs_owned,
                        v_k,
                    )? {
                    Some(h) => Ok(Some(DriftDerivResult::Dense(symmetrized_square_matrix(
                        h,
                        total,
                        "joint surrogate dH shape mismatch",
                    )?))),
                    None => Err(CustomFamilyError::UnsupportedConfiguration {
                        reason: "joint surrogate dH unavailable for analytic outer gradient"
                            .to_string(),
                    }
                    .into()),
                }
            },
        );
        let family_owned = family.clone();
        let states_owned = block_states.to_vec();
        let specs_owned = specs.to_vec();
        let owned_compute_d2h = Arc::new(
            move |u: &Array1<f64>, v: &Array1<f64>| -> Result<Option<DriftDerivResult>, String> {
                match family_owned
                    .joint_outer_hyper_surrogate_hessian_second_directional_derivative_with_specs(
                        &states_owned,
                        &specs_owned,
                        u,
                        v,
                    )? {
                    Some(m) => Ok(Some(DriftDerivResult::Dense(symmetrized_square_matrix(
                        m,
                        total,
                        "joint surrogate d2H shape mismatch",
                    )?))),
                    None => Ok(None),
                }
            },
        );
        return Ok(Some(JointHessianBundle {
            source: JointHessianSource::Dense(h_joint_unpen),
            beta_flat,
            compute_dh,
            compute_dh_many: None,
            compute_d2h,
            compute_d2h_many: None,
            owned_compute_dh: Some(owned_compute_dh),
            owned_compute_dh_many: None,
            owned_compute_d2h: Some(owned_compute_d2h),
            owned_compute_d2h_many: None,
            rho_curvature_scale: 1.0,
            hessian_logdet_correction: 0.0,
        }));
    }

    Ok(None)
}


/// Build a closure computing dH[v] using exact Newton derivatives on synced states.
/// Non-finite derivative output is treated as a hard error.
/// Symmetrize-and-scale the dH Dense result, optionally rejecting non-finite
/// values first.  The borrowed factory (`exact_newton_dh_closure`) guards
/// against non-finite output (`check_finite = true`); the owned factory
/// (`exact_newton_dh_closure_owned`) historically does not (`check_finite =
/// false`).  Routing both through this helper keeps that behavioral
/// distinction explicit rather than silently divergent.
fn finalize_dh_dense(
    h: Array2<f64>,
    total: usize,
    scale: f64,
    check_finite: bool,
) -> Result<Option<DriftDerivResult>, String> {
    if check_finite && h.iter().any(|v| !v.is_finite()) {
        return Err(CustomFamilyError::NumericalFailure {
            reason: "joint exact-newton dH returned non-finite values".to_string(),
        }
        .into());
    }
    let mut sym = symmetrized_square_matrix(h, total, "joint exact-newton dH shape mismatch")?;
    if scale != 1.0 {
        sym *= scale;
    }
    Ok(Some(DriftDerivResult::Dense(sym)))
}


/// Single source of truth for the dH[v] three-way dispatch shared by the
/// borrowed (`exact_newton_dh_closure`) and owned
/// (`exact_newton_dh_closure_owned`) closure factories.  The `check_finite`
/// flag preserves the lone behavioral difference between the two (the borrowed
/// variant rejects non-finite dense output, the owned variant does not); all
/// other logic — outer-curvature path, workspace-operator fast path, and
/// joint-Hessian fallback — is identical and lives here once.
fn exact_newton_dh_apply<F: CustomFamily + Sync>(
    family: &F,
    synced_states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    total: usize,
    use_outer_curvature_derivatives: bool,
    scale: f64,
    workspace: Option<&Arc<dyn ExactNewtonJointHessianWorkspace>>,
    check_finite: bool,
    v_k: &Array1<f64>,
) -> Result<Option<DriftDerivResult>, String> {
    // `v_k` is ALREADY the perturbation direction `δβ` the caller wants the
    // directional Hessian derivative evaluated along. The `HessianDerivativeProvider`s
    // (`BorrowedJointDerivProvider`/`OwnedJointDerivProvider`) own the implicit-
    // function-theorem sign `δβ = −H⁻¹(A_k β̂)` and negate before calling this
    // closure (matching `exact_newton_d2h_apply` and the owned `_many` closure,
    // which also pass the direction straight through). Re-negating here would
    // double-negate `D_β H[δβ]`, flipping the mode-response drift in the outer
    // LAML trace `½ tr(K · (B_i + D_β H[δβ_i]))` and desynchronising the analytic
    // outer gradient from its objective for every β-dependent-Hessian exact
    // family (spatial-adaptive, survival/bernoulli marginal-slope). Pass through.
    let mode_response = v_k.clone();
    if use_outer_curvature_derivatives {
        let h_rho = family.exact_newton_outer_curvature_directional_derivative_with_specs(
            synced_states,
            specs,
            &mode_response,
        )?;
        return match h_rho {
            Some(h) => finalize_dh_dense(h, total, scale, check_finite),
            None => Err(CustomFamilyError::UnsupportedConfiguration {
                reason: "joint exact-newton dH unavailable for analytic outer gradient".to_string(),
            }
            .into()),
        };
    }

    if let Some(workspace) = workspace
        && let Some(operator) = workspace.directional_derivative_operator(&mode_response)?
    {
        return Ok(Some(scale_drift_deriv_result(
            DriftDerivResult::Operator(operator),
            scale,
        )));
    }

    match family.exact_newton_joint_hessian_directional_derivative_with_specs(
        synced_states,
        specs,
        &mode_response,
    )? {
        Some(h) => finalize_dh_dense(h, total, scale, check_finite),
        None => Err(CustomFamilyError::UnsupportedConfiguration {
            reason: "joint exact-newton dH unavailable for analytic outer gradient".to_string(),
        }
        .into()),
    }
}


fn exact_newton_dh_closure<'a, F: CustomFamily + Sync>(
    family: &'a F,
    synced_states: Arc<Vec<ParameterBlockState>>,
    specs: &'a [ParameterBlockSpec],
    total: usize,
    use_outer_curvature_derivatives: bool,
    scale: f64,
    workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
) -> impl Fn(&Array1<f64>) -> Result<Option<DriftDerivResult>, String> + Send + Sync + 'a {
    move |v_k: &Array1<f64>| {
        exact_newton_dh_apply(
            family,
            synced_states.as_ref(),
            specs,
            total,
            use_outer_curvature_derivatives,
            scale,
            workspace.as_ref(),
            true,
            v_k,
        )
    }
}


fn exact_newton_dh_many_closure<'a>(
    scale: f64,
    workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
) -> Option<Box<DriftDerivManyFn<'a>>> {
    let workspace = workspace?;
    Some(Box::new(move |directions: &[Array1<f64>]| {
        // `directions` are already the perturbation directions `δβ`; the provider
        // owns the IFT sign and pre-negates (see `exact_newton_dh_apply`). The
        // owned `_many` counterpart passes them straight through, so this borrowed
        // path must too — re-negating here double-flips the mode-response drift.
        workspace
            .directional_derivative_operators(directions)?
            .into_iter()
            .map(|maybe_operator| {
                Ok(maybe_operator.map(|operator| {
                    scale_drift_deriv_result(DriftDerivResult::Operator(operator), scale)
                }))
            })
            .collect()
    }))
}


/// Single source of truth for the d²H[u,v] three-way dispatch shared by the
/// borrowed (`exact_newton_d2h_closure`) and owned
/// (`exact_newton_d2h_closure_owned`) closure factories.  Takes references for
/// `family`/`specs` so both ownership flavors can call it; the only difference
/// between the two factories is borrow-vs-own plumbing, which lives in the
/// wrappers, not here.
fn exact_newton_d2h_apply<F: CustomFamily + Sync>(
    family: &F,
    synced_states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    total: usize,
    use_outer_curvature_derivatives: bool,
    scale: f64,
    workspace: Option<&Arc<dyn ExactNewtonJointHessianWorkspace>>,
    u: &Array1<f64>,
    v: &Array1<f64>,
) -> Result<Option<DriftDerivResult>, String> {
    if use_outer_curvature_derivatives {
        return match family.exact_newton_outer_curvature_second_directional_derivative_with_specs(
            synced_states,
            specs,
            u,
            v,
        )? {
            Some(m) => {
                let mut sym =
                    symmetrized_square_matrix(m, total, "joint exact-newton d2H shape mismatch")?;
                if scale != 1.0 {
                    sym *= scale;
                }
                Ok(Some(DriftDerivResult::Dense(sym)))
            }
            None => Ok(None),
        };
    }

    if let Some(workspace) = workspace
        && let Some(operator) = workspace.second_directional_derivative_operator(u, v)?
    {
        return Ok(Some(scale_drift_deriv_result(
            DriftDerivResult::Operator(operator),
            scale,
        )));
    }

    match family.exact_newton_joint_hessian_second_directional_derivative_with_specs(
        synced_states,
        specs,
        u,
        v,
    )? {
        Some(m) => {
            let mut sym =
                symmetrized_square_matrix(m, total, "joint exact-newton d2H shape mismatch")?;
            if scale != 1.0 {
                sym *= scale;
            }
            Ok(Some(DriftDerivResult::Dense(sym)))
        }
        None => Ok(None),
    }
}


/// Build a closure computing d²H[u,v] using exact Newton derivatives on synced states.
fn exact_newton_d2h_closure<'a, F: CustomFamily + Sync>(
    family: &'a F,
    synced_states: Arc<Vec<ParameterBlockState>>,
    specs: &'a [ParameterBlockSpec],
    total: usize,
    use_outer_curvature_derivatives: bool,
    scale: f64,
    workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
) -> impl Fn(&Array1<f64>, &Array1<f64>) -> Result<Option<DriftDerivResult>, String> + Send + Sync + 'a
{
    move |u: &Array1<f64>, v: &Array1<f64>| {
        exact_newton_d2h_apply(
            family,
            synced_states.as_ref(),
            specs,
            total,
            use_outer_curvature_derivatives,
            scale,
            workspace.as_ref(),
            u,
            v,
        )
    }
}


fn exact_newton_d2h_many_closure<'a>(
    scale: f64,
    workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
) -> Option<Box<DriftSecondDerivManyFn<'a>>> {
    let workspace = workspace?;
    Some(Box::new(move |pairs: &[(Array1<f64>, Array1<f64>)]| {
        workspace
            .second_directional_derivative_operators(pairs)?
            .into_iter()
            .map(|maybe_operator| {
                Ok(maybe_operator.map(|operator| {
                    scale_drift_deriv_result(DriftDerivResult::Operator(operator), scale)
                }))
            })
            .collect()
    }))
}


fn exact_newton_dh_closure_owned<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: F,
    synced_states: Arc<Vec<ParameterBlockState>>,
    specs: Vec<ParameterBlockSpec>,
    total: usize,
    use_outer_curvature_derivatives: bool,
    scale: f64,
    workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
) -> Arc<dyn Fn(&Array1<f64>) -> Result<Option<DriftDerivResult>, String> + Send + Sync> {
    Arc::new(move |v_k: &Array1<f64>| {
        exact_newton_dh_apply(
            &family,
            synced_states.as_ref(),
            &specs,
            total,
            use_outer_curvature_derivatives,
            scale,
            workspace.as_ref(),
            false,
            v_k,
        )
    })
}


fn exact_newton_dh_many_closure_owned(
    scale: f64,
    workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
) -> Option<
    Arc<dyn Fn(&[Array1<f64>]) -> Result<Vec<Option<DriftDerivResult>>, String> + Send + Sync>,
> {
    let workspace = workspace?;
    Some(Arc::new(move |directions: &[Array1<f64>]| {
        workspace
            .directional_derivative_operators(directions)?
            .into_iter()
            .map(|maybe_operator| {
                Ok(maybe_operator.map(|operator| {
                    scale_drift_deriv_result(DriftDerivResult::Operator(operator), scale)
                }))
            })
            .collect()
    }))
}


fn exact_newton_d2h_closure_owned<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: F,
    synced_states: Arc<Vec<ParameterBlockState>>,
    specs: Vec<ParameterBlockSpec>,
    total: usize,
    use_outer_curvature_derivatives: bool,
    scale: f64,
    workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
) -> Arc<dyn Fn(&Array1<f64>, &Array1<f64>) -> Result<Option<DriftDerivResult>, String> + Send + Sync>
{
    Arc::new(move |u: &Array1<f64>, v: &Array1<f64>| {
        exact_newton_d2h_apply(
            &family,
            synced_states.as_ref(),
            &specs,
            total,
            use_outer_curvature_derivatives,
            scale,
            workspace.as_ref(),
            u,
            v,
        )
    })
}


fn exact_newton_d2h_many_closure_owned(
    scale: f64,
    workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
) -> Option<
    Arc<
        dyn Fn(&[(Array1<f64>, Array1<f64>)]) -> Result<Vec<Option<DriftDerivResult>>, String>
            + Send
            + Sync,
    >,
> {
    let workspace = workspace?;
    Some(Arc::new(move |pairs: &[(Array1<f64>, Array1<f64>)]| {
        workspace
            .second_directional_derivative_operators(pairs)?
            .into_iter()
            .map(|maybe_operator| {
                Ok(maybe_operator.map(|operator| {
                    scale_drift_deriv_result(DriftDerivResult::Operator(operator), scale)
                }))
            })
            .collect()
    }))
}


fn strict_solve_spd(matrix: &Array2<f64>, rhs: &Array1<f64>) -> Result<Array1<f64>, String> {
    let mut sym = matrix.clone();
    symmetrize_dense_in_place(&mut sym);
    let chol = sym
        .cholesky(Side::Lower)
        .map_err(|_| "strict pseudo-laplace SPD solve failed".to_string())?;
    Ok(chol.solvevec(rhs))
}


/// Statistics about a Levenberg-Marquardt-style δ-ridge SPD continuation.
/// Recorded by `strict_solve_spd_with_lm_continuation` and surfaced for
/// diagnostics — a recurring need for nontrivial ridges signals fragile
/// curvature that the controller may need to escalate.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct StrictSpdLmStats {
    /// δ value finally used (0.0 means the bare strict solve succeeded).
    pub(crate) delta_used: f64,
    /// Number of escalations performed before Cholesky succeeded.
    pub(crate) escalations: usize,
}


/// Strict-mode SPD solve with internal Levenberg-Marquardt δ-ridge
/// continuation: solves `(H + δI) x = b` with δ escalated geometrically
/// until the Cholesky succeeds.  The bare `strict_solve_spd` is unchanged —
/// callers that need strict semantics keep them.  Callers that want
/// fail-soft Newton on a fragile geometry (e.g. spatial-adaptive seed
/// evaluation) use this wrapper to avoid bouncing the entire seed on a
/// numerically-indefinite block.
///
/// Schedule: δ₀ = max(ε · ‖H‖₁ / p, 1e-12); growth ×10 per step; capped
/// at MAX_ESCALATIONS escalations.  The cap prevents runaway curvature
/// from producing arbitrary ridges; if the cap is hit, the bare strict
/// error propagates so the caller can route to a different optimization
/// path (e.g. sparse/gradient-only standard REML at full data).
/// Shared escalation/ridge-growth schedule used by the three
/// `strict_*_spd_with_lm_continuation` helpers. Hoisted here so a single
/// change updates the solve / inverse / logdet paths in lockstep.
const STRICT_SPD_LM_MAX_ESCALATIONS: usize = 16;

const STRICT_SPD_LM_RIDGE_GROWTH: f64 = 10.0;


/// Floor applied to IRLS working weights so downstream divisions cannot hit
/// exact zero. Used as the default `minweight` in `CustomFamilyOptions` and
/// mirrored in tests that override it.
///
/// Sourced from the canonical PIRLS positive-weight floor
/// ([`crate::solver::pirls::MIN_WEIGHT`] = `1e-12`) so every floored family
/// shares one definition; this alias keeps the descriptive local name at the
/// `minweight` defaults.
const CUSTOM_FAMILY_WEIGHT_FLOOR: f64 = crate::solver::pirls::MIN_WEIGHT;


/// Default initial ridge δ for the explicit-stabilization Cholesky escalation
/// schedule. Enters the quadratic term, the Laplace Hessian, and the penalty
/// log-determinant via the active `RidgePolicy`.
const CUSTOM_FAMILY_RIDGE_FLOOR: f64 = 1e-12;


/// Relative eigenvalue floor used wherever an eigendecomposition needs to
/// distinguish "real" curvature from noise: `eps_floor = EVAL_FLOOR · max|λ|`.
/// Applied uniformly in the strict-SPD LM eigen fallback, positive-part
/// pseudo-inverse, and penalty-direction projection.
const CUSTOM_FAMILY_EVAL_FLOOR: f64 = 1e-12;


/// Absolute relative-condition guard used to prevent the eigen / spectral
/// floors from collapsing to zero when `max|λ|` is itself tiny. Combined with
/// `CUSTOM_FAMILY_EVAL_FLOOR · max|λ|` via `.max(...)`.
const CUSTOM_FAMILY_CONDITION_RELATIVE_FLOOR: f64 = 1e-14;


/// Shared engine: try the bare strict path, fall through to an escalating
/// LM δ-ridge Cholesky, and finally an eigen-floor fallback that clamps every
/// eigenvalue from below at `eps_floor = 1e-12 · max|λ|`. Each caller
/// (solve / inverse / logdet) supplies the three operation-specific closures.
///
/// Centralizing the LM/eigen scaffolding here both removes ~180 lines of
/// near-duplicated code and guarantees the three sibling helpers stay in
/// lockstep — any future change to the schedule, the trace_scale heuristic,
/// or the eigen-floor logic now lives in exactly one place.
fn strict_spd_lm_engine<R>(
    matrix: &Array2<f64>,
    op_label: &'static str,
    empty: R,
    bare_path: impl FnOnce(&Array2<f64>) -> Result<R, String>,
    process_chol: impl FnOnce(&crate::faer_ndarray::FaerCholeskyFactor) -> R,
    process_eigen: impl FnOnce(&Array1<f64>, &Array2<f64>, f64) -> R,
) -> Result<(R, StrictSpdLmStats), String> {
    if let Ok(r) = bare_path(matrix) {
        return Ok((r, StrictSpdLmStats::default()));
    }

    let p = matrix.nrows();
    if p == 0 {
        return Ok((empty, StrictSpdLmStats::default()));
    }
    let mut sym = matrix.clone();
    symmetrize_dense_in_place(&mut sym);
    let trace_scale = (0..p).map(|i| sym[[i, i]].abs()).sum::<f64>() / (p as f64);
    let delta0 = (f64::EPSILON * trace_scale.max(1.0)).max(CUSTOM_FAMILY_RIDGE_FLOOR);

    let mut delta = delta0;
    for escalation in 1..=STRICT_SPD_LM_MAX_ESCALATIONS {
        let mut ridged = sym.clone();
        for i in 0..p {
            ridged[[i, i]] += delta;
        }
        if let Ok(chol) = ridged.cholesky(Side::Lower) {
            return Ok((
                process_chol(&chol),
                StrictSpdLmStats {
                    delta_used: delta,
                    escalations: escalation,
                },
            ));
        }
        delta *= STRICT_SPD_LM_RIDGE_GROWTH;
    }

    // δ-ridge schedule exhausted; fall back to rank-aware eigen-floor handling.
    // Floors every eigenvalue at `eps_floor = 1e-12 · max|λ|` so well-conditioned
    // modes are resolved exactly and rank-deficient directions are handled with
    // controlled curvature, preventing the spatial-adaptive pilot from collapsing
    // to a cold full-data run.
    let max_esc = STRICT_SPD_LM_MAX_ESCALATIONS;
    let (evals, evecs) = FaerEigh::eigh(&sym, Side::Lower).map_err(|e| {
        format!(
            "{op_label} failed even with LM δ-ridge continuation \
             (escalated {max_esc} times to δ={delta:.3e}, trace_scale={trace_scale:.3e}); \
             eigen-floor fallback also failed: {e}"
        )
    })?;
    let max_abs_eval = evals.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    let eps_floor = (CUSTOM_FAMILY_EVAL_FLOOR * max_abs_eval).max(1e-300);
    Ok((
        process_eigen(&evals, &evecs, eps_floor),
        StrictSpdLmStats {
            delta_used: delta,
            escalations: STRICT_SPD_LM_MAX_ESCALATIONS + 1,
        },
    ))
}


pub(crate) fn strict_solve_spd_with_lm_continuation(
    matrix: &Array2<f64>,
    rhs: &Array1<f64>,
) -> Result<(Array1<f64>, StrictSpdLmStats), String> {
    let p = matrix.nrows();
    strict_spd_lm_engine(
        matrix,
        "strict pseudo-laplace SPD solve",
        Array1::<f64>::zeros(0),
        |m| strict_solve_spd(m, rhs),
        |chol| chol.solvevec(rhs),
        |evals, evecs, eps_floor| {
            // x = Q diag(1/Λ̃) Qᵀ rhs.
            let mut q_t_rhs = Array1::<f64>::zeros(p);
            for k in 0..p {
                let mut acc = 0.0;
                for i in 0..p {
                    acc += evecs[[i, k]] * rhs[i];
                }
                q_t_rhs[k] = acc / evals[k].max(eps_floor);
            }
            let mut x = Array1::<f64>::zeros(p);
            for i in 0..p {
                let mut acc = 0.0;
                for k in 0..p {
                    acc += evecs[[i, k]] * q_t_rhs[k];
                }
                x[i] = acc;
            }
            x
        },
    )
}


/// Exact pseudo-Laplace log-determinant `log|H + S_λ|` of the REML/LAML
/// objective, computed from the eigenspectrum with **no δ-ridge** so the value
/// stays on the same objective as the analytic gradient `tr((H+S_λ)⁻¹ ·)`
/// (gam#748).
///
/// The earlier strict path returned `log|H + S_λ + δI|` with `δ = δ(ρ)`
/// escalated geometrically until factorization succeeded. That makes `V(ρ)`
/// carry a ρ-dependent, discontinuous `δ(ρ)` the analytic derivatives ignore —
/// exactly the objective/derivative mismatch the
/// operator-dense path's own comment forbids ("mixing an approximate
/// determinant with exact traces gives ARC a Hessian for a different
/// objective"). The strict path now computes one honest quantity:
///
/// - eigendecompose the symmetrised `H + S_λ`;
/// - **reject** (return `Err`) when any eigenvalue is genuinely negative
///   (`λ < −tol`). An indefinite joint coefficient Hessian is a real defect
///   (a non-stationary inner β or a mis-signed curvature block); rejecting it
///   tells the outer optimizer to step back, instead of masking it with a
///   biased finite number;
/// - sum `Σ_{λ > tol} log λ` — the exact pseudo-logdet on the positive
///   eigenspace, which is `C∞` in ρ because the positive eigenspace of a PSD
///   `S(ρ)=Σ e^{ρ_k} S_k` is structurally fixed. A near-zero band `[−tol, tol]`
///   (a structural null space) is simply not in `range` and contributes no
///   term, matching the projected `tr` derivative; a near-singular-but-positive
///   curvature is accepted exactly as the historical Cholesky strict path did.
fn strict_exact_pseudo_logdet(
    matrix: &Array2<f64>,
    accumulation_depth: usize,
) -> Result<f64, String> {
    let mut sym = matrix.clone();
    symmetrize_dense_in_place(&mut sym);
    let (evals, _) = FaerEigh::eigh(&sym, Side::Lower)
        .map_err(|e| format!("strict pseudo-laplace eigendecomposition failed: {e}"))?;
    let p = sym.nrows();
    let max_abs_eval = evals.iter().fold(0.0_f64, |acc, &ev| acc.max(ev.abs()));
    // Bauer-Fike: |δσ| ≤ p·‖δH‖_∞; n-term fma roundoff gives ‖δH‖_∞ ≤ ε·n·‖H‖,
    // so σ_noise ≤ ε·n·p·‖H‖₂. Tenfold slack absorbs sign cancellations,
    // and a 100·ε floor handles the ‖H‖→0 limit. This `neg_tol` is the
    // INDEFINITENESS-rejection band only: an eigenvalue below `−neg_tol` is a
    // genuine negative curvature (non-stationary β / mis-signed block) and is
    // rejected, not masked (gam#748).
    let eps = f64::EPSILON;
    let eps_np = eps * (accumulation_depth as f64) * (p as f64);
    // `neg_tol` is the INDEFINITENESS-rejection band only: an eigenvalue below
    // `−neg_tol` is a genuine negative curvature (non-stationary β / mis-signed
    // block) and is rejected, not masked (gam#748).
    let neg_tol = (10.0 * eps_np * max_abs_eval).max(100.0 * eps);
    // POSITIVE-eigenspace inclusion cutoff for the pseudo-logdet sum. This MUST
    // be byte-identical to the cutoff the analytic REML gradient's trace kernel
    // uses (`positive_eigenvalue_threshold`, the `range(H+Sλ)` Moore–Penrose
    // pinv drop in `joint_penalty_subspace_trace_parts`), or the LAML VALUE
    // `½ log|H+Sλ|₊` and its analytic GRADIENT `½ tr((H+Sλ)⁺ ∂Sλ)` are evaluated
    // over DIFFERENT subspaces and describe DIFFERENT objectives — the "mixing
    // an approximate determinant with exact traces gives ARC a Hessian for a
    // different objective" trap (gam#748).
    //
    // Historically this sum used the Bauer–Fike `neg_tol = 10·ε·n·p·‖H‖`, a
    // factor of ~n/10 LARGER than the kernel's `100·ε·p·‖H‖`. At an oversmoothed
    // marginal-slope ρ probe a penalty-null trend eigenvalue lands in the band
    // `(100·ε·p·‖H‖, 10·ε·n·p·‖H‖)`: DROPPED from the value logdet but KEPT in
    // the gradient kernel, so the analytic outer gradient is the derivative of a
    // different objective than the value. ARC's predicted descent then never
    // matches the actual objective change and the outer optimizer freezes
    // (constant ‖g‖, stuck cost — gam#808). Sharing the kernel's threshold here
    // removes the desync at the source; both are `C∞` in ρ (the positive
    // eigenspace of a PSD-shifted Hessian is structurally fixed).
    let pos_tol = positive_eigenvalue_threshold(evals.as_slice().unwrap());
    if evals.iter().any(|&ev| ev < -neg_tol) {
        let min_eval = evals.iter().copied().fold(f64::INFINITY, f64::min);
        let below = evals.iter().filter(|&&ev| ev < -neg_tol).count();
        return Err(CustomFamilyError::NumericalFailure {
            reason: format!(
                "strict pseudo-laplace logdet: {below} eigenvalue(s) below -neg_tol \
             (min(λ)={min_eval:.6e}, max|λ|={max_abs_eval:.6e}, neg_tol={neg_tol:.6e}, εnp={eps_np:.6e}); \
             indefinite joint coefficient Hessian rejected (no δ-ridge masking, gam#748)"
            ),
        }
        .into());
    }
    Ok(evals
        .iter()
        .copied()
        .filter(|&ev| ev > pos_tol)
        .map(f64::ln)
        .sum())
}


fn pinv_positive_part(matrix: &Array2<f64>, ridge_floor: f64) -> Result<Array2<f64>, String> {
    let mut sym = matrix.clone();
    symmetrize_dense_in_place(&mut sym);
    let (eigenvalues, eigenvectors) = sym
        .eigh(Side::Lower)
        .map_err(|e| format!("positive-part covariance eigendecomposition failed: {e}"))?;
    let max_abs_eigenvalue = eigenvalues.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    let tol = (max_abs_eigenvalue * CUSTOM_FAMILY_EVAL_FLOOR)
        .max(ridge_floor.max(CUSTOM_FAMILY_CONDITION_RELATIVE_FLOOR));
    let p = matrix.nrows();
    let mut pinv = Array2::<f64>::zeros((p, p));
    for (k, &ev) in eigenvalues.iter().enumerate() {
        if ev <= tol {
            continue;
        }
        let inv_ev = 1.0 / ev;
        for i in 0..p {
            let vi = eigenvectors[[i, k]];
            for j in 0..p {
                pinv[[i, j]] += inv_ev * vi * eigenvectors[[j, k]];
            }
        }
    }
    symmetrize_dense_in_place(&mut pinv);
    Ok(pinv)
}


fn include_exact_newton_logdet_h<F: CustomFamily + ?Sized>(
    family: &F,
    options: &BlockwiseFitOptions,
) -> bool {
    options.use_remlobjective
        && matches!(
            family.exact_newton_outerobjective(),
            ExactNewtonOuterObjective::RidgedQuadraticReml
                | ExactNewtonOuterObjective::StrictPseudoLaplace
        )
}


pub(crate) fn custom_family_outer_derivatives<F: CustomFamily + ?Sized>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
) -> (
    crate::solver::outer_strategy::Derivative,
    crate::solver::outer_strategy::DeclaredHessianForm,
) {
    use crate::solver::outer_strategy::{DeclaredHessianForm, Derivative};

    // The capability-vs-policy split: capability tells us *what the family
    // can compute*; policy tells us *what we should ask for at this size*.
    //
    // For the outer-strategy declaration here we have only `specs` and
    // `options` (no resolved psi_dim), so policy is queried at
    // psi_dim = 0 — the gradient/Hessian forms returned here are the
    // pre-psi declarations consumed by the outer planner ladder. The
    // per-iter clamp in `optimize_spatial_length_scale_exact_joint`
    // consults `outer_derivative_policy` again with the realized
    // psi_dim for the κ optimizer.
    let policy = family.outer_derivative_policy(specs, 0, options);
    let gradient = if policy.capability.has_gradient() {
        Derivative::Analytic
    } else {
        Derivative::Unavailable
    };
    // The analytic outer Hessian is routed to ARC whenever the realized family
    // exposes second-order calculus. Matrix-free Hessian support is a
    // representation capability used by the evaluator; it must not be hidden
    // from the outer optimizer by a cost-based first-order policy.
    let hessian = if options.use_outer_hessian
        && include_exact_newton_logdet_h(family, options)
        && policy.capability.has_hessian()
    {
        DeclaredHessianForm::Either
    } else {
        DeclaredHessianForm::Unavailable
    };

    (gradient, hessian)
}


fn include_exact_newton_logdet_s<F: CustomFamily + ?Sized>(
    family: &F,
    options: &BlockwiseFitOptions,
) -> bool {
    family.exact_newton_outerobjective() == ExactNewtonOuterObjective::RidgedQuadraticReml
        && options.use_remlobjective
}


fn use_exact_newton_strict_spd<F: CustomFamily + ?Sized>(family: &F) -> bool {
    family.exact_newton_outerobjective() == ExactNewtonOuterObjective::StrictPseudoLaplace
}


fn blockwise_logdet_terms<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &mut [ParameterBlockState],
    block_log_lambdas: &[Array1<f64>],
    options: &BlockwiseFitOptions,
) -> Result<(f64, f64), String> {
    blockwise_logdet_terms_with_workspace(family, specs, states, block_log_lambdas, options, None)
}


fn blockwise_logdet_terms_with_workspace<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &mut [ParameterBlockState],
    block_log_lambdas: &[Array1<f64>],
    options: &BlockwiseFitOptions,
    preferred_workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
) -> Result<(f64, f64), String> {
    let include_logdet_h = include_exact_newton_logdet_h(family, options);
    let include_logdet_s = include_exact_newton_logdet_s(family, options);
    if !include_logdet_h && !include_logdet_s {
        return Ok((0.0, 0.0));
    }
    let strict_spd = use_exact_newton_strict_spd(family);
    refresh_all_block_etas(family, specs, states)?;
    let ranges = block_param_ranges(specs);
    let total = ranges.last().map(|(_, e)| *e).unwrap_or(0);
    // Universal full-span robustness: the outer REML logdet of the
    // penalized Hessian must use the SAME Jeffreys-augmented Hessian
    // `H + S_λ + H_Φ` the inner Newton converged on, or the LAML score and its
    // analytic derivatives describe a different objective. Compute `H_Φ` once
    // over the full-span basis `Z_J` and add it into whichever
    // logdet path runs below. `None` ⇒ no logdet-H contribution (logdet-S only).
    // Cheap matrix-free conditioning pre-check for the OUTER logdet H_Φ. When a
    // matrix-free workspace exposes the Hessian-vector product, bound the joint
    // information's spectrum from a few matvecs (no dense H, no O(p³) eigh): if it
    // certifies well-conditioned the exact gate is certain to return H_Φ = 0, so
    // we skip the whole dense formation and use `None` (no logdet-H Jeffreys
    // contribution), byte-identical to the gated-off path. This keeps the outer
    // LAML logdet consistent with the inner solve (which also gated the term off
    // on the same well-conditioned geometry) while preserving the matrix-free path
    // at outer-eval scale. Returns `false`/unsure ⇒ exact formation below.
    //
    // EXPECTED-INFORMATION GUARD (gam#1020): the matvec certifies the
    // OBSERVED Hessian's conditioning; when the family overrides the Jeffreys
    // information with the expected Fisher information the certificate does
    // not transfer (observed grows on saturated rows where expected decays),
    // so the pre-check is bypassed and the exact gate always runs.
    let outer_precheck_eligible = include_logdet_h
        && family.joint_jeffreys_information_matches_observed_hessian()
        && total >= crate::estimate::reml::jeffreys_subspace::CHEAP_CONDITIONING_PRECHECK_MIN_DIM;
    let outer_jeffreys_precheck_skips = match preferred_workspace.as_ref() {
        Some(ws) if outer_precheck_eligible && ws.hessian_matvec_available() => {
            let hv = |v: &Array1<f64>| -> Result<Array1<f64>, String> {
                match ws.hessian_matvec(v)? {
                    Some(out) if out.len() == total => Ok(out),
                    // Workspace declined this matvec ⇒ cannot certify ⇒ do not skip.
                    // Return a non-finite sentinel so the cheap estimator bails to
                    // the conservative `false` (never skip on an unresolved apply).
                    _ => Ok(Array1::from_elem(total, f64::NAN)),
                }
            };
            crate::estimate::reml::jeffreys_subspace::jeffreys_term_skippable_via_matvec(hv, total)
                .unwrap_or(false)
        }
        _ => false,
    };
    let logdet_jeffreys_hphi: Option<Array2<f64>> = if include_logdet_h
        && !outer_jeffreys_precheck_skips
        && !options.seed_screening
        && family.joint_jeffreys_term_required()
    {
        // Skipped during seed screening: this per-axis Jeffreys curvature
        // (O(p · per-axis-Hdot)) augments the outer LAML logdet `½ log|H+Sλ+H_Φ|`,
        // a refinement the screening SCORE does not need. Screening ranks seeds by
        // the un-augmented `½ log|H+Sλ|` plus the value-only Firth penalty already
        // in `penalty_value`; the load-bearing H_Φ is restored for the real fit
        // (gam#729/#808).
        match build_joint_jeffreys_subspace(specs, &ranges)? {
            Some(z_joint) => {
                custom_family_joint_jeffreys_term(family, states, specs, &ranges, &z_joint)?
                    .map(|(_phi, _grad, hphi)| hphi)
            }
            None => None,
        }
    } else {
        None
    };
    let compute_block_logdet_term = |b: usize| -> Result<(Array2<f64>, f64), String> {
        let spec = &specs[b];
        let (start, end) = ranges[b];
        let p = end - start;
        let lambdas = block_log_lambdas[b].mapv(f64::exp);
        let mut s_lambda = Array2::<f64>::zeros((p, p));
        for (k, s) in spec.penalties.iter().enumerate() {
            s.add_scaled_to(lambdas[k], &mut s_lambda);
        }
        let block_logdet = if include_logdet_s {
            // Pseudo-logdet of S_λ on the positive eigenspace.
            //
            // CONSISTENCY REQUIREMENT (gam#752/#748/#808 class): this VALUE is
            // the `log|S_λ|₊` term of the outer REML/LAML objective, and its
            // ρ-gradient is supplied separately by
            // `compute_block_penalty_logdet_derivs`, which differentiates the
            // canonical `PenaltyPseudologdet`. If the value used a *different*
            // positive/null eigenspace split (e.g. structural-count `skip(m0)`
            // by COUNT, or a ridge-blind `positive_eigenvalue_threshold`) than
            // the gradient's by-magnitude `> ridge + noise_band` rule, the
            // outer optimizer would see an objective and a gradient that
            // describe different functions near the ridge boundary (a barely-
            // active mode `λ_k σ_k → 0` whose ridged eigenvalue dips below
            // `ridge + noise_band` is kept by the count rule but dropped by the
            // magnitude rule). To guarantee value↔gradient agree by
            // construction, compute the value from the SAME canonical
            // `PenaltyPseudologdet` the gradient differentiates, with the same
            // dense penalty components, the same λ, and the same ridge.
            let ridge = if options.ridge_policy.include_penalty_logdet {
                effective_solverridge(options.ridge_floor)
            } else {
                0.0
            };
            let penalties_dense: Vec<Array2<f64>> =
                spec.penalties.iter().map(|pen| pen.to_dense()).collect();
            let lambdas_vec: Vec<f64> = lambdas.to_vec();
            match crate::estimate::reml::penalty_logdet::PenaltyPseudologdet::from_components(
                &penalties_dense,
                &lambdas_vec,
                ridge,
            ) {
                Ok(pld) => pld.value(),
                Err(eigh_err_msg) => {
                    // `from_components` only fails when the single internal
                    // eigendecomposition fails, which for PSD penalties is
                    // purely numerical. Fall back to Cholesky on the ridged
                    // matrix (which should be SPD). The Cholesky logdet
                    // includes null-space contributions (~m₀ × ln(ridge)),
                    // a smooth bias that does not corrupt the REML gradient.
                    let mut s_for_logdet = s_lambda.clone();
                    if ridge > 0.0 {
                        for i in 0..p {
                            s_for_logdet[[i, i]] += ridge;
                        }
                    }
                    penalty_logdet_cholesky_fallback(&s_for_logdet, ridge, b, p, &eigh_err_msg)?
                }
            }
        } else {
            0.0
        };
        Ok((s_lambda, block_logdet))
    };

    // Per-block penalty assembly and eigendecomposition are independent.
    // Use rayon only from non-rayon callers so inner operator/eigendecomp work
    // does not nest under an existing worker. Collecting an indexed range into
    // a Vec preserves block order; totals are accumulated sequentially below
    // to keep floating-point summation deterministic.
    let block_terms: Vec<Result<(Array2<f64>, f64), String>> =
        if specs.len() > 1 && rayon::current_thread_index().is_none() {
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            (0..specs.len())
                .into_par_iter()
                .map(compute_block_logdet_term)
                .collect()
        } else {
            (0..specs.len()).map(compute_block_logdet_term).collect()
        };
    let mut s_lambdas = Vec::with_capacity(block_terms.len());
    let mut penalty_logdet_s_total = 0.0;
    for block_term in block_terms {
        let (s_lambda, block_logdet) = block_term?;
        s_lambdas.push(s_lambda);
        penalty_logdet_s_total += block_logdet;
    }
    if !include_logdet_h {
        return Ok((0.0, penalty_logdet_s_total));
    }
    // Try the shared scale-aware exact curvature path first.
    if let Some(curvature) = family.exact_newton_outer_curvature(states)? {
        let mut h_joint = symmetrized_square_matrix(
            curvature.hessian,
            total,
            "joint exact-newton Hessian validation in logdet terms (rescaled)",
        )?;
        for (b, s_lambda) in s_lambdas.iter().enumerate() {
            let (start, end) = ranges[b];
            h_joint
                .slice_mut(ndarray::s![start..end, start..end])
                .scaled_add(curvature.rho_curvature_scale, s_lambda);
        }
        if let Some(hphi) = logdet_jeffreys_hphi.as_ref() {
            h_joint.scaled_add(curvature.rho_curvature_scale, hphi);
        }
        let logdet_h_scaled = if strict_spd {
            strict_exact_pseudo_logdet(&h_joint, joint_observation_count(states))?
        } else {
            stable_logdet_with_ridge_policy(
                &h_joint,
                options.ridge_floor * curvature.rho_curvature_scale,
                options.ridge_policy,
            )?
        };
        let logdet_h_total = logdet_h_scaled + curvature.hessian_logdet_correction;
        return Ok((logdet_h_total, penalty_logdet_s_total));
    }
    let exact_joint_source = if let Some(workspace) = preferred_workspace.as_ref() {
        exact_newton_joint_hessian_source_from_workspace(
            workspace,
            total,
            MaterializationIntent::LogdetFactorization,
            "joint exact-newton operator mismatch in logdet terms",
        )?
    } else if !strict_spd && use_joint_matrix_free_path(total, joint_observation_count(states)) {
        family
            .exact_newton_joint_hessian_workspace_with_options(states, specs, options)?
            .as_ref()
            .map(|workspace| {
                exact_newton_joint_hessian_source_from_workspace(
                    workspace,
                    total,
                    MaterializationIntent::LogdetFactorization,
                    "joint exact-newton operator mismatch in logdet terms",
                )
            })
            .transpose()?
            .flatten()
    } else {
        None
    };
    if let Some(source) = exact_joint_source {
        // Exact determinant of H + S_λ for operator-backed coefficient Hessians.
        //
        // The REML gradient and Hessian use analytic trace identities such as
        // ∂ log|A(θ)| = tr(A⁻¹ A_θ).  Mixing an approximate determinant with
        // exact traces violates that identity and gives ARC a Hessian for a
        // different objective.  Materializing the coefficient Hessian by
        // canonical-basis HVPs keeps the objective/derivative pair exact.  At
        // large-scale CTN scale `total` is a few hundred, so this is sub-MiB; the
        // materializer below refuses oversized systems before allocation.
        let mut h_joint = materialize_joint_hessian_source(
            &source,
            total,
            "joint exact-newton operator dense logdet materialization",
        )?;
        for (b, s_lambda) in s_lambdas.iter().enumerate() {
            let (start, end) = ranges[b];
            h_joint
                .slice_mut(ndarray::s![start..end, start..end])
                .scaled_add(1.0, s_lambda);
        }
        if let Some(hphi) = logdet_jeffreys_hphi.as_ref() {
            h_joint.scaled_add(1.0, hphi);
        }
        let logdet_h_total = if strict_spd {
            strict_exact_pseudo_logdet(&h_joint, joint_observation_count(states))?
        } else {
            stable_logdet_with_ridge_policy(&h_joint, options.ridge_floor, options.ridge_policy)?
        };
        return Ok((logdet_h_total, penalty_logdet_s_total));
    }
    // Fallback: try the non-rescaled symmetrized path (for families that
    // don't implement exact_newton_outer_curvature but do provide
    // a plain joint Hessian).
    if let Some(mut h_joint) = exact_newton_joint_hessian_symmetrized(
        family,
        states,
        specs,
        total,
        "joint exact-newton Hessian validation in logdet terms",
    )? {
        for (b, s_lambda) in s_lambdas.iter().enumerate() {
            let (start, end) = ranges[b];
            h_joint
                .slice_mut(ndarray::s![start..end, start..end])
                .scaled_add(1.0, s_lambda);
        }
        if let Some(hphi) = logdet_jeffreys_hphi.as_ref() {
            h_joint.scaled_add(1.0, hphi);
        }
        let logdet_h_total = if strict_spd {
            strict_exact_pseudo_logdet(&h_joint, joint_observation_count(states))?
        } else {
            stable_logdet_with_ridge_policy(&h_joint, options.ridge_floor, options.ridge_policy)?
        };
        return Ok((logdet_h_total, penalty_logdet_s_total));
    }

    let eval = family.evaluate(states)?;
    if eval.blockworking_sets.len() != specs.len() {
        return Err(format!(
            "family returned {} block working sets, expected {}",
            eval.blockworking_sets.len(),
            specs.len()
        ));
    }

    let mut logdet_h_total = 0.0;
    let logdet_s_total = penalty_logdet_s_total;
    for b in 0..specs.len() {
        let spec = &specs[b];
        let work = &eval.blockworking_sets[b];
        let p = spec.design.ncols();
        let xtwx = match work {
            BlockWorkingSet::Diagonal {
                working_response: _,
                working_weights,
            } => with_block_geometry(family, states, spec, b, |x_dyn, _| {
                let w = floor_positiveworking_weights(working_weights, options.minweight);
                let (xtwx, _) = weighted_normal_equations(x_dyn, &w, None)?;
                Ok(xtwx)
            })?,
            BlockWorkingSet::ExactNewton {
                gradient: _,
                hessian,
            } => {
                if hessian.nrows() != p || hessian.ncols() != p {
                    return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                        "block {b} exact-newton Hessian shape mismatch: got {}x{}, expected {}x{}",
                        hessian.nrows(),
                        hessian.ncols(),
                        p,
                        p
                    ) }.into());
                }
                hessian.to_dense()
            }
        };

        let s_lambda = &s_lambdas[b];

        let mut h = xtwx;
        h += s_lambda;
        logdet_h_total += if strict_spd {
            strict_exact_pseudo_logdet(&h, joint_observation_count(states))?
        } else {
            stable_logdet_with_ridge_policy(&h, options.ridge_floor, options.ridge_policy)?
        };
    }
    Ok((logdet_h_total, logdet_s_total))
}


/// Snapshot of a single block's eta for line-search rollback.
///
/// Created from a specific block's state; can only restore to or update
/// that same block.  There is no shared buffer across blocks, so
/// cross-block length confusion is structurally impossible.
struct BlockEtaCheckpoint {
    saved: Array1<f64>,
}


impl BlockEtaCheckpoint {
    /// Capture the current eta of `state`.
    fn capture(state: &ParameterBlockState) -> Self {
        Self {
            saved: state.eta.clone(),
        }
    }

    /// Capture into a pre-allocated buffer, returning the filled checkpoint.
    /// The buffer is taken (O(1) move) and filled with eta's data (O(n) copy).
    fn capture_reuse(state: &ParameterBlockState, buf: &mut Array1<f64>) -> Self {
        if buf.len() == state.eta.len() {
            buf.assign(&state.eta);
            Self {
                saved: std::mem::take(buf),
            }
        } else {
            Self::capture(state)
        }
    }

    /// Return the internal buffer for recycling.
    fn into_buffer(self) -> Array1<f64> {
        self.saved
    }

    /// Restore: `state.eta = saved`.
    fn restore_eta(&self, state: &mut ParameterBlockState) {
        state.eta.assign(&self.saved);
    }

    /// Incremental update: `state.eta = saved + alpha * direction`.
    fn restore_eta_with_step(
        &self,
        state: &mut ParameterBlockState,
        alpha: f64,
        direction: &Array1<f64>,
    ) {
        // In-place: eta = eta_backup + alpha * xd (zero allocations).
        state.eta.assign(&self.saved);
        state.eta.scaled_add(alpha, direction);
    }
}


/// Classification of which branch of the trust-region radius policy
/// fired on a single update — surfaced in cycle logs so it is possible
/// to tell at a glance whether the inner solver is being throttled by
/// the TR (e.g., `RejectFloor`/`ShrinkOnRejection`) or, conversely,
/// whether the step is sitting well inside the region (`HoldInside`)
/// so the slow convergence is NOT a TR-policy issue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum JointTrustRegionDecision {
    /// `rho > 0.75` AND `step_norm >= 0.99 * old_radius` — model is good
    /// AND the step is at the TR boundary, so doubling reflects a real
    /// constraint that was just lifted.
    GrowAtBoundary,
    /// `rho > 0.75` but the step is well inside the region; radius held
    /// because no evidence the TR was constraining the step.  When the
    /// inner is converging linearly and this branch fires every cycle,
    /// the TR is NOT the bottleneck — Newton itself is finding short
    /// steps for a reason unrelated to the trust radius.
    HoldInside,
    /// `0.25 <= rho <= 0.75` (moderate model fidelity) — radius held.
    HoldModerate,
    /// `rho < 0.25` but step accepted (positive descent above noise).
    /// Radius shrunk to a quarter to be more conservative next cycle.
    ShrinkOnMarginalAccept,
    /// Step rejected — radius shrunk and capped at half the proposed
    /// step norm so a re-proposal is constrained inside the rejected
    /// region.
    ShrinkOnRejection,
    /// Radius was already at the floor before this update.  Persistent
    /// `RejectFloor` is the unambiguous signal of a degenerate ρ region.
    RejectFloor,
}


impl JointTrustRegionDecision {
    fn label(&self) -> &'static str {
        match self {
            Self::GrowAtBoundary => "grow_at_boundary",
            Self::HoldInside => "hold_inside",
            Self::HoldModerate => "hold_moderate",
            Self::ShrinkOnMarginalAccept => "shrink_marginal_accept",
            Self::ShrinkOnRejection => "shrink_reject",
            Self::RejectFloor => "reject_floor",
        }
    }
}


#[derive(Clone, Copy, Debug)]
struct JointTrustRegionUpdate {
    rho: f64,
    radius: f64,
    accepted: bool,
    decision: JointTrustRegionDecision,
}


fn update_joint_trust_region_radius(
    old_radius: f64,
    step_norm: f64,
    actual_reduction: f64,
    predicted_reduction: f64,
    objective_scale: f64,
) -> JointTrustRegionUpdate {
    // Floating-point noise floor relative to the objective magnitude.
    // When both the model-predicted and the realized reductions are at
    // this scale, their sign is dominated by round-off in the
    // log-likelihood evaluation rather than by genuine descent or
    // ascent; rejecting on that sign would discard a perfectly
    // converged step. Mirrors the noise-floor handling in
    // src/solver/pirls.rs (see the analogous `noise_floor` block).
    let noise_floor = objective_scale.abs().max(1.0) * 1e-14;
    let predicted_finite_positive =
        predicted_reduction > noise_floor && predicted_reduction.is_finite();
    let rho = if actual_reduction.abs() <= noise_floor {
        // The realized objective change is at the floating-point round-off floor:
        // the step neither helped nor hurt beyond noise, so it is a numerically
        // neutral (converged) step. Treat it as `rho = 1` REGARDLESS of whether
        // `predicted_reduction` happens to sit just above the floor. The previous
        // form only took this branch when `!predicted_finite_positive`; when a
        // tiny-but-valid Newton step near a flat-objective optimum produced
        // `predicted_reduction` marginally above `noise_floor` while
        // `actual_reduction` was within it, it divided two round-off-level
        // quantities and got a spurious negative `rho`, rejecting the step and
        // ratcheting the trust radius to the floor — pinning the solve far below
        // the (small, valid) Newton step it needed (gam#797 last-mile; same
        // pinning family observed on clustered bernoulli). Keying neutrality on the
        // *actual* reduction is the correct round-off guard.
        1.0
    } else if predicted_finite_positive {
        actual_reduction / predicted_reduction
    } else {
        f64::NEG_INFINITY
    };
    let accepted = rho.is_finite() && rho > 0.0 && actual_reduction >= -noise_floor;
    let mut radius = old_radius;
    let decision: JointTrustRegionDecision;
    if !accepted {
        radius *= 0.25;
        if step_norm.is_finite() && step_norm > 0.0 {
            radius = radius.min(0.5 * step_norm);
        }
        decision = JointTrustRegionDecision::ShrinkOnRejection;
    } else if rho < 0.25 {
        radius *= 0.25;
        decision = JointTrustRegionDecision::ShrinkOnMarginalAccept;
    } else if rho > 0.75 && step_norm >= 0.99 * old_radius {
        radius *= 2.0;
        decision = JointTrustRegionDecision::GrowAtBoundary;
    } else if rho > 0.75 {
        decision = JointTrustRegionDecision::HoldInside;
    } else {
        decision = JointTrustRegionDecision::HoldModerate;
    }
    if !radius.is_finite() || radius <= 0.0 {
        radius = 1.0e-12;
    }
    let clamped_radius = radius.clamp(1.0e-12, 1.0e6);
    // Promote to RejectFloor if we landed at the absolute floor.  The
    // base classification is preserved up to this final clamp; the
    // floor classification is just a stronger label that captures the
    // "no descent direction exists at this radius" signal.
    let final_decision = if clamped_radius <= 1.0e-12 + f64::EPSILON
        && matches!(
            decision,
            JointTrustRegionDecision::ShrinkOnRejection
                | JointTrustRegionDecision::ShrinkOnMarginalAccept
        ) {
        JointTrustRegionDecision::RejectFloor
    } else {
        decision
    };
    JointTrustRegionUpdate {
        rho,
        radius: clamped_radius,
        accepted,
        decision: final_decision,
    }
}


fn joint_objective_roundoff_slack(old_objective: f64, trial_objective: f64) -> f64 {
    (64.0 * f64::EPSILON * (1.0 + old_objective.abs() + trial_objective.abs())).max(1.0e-10)
}


// True iff the line search detected a noise-level realized reduction (i.e.
// the trial step neither helped nor hurt the objective beyond round-off)
// AND the local quadratic model agrees that no further descent is available
// within tolerance. `actual_reduction <= 0` is kept (not made sign-symmetric)
// because at rank-deficient optima (σ_min(H) ≲ ε_machine) the outer-gradient
// FD identity requires β trajectories to be CONSISTENT across λ probes —
// accepting positive-noise-level reductions exits the loop one attempt
// earlier than the negative case and decorrelates the null-space drift
// between consecutive REML evaluations. Concretely:
// `outer_lamlgradient_matches_finite_differencewhen_joint_exact_path_is_active`
// at HardPseudo σ_min ~ 1e-10 fails when symmetric. The asymmetric guard
// preserves the spin avoidance for the common (negative-noise) case at
// large scale while leaving the rank-deficient FD identity intact.
fn joint_objective_floor_reached(
    old_objective: f64,
    trial_objective: f64,
    actual_reduction: f64,
    predicted_reduction: f64,
    objective_tol: f64,
) -> bool {
    trial_objective.is_finite()
        && actual_reduction <= 0.0
        && actual_reduction.abs() <= joint_objective_roundoff_slack(old_objective, trial_objective)
        && predicted_reduction.is_finite()
        && predicted_reduction
            <= objective_tol.max(joint_objective_roundoff_slack(
                old_objective,
                trial_objective,
            ))
}


/// True iff the joint-Newton proposal is already at the step-tolerance floor —
/// the unclamped Newton step's inf-norm is within `STEP_FLOOR_CERT_FACTOR ×
/// step_tol` (the same round-off band the constrained-stationary certificate
/// uses for "a hair above tol"). At the floor the iterate is doing KKT polishing
/// on a flat objective, so a `predicted_reduction = rhs·δ − ½δᵀHδ ≤ 0` is the
/// SIGN of two near-equal O(step²) quantities (round-off), NOT a model-invalid
/// descent direction; the preconditioned-descent substitution must be suppressed
/// there or it replaces the tiny polishing step with an objective-descent step
/// that catapults the KKT residual off the near-converged iterate (gam#787 binary
/// matern centers=12: residual 1.7e-4 → 4.7e-1, never recovers).
fn joint_proposal_at_step_floor(proposal_step_inf: f64, step_tol: f64) -> bool {
    const STEP_FLOOR_CERT_FACTOR: f64 = 4.0;
    proposal_step_inf.is_finite()
        && step_tol.is_finite()
        && proposal_step_inf <= STEP_FLOOR_CERT_FACTOR * step_tol
}


fn joint_trust_region_metric_step_norm(delta: &Array1<f64>, metric_diag: &Array1<f64>) -> f64 {
    assert_eq!(delta.len(), metric_diag.len());
    joint_trust_region_metric_step_norm_view(delta.view(), metric_diag.view())
}


fn joint_trust_region_metric_step_norm_view(
    delta: ArrayView1<f64>,
    metric_diag: ArrayView1<f64>,
) -> f64 {
    assert_eq!(delta.len(), metric_diag.len());
    delta
        .iter()
        .zip(metric_diag.iter())
        .map(|(step, weight)| step * step * positive_joint_diagonal_entry(*weight))
        .sum::<f64>()
        .sqrt()
}


fn joint_trust_region_block_metric_norms(
    delta: &Array1<f64>,
    ranges: &[(usize, usize)],
    metric_diag: &Array1<f64>,
) -> Vec<f64> {
    assert_eq!(delta.len(), metric_diag.len());
    ranges
        .iter()
        .map(|(start, end)| {
            joint_trust_region_metric_step_norm_view(
                delta.slice(s![*start..*end]),
                metric_diag.slice(s![*start..*end]),
            )
        })
        .collect()
}


fn truncate_joint_step_to_block_metric_radii(
    delta: &mut Array1<f64>,
    ranges: &[(usize, usize)],
    metric_diag: &Array1<f64>,
    block_radii: &[f64],
) -> Vec<f64> {
    assert_eq!(ranges.len(), block_radii.len());
    assert_eq!(delta.len(), metric_diag.len());
    let mut norms = Vec::with_capacity(ranges.len());
    for (block_idx, (start, end)) in ranges.iter().copied().enumerate() {
        let metric_view = metric_diag.slice(s![start..end]);
        let mut block = delta.slice_mut(s![start..end]);
        let norm = joint_trust_region_metric_step_norm_view(block.view(), metric_view);
        let radius = block_radii[block_idx];
        if norm.is_finite() && norm > radius && radius > 0.0 {
            block.mapv_inplace(|v| v * (radius / norm));
            norms.push(radius);
        } else {
            norms.push(norm);
        }
    }
    norms
}


fn joint_block_step_hit_trust_boundary(step_norm: f64, radius: f64) -> bool {
    step_norm.is_finite() && radius > 0.0 && step_norm >= 0.99 * radius
}


/// Per-block dogleg step (Powell, blending the Cauchy and Newton points within
/// the block's M-metric trust radius). This is the principled globalization for
/// the coupled location-scale inner Newton (gam#826/#808): box-truncating the
/// Newton step alone freezes progress when the spectral solve is degenerate at
/// the oversmoothed seed — the high-curvature `log_sigma` block has
/// `λ ~ exp(2·ρ_bound)` so its Newton component is `O(g/λ) ≈ 5e-21`, the
/// mean/trend blocks get isotropically shrunk to the radius, and the residual
/// stalls while β barely moves. The dogleg always includes the Cauchy leg
/// (the model-minimizing steepest-descent step in the block metric), so the
/// realized decrease is at least the Cauchy decrease whenever the block
/// gradient is nonzero — progress is guaranteed even when the Newton step is
/// numerically frozen. Inside the radius the dogleg returns the exact Newton
/// step, so the converged β, the KKT certificate, and the well-conditioned /
/// #729 endgame are byte-identical to the undamped solve.
///
/// Inputs per block `b`:
///   * `newton[start..end]`  — Newton (spectral) step block `δ_N`.
///   * `cauchy[start..end]`  — the FULL (unconstrained) Cauchy block
///     `δ_C = τ·p_sd`, where `p_sd = M⁻¹·rhs` is the M-metric steepest-descent
///     direction of the model and `τ` minimizes the model along it; precomputed
///     once per cycle by `joint_cauchy_step` (the curvature `p_sd·H·p_sd` needs
///     a coupled Hessian-vector product, so it must be hoisted out of the
///     radius-shrink loop).
///   * `radius`              — the block's current M-metric trust radius.
///
/// Returns the block step norms in the M-metric (same contract as
/// `truncate_joint_step_to_block_metric_radii`) and overwrites `out` with the
/// dogleg blend per block.
fn joint_dogleg_step_to_block_metric_radii(
    newton: &Array1<f64>,
    cauchy: &Array1<f64>,
    ranges: &[(usize, usize)],
    metric_diag: &Array1<f64>,
    block_radii: &[f64],
    out: &mut Array1<f64>,
) -> Vec<f64> {
    assert_eq!(ranges.len(), block_radii.len());
    assert_eq!(newton.len(), metric_diag.len());
    assert_eq!(cauchy.len(), metric_diag.len());
    assert_eq!(out.len(), metric_diag.len());
    let mut norms = Vec::with_capacity(ranges.len());
    for (block_idx, (start, end)) in ranges.iter().copied().enumerate() {
        let metric_view = metric_diag.slice(s![start..end]);
        let newton_b = newton.slice(s![start..end]);
        let cauchy_b = cauchy.slice(s![start..end]);
        let radius = block_radii[block_idx];
        let newton_norm = joint_trust_region_metric_step_norm_view(newton_b, metric_view);
        let cauchy_norm = joint_trust_region_metric_step_norm_view(cauchy_b, metric_view);
        let mut out_b = out.slice_mut(s![start..end]);

        // Degenerate radius (non-finite or non-positive): nothing moves.
        if !radius.is_finite() || radius <= 0.0 {
            out_b.fill(0.0);
            norms.push(0.0);
            continue;
        }

        // Newton step (or a non-finite Cauchy fallback) inside the radius: take
        // the exact Newton step. This is the only branch a well-conditioned /
        // converging fit ever reaches near the optimum, so the endgame numerics
        // are unchanged.
        if newton_norm.is_finite() && newton_norm <= radius {
            out_b.assign(&newton_b);
            norms.push(newton_norm);
            continue;
        }

        // Cauchy leg longer than the radius (or Newton/Cauchy not comparable):
        // scale the Cauchy step to the boundary. When the Cauchy step itself is
        // unusable, fall back to scaling the Newton step (pre-dogleg behavior).
        if !(cauchy_norm.is_finite() && cauchy_norm > 0.0) {
            let scale = if newton_norm.is_finite() && newton_norm > 0.0 {
                radius / newton_norm
            } else {
                0.0
            };
            out_b.assign(&newton_b);
            out_b.mapv_inplace(|v| v * scale);
            norms.push(if scale > 0.0 { radius } else { 0.0 });
            continue;
        }
        if cauchy_norm >= radius {
            let scale = radius / cauchy_norm;
            out_b.assign(&cauchy_b);
            out_b.mapv_inplace(|v| v * scale);
            norms.push(radius);
            continue;
        }

        // Dogleg blend: δ(θ) = δ_C + θ·(δ_N − δ_C), θ ∈ [0,1], pick θ so
        // ‖δ(θ)‖_M = radius. Solve the quadratic ‖δ_C + θ·d‖²_M = radius² with
        // d = δ_N − δ_C, a = ‖d‖²_M, b = 2·⟨δ_C, d⟩_M, c = ‖δ_C‖²_M − radius².
        let mut a = 0.0_f64;
        let mut b = 0.0_f64;
        for ((cb, nb), w) in cauchy_b.iter().zip(newton_b.iter()).zip(metric_view.iter()) {
            let m = positive_joint_diagonal_entry(*w);
            let d = nb - cb;
            a += m * d * d;
            b += 2.0 * m * cb * d;
        }
        let c = cauchy_norm * cauchy_norm - radius * radius;
        // a > 0 because δ_N ≠ δ_C here (Newton outside, Cauchy inside the
        // radius). Largest root in [0,1] keeps the step on the dogleg path.
        let disc = (b * b - 4.0 * a * c).max(0.0);
        let theta = if a > 0.0 {
            ((-b + disc.sqrt()) / (2.0 * a)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        for ((o, cb), nb) in out_b.iter_mut().zip(cauchy_b.iter()).zip(newton_b.iter()) {
            *o = cb + theta * (nb - cb);
        }
        let norm = joint_trust_region_metric_step_norm_view(out_b.view(), metric_view);
        norms.push(norm);
    }
    norms
}


/// Unconstrained Cauchy point of the joint penalized quadratic model in the
/// block-diagonal M-metric: `δ_C = τ·p_sd` with `p_sd = M⁻¹·rhs` (the M-metric
/// steepest-descent direction of the model `m(δ) = −rhs·δ + ½·δᵀHδ` at δ=0)
/// and `τ = (rhs·p_sd)/(p_sd·H·p_sd)` minimizing the model along `p_sd`. When
/// the curvature `p_sd·H·p_sd ≤ 0` the model is unbounded below along `p_sd`,
/// so `δ_C` is just `p_sd` (the dogleg's boundary scaling then takes it to the
/// radius — a descent step on the indefinite/flat direction). `h_psd` must be
/// `H_pen·p_sd` for the SAME penalized (and Firth-augmented, when armed) Hessian
/// the trust-region model uses, so the dogleg path is consistent with the
/// accept/reject quadratic.
fn joint_cauchy_step(rhs: &Array1<f64>, p_sd: &Array1<f64>, h_psd: &Array1<f64>) -> Array1<f64> {
    let directional = rhs.dot(p_sd);
    if !directional.is_finite() || directional <= 0.0 {
        // `p_sd` is not an ascent direction of −m (no descent on the objective);
        // emit a zero Cauchy step so the dogleg falls back to the Newton leg.
        return Array1::zeros(p_sd.len());
    }
    let curvature = p_sd.dot(h_psd);
    let mut delta = p_sd.clone();
    if curvature.is_finite() && curvature > 0.0 {
        let tau = directional / curvature;
        if tau.is_finite() && tau > 0.0 {
            delta.mapv_inplace(|v| tau * v);
        }
    }
    // Non-positive curvature: leave δ_C = p_sd; the dogleg scales it to the
    // trust boundary (the model decreases without bound along p_sd there).
    delta
}


fn shrink_active_joint_block_trust_radii(
    block_radii: &mut [f64],
    block_step_norms: &[f64],
    factor: f64,
) -> f64 {
    assert_eq!(block_radii.len(), block_step_norms.len());
    // Joint-Newton step-rejection radius shrink. Must guarantee strict
    // monotone decrease of `max(block_radii)` until the floor, otherwise the
    // next trust-region attempt computes a step byte-identical to the rejected
    // one and the inner loop stalls forever (gam joint-Newton fully-rejected
    // cycles, root cause behind the 8-cycle bail at FULLY_REJECTED_STALL_MAX_CYCLES).
    //
    // Two cooperating mechanisms:
    //   * For every block that participates in the shrink, the new radius is
    //     pulled below the rejected step's magnitude (`0.5 · step_norm`),
    //     matching the analogous clamp in `update_joint_trust_region_radius`'s
    //     reject branch. This forces the next step to be strictly smaller
    //     than the current one even when `radius * factor` is still larger
    //     than `step_norm` (which happens whenever the dogleg/truncate path
    //     returned a Newton step shorter than the block's radius).
    //   * Block participation: by default only shrink blocks whose step hit
    //     the per-block trust boundary (the boundary block was the one the
    //     trust radius actually constrained — interior blocks took their
    //     natural Newton step and shrinking their radius is wasted). BUT when
    //     every boundary block already sits at the 1e-12 floor, further
    //     shrinking those blocks is a no-op (they'd just re-clamp to the
    //     floor), so we *must* shrink the interior blocks instead to actually
    //     change the joint step. Without this carve-out the deadlock was:
    //     boundary block pinned at 1e-12, interior block radius held at its
    //     pre-stall value, `max(block_radii)` held by the interior block, the
    //     dogleg/truncate produces an identical joint δ every cycle, every
    //     trust attempt rejects on the same objective check, the cycle burns
    //     to `inner_loop_hard_ceiling` (1200) cycles wasting ~120 s per
    //     outer ρ-evaluation — the Rust CI Test hang and the
    //     `rust_margslope_aniso_duchon16d_*` large-scale 2400 s timeout.
    const RADIUS_FLOOR: f64 = 1.0e-12;
    let any_boundary_block = block_radii
        .iter()
        .zip(block_step_norms)
        .any(|(radius, step_norm)| joint_block_step_hit_trust_boundary(*step_norm, *radius));
    let all_boundary_blocks_at_floor = any_boundary_block
        && block_radii
            .iter()
            .zip(block_step_norms)
            .filter(|(radius, step_norm)| {
                joint_block_step_hit_trust_boundary(**step_norm, **radius)
            })
            .all(|(radius, _)| *radius <= RADIUS_FLOOR * (1.0 + 1.0e-12));
    // Snapshot the joint max BEFORE the shrink loop so the max-holding
    // block(s) — boundary OR interior — always participate. The
    // Moré–Sorensen inner step uses the SCALAR
    // `joint_trust_radius = max(block_radii)` as its trust constraint
    // (`spectrum.trust_region_step(joint_trust_radius)`); if the max-holder
    // is an interior block whose step came in well below its per-block
    // radius, the original boundary-only rule left the joint max held, the
    // MS solve re-computed a byte-identical rejected step, and the inner
    // loop stalled at `inner_loop_hard_ceiling`. Surfaced as the ~2-hour
    // Rust CI test hang where cycles 117..305+ all logged
    // `r=1.562e-2 (held) decision=shrink_reject |δ|=1.562e-2 |δ|∞=1.052e-4`
    // identically: the boundary block's per-block radius collapsed toward
    // the floor without ever changing the scalar joint radius, and the
    // existing `all_boundary_blocks_at_floor` carve-out was unreachable
    // because the boundary block kept getting deeper than the interior
    // max-holder. Forcing the max-holder to participate makes the scalar
    // `max(block_radii)` strictly decrease on every rejected attempt until
    // the floor, after which the fully-rejected stall guard
    // (`FULLY_REJECTED_STALL_MAX_CYCLES`) takes over and bails the cycle
    // cleanly.
    let max_radius_before = block_radii.iter().copied().fold(0.0_f64, f64::max);
    for (radius, step_norm) in block_radii.iter_mut().zip(block_step_norms) {
        let at_boundary = joint_block_step_hit_trust_boundary(*step_norm, *radius);
        let holds_max = max_radius_before > 0.0
            && max_radius_before.is_finite()
            && *radius >= max_radius_before * (1.0 - 1.0e-12);
        let participates = if all_boundary_blocks_at_floor {
            // Boundary-at-floor stall: the boundary blocks cannot shrink any
            // further, so participate every block (including interior ones)
            // so the joint step magnitude actually changes.
            true
        } else if any_boundary_block {
            at_boundary || holds_max
        } else {
            true
        };
        if participates {
            let mut new_radius = *radius * factor;
            if step_norm.is_finite() && *step_norm > 0.0 {
                new_radius = new_radius.min(0.5 * *step_norm);
            }
            *radius = new_radius.clamp(RADIUS_FLOOR, 1.0e6);
        }
    }
    block_radii.iter().copied().fold(0.0_f64, f64::max)
}


fn apply_joint_feasibility_limit<F: CustomFamily + ?Sized>(
    family: &F,
    states: &[ParameterBlockState],
    ranges: &[(usize, usize)],
    trial_delta: &mut Array1<f64>,
) -> Result<bool, String> {
    // Collect each block's feasibility α and apply the *minimum* to the
    // JOINT trial step, not to each block in isolation.
    //
    // The joint Newton direction δ̂ = H⁻¹(−g) is the unique descent direction
    // for the local quadratic model up to a positive scalar; any α·δ̂ with
    // α ∈ (0, 1] is still a descent direction on the joint objective.
    // Scaling ONLY one block by α produces (α·δ̂_A, δ̂_B, …), which is
    // neither δ̂ nor α·δ̂ and is not, in general, a descent direction on
    // the joint quadratic.
    //
    // Production survival_marginal_slope failure mode at large scale:
    // the time block returned α ≈ 1e-4 (monotonicity guard); per-block
    // scaling crushed δ_time to ~2.3e-4 while logslope kept its full
    // unconstrained Newton step. The joint step was no longer a Newton
    // direction; the time-block gradient stayed at ‖g_time‖ ≈ 5.6e8 for
    // the next 15+ cycles, triggering the linearized-rate stall
    // early-exit on every outer seed.
    //
    // Scaling the joint step by min α preserves Newton direction; the
    // trust-region/line-search already chooses the appropriate step size
    // within direction, this barrier check just enforces feasibility on
    // top of that direction.
    let mut joint_alpha = 1.0_f64;
    let mut limiting_block: Option<usize> = None;
    for (block_idx, (start, end)) in ranges.iter().copied().enumerate() {
        let block_delta = trial_delta.slice(s![start..end]).to_owned();
        if let Some(alpha_max) = family.max_feasible_step_size(states, block_idx, &block_delta)? {
            if !alpha_max.is_finite() || alpha_max <= 0.0 {
                return Err(format!(
                    "joint Newton block {block_idx} has no positive feasible step"
                ));
            }
            if alpha_max < joint_alpha {
                joint_alpha = alpha_max;
                limiting_block = Some(block_idx);
            }
        }
    }
    if joint_alpha < 1.0 {
        trial_delta.mapv_inplace(|v| joint_alpha * v);
        log::debug!(
            "[PIRLS/joint-Newton] feasibility scaled joint step by α={:.3e} (block {:?} binding)",
            joint_alpha,
            limiting_block,
        );
        Ok(true)
    } else {
        Ok(false)
    }
}


fn joint_inner_kkt_converged(residual: f64, residual_tol: f64) -> bool {
    residual.is_finite() && residual_tol.is_finite() && residual <= residual_tol
}


/// Per-iterate diagnostic snapshot assembled when the joint Newton inner solve
/// refuses to certify constrained-stationarity. The report breaks the failure
/// down by block (so the offending smooth can be named), records the H_pen
/// eigenvalue spectrum (so rank-deficiency in the penalized Hessian is
/// detectable from logs), and classifies the refusal so downstream tooling
/// can act without re-deriving the cert math.
#[derive(Clone, Debug)]
struct KktRefusalReport {
    block_names: Vec<String>,
    block_widths: Vec<usize>,
    block_beta_inf: Vec<f64>,
    block_grad_inf: Vec<f64>,
    block_penalty_grad_inf: Vec<f64>,
    block_residual_inf: Vec<f64>,
    block_carrying_residual: Option<usize>,

    hpen_eigenvalues_sorted_desc: Vec<f64>,
    hpen_min_abs_eigenvalue: f64,
    hpen_max_abs_eigenvalue: f64,
    hpen_condition_number: f64,
    hpen_nullity_at_rank_tol: usize,
    hpen_rank_tol: f64,
    hpen_null_gradient_inf: f64,
    hpen_null_vector_block_inf: Vec<f64>,
    hpen_null_vector_carrying_block: Option<usize>,

    active_set_rows_total: usize,
    accepted_step_inf: f64,
    proposal_step_inf: f64,
    trust_radius: f64,
    cycle: usize,

    residual_tol: f64,
    obj_tol: f64,
    step_tol: f64,

    linearized_rel: f64,
    scalar_model_relerr: f64,
    objective_change: f64,
    projected_residual_inf: f64,

    diagnosis: KktRefusalDiagnosis,
}


/// Three-way classification of why the cert refused, computed from the
/// H_pen spectrum and the projected residual at the refusing iterate.
/// `RankDeficientHPen` is the regression canary the nullspace lead's
/// smooth-construction rework is intended to eliminate; keep this variant
/// intact when extending — it doubles as the user-facing signal for
/// "an unconstrained polynomial null space slipped past absorption."
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum KktRefusalDiagnosis {
    RankDeficientHPen,
    PhantomMultiplierWithWellConditionedH,
    ActiveSetIncomplete,
    /// Cross-block identifiability aliasing surfaced mid-inner-solve
    /// (e.g., a binding active set materialised a 2-way alias that
    /// the pre-fit audit could not see at the cold design). The fix
    /// is structural — drop or reparameterise the aliased block;
    /// rho-anneal will not recover.
    AliasingDetectedAtFit,
}


impl KktRefusalDiagnosis {
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            KktRefusalDiagnosis::RankDeficientHPen => "rank_deficient_H_pen",
            KktRefusalDiagnosis::PhantomMultiplierWithWellConditionedH => {
                "phantom_multiplier_with_well_conditioned_H"
            }
            KktRefusalDiagnosis::ActiveSetIncomplete => "active_set_incomplete",
            KktRefusalDiagnosis::AliasingDetectedAtFit => "aliasing_detected_at_fit",
        }
    }

    /// Parse the textual `diagnosis:` field embedded in the structured
    /// bubbled error string. Returns `None` when no recognised label is
    /// present (legacy / non-cert-refusal error strings).
    pub(crate) fn parse_from_error(message: &str) -> Option<Self> {
        let marker = "diagnosis: ";
        let start = message.rfind(marker)? + marker.len();
        let tail = &message[start..];
        let end = tail
            .find(|c: char| c == ';' || c == '\n' || c == ' ')
            .unwrap_or(tail.len());
        match &tail[..end] {
            "rank_deficient_H_pen" => Some(KktRefusalDiagnosis::RankDeficientHPen),
            "phantom_multiplier_with_well_conditioned_H" => {
                Some(KktRefusalDiagnosis::PhantomMultiplierWithWellConditionedH)
            }
            "active_set_incomplete" => Some(KktRefusalDiagnosis::ActiveSetIncomplete),
            "aliasing_detected_at_fit" => Some(KktRefusalDiagnosis::AliasingDetectedAtFit),
            _ => None,
        }
    }

    fn guidance(self) -> &'static str {
        match self {
            KktRefusalDiagnosis::RankDeficientHPen => {
                "check whether the named block has a structural or numerical null direction \
                 not identified by the likelihood/penalty combination; for Duchon-style \
                 smooths this may be a polynomial null space, while marginal-slope fits can \
                 also expose callback-owned weak directions"
            }
            KktRefusalDiagnosis::PhantomMultiplierWithWellConditionedH => {
                "check whether the named block has a near-separated or weakly identified \
                 direction despite a well-conditioned penalized Hessian; in marginal-slope \
                 fits this often indicates marginal/logslope coupling rather than a \
                 Matérn/Duchon polynomial-nullspace failure"
            }
            KktRefusalDiagnosis::ActiveSetIncomplete => {
                "check whether the named block's linear constraints need an additional \
                 active row or a tighter constrained re-solve; this is an active-set \
                 certification failure, not a polynomial-nullspace diagnosis"
            }
            KktRefusalDiagnosis::AliasingDetectedAtFit => {
                "check whether the named block aliases another block after runtime \
                 constraints or callbacks materialize; drop or reparameterize the aliased \
                 direction before fitting"
            }
        }
    }
}


/// Relative rank tolerance applied to `|λ|/λ_max` when counting the
/// nullity of `H_pen`. Matches the threshold the surrounding REML
/// penalty-rank machinery uses for "structurally zero".
const KKT_REFUSAL_RANK_TOL: f64 = 1e-10;


/// Joint width above which the pairwise Jeffreys second-order endgame
/// completion fallback (`p(p+1)/2` exact second-directional joint-Hessian
/// calls per endgame cycle, gam#979) is not attempted. Wide systems may still
/// get the exact completion from a family-provided contracted trace hook.
const JEFFREYS_COMPLETION_MAX_P: usize = 64;


/// Residual band (as a multiple of the KKT residual tolerance) inside which
/// the inner joint Newton is considered to be in its convergence ENDGAME and
/// the exact Jeffreys second-order completion is added to the step model
/// (gam#979). Far from the mode the trust region globalizes any model, so
/// exactness buys nothing there; in the endgame it converts the
/// divided-difference model's linear sawtooth into quadratic convergence.
const JEFFREYS_COMPLETION_RESIDUAL_BAND: f64 = 300.0;


/// Self-vanishing Levenberg–Marquardt damping factor for the range-restricted
/// spectral Newton step (`solve_joint_newton_step_on_spectral_range`). The
/// caller forms the residual-scaled magnitude
/// `μ = JOINT_SPECTRAL_LEVENBERG_FACTOR · ‖∇L − Sβ‖∞`, which the solve converts
/// to a DIMENSIONLESS, scale-invariant Marquardt damping `ν = μ / λ_max` applied
/// MULTIPLICATIVELY to each range curvature (`curvature·(1 + ν)`), not added
/// (`curvature + μ`). The multiplicative form is essential on a coupled
/// location-scale joint Hessian whose spectrum spans the penalty scale
/// (`λ ~ e²⁴` at the oversmoothed seed) and the likelihood scale (the
/// mean/wiggle XᵀWX curvature): an ADDITIVE μ — set by the penalty-inflated
/// residual — swamps the small likelihood curvature and freezes that block
/// (#826), whereas the multiplicative `1/(1+ν)` throttle is identical across all
/// scales so no block stalls. Both forms cap the unbounded `component/λ` step
/// along near-singular (ill-conditioned but above-`KKT_REFUSAL_RANK_TOL`)
/// eigen-directions — the modes that make the undamped step oscillate — and both
/// vanish as the iterate converges (`ν → 0`), recovering the exact Moore–Penrose
/// Newton step so the KKT fixed point and the well-identified fast path are
/// unchanged. `1e-3` keeps the damping two to three orders below the dominant
/// curvature on a well-conditioned problem.
const JOINT_SPECTRAL_LEVENBERG_FACTOR: f64 = 1.0e-3;


/// Condition number above which a FULL-RANK (`nullity == 0`) penalized Hessian is
/// treated as ill-conditioned enough that a family opting into
/// [`CustomFamily::levenberg_on_ill_conditioning`] gets the self-vanishing
/// constrained-QP Levenberg floor (gam#1040). Below it the active-set QP minimiser
/// is well-determined and the EXACT undamped Newton/KKT solve keeps its quadratic
/// convergence; the survival marginal-slope joint sits at cond ≈ 5.8e6, far above
/// this gate, while a tiny well-conditioned constrained AFT sits well below it.
const LEVENBERG_ILL_CONDITIONING_THRESHOLD: f64 = 1.0e4;


#[derive(Clone, Debug)]
struct JointSpectralNewtonStep {
    delta: Array1<f64>,
    range_rhs_inf: f64,
    null_rhs_inf: f64,
    lambda_max_abs: f64,
    lambda_min_positive: f64,
    nullity: usize,
    rank_tol: f64,
    /// Number of eigen-directions whose curvature was negative (beyond the
    /// rank cutoff) and was reflected to `|λ|` to form a modified-Newton
    /// descent step. Zero for a genuinely positive-semidefinite model.
    reflected_negative_modes: usize,
    /// Most negative eigenvalue encountered (≤ 0); `0.0` when the model was
    /// positive-semidefinite within the rank cutoff.
    most_negative_eigenvalue: f64,
}


/// Production home for the exact trust-region engine ([`WhitenedHessianSpectrum`]),
/// wired into the unconstrained dense-spectral joint-Newton step in
/// `inner_blockwise_fit` (gam#979). Kept in its own module so the engine's
/// helpers stay namespaced; the parent reaches it via `whitened_spectrum::`.
mod whitened_spectrum {
    use super::*;

    /// Eigendecomposition of the metric-whitened penalized Hessian, retained so
    /// every trust-radius shrink within one Newton cycle re-solves the
    /// trust-region subproblem from the SAME `O(p³)` factorization at `O(p)` cost.
    ///
    /// # Why this exists (gam#979)
    ///
    /// The coupled marginal↔logslope inner Newton needs ONE globalization, not a
    /// stack of approximations. Historically the joint step was a *modified-Newton*
    /// (reflect indefinite eigenvalues to `|λ|`) wrapped in a *heuristically gated*
    /// multiplicative Marquardt damping (engaged on `nullity>0`, or condition number
    /// over a threshold, or after N non-improving cycles) and then a *dogleg* between
    /// that step and the Cauchy point, truncated to per-block step-norm trust radii.
    /// Each piece approximates a different facet of the one exact object below, and
    /// each had to be gated so it would not re-break the case another piece was added
    /// for (#826 vs #808 vs #733/#734 vs #787). When none of the gates matched the
    /// operating point — well-conditioned `H_pen`, yet a coupled near-aliased
    /// direction with a huge raw Newton component — the truncated direction made only
    /// Cauchy-sized progress, the gain ratio never justified growing the radius, and
    /// the residual crawled for hundreds of cycles (the #979 "phantom multiplier"
    /// grind / survival hang).
    ///
    /// [`Self::trust_region_step`] replaces all of that with the *exact* solution of
    /// the trust-region subproblem
    ///   minimize  `−rhsᵀδ + ½ δᵀ H_pen δ`   subject to  `‖δ‖_D ≤ r`,
    /// via the Moré–Sorensen characterization: the minimizer is `δ(λ) = (H_pen +
    /// λD)⁻¹ rhs` for the unique `λ ≥ max(0, −γ_min)` with `‖δ(λ)‖_D = r` (or `λ = 0`
    /// when the Newton step is interior and `H_pen ≻ 0`). Working in the `D`-metric
    /// generalized eigenbasis this is a scalar secular equation in `λ`, solved by a
    /// safeguarded Newton iteration on the already-computed spectrum. Properties that
    /// make it the right object:
    ///   * indefiniteness is handled exactly (`λ ≥ −γ_min` makes `H_pen+λD ⪰ 0` on
    ///     the boundary — no reflection heuristic, no negative-curvature special case
    ///     other than the rigorous hard case);
    ///   * the damping `λ` is determined by the trust radius, not by nullity /
    ///     condition / stall gates — those gates disappear;
    ///   * it self-vanishes: at the KKT fixed point `rhs → 0 ⇒ δ → 0`, and once the
    ///     iterate is in a region where `H_pen ≻ 0` the Newton step goes interior so
    ///     `λ = 0` and convergence is quadratic — the converged β, the KKT
    ///     certificate, and the REML/LAML the residual feeds are byte-identical to an
    ///     undamped exact-Newton solve;
    ///   * it is affine covariant in the `D` metric, so blocks at wildly different
    ///     curvature scales (the penalty `λ ~ e²⁴` modes vs the `XᵀWX` likelihood
    ///     modes at an oversmoothed seed) are damped uniformly by `1/(γ_k+λ)` — the
    ///     scale-invariance the per-block radii and the multiplicative-Marquardt form
    ///     were each hand-built to approximate.
    ///
    /// The genuine penalty null space (`|γ_k| ≤ null_cutoff`) is still projected out
    /// (the gam#553 Moore–Penrose range restriction): an unidentified gauge direction
    /// carries no finite Newton step and is left unchanged, its stationarity-residual
    /// component reported via [`JointSpectralNewtonStep::null_rhs_inf`].
    pub(super) struct WhitenedHessianSpectrum {
        /// Generalized eigenvalues `γ_k` of `(H_pen, D)` = eigenvalues of the
        /// whitened matrix `A = D^{-1/2} H_pen D^{-1/2}`.
        gamma: Array1<f64>,
        /// Whitened eigenvectors `v_k` (columns) of `A`.
        evecs: Array2<f64>,
        /// rhs in the whitened eigenbasis: `c_k = v_kᵀ D^{-1/2} rhs`.
        c: Array1<f64>,
        /// `D^{-1/2}` diagonal, mapping a whitened step `η` back to `δ = D^{-1/2} η`.
        d_inv_sqrt: Array1<f64>,
        /// `max_k |γ_k|` (the curvature scale; `D`-whitened).
        lambda_max_abs: f64,
        /// Curvature magnitude at/below which a direction is treated as genuinely
        /// unidentified (penalty null space) and dropped from the step.
        null_cutoff: f64,
    }

    impl WhitenedHessianSpectrum {
        /// Eigendecompose the `D`-whitened penalized Hessian once. `metric_diag`
        /// supplies the positive trust-region metric `D` (each entry is passed
        /// through [`positive_joint_diagonal_entry`] so a non-positive curvature
        /// estimate becomes a safe positive scale). `rank_tol` is the relative
        /// near-singularity cutoff; the genuine numerical-rank floor is derived from
        /// the whitened spectrum exactly as the legacy spectral solve did.
        pub(super) fn decompose(
            h_pen: &Array2<f64>,
            rhs: &Array1<f64>,
            metric_diag: &Array1<f64>,
            rank_tol: f64,
        ) -> Result<Self, String> {
            let p = h_pen.nrows();
            if h_pen.ncols() != p || rhs.len() != p || metric_diag.len() != p {
                return Err(format!(
                    "whitened trust-region decomposition dimension mismatch: H={}x{}, rhs={}, metric={}",
                    h_pen.nrows(),
                    h_pen.ncols(),
                    rhs.len(),
                    metric_diag.len()
                ));
            }
            let d_inv_sqrt = Array1::from_iter(
                metric_diag
                    .iter()
                    .map(|w| 1.0 / positive_joint_diagonal_entry(*w).sqrt()),
            );
            // A = D^{-1/2} H D^{-1/2}; symmetric since H is symmetric and D diagonal.
            let mut a = Array2::<f64>::zeros((p, p));
            for i in 0..p {
                for j in 0..p {
                    a[[i, j]] = h_pen[[i, j]] * d_inv_sqrt[i] * d_inv_sqrt[j];
                }
            }
            symmetrize_dense_in_place(&mut a);
            let (gamma, evecs) = FaerEigh::eigh(&a, Side::Lower)
                .map_err(|e| format!("whitened trust-region eigendecomposition failed: {e}"))?;
            // c = Vᵀ (D^{-1/2} rhs).
            let whitened_rhs = &d_inv_sqrt * rhs;
            let c = evecs.t().dot(&whitened_rhs);
            let lambda_max_abs = gamma.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            let numerical_floor = lambda_max_abs * (p as f64).sqrt() * f64::EPSILON;
            let cutoff = rank_tol * lambda_max_abs;
            let null_cutoff = cutoff.min(numerical_floor);
            Ok(Self {
                gamma,
                evecs,
                c,
                d_inv_sqrt,
                lambda_max_abs,
                null_cutoff,
            })
        }

        /// `‖η(λ)‖²_2 = Σ_{identified k} c_k² / (γ_k + λ)²` — the squared `D`-metric
        /// norm of the trial step as a function of the Levenberg shift `λ`. Only
        /// identified (above-`null_cutoff`) modes participate; the null space carries
        /// no step.
        fn step_norm_sq(&self, lambda: f64) -> f64 {
            let mut acc = 0.0;
            for k in 0..self.gamma.len() {
                if self.gamma[k].abs() <= self.null_cutoff {
                    continue;
                }
                let denom = self.gamma[k] + lambda;
                if denom.abs() <= f64::MIN_POSITIVE {
                    return f64::INFINITY;
                }
                let t = self.c[k] / denom;
                acc += t * t;
            }
            acc
        }

        /// Predicted objective decrease of the *unconstrained* (infinite trust
        /// radius) modified-Newton step on the quadratic model — the Newton
        /// decrement `½ Σ_{identified k} c_k² / |γ_k|`.
        ///
        /// For the exact Newton step `δ = Σ c_k/γ_k v_k` (range modes only) the
        /// model decrease `m(0) − m(δ) = rhsᵀδ − ½δᵀH_pen δ` evaluates to
        /// `½ Σ c_k²/γ_k`; reflecting negative-curvature modes to `|γ_k|` (the
        /// modified-Newton descent step `trust_region_step` takes) makes every
        /// term a genuine decrease, so the unconstrained model can reduce the
        /// objective by AT MOST this amount. The null space (`|γ_k| ≤
        /// null_cutoff`) carries no step and no decrease and is skipped — its
        /// residual mass is a free gauge direction the outer IFT projects out
        /// (gam#553).
        ///
        /// This is the curvature-aware convergence quantity the coupled
        /// joint-Newton needs on a weakly-identified (near-flat) carrying block
        /// (survival marginal↔logslope, link-wiggle, location-scale — gam#1040 /
        /// gam#1088): a large penalized stationarity residual `‖∇L − Sβ‖∞` along
        /// a low-curvature direction (`g` large, `γ` tiny) gives an enormous raw
        /// Newton step that the trust region clamps, so the residual- and
        /// step-norm gates never close and the loop grinds to the cycle ceiling
        /// — yet the *achievable* objective improvement `g²/(2γ)` may be far
        /// below `objective_tol`. When it is, the iterate IS the penalized
        /// optimum to within tolerance (Conn–Gould–Toint, *Trust-Region
        /// Methods*, Thm 6.4.6): no step the model can resolve lowers the
        /// objective by more than `objective_tol`, so continuing is wall-clock
        /// waste. A genuine defect (real curvature AND large gradient) produces
        /// a LARGE decrement, so this never masks one.
        pub(super) fn newton_decrement(&self) -> f64 {
            let mut acc = 0.0;
            for k in 0..self.gamma.len() {
                let abs_gamma = self.gamma[k].abs();
                if abs_gamma <= self.null_cutoff {
                    continue;
                }
                acc += self.c[k] * self.c[k] / abs_gamma;
            }
            0.5 * acc
        }

        /// Assemble the whitened step `η(λ) = Σ c_k/(γ_k+λ) v_k` over identified
        /// modes and map it back to `δ = D^{-1/2} η`. Returns `(δ, range_rhs_inf,
        /// null_rhs_inf, nullity, lambda_min_positive, reflected_negative_modes,
        /// most_negative)` diagnostics consistent with the legacy spectral step.
        fn assemble(
            &self,
            lambda: f64,
            extra_min_mode: Option<(usize, f64)>,
        ) -> JointSpectralNewtonStep {
            let p = self.gamma.len();
            let mut eta = Array1::<f64>::zeros(p);
            let mut range_rhs_inf = 0.0_f64;
            let mut null_rhs_inf = 0.0_f64;
            let mut lambda_min_positive = f64::INFINITY;
            let mut nullity = 0usize;
            let mut reflected_negative_modes = 0usize;
            let mut most_negative = 0.0_f64;
            for k in 0..p {
                let g = self.gamma[k];
                if g.abs() <= self.null_cutoff {
                    nullity += 1;
                    null_rhs_inf = null_rhs_inf.max(self.c[k].abs());
                    continue;
                }
                range_rhs_inf = range_rhs_inf.max(self.c[k].abs());
                if g < 0.0 {
                    reflected_negative_modes += 1;
                    most_negative = most_negative.min(g);
                } else {
                    lambda_min_positive = lambda_min_positive.min(g);
                }
                let denom = g + lambda;
                if denom.abs() > f64::MIN_POSITIVE {
                    let coeff = self.c[k] / denom;
                    for i in 0..p {
                        eta[i] += coeff * self.evecs[[i, k]];
                    }
                }
            }
            // Hard case: add τ·v_min along a minimal-curvature eigenvector to reach
            // the trust boundary when rhs has no component there.
            if let Some((k_min, tau)) = extra_min_mode {
                for i in 0..p {
                    eta[i] += tau * self.evecs[[i, k_min]];
                }
            }
            // δ = D^{-1/2} η.
            let delta = &self.d_inv_sqrt * &eta;
            JointSpectralNewtonStep {
                delta,
                range_rhs_inf,
                null_rhs_inf,
                lambda_max_abs: self.lambda_max_abs,
                lambda_min_positive,
                nullity,
                rank_tol: KKT_REFUSAL_RANK_TOL,
                reflected_negative_modes,
                most_negative_eigenvalue: most_negative,
            }
        }

        /// Exact solution of the trust-region subproblem inside the `D`-metric ball
        /// of radius `trust_radius`. When `trust_radius` is non-finite or `≤ 0` the
        /// unconstrained (Moore–Penrose, range-restricted) Newton step is returned —
        /// i.e. the caller opted out of the trust region.
        pub(super) fn trust_region_step(&self, trust_radius: f64) -> JointSpectralNewtonStep {
            // Smallest identified curvature (signed). Empty identified set ⇒ pure
            // null space ⇒ zero step.
            let mut gamma_min_id = f64::INFINITY;
            let mut any_identified = false;
            for k in 0..self.gamma.len() {
                if self.gamma[k].abs() <= self.null_cutoff {
                    continue;
                }
                any_identified = true;
                gamma_min_id = gamma_min_id.min(self.gamma[k]);
            }
            if !any_identified {
                return self.assemble(0.0, None);
            }

            let unconstrained_radius = !(trust_radius.is_finite() && trust_radius > 0.0);
            // Interior Newton step is admissible only when the model is convex on the
            // identified range (γ_min > 0); then λ = 0 gives the exact Newton step.
            if gamma_min_id > 0.0 {
                let newton_norm = self.step_norm_sq(0.0).sqrt();
                if unconstrained_radius || newton_norm <= trust_radius {
                    return self.assemble(0.0, None);
                }
            } else if unconstrained_radius {
                // No trust region but an indefinite/semidefinite model: the
                // unconstrained problem is unbounded below. Fall back to the
                // reflected modified-Newton step (|γ| curvature) so the caller still
                // receives a finite descent direction; the downstream accept/reject
                // validates it. This path is only hit when a caller explicitly
                // disables the trust region on an indefinite model.
                return self.assemble_reflected();
            }

            // Boundary solution: find λ ≥ λ_lo with ‖η(λ)‖ = trust_radius.
            let lambda_lo = (-gamma_min_id).max(0.0);
            // Hard case detection: is rhs orthogonal to the minimal-curvature
            // eigenspace? If so ‖η(λ_lo)‖ is finite and may be below the radius.
            let min_mode_tol = self.null_cutoff.max(self.lambda_max_abs * 1e-12);
            let mut hard_case_component_sq = 0.0;
            let mut k_min_witness = None;
            for k in 0..self.gamma.len() {
                if self.gamma[k].abs() <= self.null_cutoff {
                    continue;
                }
                if (self.gamma[k] - gamma_min_id).abs() <= min_mode_tol {
                    hard_case_component_sq += self.c[k] * self.c[k];
                    k_min_witness = Some(k);
                }
            }
            // Evaluate the norm just above the pole. With a real rhs component at the
            // minimal mode the norm diverges at λ_lo, so the secular root is interior
            // to (λ_lo, ∞) and a small relative offset brackets it. With no such
            // component (hard case) the norm at λ_lo is finite.
            let lambda_lo_eval = lambda_lo + self.lambda_max_abs.max(1.0) * 1e-12;
            if hard_case_component_sq <= (self.lambda_max_abs.max(1.0) * 1e-12).powi(2) {
                let norm_at_lo = self.step_norm_sq(lambda_lo_eval).sqrt();
                if norm_at_lo < trust_radius {
                    // Hard case: λ = λ_lo, then add τ·v_min to reach the boundary.
                    if let Some(k_min) = k_min_witness {
                        let deficit =
                            (trust_radius * trust_radius - norm_at_lo * norm_at_lo).max(0.0);
                        let tau = deficit.sqrt();
                        return self.assemble(lambda_lo, Some((k_min, tau)));
                    }
                    return self.assemble(lambda_lo, None);
                }
            }
            // Safeguarded Newton on φ(λ) = 1/‖η(λ)‖ − 1/r (well-behaved, ~linear),
            // bracketed in [lo, hi]. φ is increasing in λ (‖η‖ decreasing), φ(lo)<0,
            // and we grow hi until φ(hi)>0.
            let target = trust_radius;
            let mut lo = lambda_lo_eval;
            let mut hi = lambda_lo_eval.max(self.lambda_max_abs).max(1.0);
            let mut grow_guard = 0;
            while self.step_norm_sq(hi).sqrt() > target && grow_guard < 200 {
                hi *= 2.0;
                grow_guard += 1;
            }
            let mut lambda = 0.5 * (lo + hi);
            for _ in 0..100 {
                let q = self.step_norm_sq(lambda);
                let norm = q.sqrt();
                if !norm.is_finite() {
                    lo = lambda;
                    lambda = 0.5 * (lo + hi);
                    continue;
                }
                // Maintain the bracket on φ(λ) = 1/norm − 1/target.
                if norm > target {
                    lo = lambda;
                } else {
                    hi = lambda;
                }
                let phi = 1.0 / norm - 1.0 / target;
                if phi.abs() <= 1e-12 / target {
                    break;
                }
                // q'(λ) = -2 Σ c_k²/(γ_k+λ)³ ⇒ d/dλ (1/norm) = -½ q^{-3/2} q'.
                let mut q_prime = 0.0;
                for k in 0..self.gamma.len() {
                    if self.gamma[k].abs() <= self.null_cutoff {
                        continue;
                    }
                    let denom = self.gamma[k] + lambda;
                    if denom.abs() <= f64::MIN_POSITIVE {
                        continue;
                    }
                    q_prime += -2.0 * self.c[k] * self.c[k] / (denom * denom * denom);
                }
                let phi_prime = -0.5 * q.powf(-1.5) * q_prime;
                let next = if phi_prime.abs() > f64::MIN_POSITIVE {
                    lambda - phi / phi_prime
                } else {
                    0.5 * (lo + hi)
                };
                // Safeguard into the bracket.
                lambda = if next.is_finite() && next > lo && next < hi {
                    next
                } else {
                    0.5 * (lo + hi)
                };
                if (hi - lo) <= 1e-14 * (1.0 + hi.abs()) {
                    break;
                }
            }
            self.assemble(lambda, None)
        }

        /// Reflected modified-Newton step (`|γ_k|` curvature, no trust region). Only
        /// used when a caller disables the trust region on an indefinite model — the
        /// trust-region path proper never reflects.
        fn assemble_reflected(&self) -> JointSpectralNewtonStep {
            let p = self.gamma.len();
            let mut eta = Array1::<f64>::zeros(p);
            let mut range_rhs_inf = 0.0_f64;
            let mut null_rhs_inf = 0.0_f64;
            let mut lambda_min_positive = f64::INFINITY;
            let mut nullity = 0usize;
            let mut reflected_negative_modes = 0usize;
            let mut most_negative = 0.0_f64;
            for k in 0..p {
                let g = self.gamma[k];
                if g.abs() <= self.null_cutoff {
                    nullity += 1;
                    null_rhs_inf = null_rhs_inf.max(self.c[k].abs());
                    continue;
                }
                range_rhs_inf = range_rhs_inf.max(self.c[k].abs());
                let curvature = if g < 0.0 {
                    reflected_negative_modes += 1;
                    most_negative = most_negative.min(g);
                    g.abs()
                } else {
                    lambda_min_positive = lambda_min_positive.min(g);
                    g
                };
                let coeff = self.c[k] / curvature;
                for i in 0..p {
                    eta[i] += coeff * self.evecs[[i, k]];
                }
            }
            let delta = &self.d_inv_sqrt * &eta;
            JointSpectralNewtonStep {
                delta,
                range_rhs_inf,
                null_rhs_inf,
                lambda_max_abs: self.lambda_max_abs,
                lambda_min_positive,
                nullity,
                rank_tol: KKT_REFUSAL_RANK_TOL,
                reflected_negative_modes,
                most_negative_eigenvalue: most_negative,
            }
        }
    }
}


#[cfg(test)]
mod trust_region_subproblem_tests {
    use super::whitened_spectrum::WhitenedHessianSpectrum;
    use super::*;
    use ndarray::array;

    fn metric_norm(delta: &Array1<f64>, d: &Array1<f64>) -> f64 {
        delta
            .iter()
            .zip(d.iter())
            .map(|(x, w)| x * x * positive_joint_diagonal_entry(*w))
            .sum::<f64>()
            .sqrt()
    }

    /// Interior case: a positive-definite model with a generous trust radius
    /// must return the exact (full) Newton step `H⁻¹ rhs`, i.e. λ = 0.
    #[test]
    fn interior_returns_exact_newton_step() {
        let h = array![[3.0, 1.0], [1.0, 2.0]];
        let rhs = array![1.0, -2.0];
        let d = array![1.0, 1.0];
        let spec = WhitenedHessianSpectrum::decompose(&h, &rhs, &d, KKT_REFUSAL_RANK_TOL).unwrap();
        let step = spec.trust_region_step(1e6);
        // Exact Newton: H δ = rhs.
        let resid = h.dot(&step.delta) - &rhs;
        assert!(
            resid.iter().all(|v| v.abs() < 1e-10),
            "interior step must solve H δ = rhs exactly, residual {resid:?}"
        );
    }

    /// Boundary case: a tight radius forces `‖δ‖_D = r` and the KKT condition
    /// `(H + λD) δ = rhs` with `λ > 0`.
    #[test]
    fn boundary_satisfies_more_sorensen_kkt() {
        let h = array![[3.0, 1.0], [1.0, 2.0]];
        let rhs = array![1.0, -2.0];
        let d = array![1.0, 1.0];
        let spec = WhitenedHessianSpectrum::decompose(&h, &rhs, &d, KKT_REFUSAL_RANK_TOL).unwrap();
        let r = 0.3;
        let step = spec.trust_region_step(r);
        let norm = metric_norm(&step.delta, &d);
        assert!(
            (norm - r).abs() < 1e-8,
            "boundary step must lie on the trust boundary: ‖δ‖_D={norm} vs r={r}"
        );
        // Recover λ from one coordinate of (H+λD)δ = rhs and check the whole
        // system is satisfied at that λ.
        let hd = h.dot(&step.delta);
        // Solve λ minimizing ‖(H+λD)δ − rhs‖ in least squares over the single
        // scalar λ: λ* = (Dδ)·(rhs − Hδ) / (Dδ)·(Dδ).
        let dd = &d * &step.delta;
        let lam = dd.dot(&(&rhs - &hd)) / dd.dot(&dd);
        assert!(lam > 0.0, "boundary multiplier must be positive, got {lam}");
        let resid = &hd + &(lam * &dd) - &rhs;
        assert!(
            resid.iter().all(|v| v.abs() < 1e-7),
            "(H+λD)δ = rhs must hold at the recovered λ={lam}, residual {resid:?}"
        );
    }

    /// Indefinite model: the exact subproblem still returns a finite boundary
    /// step that is a descent direction (rhsᵀδ > 0) and lies on the boundary.
    #[test]
    fn indefinite_model_returns_descent_step_on_boundary() {
        // Eigenvalues +4 and -1: genuinely indefinite.
        let h = array![[1.5, 2.5], [2.5, 1.5]];
        let rhs = array![1.0, 0.4];
        let d = array![1.0, 1.0];
        let spec = WhitenedHessianSpectrum::decompose(&h, &rhs, &d, KKT_REFUSAL_RANK_TOL).unwrap();
        let r = 0.7;
        let step = spec.trust_region_step(r);
        assert!(step.reflected_negative_modes >= 1 || step.most_negative_eigenvalue < 0.0);
        let norm = metric_norm(&step.delta, &d);
        assert!(
            (norm - r).abs() < 1e-7,
            "indefinite boundary step ‖δ‖_D={norm} vs r={r}"
        );
        assert!(
            rhs.dot(&step.delta) > 0.0,
            "step must be a descent direction for −rhsᵀδ + ½δᵀHδ (rhsᵀδ>0)"
        );
        // (H+λD) must be PSD at the chosen λ (most negative eigenvalue ≥ -λ).
        let dd = &d * &step.delta;
        let lam = dd.dot(&(&rhs - &h.dot(&step.delta))) / dd.dot(&dd);
        assert!(lam >= 1.0 - 1e-6, "λ must dominate -γ_min=1, got {lam}");
    }

    /// Self-vanishing: as rhs → 0 the step → 0 regardless of the radius, so the
    /// converged β and the KKT fixed point are unchanged by the globalization.
    #[test]
    fn step_vanishes_as_rhs_vanishes() {
        let h = array![[3.0, 1.0], [1.0, 2.0]];
        let rhs = array![1e-13, -2e-13];
        let d = array![1.0, 1.0];
        let spec = WhitenedHessianSpectrum::decompose(&h, &rhs, &d, KKT_REFUSAL_RANK_TOL).unwrap();
        let step = spec.trust_region_step(0.5);
        assert!(
            step.delta.iter().all(|v| v.abs() < 1e-11),
            "near-zero rhs must give near-zero step, got {:?}",
            step.delta
        );
    }

    /// Null space: a genuinely zero-curvature direction is dropped from the step
    /// (Moore–Penrose range restriction) and reported via `null_rhs_inf`.
    #[test]
    fn null_direction_is_dropped_and_reported() {
        // Second coordinate has zero curvature; rhs has mass there.
        let h = array![[2.0, 0.0], [0.0, 0.0]];
        let rhs = array![1.0, 0.5];
        let d = array![1.0, 1.0];
        let spec = WhitenedHessianSpectrum::decompose(&h, &rhs, &d, KKT_REFUSAL_RANK_TOL).unwrap();
        let step = spec.trust_region_step(1e6);
        assert_eq!(step.nullity, 1, "one null direction expected");
        assert!(
            step.null_rhs_inf >= 0.5 - 1e-9,
            "null-space rhs component must be reported, got {}",
            step.null_rhs_inf
        );
        // The identified direction takes its exact Newton component (1/2).
        assert!((step.delta[0] - 0.5).abs() < 1e-10);
        assert!(step.delta[1].abs() < 1e-10, "null coordinate left at 0");
    }

    /// Non-identity metric: the boundary is measured in the `D` norm, so a step
    /// with a large lightly-weighted coordinate is admissible.
    #[test]
    fn respects_non_identity_metric() {
        let h = array![[2.0, 0.0], [0.0, 8.0]];
        let rhs = array![1.0, 1.0];
        let d = array![1.0, 16.0];
        let spec = WhitenedHessianSpectrum::decompose(&h, &rhs, &d, KKT_REFUSAL_RANK_TOL).unwrap();
        let r = 0.2;
        let step = spec.trust_region_step(r);
        let norm = metric_norm(&step.delta, &d);
        assert!(
            (norm - r).abs() < 1e-8,
            "step must lie on the D-metric boundary, ‖δ‖_D={norm} vs r={r}"
        );
    }

    /// Shrinking the radius re-solves the subproblem (the direction bends toward
    /// the gradient) rather than rescaling a fixed direction — the property the
    /// dogleg/truncation lacked. A halved radius must not merely halve the step.
    #[test]
    fn radius_shrink_bends_direction_not_just_scale() {
        let h = array![[50.0, 0.0], [0.0, 0.5]];
        let rhs = array![1.0, 1.0];
        let d = array![1.0, 1.0];
        let spec = WhitenedHessianSpectrum::decompose(&h, &rhs, &d, KKT_REFUSAL_RANK_TOL).unwrap();
        let big = spec.trust_region_step(1.0).delta;
        let small = spec.trust_region_step(0.25).delta;
        // Direction (unit vectors) must differ: a pure truncation keeps the
        // direction fixed; the exact subproblem rotates toward the steep mode.
        let big_u = &big / metric_norm(&big, &d);
        let small_u = &small / metric_norm(&small, &d);
        let cos = big_u.dot(&small_u);
        assert!(
            cos < 0.9999,
            "exact TR step must bend the direction under radius shrink (cos={cos})"
        );
    }
}


/// Numerical nullity of a symmetric penalized Hessian at the shared
/// `KKT_REFUSAL_RANK_TOL` relative cutoff (the same threshold the spectral
/// range solve and the REML penalty-rank machinery use). Returns `None` only
/// when the eigendecomposition fails or the matrix is the zero matrix (no
/// finite curvature scale to normalize against); callers treat a `None` as
/// "could not certify full rank" and fall back to the conservative (damped)
/// path.
///
/// This exists so the CONSTRAINED active-set QP branch can decide whether the
/// joint design is genuinely rank-deficient (`nullity > 0` ⇒ an unidentified
/// gauge direction that needs the self-vanishing Levenberg floor to make the
/// QP minimizer unique) or fully identified (`nullity == 0` ⇒ the exact,
/// undamped Newton/KKT step is well-posed and converges quadratically). The
/// spectral-range branch already gets this for free via
/// `JointSpectralNewtonStep::nullity`; the constrained branch never runs the
/// eigensolve otherwise, so it computes it here on the already-penalized `lhs`.
/// PSD part of a symmetric matrix: eigendecompose and clamp negative
/// eigenvalues to zero. Used by the step consumers that REQUIRE a convex
/// model (the constrained active-set QP and the SPD-PCG matvec) when folding
/// the exact divided-difference Jeffreys curvature `H_Φ`, which is indefinite
/// exactly where `Φ` is (gam#979). On a PSD input this is the identity (up to
/// eigendecomposition round-off). Falls back to the zero matrix if the
/// eigendecomposition fails — the safe unaugmented step, never a wrong one.
fn symmetric_psd_projection(matrix: &Array2<f64>) -> Array2<f64> {
    let p = matrix.nrows();
    let mut sym = matrix.clone();
    symmetrize_dense_in_place(&mut sym);
    let Ok((evals, evecs)) = FaerEigh::eigh(&sym, Side::Lower) else {
        return Array2::zeros((p, p));
    };
    if evals.iter().all(|lam| *lam >= 0.0) {
        return sym;
    }
    let clamped = Array1::from_iter(evals.iter().map(|lam| lam.max(0.0)));
    let scaled = &evecs * &clamped.view().insert_axis(ndarray::Axis(0));
    scaled.dot(&evecs.t())
}


fn symmetric_penalized_hessian_nullity(lhs: &Array2<f64>) -> Option<usize> {
    let p = lhs.nrows();
    if p == 0 || lhs.ncols() != p {
        return Some(0);
    }
    let (evals, _) = FaerEigh::eigh(lhs, Side::Lower).ok()?;
    let max_abs = evals.iter().map(|x: &f64| x.abs()).fold(0.0_f64, f64::max);
    if !(max_abs.is_finite() && max_abs > 0.0) {
        return None;
    }
    let cutoff = KKT_REFUSAL_RANK_TOL * max_abs;
    Some(evals.iter().filter(|x| x.abs() < cutoff).count())
}


/// Numerical null-space count AND the (range-space) condition number of a
/// symmetric penalized Hessian, from ONE eigendecomposition.
///
/// `nullity` counts eigenvalues below the shared rank tolerance (`< tol·λmax`);
/// `condition` is `λmax / λmin_range` where `λmin_range` is the smallest
/// eigenvalue magnitude ABOVE that tolerance (the smallest *identified*
/// curvature). On a full-rank-but-ill-conditioned `H_pen` (`nullity == 0`,
/// `condition` large) the constrained active-set QP minimiser is unique only up
/// to round-off along the near-null mode, so without a Levenberg floor the
/// active set slides an O(1) proposal step every cycle and the inner solve
/// grinds its whole budget — the survival marginal-slope hang (gam#1040). Both
/// signals come from the single `eigh` the nullity check already paid for.
fn symmetric_penalized_hessian_nullity_and_condition(
    lhs: &Array2<f64>,
) -> Option<(usize, f64)> {
    let p = lhs.nrows();
    if p == 0 || lhs.ncols() != p {
        return Some((0, 1.0));
    }
    let (evals, _) = FaerEigh::eigh(lhs, Side::Lower).ok()?;
    let max_abs = evals.iter().map(|x: &f64| x.abs()).fold(0.0_f64, f64::max);
    if !(max_abs.is_finite() && max_abs > 0.0) {
        return None;
    }
    let cutoff = KKT_REFUSAL_RANK_TOL * max_abs;
    let nullity = evals.iter().filter(|x| x.abs() < cutoff).count();
    // Smallest identified (range-space) eigenvalue magnitude: the floor for the
    // condition number. Eigenvalues below `cutoff` are the null space, excluded.
    let min_range = evals
        .iter()
        .map(|x: &f64| x.abs())
        .filter(|&m| m >= cutoff && m > 0.0)
        .fold(f64::INFINITY, f64::min);
    let condition = if min_range.is_finite() && min_range > 0.0 {
        max_abs / min_range
    } else {
        f64::INFINITY
    };
    Some((nullity, condition))
}


#[allow(clippy::too_many_arguments)]
fn compute_kkt_refusal_report(
    cycle: usize,
    states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    s_lambdas: &[Array2<f64>],
    ranges: &[(usize, usize)],
    cached_joint_gradient: Option<&Array1<f64>>,
    cached_active_sets: &[Option<Vec<usize>>],
    block_constraints: &[Option<LinearInequalityConstraints>],
    joint_hessian_source: Option<&JointHessianSource>,
    total_p: usize,
    ridge: f64,
    ridge_policy: RidgePolicy,
    accepted_step_inf: f64,
    proposal_step_inf: f64,
    trust_radius: f64,
    residual_tol: f64,
    obj_tol: f64,
    step_tol: f64,
    objective_change: f64,
    projected_residual_inf: f64,
    math: Option<&JointNewtonMathDiagnostic>,
) -> KktRefusalReport {
    let block_names: Vec<String> = specs.iter().map(|s| s.name.clone()).collect();
    let block_widths: Vec<usize> = states.iter().map(|s| s.beta.len()).collect();
    let block_beta_inf: Vec<f64> = states
        .iter()
        .map(|s| s.beta.iter().map(|x: &f64| x.abs()).fold(0.0_f64, f64::max))
        .collect();

    let block_grad_inf: Vec<f64> = match cached_joint_gradient {
        Some(joint_grad) => {
            let mut acc = 0usize;
            states
                .iter()
                .map(|s| {
                    let n = s.beta.len();
                    let end = (acc + n).min(joint_grad.len());
                    // A width-0 block (e.g. a constant-scale `noise_formula="1"`
                    // log_sigma channel collapsed to zero free coefficients,
                    // gam#553) has no gradient and a zero residual — report 0.0,
                    // not the NaN sentinel. The NaN sentinel is reserved for a
                    // genuine layout mismatch: a positive-width block whose
                    // coordinates fall past the end of the joint gradient.
                    let nrm = if n == 0 {
                        0.0
                    } else if acc < end {
                        joint_grad
                            .slice(ndarray::s![acc..end])
                            .iter()
                            .map(|x: &f64| x.abs())
                            .fold(0.0_f64, f64::max)
                    } else {
                        f64::NAN
                    };
                    acc += n;
                    nrm
                })
                .collect()
        }
        None => vec![f64::NAN; states.len()],
    };

    let block_penalty_grad_inf: Vec<f64> = ranges
        .iter()
        .enumerate()
        .map(|(b, _)| {
            let mut penalty_block = s_lambdas[b].dot(&states[b].beta);
            if ridge_policy.include_quadratic_penalty && ridge > 0.0 {
                penalty_block += &states[b].beta.mapv(|v| ridge * v);
            }
            penalty_block
                .iter()
                .map(|x: &f64| x.abs())
                .fold(0.0_f64, f64::max)
        })
        .collect();

    let residual_vec_opt = cached_joint_gradient.and_then(|joint_grad| {
        exact_newton_joint_projected_stationarity_vector_from_gradient(
            joint_grad,
            states,
            specs,
            s_lambdas,
            ridge,
            ridge_policy,
            block_constraints,
            Some(cached_active_sets),
        )
        .ok()
    });
    let block_residual_inf: Vec<f64> = match residual_vec_opt.as_ref() {
        Some(residual) => ranges
            .iter()
            .map(|(start, end)| {
                // A zero-width block (start == end) has no residual of its own;
                // an empty `fold` would report a spurious `0.0`. Mark it `NaN`
                // so the `is_finite()` filter below excludes it from the
                // carrying-block selection (it cannot carry residual it has no
                // parameters for).
                if start >= end {
                    f64::NAN
                } else {
                    residual
                        .slice(ndarray::s![*start..*end])
                        .iter()
                        .map(|x: &f64| x.abs())
                        .fold(0.0_f64, f64::max)
                }
            })
            .collect(),
        None => vec![f64::NAN; states.len()],
    };
    let block_carrying_residual = block_residual_inf
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i);

    let mut hpen_eigenvalues_sorted_desc: Vec<f64> = Vec::new();
    let mut hpen_min_abs_eigenvalue = f64::NAN;
    let mut hpen_max_abs_eigenvalue = f64::NAN;
    let mut hpen_condition_number = f64::NAN;
    let mut hpen_nullity_at_rank_tol = 0usize;
    let mut hpen_null_gradient_inf = f64::NAN;
    let mut hpen_null_vector_block_inf = Vec::new();
    let mut hpen_null_vector_carrying_block = None;
    if total_p > 0
        && let Some(source) = joint_hessian_source
        && let Ok(mut h_joint) =
            materialize_joint_hessian_source(source, total_p, "KKT refusal diagnostic spectrum")
    {
        let model_diagonal_ridge = if ridge_policy.include_quadratic_penalty && ridge > 0.0 {
            ridge
        } else {
            0.0
        };
        add_joint_penalty_to_matrix(&mut h_joint, ranges, s_lambdas, model_diagonal_ridge, None);
        symmetrize_dense_in_place(&mut h_joint);
        if let Ok((evals, evecs)) = FaerEigh::eigh(&h_joint, Side::Lower) {
            let mut sorted: Vec<f64> = evals.iter().copied().collect();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let max_abs = sorted.iter().map(|x: &f64| x.abs()).fold(0.0_f64, f64::max);
            let min_abs = sorted
                .iter()
                .map(|x: &f64| x.abs())
                .fold(f64::INFINITY, f64::min);
            let cutoff = KKT_REFUSAL_RANK_TOL * max_abs;
            hpen_nullity_at_rank_tol = sorted.iter().filter(|x| x.abs() < cutoff).count();
            hpen_max_abs_eigenvalue = max_abs;
            hpen_min_abs_eigenvalue = if min_abs.is_finite() {
                min_abs
            } else {
                f64::NAN
            };
            hpen_condition_number = if min_abs > 0.0 && min_abs.is_finite() {
                max_abs / min_abs
            } else {
                f64::INFINITY
            };
            if let Some(residual) = residual_vec_opt.as_ref()
                && residual.len() == total_p
                && hpen_nullity_at_rank_tol > 0
            {
                let mut best_component = 0.0_f64;
                let mut best_block_inf = vec![0.0_f64; ranges.len()];
                for k in 0..evals.len() {
                    if evals[k].abs() >= cutoff {
                        continue;
                    }
                    let component = evecs.column(k).dot(residual).abs();
                    if component > best_component {
                        best_component = component;
                        best_block_inf.clear();
                        best_block_inf.extend(ranges.iter().map(|(start, end)| {
                            evecs
                                .slice(ndarray::s![*start..*end, k])
                                .iter()
                                .map(|x: &f64| x.abs())
                                .fold(0.0_f64, f64::max)
                        }));
                    }
                }
                hpen_null_gradient_inf = best_component;
                hpen_null_vector_block_inf = best_block_inf;
                hpen_null_vector_carrying_block = hpen_null_vector_block_inf
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| v.is_finite())
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i);
            }
            hpen_eigenvalues_sorted_desc = sorted;
        }
    }

    let active_set_rows_total: usize = cached_active_sets
        .iter()
        .map(|maybe_rows| maybe_rows.as_ref().map(|v| v.len()).unwrap_or(0))
        .sum();
    let any_block_has_constraints = block_constraints.iter().any(|c| c.is_some());

    let diagnosis = if hpen_nullity_at_rank_tol > 0 {
        KktRefusalDiagnosis::RankDeficientHPen
    } else if any_block_has_constraints
        && cached_active_sets.iter().any(|s| s.is_some())
        && projected_residual_inf > residual_tol
    {
        // Well-conditioned H_pen, the user has bound constraints, the current
        // active set already pinned some rows, yet the projected residual is
        // still many tolerances above the threshold. The cert refused
        // *because* the projection captured part of the multiplier but not
        // all of it — i.e. the active set is missing a row.
        KktRefusalDiagnosis::ActiveSetIncomplete
    } else {
        KktRefusalDiagnosis::PhantomMultiplierWithWellConditionedH
    };

    KktRefusalReport {
        block_names,
        block_widths,
        block_beta_inf,
        block_grad_inf,
        block_penalty_grad_inf,
        block_residual_inf,
        block_carrying_residual,
        hpen_eigenvalues_sorted_desc,
        hpen_min_abs_eigenvalue,
        hpen_max_abs_eigenvalue,
        hpen_condition_number,
        hpen_nullity_at_rank_tol,
        hpen_rank_tol: KKT_REFUSAL_RANK_TOL,
        hpen_null_gradient_inf,
        hpen_null_vector_block_inf,
        hpen_null_vector_carrying_block,
        active_set_rows_total,
        accepted_step_inf,
        proposal_step_inf,
        trust_radius,
        cycle,
        residual_tol,
        obj_tol,
        step_tol,
        linearized_rel: math
            .map(JointNewtonMathDiagnostic::linearized_rel)
            .unwrap_or(f64::NAN),
        scalar_model_relerr: math
            .map(JointNewtonMathDiagnostic::scalar_model_relative_error)
            .unwrap_or(f64::NAN),
        objective_change,
        projected_residual_inf,
        diagnosis,
    }
}


impl KktRefusalReport {
    fn carrying_block_label(&self) -> String {
        match self.block_carrying_residual {
            Some(idx) => format!(
                "{} (idx={}, |g|={:.3e}, |Sβ|={:.3e}, |∇L-Sβ|={:.3e}, |β|={:.3e}, width={})",
                self.block_names.get(idx).map(String::as_str).unwrap_or("?"),
                idx,
                self.block_grad_inf.get(idx).copied().unwrap_or(f64::NAN),
                self.block_penalty_grad_inf
                    .get(idx)
                    .copied()
                    .unwrap_or(f64::NAN),
                self.block_residual_inf
                    .get(idx)
                    .copied()
                    .unwrap_or(f64::NAN),
                self.block_beta_inf.get(idx).copied().unwrap_or(f64::NAN),
                self.block_widths.get(idx).copied().unwrap_or(0),
            ),
            None => "<no block carries finite residual>".to_string(),
        }
    }

    fn beta_inf(&self) -> f64 {
        self.block_beta_inf.iter().copied().fold(0.0_f64, f64::max)
    }

    fn null_direction_label(&self) -> String {
        match self.hpen_null_vector_carrying_block {
            Some(idx) => format!(
                "{} (idx={}, |u_block|∞={:.3e}, |uᵀg_proj|={:.3e})",
                self.block_names.get(idx).map(String::as_str).unwrap_or("?"),
                idx,
                self.hpen_null_vector_block_inf
                    .get(idx)
                    .copied()
                    .unwrap_or(f64::NAN),
                self.hpen_null_gradient_inf,
            ),
            None => format!("none (|uᵀg_proj|={:.3e})", self.hpen_null_gradient_inf),
        }
    }

    /// Multi-line structured log emitted at the cert REFUSED site. The
    /// per-block residual / eigenspectrum / diagnosis breakdown is what
    /// makes the failure actionable (vs the legacy one-liner that only
    /// reported aggregate residual + cert math).
    fn format_structured_log(&self, four_tol: f64) -> String {
        format!(
            "[PIRLS/joint-Newton convergence] cycle {:>3} | cert REFUSED: residual={:.3e} > tol={:.3e} (cert)\n  \
             carrying-block: {}\n  \
             block_names={:?}, block_widths={:?}, block_grad_inf={:?}, block_penalty_grad_inf={:?}, block_residual_inf={:?}\n  \
             H_pen spectrum: λ_max={:.3e}, λ_min={:.3e}, cond={:.3e}, nullity@{:.0e}={} (of {} eigenvalues)\n  \
             free-null diagnostic: {}\n  \
             cert math: linearized_rel={:.3e}, scalar_relerr={:.3e}, |Δobj|={:.3e} (tol={:.3e}), accepted_step_inf={:.3e} (tol={:.3e}), proposal_step_inf={:.3e}, trust_radius={:.3e}, |β|∞={:.3e}, active_set_rows_total={}\n  \
             diagnosis: {}",
            self.cycle,
            self.projected_residual_inf,
            four_tol,
            self.carrying_block_label(),
            self.block_names,
            self.block_widths,
            self.block_grad_inf,
            self.block_penalty_grad_inf,
            self.block_residual_inf,
            self.hpen_max_abs_eigenvalue,
            self.hpen_min_abs_eigenvalue,
            self.hpen_condition_number,
            self.hpen_rank_tol,
            self.hpen_nullity_at_rank_tol,
            self.hpen_eigenvalues_sorted_desc.len(),
            self.null_direction_label(),
            self.linearized_rel,
            self.scalar_model_relerr,
            self.objective_change,
            self.obj_tol,
            self.accepted_step_inf,
            self.step_tol,
            self.proposal_step_inf,
            self.trust_radius,
            self.beta_inf(),
            self.active_set_rows_total,
            self.diagnosis.as_str(),
        )
    }

    /// Single-string formatter used by the bubbled error returned from
    /// the inner solver, where the caller wants one self-contained line
    /// even though the data is structured.
    fn format_bubbled_error(&self) -> String {
        let carrying = self.carrying_block_label();
        format!(
            "cycle={} cert REFUSED: residual={:.3e} > tol={:.3e}; \
             carrying-block: {}; block_names={:?}, block_widths={:?}, \
             block_grad_inf={:?}, block_penalty_grad_inf={:?}, block_residual_inf={:?}; \
             H_pen spectrum: λ_max={:.3e}, λ_min={:.3e}, cond={:.3e}, nullity@{:.0e}={}/{}; \
             free-null diagnostic: {}; \
             cert math: linearized_rel={:.3e}, scalar_relerr={:.3e}, |Δobj|={:.3e}, \
             accepted_step_inf={:.3e}, proposal_step_inf={:.3e}, trust_radius={:.3e}, \
             |β|∞={:.3e}, active_set_rows_total={}; diagnosis: {}; {}",
            self.cycle,
            self.projected_residual_inf,
            4.0 * self.residual_tol,
            carrying,
            self.block_names,
            self.block_widths,
            self.block_grad_inf,
            self.block_penalty_grad_inf,
            self.block_residual_inf,
            self.hpen_max_abs_eigenvalue,
            self.hpen_min_abs_eigenvalue,
            self.hpen_condition_number,
            self.hpen_rank_tol,
            self.hpen_nullity_at_rank_tol,
            self.hpen_eigenvalues_sorted_desc.len(),
            self.null_direction_label(),
            self.linearized_rel,
            self.scalar_model_relerr,
            self.objective_change,
            self.accepted_step_inf,
            self.proposal_step_inf,
            self.trust_radius,
            self.beta_inf(),
            self.active_set_rows_total,
            self.diagnosis.as_str(),
            self.diagnosis.guidance(),
        )
    }
}


const JOINT_PCG_REL_TOL: f64 = 1e-8;

const PCG_ETA_MAX: f64 = 1.0e-1;

const PCG_ETA_MIN: f64 = 1.0e-8;

const PCG_GAMMA: f64 = 0.9;

const PCG_ALPHA: f64 = 1.618_033_988_749_895;


/// Eisenstat–Walker adaptive forcing term for the inner PCG tolerance:
/// when the previous outer KKT residual is known, scale the next inner
/// solve's relative tolerance by `γ·(‖r_cur‖/‖r_prev‖)^α`, clamped to
/// `[PCG_ETA_MIN, PCG_ETA_MAX]`. On the first cycle (no previous
/// residual) we use the loose `PCG_ETA_MAX` to avoid over-solving when
/// the iterate is far from the optimum.
fn joint_pcg_eisenstat_walker_forcing(prev_kkt_norm: Option<f64>, current_kkt_norm: f64) -> f64 {
    if !current_kkt_norm.is_finite() || current_kkt_norm < 0.0 {
        return JOINT_PCG_REL_TOL;
    }
    let Some(prev_kkt_norm) = prev_kkt_norm else {
        return PCG_ETA_MAX;
    };
    if !prev_kkt_norm.is_finite() || prev_kkt_norm <= 0.0 {
        return JOINT_PCG_REL_TOL;
    }
    let ratio = current_kkt_norm / prev_kkt_norm;
    if !ratio.is_finite() || ratio < 0.0 {
        return JOINT_PCG_REL_TOL;
    }
    (PCG_GAMMA * ratio.powf(PCG_ALPHA)).clamp(PCG_ETA_MIN, PCG_ETA_MAX)
}


fn apply_joint_penalized_hessian_into(
    source: &JointHessianSource,
    ranges: &[(usize, usize)],
    s_lambdas: &[Array2<f64>],
    diagonal_ridge: f64,
    vector: &Array1<f64>,
    out: &mut Array1<f64>,
    joint_full_width: Option<&crate::families::joint_penalty::JointPenaltyBundle>,
) -> Result<(), String> {
    let mut penalty = Array1::<f64>::zeros(vector.len());
    apply_joint_penalized_hessian_into_with_workspace(
        source,
        ranges,
        s_lambdas,
        diagonal_ridge,
        vector,
        out,
        &mut penalty,
        joint_full_width,
    )
}


/// Variant of [`apply_joint_penalized_hessian_into`] that reuses a
/// caller-supplied scratch buffer for the penalty term instead of
/// allocating per call.  Use this in hot loops (e.g. the trust-region
/// trial loop) where `penalty_scratch` and the output `out` are hoisted
/// outside the loop and reused across attempts.
///
/// `penalty_scratch` must have the same length as `vector`; its contents
/// are overwritten on every call.
fn apply_joint_penalized_hessian_into_with_workspace(
    source: &JointHessianSource,
    ranges: &[(usize, usize)],
    s_lambdas: &[Array2<f64>],
    diagonal_ridge: f64,
    vector: &Array1<f64>,
    out: &mut Array1<f64>,
    penalty_scratch: &mut Array1<f64>,
    joint_full_width: Option<&crate::families::joint_penalty::JointPenaltyBundle>,
) -> Result<(), String> {
    match source {
        JointHessianSource::Dense(h_joint) => {
            crate::faer_ndarray::fast_av_view_into(h_joint, vector, out.view_mut());
        }
        JointHessianSource::Operator { apply_into, .. } => {
            apply_into(vector, out)?;
        }
    }
    penalty_scratch.fill(0.0);
    apply_joint_block_penalty_into(
        ranges,
        s_lambdas,
        vector,
        diagonal_ridge,
        penalty_scratch,
        joint_full_width,
    );
    *out += &*penalty_scratch;
    Ok(())
}


fn stabilized_joint_solver_diagonal_ridge<F: CustomFamily + ?Sized>(
    family: &F,
    source: &JointHessianSource,
    ranges: &[(usize, usize)],
    s_lambdas: &[Array2<f64>],
    base_diagonal_ridge: f64,
    ridge_floor: f64,
    joint_full_width: Option<&crate::families::joint_penalty::JointPenaltyBundle>,
) -> f64 {
    if use_exact_newton_strict_spd(family) {
        return base_diagonal_ridge;
    }
    let JointHessianSource::Dense(h_joint) = source else {
        return base_diagonal_ridge;
    };
    let mut lhs = h_joint.clone();
    add_joint_penalty_to_matrix(
        &mut lhs,
        ranges,
        s_lambdas,
        base_diagonal_ridge,
        joint_full_width,
    );
    let shift = exact_newton_stabilizing_shift(&lhs, ridge_floor).unwrap_or(0.0);
    if shift > 0.0 {
        log::debug!(
            "[PIRLS/joint-Newton] stabilized dense penalized Hessian with diagonal shift {:.3e}",
            shift
        );
    }
    base_diagonal_ridge + shift
}


fn joint_quadratic_predicted_reduction(
    rhs: &Array1<f64>,
    hpen_delta: &Array1<f64>,
    delta: &Array1<f64>,
) -> f64 {
    rhs.dot(delta) - 0.5 * delta.dot(hpen_delta)
}


fn joint_preconditioned_descent_delta(
    source: &JointHessianSource,
    ranges: &[(usize, usize)],
    s_lambdas: &[Array2<f64>],
    diagonal_ridge: f64,
    rhs: &Array1<f64>,
    joint_full_width: Option<&crate::families::joint_penalty::JointPenaltyBundle>,
) -> Result<Array1<f64>, String> {
    let base_diagonal = match source {
        JointHessianSource::Dense(h_joint) => h_joint.diag().to_owned(),
        JointHessianSource::Operator { diagonal, .. } => diagonal.clone(),
    };
    let preconditioner = joint_penalty_preconditioner_diag(
        &base_diagonal,
        ranges,
        s_lambdas,
        diagonal_ridge,
        joint_full_width,
    );
    let mut delta = rhs / &preconditioner;
    if !delta.iter().all(|v| v.is_finite()) || rhs.dot(&delta) <= 0.0 {
        delta.assign(rhs);
    }
    let directional = rhs.dot(&delta);
    if directional.is_finite() && directional > 0.0 {
        let mut hpen_delta = Array1::<f64>::zeros(rhs.len());
        apply_joint_penalized_hessian_into(
            source,
            ranges,
            s_lambdas,
            diagonal_ridge,
            &delta,
            &mut hpen_delta,
            joint_full_width,
        )?;
        let curvature = delta.dot(&hpen_delta);
        if curvature.is_finite() && curvature > 0.0 {
            let alpha = (directional / curvature).clamp(1.0e-12, 1.0);
            delta.mapv_inplace(|v| alpha * v);
        }
    }
    Ok(delta)
}


fn joint_line_search_log_likelihood<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    line_search_options: &BlockwiseFitOptions,
    states: &[ParameterBlockState],
) -> Result<(f64, Option<Arc<dyn ExactNewtonJointHessianWorkspace>>), String> {
    family
        .log_likelihood_only_with_options(states, line_search_options)
        .map(|log_likelihood| (log_likelihood, None))
}


fn coefficient_line_search_options(
    options: &BlockwiseFitOptions,
    early_exit_threshold: f64,
) -> BlockwiseFitOptions {
    let mut line_search_options = options.clone();
    // Preserve `outer_score_subsample` so the trial-objective and the
    // Hessian/gradient share a row measure: the trust-region ratio
    // ρ = [F(β) − F(β + δ)] / [−g·δ − ½·δᵀHδ] is only valid when
    // numerator and denominator evaluate the same measure. Disable
    // *auto*-install so no mid-iteration mask rebuild can occur, and
    // tag scope=InnerCoefficient so any sibling auto-install path that
    // somehow gets reached bails out (cf. `install_auto_outer_subsample_options`).
    line_search_options.auto_outer_subsample = false;
    line_search_options.outer_eval_context =
        options
            .outer_eval_context
            .as_ref()
            .map(|ctx| OuterEvalContext {
                rho: ctx.rho.clone(),
                eval_id: ctx.eval_id,
                scope: EvalScope::InnerCoefficient,
            });
    line_search_options.early_exit_threshold = Some(early_exit_threshold);
    line_search_options
}


type JointGradientLoad = (
    f64,
    Option<Array1<f64>>,
    Option<FamilyEvaluation>,
    Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
);


fn load_joint_gradient_evaluation<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    states: &[ParameterBlockState],
    prefer_workspace: bool,
    preferred_workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
) -> Result<JointGradientLoad, String> {
    let workspace = match preferred_workspace {
        Some(workspace) => Some(workspace),
        None if prefer_workspace && family.inner_joint_workspace_gradient_available(specs) => {
            family.exact_newton_joint_hessian_workspace_with_options(states, specs, options)?
        }
        None => None,
    };
    if let Some(workspace_ref) = workspace.as_ref()
        && let Some(joint_eval) = workspace_ref.joint_gradient_evaluation()?
    {
        return Ok((
            joint_eval.log_likelihood,
            Some(joint_eval.gradient),
            None,
            Some(Arc::clone(workspace_ref)),
        ));
    }
    if let Some(joint_eval) = family.exact_newton_joint_gradient_evaluation(states, specs)? {
        return Ok((
            joint_eval.log_likelihood,
            Some(joint_eval.gradient),
            None,
            workspace,
        ));
    }
    let eval = family.evaluate(states)?;
    let log_likelihood = eval.log_likelihood;
    let gradient = exact_newton_joint_gradient_from_eval(&eval, specs, states)?;
    Ok((log_likelihood, gradient, Some(eval), workspace))
}


fn require_projected_kkt_residual(
    residual: Option<ProjectedKktResidual>,
    context: &str,
) -> Result<ProjectedKktResidual, String> {
    match residual {
        Some(residual) => Ok(residual),
        None => Err(CustomFamilyError::UnsupportedConfiguration { reason: format!(
            "{context}: converged joint-Newton exact inner solve did not produce a projected KKT \
             residual; refusing to assemble REML/LAML derivatives without the IFT correction input"
        ) }.into()),
    }
}


#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ConstrainedStationaryCertificate {
    NotCandidate,
    Accept,
    RefusePhantomMultiplier,
}


#[derive(Clone, Debug)]
struct JointNewtonMathDiagnostic {
    old_kkt_inf: f64,
    linearized_next_kkt_inf: f64,
    predicted_reduction: f64,
    actual_reduction: f64,
    trust_ratio: f64,
    step_inf: f64,
    proposal_inf: f64,
}


impl JointNewtonMathDiagnostic {
    fn scalar_model_relative_error(&self) -> f64 {
        (self.actual_reduction - self.predicted_reduction).abs()
            / self.predicted_reduction.abs().max(1.0)
    }

    fn linearized_rel(&self) -> f64 {
        self.linearized_next_kkt_inf / (1.0 + self.old_kkt_inf)
    }
}


fn constrained_stationary_certificate_decision(
    math: &JointNewtonMathDiagnostic,
    objective_change: f64,
    objective_tol: f64,
    step_tol: f64,
    geometric_tail_bound: Option<f64>,
    residual: f64,
    residual_tol: f64,
) -> ConstrainedStationaryCertificate {
    let linearized_rel = math.linearized_rel();
    let scalar_model_relerr = math.scalar_model_relative_error();
    let objective_exhausted = objective_change <= objective_tol
        || geometric_tail_bound.is_some_and(|tail| tail <= objective_tol);
    let step_exhausted =
        math.step_inf.is_finite() && step_tol.is_finite() && math.step_inf <= step_tol;

    if !(objective_exhausted
        && step_exhausted
        && linearized_rel >= 0.5
        && scalar_model_relerr <= 1e-3)
    {
        return ConstrainedStationaryCertificate::NotCandidate;
    }

    // A large linearized residual can mean either an honest active-set
    // multiplier or an H-null/rank-deficient direction that Newton cannot
    // move. Only the projected KKT residual distinguishes those cases. This
    // small tolerance band is intentionally tied to the inner residual
    // tolerance, because this branch is allowed to certify convergence only
    // when the active-set projection has actually captured the multiplier.
    //
    // The band is a small MULTIPLE of `residual_tol`, not exactly `1x`: this
    // branch fires only once the iterate is already proven stationary (objective
    // exhausted, step exhausted, `linearized_rel >= 0.5` so the residual is
    // multiplier/null mass not a gradient defect, `scalar_relerr <= 1e-3` so the
    // quadratic model is exact). There the active-projected residual stalls at the
    // conditioning/round-off floor — for the survival baseline-hazard block
    // (well-conditioned after the data-seeded baseline, gam#797) it floors a hair
    // above the scale-relative `residual_tol`, so demanding exactly `<= tol` leaves
    // a fully-stationary iterate uncertified. A `4x` band certifies the genuinely
    // converged iterate while still rejecting a residual orders of magnitude above
    // tolerance (a real defect), the only case this guard must catch.
    let cert_residual_factor = 4.0;
    if residual.is_finite() && residual <= cert_residual_factor * residual_tol {
        ConstrainedStationaryCertificate::Accept
    } else {
        ConstrainedStationaryCertificate::RefusePhantomMultiplier
    }
}


/// True iff the recent KKT-residual tail (`history`, oldest→newest) shows STEADY
/// geometric descent: every consecutive pair strictly decreased by at least the
/// factor `(1 - min_drop)` over the whole window.
///
/// This distinguishes a still-converging Newton direction from a genuine
/// multiplier/null plateau at the certificate-refusal gate (gam#787 duchon
/// centers≥20). The constrained-stationary refusal fires on a flat objective +
/// `linearized_rel ≥ 0.5`, but those signals ALSO hold for a logslope block
/// whose residual is dropping by a steady factor each cycle (objective already
/// at its Φ-bounded floor while the KKT residual still polishes): refusing there
/// rejects the seed a few cycles short of `residual_tol`. Requiring a STEADY
/// drop over `≥ window` cycles (not a single lucky decrease) keeps a noisy
/// near-plateau from being falsely extended, and the inner cycle cap still
/// bounds the extra work.
fn residual_in_steady_geometric_descent(history: &std::collections::VecDeque<f64>) -> bool {
    let window = history.len();
    if window < 3 {
        return false;
    }
    let min_drop = 0.1; // each cycle must cut the residual by ≥ 10%.
    history
        .iter()
        .zip(history.iter().skip(1))
        .all(|(prev, next)| {
            prev.is_finite() && next.is_finite() && *prev > 0.0 && *next < (1.0 - min_drop) * *prev
        })
}


/// Inf-norm of the active-set-projected stationarity residual restricted to the
/// **range** of the joint penalized Hessian `H_pen = H + S(λ) + ridge·I`.
///
/// A penalized smooth whose penalty has a polynomial null space the censored /
/// location-scale data does not pin down (TP / Bernstein trend directions in a
/// survival `time_transform` or `log_sigma` channel, gam#553) leaves a residual
/// that lives entirely in `ker(H_pen)`: along that direction the objective has
/// neither curvature nor a constraint, so it is a genuinely *free* gauge
/// direction, not an unresolved KKT defect. The total residual inf-norm then
/// stays large forever and the phantom-multiplier refusal never clears, aborting
/// the fit at REML startup even though the iterate is stationary on the entire
/// identifiable (range) subspace.
///
/// The downstream outer IFT trace already removes the null-space component via
/// the projected pseudo-inverse `U_S·H_proj⁻¹·U_Sᵀ`, so only a *range-space*
/// residual component can bias the envelope gradient (see the "do NOT
/// soft-accept" investigation note at the certifier call site). This returns the
/// range-space inf-norm so the certifier can accept iff that — the only part
/// that matters for outer correctness — is at tolerance, while a real defect
/// (residual with mass in the curved subspace) still refuses.
///
/// Returns `None` when the penalized Hessian cannot be materialized or
/// eigendecomposed, or carries no numerical null space — in which case the
/// caller keeps the strict total-residual refusal (no null space ⇒ range = all).
fn projected_residual_range_space_inf(
    projected_residual: &Array1<f64>,
    joint_hessian_source: &JointHessianSource,
    ranges: &[(usize, usize)],
    s_lambdas: &[Array2<f64>],
    ridge: f64,
    ridge_policy: RidgePolicy,
    total_p: usize,
) -> Option<f64> {
    if total_p == 0 || projected_residual.len() != total_p {
        return None;
    }
    let mut h_joint = materialize_joint_hessian_source(
        joint_hessian_source,
        total_p,
        "penalty-null-space certificate spectrum",
    )
    .ok()?;
    let model_diagonal_ridge = if ridge_policy.include_quadratic_penalty && ridge > 0.0 {
        ridge
    } else {
        0.0
    };
    add_joint_penalty_to_matrix(&mut h_joint, ranges, s_lambdas, model_diagonal_ridge, None);
    symmetrize_dense_in_place(&mut h_joint);
    let (evals, evecs) = FaerEigh::eigh(&h_joint, Side::Lower).ok()?;
    let max_abs = evals.iter().map(|x: &f64| x.abs()).fold(0.0_f64, f64::max);
    if !(max_abs.is_finite() && max_abs > 0.0) {
        return None;
    }
    let cutoff = KKT_REFUSAL_RANK_TOL * max_abs;
    let nullity = evals.iter().filter(|x| x.abs() < cutoff).count();
    if nullity == 0 {
        // No data-unconstrained null space — the range is the whole space, so
        // the strict total-residual refusal already governs. Signal "no relief".
        return None;
    }
    // Range-space residual = residual minus its projection onto ker(H_pen).
    // Equivalently, accumulate the residual's coordinates along every
    // range-space (|λ| ≥ cutoff) eigenvector. The eigenbasis is orthonormal,
    // so ‖P_range r‖∞ is read off the reconstructed range component.
    let mut range_component = Array1::<f64>::zeros(total_p);
    for k in 0..evals.len() {
        if evals[k].abs() < cutoff {
            continue;
        }
        let coeff = evecs.column(k).dot(projected_residual);
        range_component.scaled_add(coeff, &evecs.column(k));
    }
    Some(
        range_component
            .iter()
            .map(|x: &f64| x.abs())
            .fold(0.0_f64, f64::max),
    )
}
