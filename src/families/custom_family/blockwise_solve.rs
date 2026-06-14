//! The inner block-coordinate solve: per-block updaters (diagonal / exact-Newton),
//! weighted normal equations, linear-constraint + lower-bound assembly and active
//! sets, block penalty/metric helpers, the total-quadratic-penalty objective, and
//! the SPD logdet / strict-solve / pseudo-inverse numeric kernels. Also the
//! labeled-rho aggregation/pullback helpers that drive the outer eval.

use super::*;

pub(crate) fn aggregate_labeled_hessian(
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
pub(crate) fn rho_prior_cost_gradient_hessian(
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

pub(crate) fn add_labeled_rho_prior_to_outer_eval(
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

pub(crate) fn physical_warm_start_for_labeled(
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

pub(crate) fn pullback_labeled_outer_eval(
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

pub(crate) fn outerobjectivegradienthessian_labeled<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
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

pub(crate) fn custom_family_seed_screening_proxy_labeled<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
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

pub(crate) fn split_log_lambdas(
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

pub(crate) fn buildblock_states<F: CustomFamily + Clone + Send + Sync + 'static>(
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

pub(crate) fn refresh_all_block_etas<F: CustomFamily + Clone + Send + Sync + 'static>(
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

pub(crate) fn refresh_single_block_eta<F: CustomFamily + Clone + Send + Sync + 'static>(
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
pub(crate) fn capped_inner_max_cycles(options: &BlockwiseFitOptions, base_cycles: usize) -> usize {
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

pub(crate) fn weighted_normal_equations(
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
pub(crate) fn exact_newton_stabilizing_shift(
    lhs_dense: &Array2<f64>,
    ridge_floor: f64,
) -> Option<f64> {
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

pub(crate) fn stabilize_exact_newton_lhs_in_place<F: CustomFamily + ?Sized>(
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

pub(crate) fn shift_linear_constraints_to_delta(
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

pub(crate) fn collect_block_linear_constraints<F: CustomFamily + ?Sized>(
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

pub(crate) fn reject_constrained_post_update_repair(
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

pub(crate) fn assemble_joint_linear_constraints(
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

pub(crate) fn flatten_joint_active_set(
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

pub(crate) fn scatter_joint_active_set(
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
pub(crate) fn assemble_active_constraint_block(
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

pub(crate) struct SimpleLowerBounds {
    lower_bounds: Array1<f64>,
    row_to_coeff: Vec<usize>,
    coeff_to_row: Vec<Option<usize>>,
}

pub(crate) fn extract_simple_lower_bounds(
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

pub(crate) fn lower_bound_active_rows_to_coeffs(
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

pub(crate) fn lower_bound_active_coeffs_to_rows(
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

pub(crate) fn lower_bound_active_coeffs_from_solution(
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

pub(crate) fn project_to_lower_bounds(beta: &mut Array1<f64>, lower_bounds: &Array1<f64>) {
    for i in 0..beta.len() {
        let lower = lower_bounds[i];
        if lower.is_finite() && beta[i] < lower {
            beta[i] = lower;
        }
    }
}

pub(crate) fn solve_quadratic_with_simple_lower_bounds(
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

pub(crate) fn normalize_active_set(mut active_set: Vec<usize>) -> Option<Vec<usize>> {
    active_set.sort_unstable();
    active_set.dedup();
    if active_set.is_empty() {
        None
    } else {
        Some(active_set)
    }
}

pub(crate) fn normalize_active_sets(
    active_sets: Vec<Option<Vec<usize>>>,
) -> Vec<Option<Vec<usize>>> {
    active_sets
        .into_iter()
        .map(|active_set| active_set.and_then(normalize_active_set))
        .collect()
}

pub(crate) struct BlockUpdateContext<'a> {
    family: &'a dyn CustomFamily,
    states: &'a [ParameterBlockState],
    spec: &'a ParameterBlockSpec,
    block_idx: usize,
    s_lambda: &'a Array2<f64>,
    options: &'a BlockwiseFitOptions,
    linear_constraints: Option<&'a LinearInequalityConstraints>,
    cached_active_set: Option<&'a [usize]>,
}

pub(crate) struct BlockUpdateResult {
    beta_new_raw: Array1<f64>,
    active_set: Option<Vec<usize>>,
}

#[inline]
pub(crate) fn floor_positiveworking_weights(
    working_weights: &Array1<f64>,
    minweight: f64,
) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(working_weights.len());
    ndarray::Zip::from(&mut out)
        .and(working_weights)
        .par_for_each(|o, &wi| *o = if wi <= 0.0 { 0.0 } else { wi.max(minweight) });
    out
}

pub(crate) trait ParameterBlockUpdater {
    fn compute_update_step(
        &self,
        ctx: &BlockUpdateContext<'_>,
    ) -> Result<BlockUpdateResult, String>;
}

pub(crate) struct DiagonalBlockUpdater<'a> {
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

pub(crate) struct ExactNewtonBlockUpdater<'a> {
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

pub(crate) fn check_linear_feasibility(
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
pub(crate) fn effective_solverridge(ridge_floor: f64) -> f64 {
    ridge_floor.max(1e-15)
}

pub(crate) fn block_quadratic_penalty(
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

pub(crate) fn block_penalized_hessian_vector(
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

pub(crate) fn symmetric_matrix_diagonal(matrix: &SymmetricMatrix) -> Array1<f64> {
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

pub(crate) fn block_penalized_metric_diagonal(
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

pub(crate) fn block_penalized_metric_norm(
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

pub(crate) fn truncate_block_step_to_metric_radius(
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

pub(crate) const TOTAL_QUADRATIC_PENALTY_PAR_MIN_BLOCKS: usize = 4;

// Avoid Rayon overhead for a few tiny blocks; this approximates the dense
// mat-vec work in βᵀSβ before splitting independent block penalties.
pub(crate) const TOTAL_QUADRATIC_PENALTY_PAR_MIN_DENSE_WORK: usize = 16_384;

pub(crate) fn total_quadratic_penalty_parallel_worthwhile(
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

pub(crate) fn total_quadratic_penalty(
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
pub(crate) fn smooth_regularized_logdet_hessian_finite_check(
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
pub(crate) fn validate_block_hessians_finite(eval: &FamilyEvaluation) -> Result<(), String> {
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

pub(crate) fn stable_logdet_with_ridge_policy(
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
pub(crate) fn try_cholesky_with_escalating_ridge<R>(
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
pub(crate) fn penalty_logdet_cholesky_fallback(
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

pub(crate) fn resolved_ridge_determinant_mode(
    ridge_policy: RidgePolicy,
    dim: usize,
) -> RidgeDeterminantMode {
    assert!(
        dim.checked_add(1).is_some(),
        "ridge determinant dimension overflow"
    );
    match ridge_policy.determinant_mode {
        RidgeDeterminantMode::Auto => RidgeDeterminantMode::Full,
        mode => mode,
    }
}

pub(crate) fn inverse_spdwith_retry(
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

pub(crate) fn validate_flat_direction_length(
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

pub(crate) fn strict_solve_spd(
    matrix: &Array2<f64>,
    rhs: &Array1<f64>,
) -> Result<Array1<f64>, String> {
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
pub(crate) const STRICT_SPD_LM_MAX_ESCALATIONS: usize = 16;

pub(crate) const STRICT_SPD_LM_RIDGE_GROWTH: f64 = 10.0;

/// Floor applied to IRLS working weights so downstream divisions cannot hit
/// exact zero. Used as the default `minweight` in `CustomFamilyOptions` and
/// mirrored in tests that override it.
///
/// Sourced from the canonical PIRLS positive-weight floor
/// ([`crate::solver::pirls::MIN_WEIGHT`] = `1e-12`) so every floored family
/// shares one definition; this alias keeps the descriptive local name at the
/// `minweight` defaults.
pub(crate) const CUSTOM_FAMILY_WEIGHT_FLOOR: f64 = crate::solver::pirls::MIN_WEIGHT;

/// Default initial ridge δ for the explicit-stabilization Cholesky escalation
/// schedule. Enters the quadratic term, the Laplace Hessian, and the penalty
/// log-determinant via the active `RidgePolicy`.
pub(crate) const CUSTOM_FAMILY_RIDGE_FLOOR: f64 = 1e-12;

/// Relative eigenvalue floor used wherever an eigendecomposition needs to
/// distinguish "real" curvature from noise: `eps_floor = EVAL_FLOOR · max|λ|`.
/// Applied uniformly in the strict-SPD LM eigen fallback, positive-part
/// pseudo-inverse, and penalty-direction projection.
pub(crate) const CUSTOM_FAMILY_EVAL_FLOOR: f64 = 1e-12;

/// Absolute relative-condition guard used to prevent the eigen / spectral
/// floors from collapsing to zero when `max|λ|` is itself tiny. Combined with
/// `CUSTOM_FAMILY_EVAL_FLOOR · max|λ|` via `.max(...)`.
pub(crate) const CUSTOM_FAMILY_CONDITION_RELATIVE_FLOOR: f64 = 1e-14;

/// Shared engine: try the bare strict path, fall through to an escalating
/// LM δ-ridge Cholesky, and finally an eigen-floor fallback that clamps every
/// eigenvalue from below at `eps_floor = 1e-12 · max|λ|`. Each caller
/// (solve / inverse / logdet) supplies the three operation-specific closures.
///
/// Centralizing the LM/eigen scaffolding here both removes ~180 lines of
/// near-duplicated code and guarantees the three sibling helpers stay in
/// lockstep — any future change to the schedule, the trace_scale heuristic,
/// or the eigen-floor logic now lives in exactly one place.
pub(crate) fn strict_spd_lm_engine<R>(
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
pub(crate) fn strict_exact_pseudo_logdet(
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

pub(crate) fn pinv_positive_part(
    matrix: &Array2<f64>,
    ridge_floor: f64,
) -> Result<Array2<f64>, String> {
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
pub(crate) fn symmetric_psd_projection(matrix: &Array2<f64>) -> Array2<f64> {
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

/// Modified-Newton convexification of a symmetric (penalized) Hessian: reflect
/// every negative-curvature eigen-direction to its magnitude `|λ|` and floor the
/// remaining (near-null) modes to a small positive multiple of `λmax`, returning
/// a positive-definite matrix with the SAME eigenvectors.
///
/// This is the exact negative-curvature handling the unconstrained dense-spectral
/// path already performs inside `WhitenedHessianSpectrum::assemble` (negative `γ`
/// reflected to `|γ|`). The CONSTRAINED active-set QP branch, by contrast, feeds
/// the raw penalized Hessian to `solve_quadratic_with_linear_constraints`. On the
/// survival marginal-slope flat baseline-hazard λ valley the EXACT joint NLL
/// Hessian is INDEFINITE away from the optimum (the linear baseline + the
/// z·exp(logslope) cross-coupling carry genuine negative curvature there). An
/// indefinite QP model has a direction that lowers the local quadratic objective
/// while moving AWAY from the KKT point, so the trust region — which gates on the
/// objective-reduction ratio ρ, not the stationarity residual — happily accepts
/// step after step at ρ≈1 and GROWS its radius while the stationarity residual
/// diverges (`per_block_resid[time]` 3.5e4 → 9.5e6 over 11 cycles, the gam#1040 /
/// gam#979 divergence). The self-vanishing Levenberg μ cannot rescue this: a
/// μ·I shift that is tiny relative to the most-negative eigenvalue leaves the
/// model indefinite, and a μ large enough to flip it would bias the converged β.
///
/// Reflecting (not merely clamping-to-zero as `symmetric_psd_projection` does)
/// preserves the curvature MAGNITUDE on negative modes, so the modified-Newton
/// step length matches the dense-spectral path's and the QP stays bounded (a
/// clamp-to-zero null mode would make the QP unbounded along that direction). At
/// a genuine optimum the constrained Hessian over the identified subspace is PSD,
/// so the reflection is a no-op there and the converged β is unchanged — exactly
/// the property the dense path relies on. Falls back to the (symmetrized) input
/// if the eigendecomposition fails: the conservative undamped step, never a wrong
/// one.
pub(crate) fn symmetric_negative_curvature_reflected(matrix: &Array2<f64>) -> Array2<f64> {
    let p = matrix.nrows();
    let mut sym = matrix.clone();
    symmetrize_dense_in_place(&mut sym);
    let Ok((evals, evecs)) = FaerEigh::eigh(&sym, Side::Lower) else {
        return sym;
    };
    let lambda_max_abs = evals.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    if !(lambda_max_abs.is_finite() && lambda_max_abs > 0.0) {
        return sym;
    }
    // Positive floor for near-null modes: the same relative scale the dense
    // spectral path uses for its numerical-rank cutoff, so a flat direction
    // gets a tiny-but-strictly-positive curvature (bounded QP) instead of a
    // zero one (unbounded QP). Reflecting already makes every above-floor mode
    // positive; this only lifts the genuine null modes off zero.
    let floor = lambda_max_abs * (p as f64).sqrt() * f64::EPSILON;
    let reflected = Array1::from_iter(evals.iter().map(|lam| lam.abs().max(floor)));
    let scaled = &evecs * &reflected.view().insert_axis(ndarray::Axis(0));
    scaled.dot(&evecs.t())
}

/// Smallest (signed) eigenvalue of a symmetric matrix; `NaN` if the
/// eigendecomposition fails. Diagnostic helper for the #1040 convexification
/// trace — reads the most-negative curvature so the log can confirm whether the
/// constrained-QP Hessian is indefinite and whether the reflection engaged.
pub(crate) fn symmetric_min_eigenvalue_signed(matrix: &Array2<f64>) -> f64 {
    let mut sym = matrix.clone();
    symmetrize_dense_in_place(&mut sym);
    match FaerEigh::eigh(&sym, Side::Lower) {
        Ok((evals, _)) => evals.iter().copied().fold(f64::INFINITY, f64::min),
        Err(_) => f64::NAN,
    }
}

pub(crate) fn symmetric_penalized_hessian_nullity(lhs: &Array2<f64>) -> Option<usize> {
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
pub(crate) fn symmetric_penalized_hessian_nullity_and_condition(
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
