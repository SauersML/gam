//! The inner block-coordinate solve: per-block updaters (diagonal / exact-Newton),
//! weighted normal equations, linear-constraint + lower-bound assembly and active
//! sets, block penalty/metric helpers, the total-quadratic-penalty objective, and
//! the SPD logdet / strict-solve / pseudo-inverse numeric kernels. Also the
//! labeled-rho aggregation/pullback helpers that drive the outer eval.

use super::*;
use opt::{RidgeSchedule, escalate_ridge};

/// Convert one already-semantic log-smoothing vector to physical strengths.
/// This is the only custom-family conversion seam: it rejects the first bad
/// coordinate and never clamps or floors either representation.
pub(crate) fn exact_lambdas_from_log_strengths(
    log_strengths: &Array1<f64>,
    label: &str,
) -> Result<Array1<f64>, CustomFamilyError> {
    gam_problem::checked_exp_log_strengths(log_strengths.iter().copied())
        .map(Array1::from_vec)
        .map_err(|error| CustomFamilyError::ConstraintViolation {
            reason: format!("{label}: {error}"),
        })
}

pub(crate) fn exact_lambdas_by_block(
    block_log_strengths: &[Array1<f64>],
    label: &str,
) -> Result<Vec<Array1<f64>>, CustomFamilyError> {
    block_log_strengths
        .iter()
        .enumerate()
        .map(|(block, values)| {
            exact_lambdas_from_log_strengths(values, &format!("{label} block {block}"))
        })
        .collect()
}

pub(crate) fn aggregate_labeled_hessian(
    hessian: &Array2<f64>,
    layout: &PenaltyLabelLayout,
) -> Result<Array2<f64>, String> {
    // gam#1587: the evaluator Hessian indexes the per-block physical coords
    // followed by the appended joint coords. Build the unified physical→outer map
    // over both ranges (joint always maps to a concrete outer coord).
    let n_joint = layout.joint_specs.len();
    let expected = layout.physical_count() + n_joint;
    if hessian.nrows() != expected || hessian.ncols() != expected {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: format!(
                "physical Hessian shape mismatch: got {}x{}, expected {}x{} (per-block {} + joint {})",
                hessian.nrows(),
                hessian.ncols(),
                expected,
                expected,
                layout.physical_count(),
                n_joint,
            ),
        }
        .into());
    }
    let to_outer: Vec<Option<usize>> = layout
        .physical_to_outer
        .iter()
        .copied()
        .chain(layout.joint_to_outer.iter().map(|&o| Some(o)))
        .collect();
    let mut out = Array2::<f64>::zeros((layout.initial_rho.len(), layout.initial_rho.len()));
    for (i, oi) in to_outer.iter().enumerate() {
        let Some(oi) = *oi else { continue };
        for (j, oj) in to_outer.iter().enumerate() {
            if let Some(oj) = *oj {
                out[[oi, oj]] += hessian[[i, j]];
            }
        }
    }
    Ok(out)
}

/// Adapter over the shared [`rho_prior_eval`](gam_solve::rho_prior_eval)
/// engine using the custom-family invalid-prior policy
/// (`HardError`): the prior math is shared with the REML/LAML runtime, and a
/// malformed prior surfaces as a structured [`CustomFamilyError`] rather than
/// being folded into the objective.
pub(crate) fn rho_prior_cost_gradient_hessian(
    prior: &gam_problem::RhoPrior,
    rho: &Array1<f64>,
) -> Result<(f64, Array1<f64>, Option<Array2<f64>>), String> {
    use gam_solve::rho_prior_eval::{InvalidPriorPolicy, RhoPriorError};
    match gam_solve::rho_prior_eval::evaluate(prior, rho, InvalidPriorPolicy::HardError) {
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
    rho_prior: &gam_problem::RhoPrior,
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
    if matches!(rho_prior, gam_problem::RhoPrior::Flat) {
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
        gam_solve::objective_base::add_rho_block_dense_to_hessian(
            &mut result.outer_hessian,
            &prior_hessian,
        )?;
    }
    Ok(result)
}

pub(crate) fn physical_warm_start_for_labeled(
    warm_start: Option<&ConstrainedWarmStart>,
    physical_rho: &Array1<f64>,
    layout: &PenaltyLabelLayout,
) -> Option<ConstrainedWarmStart> {
    if !layout.physical_rho_requires_remap() {
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
    rho_prior: &gam_problem::RhoPrior,
    eval_mode: EvalMode,
) -> Result<OuterObjectiveEvalResult, String> {
    if eval_mode == EvalMode::ValueOnly {
        result.gradient = Array1::<f64>::zeros(layout.initial_rho.len());
    } else {
        result.gradient = aggregate_labeled_gradient(&result.gradient, layout)?;
    }
    if eval_mode == EvalMode::ValueGradientHessian {
        result.outer_hessian = match result.outer_hessian {
            gam_problem::HessianValue::Dense(hessian) => {
                gam_problem::HessianValue::Dense(aggregate_labeled_hessian(&hessian, layout)?)
            }
            gam_problem::HessianValue::Operator(operator) => gam_problem::HessianValue::Operator(
                Arc::new(LabeledHessianOperator::new(operator, layout)),
            ),
            gam_problem::HessianValue::Unavailable => gam_problem::HessianValue::Unavailable,
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
    rho_prior: &gam_problem::RhoPrior,
    eval_mode: EvalMode,
) -> Result<OuterObjectiveEvalResult, String> {
    let physical_rho = expand_labeled_log_lambdas(rho, layout)?;
    let physical_warm_start = physical_warm_start_for_labeled(warm_start, &physical_rho, layout);
    // gam#1587: build the per-eval joint penalty bundle from the current outer ρ
    // (each joint spec's λ pulled from its tied outer coordinate) and attach it
    // to the inner-solve options so BOTH the inner β̂ AND the outer evaluator
    // (penalty coords / logdet / operator) see the full-width centered penalty.
    // No joint specs ⇒ `options` is passed through untouched (byte-identical).
    let options_with_joint;
    let options: &BlockwiseFitOptions = if layout.joint_specs.is_empty() {
        options
    } else {
        let total_compiled: usize = specs.iter().map(|s| s.design.ncols()).sum();
        let joint_log_lambdas = layout.joint_log_lambdas(rho);
        let bundle = gam_problem::JointPenaltyBundle::new(
            std::sync::Arc::new(layout.joint_specs.clone()),
            joint_log_lambdas,
            total_compiled,
        )?;
        let mut cloned = options.clone();
        cloned.joint_penalties = Some(std::sync::Arc::new(bundle));
        options_with_joint = cloned;
        &options_with_joint
    };
    let base = outerobjectivegradienthessian_internal(
        family,
        specs,
        options,
        &layout.penalty_counts,
        &physical_rho,
        physical_warm_start.as_ref().or(warm_start),
        gam_problem::RhoPrior::Flat,
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
    rho_prior: &gam_problem::RhoPrior,
) -> Result<(f64, ConstrainedWarmStart, bool), String> {
    let physical_rho = expand_labeled_log_lambdas(rho, layout)?;
    let per_block = split_log_lambdas(&physical_rho, &layout.penalty_counts)?;
    let physical_warm_start = physical_warm_start_for_labeled(warm_start, &physical_rho, layout);
    // Seed screening only RANKS candidate seeds by their penalized inner merit;
    // it is capped and never produces the final fit. Mark the inner solve as a
    // screening solve so it skips the O(p · per-axis-Hdot) full Jeffreys/Firth
    // curvature loop and keeps only the cheap value-only Jeffreys term in the
    // score (gam#729/#808). For a K-block coupled family (Dirichlet/multinomial)
    // each per-axis directional derivative is O(K²·n·p), so paying the full term
    // for every cascade candidate over the joint width is the wrong cost class
    // and made the coupled fit non-completing in screening alone. The real fit
    // (after a seed is selected) runs with `seed_screening = false`, so the
    // load-bearing Firth curvature is fully present where it matters.
    let mut screening_options = BlockwiseFitOptions {
        seed_screening: true,
        ..options.clone()
    };
    // gam#1587: the screening inner solve must apply the same full-width joint
    // penalty so the ranked proxy objective matches the real penalized fit.
    if !layout.joint_specs.is_empty() {
        let total_compiled: usize = specs.iter().map(|s| s.design.ncols()).sum();
        let joint_log_lambdas = layout.joint_log_lambdas(rho);
        let bundle = gam_problem::JointPenaltyBundle::new(
            std::sync::Arc::new(layout.joint_specs.clone()),
            joint_log_lambdas,
            total_compiled,
        )?;
        screening_options.joint_penalties = Some(std::sync::Arc::new(bundle));
    }
    let mut inner = inner_blockwise_fit(
        family,
        specs,
        &per_block,
        &screening_options,
        physical_warm_start.as_ref().or(warm_start),
    )?;
    refresh_all_block_etas(family, specs, &mut inner.block_states)?;
    let prior_terms = rho_prior_cost_gradient_hessian(rho_prior, rho)?;
    // A capped screening iterate is deliberately not a certified coefficient
    // mode. Its Laplace determinants are therefore undefined, and using them
    // here would either rank noisy partial-fit curvature or (correctly) fail
    // once determinant construction is restricted to converged modes. Rank on
    // the same curvature-free penalized merit minimized by the inner solver.
    // Accepted inner steps decrease this quantity, so the terminal capped
    // iterate is a meaningful seed-quality signal. Full outer evaluations keep
    // requiring convergence and certified REML/LAML determinants.
    let score = checked_penalizedobjective(
        inner.log_likelihood,
        inner.penalty_value,
        0.0,
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
    // Certify the complete vector before producing any partial block output.
    // Every downstream physical conversion is therefore dominated by the
    // shared exact-domain contract even when it operates block-by-block.
    gam_problem::validate_log_strengths(flat.iter().copied()).map_err(|error| {
        CustomFamilyError::ConstraintViolation {
            reason: format!("log-smoothing vector: {error}"),
        }
    })?;
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

    let xtwx = x.xt_diag_x_signed_op(FiniteSignedWeightsView::try_from_array(w)?)?;
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
/// Stabilizing shift for a penalized joint Hessian `combined = H_data + S` whose
/// penalty `S` is positive-semidefinite by construction.
///
/// Because `S ⪰ 0`, Weyl's inequality gives `λ_min(H_data + S) ≥ λ_min(H_data)`,
/// so the lifting ridge needed to make `combined` PD is bounded by the curvature
/// of the *data* Hessian alone. We therefore take the Gershgorin lower bound on
/// `H_data` (the `gershgorin_src`) rather than on `combined`, while still using
/// `combined` for the PD Cholesky certificate.
///
/// Why this is not a micro-optimization (gam#979 survival marginal-slope hang):
/// Gershgorin's bound `min_i (H_ii − Σ_{j≠i}|H_ij|)` is only tight when the
/// off-diagonals are small relative to the diagonal. A heavily over-smoothed
/// penalty has large symmetric off-diagonals that are *balanced* by equally large
/// diagonals — the matrix is exactly PSD, but the per-row `diag − radius` can be
/// hugely negative. On the survival marginal-slope pilot the time-block penalty
/// reaches `λ ≈ 6e7`, so Gershgorin on `combined` returned `≈ −1.2e7` even though
/// the assembled penalty's true `λ_min` is `+1e-10`. The old `δ = floor − g`
/// shift then added a `~1.2e7` ridge — `~550×` the data curvature (`~2e4`) — and
/// every inner Newton step shrank to `g/(H+μ) ≈ 1e-4`, so the coupled solve
/// crawled `30+` cycles without ever certifying KKT convergence and the fit hung.
/// Bounding the shift by the data Hessian instead collapses the ridge to
/// `O(data scale)`, restoring proper Newton steps and prompt convergence, while
/// remaining a guaranteed PD certificate: `λ_min(combined + δI) ≥ λ_min(H_data)
/// + δ ≥ g + (floor − g) = floor > 0`.
pub(crate) fn exact_newton_stabilizing_shift_psd_penalized(
    combined: &Array2<f64>,
    gershgorin_src: &Array2<f64>,
    ridge_floor: f64,
) -> Option<f64> {
    stabilizing_shift_core(combined, gershgorin_src, ridge_floor)
}

/// Shared engine for the stabilizing-shift helper. `cholesky_test` is the matrix
/// that must end up positive definite; `gershgorin_src` is the matrix whose
/// Gershgorin disc lower-bounds `λ_min`.
fn stabilizing_shift_core(
    cholesky_test: &Array2<f64>,
    gershgorin_src: &Array2<f64>,
    ridge_floor: f64,
) -> Option<f64> {
    let floor = effective_solverridge(ridge_floor);
    // Fast path: already PD at zero shift ⇒ no stabilization needed. One Cholesky
    // (O(p³/3)), the common case on a well-conditioned cycle.
    if cholesky_test.cholesky(Side::Lower).is_ok() {
        return None;
    }
    // Near-singular / indefinite. We need a positive diagonal shift `δ` that makes
    // `H + δI` PD. A full eigendecomposition (the previous implementation) reads
    // the exact `λ_min` but costs `O(p³)` for ALL eigenpairs EVERY inner cycle;
    // for a coupled K-block family the shift fires almost every cycle, so that
    // dominated the inner solve (gam#729/#826).
    //
    // The Gershgorin lower bound on `λ_min` — a single `O(p²)` pass, every
    // eigenvalue lies in some disc `[H_ii − R_i, H_ii + R_i]` with
    // `R_i = Σ_{j≠i} |H_ij|`, so `λ_min ≥ min_i (H_ii − R_i) =: g` — gives a
    // *guaranteed-PD* shift `floor − g` in one pass. But on a dense, coupled data
    // Hessian (e.g. the survival marginal/logslope aliasing of gam#979, where the
    // marginal and logslope smooths share covariates and every row is full) the
    // disc radius `R_i` is enormous relative to the true spectrum, so `g` sits
    // *far* below the actual `λ_min` and `floor − g` over-shifts by an order of
    // magnitude. That inflated ridge does NOT just guarantee positive-definiteness
    // — it damps the Newton step `(H_pen + δI)⁻¹ g` in exactly the low-curvature
    // coupled directions that carry the residual, collapsing the joint-Newton
    // contraction to a slow linear crawl (persistent gain-ratio > 1 with interior,
    // never-trust-clamped steps: the ridge, not the trust region, is throttling the
    // step). The stabilizer's only job is PD-ness; step-size control belongs to the
    // trust region, so the ridge must be the *minimal* one that restores PD.
    //
    // Recover a near-minimal shift without an `O(p³)`-per-eigenpair eigh: use the
    // Gershgorin shift only as a guaranteed-PD upper bracket and bisect the PD
    // frontier with Cholesky. `cholesky(H + δI)` succeeds iff `δ > −λ_min(H)`, a
    // monotone step in `δ`, so a handful of bisections between the known-indefinite
    // `δ = 0` (the fast-path Cholesky above already failed) and the known-PD
    // Gershgorin bracket squeeze `δ` to within `2⁻ⁿ` of the minimal PD shift. Each
    // step is one `O(p³/3)` Cholesky; the iteration count is capped and only runs
    // on the indefinite cycles the fast path did not already clear, so the cost is
    // bounded and self-vanishing. The final `+ floor` restores the `≥ floor`
    // positive-definiteness margin the downstream solve relies on.
    let p = gershgorin_src.nrows();
    let mut gershgorin_min = f64::INFINITY;
    for i in 0..p {
        let diag = gershgorin_src[[i, i]];
        let mut radius = 0.0_f64;
        for j in 0..p {
            if j != i {
                radius += gershgorin_src[[i, j]].abs();
            }
        }
        gershgorin_min = gershgorin_min.min(diag - radius);
    }
    if !gershgorin_min.is_finite() {
        let diag_max = (0..cholesky_test.nrows())
            .map(|d| cholesky_test[[d, d]].abs())
            .fold(0.0_f64, f64::max);
        return Some(floor.max(diag_max * 1e-6).max(1e-6));
    }
    if gershgorin_min >= floor {
        // Gershgorin certifies PD-at-floor but the no-shift Cholesky failed
        // (round-off on a barely-PD matrix): a floor-sized shift suffices.
        return Some(floor);
    }
    // Guaranteed-PD upper bracket: `λ_min(cholesky_test + (floor − g)·I) ≥ floor`.
    let bracket = floor - gershgorin_min;
    let cholesky_pd_at = |delta: f64| -> bool {
        let mut shifted = cholesky_test.clone();
        for d in 0..shifted.nrows() {
            shifted[[d, d]] += delta;
        }
        shifted.cholesky(Side::Lower).is_ok()
    };
    // Bisect the minimal PD shift `δ*` (where Cholesky just succeeds) in
    // `(0, bracket]`. `δ = 0` is known-indefinite (fast path failed above); the
    // bracket is known-PD. Cap iterations (relative squeeze to ~2⁻¹² of the
    // bracket) so the extra Choleskys stay bounded even when the shift fires on
    // every cycle of a coupled K-block fit.
    const MAX_BISECT: usize = 12;
    let mut lo = 0.0_f64; // indefinite
    let mut hi = bracket; // PD
    for _ in 0..MAX_BISECT {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if cholesky_pd_at(mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    // `hi` is the tightest bracket known PD (λ_min(cholesky_test + hi·I) ≈ 0⁺).
    // Add `floor` to restore the strict `≥ floor` margin, clamped to the original
    // guaranteed-PD Gershgorin shift so we never exceed the conservative bound.
    Some((hi + floor).min(bracket))
}

/// Per-block exact-Newton analogue of [`stabilized_joint_solver_diagonal_ridge`]
/// (gam#979). The block left-hand side is `lhs_dense = H_data + S`, where the
/// penalty `S = s_lambda (⪰ 0)` is positive-semidefinite by construction. As in
/// the coupled joint-Newton path, Weyl's inequality gives
/// `λ_min(H_data + S) ≥ λ_min(H_data)`, so the stabilizing ridge is bounded by
/// the *data* Hessian's curvature — never the penalty's. Passing the penalized
/// `lhs_dense` as its own Gershgorin source (a plain, unpenalized-source
/// stabilizer) reproduces the survival hang on any
/// off-diagonals of `S` make `diag − radius` read a spuriously huge negative
/// `λ_min`, so `δ = floor − g` adds a giant ridge and every per-block Newton
/// step collapses to `g/(H+μ) ≈ 0`. Bounding the shift by `H_data` (the
/// `gershgorin_src`) keeps it `O(data scale)`. This is the per-block twin of the
/// coupled fix and covers the bernoulli marginal-slope (binary) arm, which runs
/// the per-block exact-Newton updater rather than the coupled dense joint solve.
pub(crate) fn stabilize_exact_newton_penalized_lhs_in_place<F: CustomFamily + ?Sized>(
    family: &F,
    lhs_dense: &mut Array2<f64>,
    data_hessian_gershgorin_src: &Array2<f64>,
    ridge_floor: f64,
) {
    if use_exact_newton_strict_spd(family) {
        return;
    }
    if let Some(shift) = exact_newton_stabilizing_shift_psd_penalized(
        lhs_dense,
        data_hessian_gershgorin_src,
        ridge_floor,
    ) {
        for d in 0..lhs_dense.nrows() {
            lhs_dense[[d, d]] += shift;
        }
    }
}

pub(crate) fn shift_linear_constraints_to_delta(
    constraints: &ConstraintSet,
    beta: &Array1<f64>,
) -> Result<ConstraintSet, String> {
    if constraints.ncols() != beta.len() {
        return Err(CustomFamilyError::ConstraintViolation {
            reason: "linear constraints: shape mismatch".to_string(),
        }
        .into());
    }
    constraints.shifted_to_delta(beta.view())
}

pub(crate) fn collect_block_linear_constraints<F: CustomFamily + ?Sized>(
    family: &F,
    states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
) -> Result<Vec<Option<ConstraintSet>>, String> {
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
    constraints: Option<&ConstraintSet>,
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
    if raw_beta.len() != constraints.ncols() {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: format!(
                "post-update constrained block '{}' (idx {block_idx}) width mismatch: beta={}, constraints={}",
                spec.name,
                raw_beta.len(),
                constraints.ncols(),
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
    // Calibrate to the constrained QP's OWN primal feasibility tolerance. The
    // active-set solve holds a binding coordinate at its boundary only up to
    // `ACTIVE_SET_PRIMAL_FEASIBILITY_TOL` (1e-8), so an accepted step can leave a
    // bound coordinate a few ULPs to ~1e-8 inside the feasible side; a post-update
    // feasibility projection that snaps such sub-tolerance slop EXACTLY onto the
    // boundary (e.g. the monotone link-wiggle β≥0 clamp) does not change the KKT
    // point to feasibility tolerance — it is not the "repair an unrepresented
    // constraint" this guard rejects. A genuine post-hoc repair moves β by ≫1e-8
    // and still fails. The previous `1e-10` band was an order of magnitude tighter
    // than the solver can deliver, so it flagged legitimate KKT-slop cleanup.
    let tol = gam_solve::active_set::ACTIVE_SET_PRIMAL_FEASIBILITY_TOL
        * (1.0 + raw_scale.max(updated_scale));
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
    block_constraints: &[Option<ConstraintSet>],
    ranges: &[(usize, usize)],
    total_p: usize,
) -> Result<Option<ConstraintSet>, String> {
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
        .map(|constraints| constraints.as_ref().map_or(0, ConstraintSet::nrows))
        .sum::<usize>();
    if total_rows == 0 {
        return Ok(None);
    }
    for (block_idx, constraints_opt) in block_constraints.iter().enumerate() {
        let Some(constraints) = constraints_opt else {
            continue;
        };
        let (start, end) = ranges[block_idx];
        if constraints.ncols() != end - start {
            return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                "joint linear constraint assembly mismatch for block {block_idx}: {} constraint columns, block width is {}",
                constraints.ncols(),
                end - start
            ) }.into());
        }
    }
    let all_dense = block_constraints
        .iter()
        .all(|constraints| matches!(constraints, None | Some(ConstraintSet::Dense(_))));
    if all_dense {
        // Explicit-row concatenation, exactly the historical joint system.
        let mut a = Array2::<f64>::zeros((total_rows, total_p));
        let mut b = Array1::<f64>::zeros(total_rows);
        let mut row_offset = 0usize;
        for (block_idx, constraints_opt) in block_constraints.iter().enumerate() {
            let Some(ConstraintSet::Dense(constraints)) = constraints_opt else {
                continue;
            };
            let (start, end) = ranges[block_idx];
            let rows = constraints.a.nrows();
            a.slice_mut(s![row_offset..(row_offset + rows), start..end])
                .assign(&constraints.a);
            b.slice_mut(s![row_offset..(row_offset + rows)])
                .assign(&constraints.b);
            row_offset += rows;
        }
        return Ok(Some(ConstraintSet::Dense(LinearInequalityConstraints {
            a,
            b,
        })));
    }
    // At least one factored member: keep the joint system factored too.
    let mut placed = Vec::new();
    for (block_idx, constraints_opt) in block_constraints.iter().enumerate() {
        let Some(constraints) = constraints_opt else {
            continue;
        };
        placed.push(gam_problem::PlacedConstraintBlock {
            col_start: ranges[block_idx].0,
            set: constraints.clone(),
        });
    }
    Ok(Some(ConstraintSet::block_diagonal(placed, total_p)?))
}

pub(crate) fn flatten_joint_active_set(
    block_active_sets: &[Option<Vec<usize>>],
    block_constraints: &[Option<ConstraintSet>],
) -> Option<Vec<usize>> {
    if block_active_sets.len() != block_constraints.len() {
        return None;
    }
    let mut offset = 0usize;
    let mut joint_active = Vec::new();
    for (active_opt, constraints_opt) in block_active_sets.iter().zip(block_constraints.iter()) {
        let rows = constraints_opt.as_ref().map_or(0, ConstraintSet::nrows);
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
    block_constraints: &[Option<ConstraintSet>],
) -> Vec<Option<Vec<usize>>> {
    let mut per_block = Vec::with_capacity(block_constraints.len());
    let mut offset = 0usize;
    for constraints_opt in block_constraints {
        let rows = constraints_opt.as_ref().map_or(0, ConstraintSet::nrows);
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
/// Widen per-block QP-recorded active sets to the full NUMERICALLY-TIGHT face
/// at the current per-block β (gam#979).
///
/// At a degenerate binding vertex the QP can leave a row with scaled slack
/// inside the primal-feasibility band OUT of its recorded active set (a
/// phantom-dual / zero-multiplier omission). Such a row is on the active face
/// all the same: every mode-geometry consumer — the Laplace tangent of the
/// curvature certificate, the active-face logdet of the LAML value, the
/// constrained covariance — must null it, or the face tangent over-counts free
/// directions and curvature normal to the near-tight row leaks in as a phantom
/// indefiniteness (the measured survival marginal-slope terminal: tangent
/// min_eig=−1.09 on a 22-dim tangent whose true tight face is smaller, so the
/// certified constrained mode is refused as "no Laplace mode"). The union is
/// per block: the caller's recorded rows plus every finite-bound row whose
/// scaled slack `(a·β−b)/‖a‖` is below `ACTIVE_SET_PRIMAL_FEASIBILITY_TOL`,
/// ascending and deduplicated.
pub(crate) fn widen_active_sets_to_tight_face(
    block_constraints: &[Option<ConstraintSet>],
    states: &[ParameterBlockState],
    cached_active_sets: &[Option<Vec<usize>>],
) -> Result<Vec<Option<Vec<usize>>>, String> {
    let feasibility_tol = gam_solve::active_set::ACTIVE_SET_PRIMAL_FEASIBILITY_TOL;
    let mut tight_active_sets: Vec<Option<Vec<usize>>> =
        Vec::with_capacity(block_constraints.len());
    for (block_idx, constraints_opt) in block_constraints.iter().enumerate() {
        let Some(constraints) = constraints_opt else {
            tight_active_sets.push(None);
            continue;
        };
        let block_values = constraints.values(states[block_idx].beta.view())?;
        let mut rows: Vec<usize> = cached_active_sets
            .get(block_idx)
            .and_then(|active| active.clone())
            .unwrap_or_default();
        for row in 0..constraints.nrows() {
            if rows.contains(&row) {
                continue;
            }
            let norm = constraints.row_norm(row)?;
            if !(norm.is_finite() && norm > 0.0) {
                continue;
            }
            let bound = constraints.bound(row)?;
            if bound == f64::NEG_INFINITY {
                continue;
            }
            if (block_values[row] - bound) / norm < feasibility_tol {
                rows.push(row);
            }
        }
        rows.sort_unstable();
        rows.dedup();
        tight_active_sets.push((!rows.is_empty()).then_some(rows));
    }
    Ok(tight_active_sets)
}

pub(crate) fn assemble_active_constraint_block(
    block_constraints: &[Option<ConstraintSet>],
    block_active_sets: &[Option<Vec<usize>>],
    ranges: &[(usize, usize)],
    total_p: usize,
) -> Option<ActiveLinearConstraintBlock> {
    if block_constraints.len() != ranges.len() || block_active_sets.len() != ranges.len() {
        return None;
    }
    let mut active_per_block: Vec<(usize, &[usize], &ConstraintSet)> = Vec::new();
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
        if constraints.ncols() != range.1 - range.0 {
            return None;
        }
        if !active.iter().all(|&r| r < constraints.nrows()) {
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
        let gathered = constraints.gather_rows(active).ok()?;
        for (gathered_row, _) in active.iter().enumerate() {
            for col in 0..block_p {
                a[[out_row, start + col]] = gathered.a[[gathered_row, col]];
            }
            out_row += 1;
        }
    }
    Some(ActiveLinearConstraintBlock { a })
}

pub(crate) struct SimpleLowerBounds {
    pub(crate) lower_bounds: Array1<f64>,
    pub(crate) row_to_coeff: Vec<usize>,
    pub(crate) coeff_to_row: Vec<Option<usize>>,
}

pub(crate) fn extract_simple_lower_bounds(
    constraints: &ConstraintSet,
    p: usize,
) -> Result<Option<SimpleLowerBounds>, String> {
    let ConstraintSet::Dense(constraints) = constraints else {
        // A factored cone couples whole covariate rows; it is never a
        // per-coordinate lower-bound system.
        return Ok(None);
    };
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
    pub(crate) family: &'a dyn CustomFamily,
    pub(crate) states: &'a [ParameterBlockState],
    pub(crate) spec: &'a ParameterBlockSpec,
    pub(crate) block_idx: usize,
    pub(crate) s_lambda: &'a Array2<f64>,
    pub(crate) options: &'a BlockwiseFitOptions,
    pub(crate) linear_constraints: Option<&'a ConstraintSet>,
    pub(crate) cached_active_set: Option<&'a [usize]>,
}

#[derive(Debug)]
pub(crate) struct BlockUpdateResult {
    pub(crate) beta_new_raw: Array1<f64>,
    pub(crate) active_set: Option<Vec<usize>>,
}

/// Certify a diagonal working-curvature vector without changing it.
///
/// Signed finite rows are valid for observed Hessians. Conditioning and local
/// indefiniteness are handled after assembly by the shared matrix-level
/// ridge/continuation policy; a row-wise floor or sign projection would alter
/// the score, Hessian, and Laplace determinant into different models.
pub(crate) fn certify_finite_working_weights(
    working_weights: &Array1<f64>,
) -> Result<&Array1<f64>, String> {
    if let Some((i, &wi)) = working_weights
        .iter()
        .enumerate()
        .find(|&(_, &wi)| !wi.is_finite())
    {
        return Err(format!(
            "invalid diagonal working weight at row {i}: {wi} (working curvature must be finite)"
        ));
    }
    Ok(working_weights)
}

pub(crate) trait ParameterBlockUpdater {
    fn compute_update_step(
        &self,
        ctx: &BlockUpdateContext<'_>,
    ) -> Result<BlockUpdateResult, String>;
}

pub(crate) struct DiagonalBlockUpdater<'a> {
    pub(crate) working_response: &'a Array1<f64>,
    pub(crate) working_weights: &'a Array1<f64>,
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

        let working_weights =
            certify_finite_working_weights(self.working_weights).map_err(|e| {
                format!(
                    "block {} ({}) diagonal solve: {e}",
                    ctx.block_idx, ctx.spec.name
                )
            })?;

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
                let (mut lhs, rhs_opt) =
                    weighted_normal_equations(x, working_weights, Some(&y_star))?;
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
                    gam_solve::active_set::solve_quadratic_with_constraint_set(
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
                    (self.working_response[i] - off[i]) * working_weights[i]
                });
                let xtwy = x.transpose_vector_multiply(&wy);
                let beta = x
                    .solve_systemwith_policy(
                        working_weights,
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
    pub(crate) gradient: &'a Array1<f64>,
    pub(crate) hessian: &'a SymmetricMatrix,
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

        // Exact-Newton Hessians are the family's analytic second derivative: a
        // non-finite entry is invalid math, not a degenerate operating point a
        // stabilizing ridge or a feasible no-op could rescue. Refuse loudly at
        // the canonical smooth-regularized logdet boundary — the same boundary
        // the family-evaluation guard and the coupled joint-Newton initial
        // iterate enforce — BEFORE assembling `H + S` and entering the
        // (constrained or unconstrained) solve, so the failure names the
        // offending entry instead of surfacing later as a non-finite Newton
        // direction (gam#1088).
        exact_newton_hessian_finite_check(self.hessian, ctx.block_idx)?;
        let lhs = self.hessian.add_dense(ctx.s_lambda)?;
        // Solve in delta-space for both constrained and unconstrained blocks.
        // That keeps the linear system consistent even when we add a
        // numerical ridge to stabilize an indefinite exact-Newton Hessian.
        let rhs_step = self.gradient - &ctx.s_lambda.dot(&ctx.states[ctx.block_idx].beta);
        let mut lhs_dense = lhs.to_dense();
        // `lhs_dense = H_data + S` is penalized by the PSD block penalty `S`.
        // Bound the stabilizing ridge by the DATA Hessian's curvature, not the
        // penalized matrix's Gershgorin disc (gam#979): an over-smoothed `S`
        // has large balanced off-diagonals that make `diag − radius` read a
        // spurious huge-negative `λ_min`, which would otherwise add a giant
        // ridge and collapse every per-block Newton step. `self.hessian` is the
        // data Hessian; use it as the Gershgorin source.
        let data_hessian_dense = self.hessian.to_dense();
        stabilize_exact_newton_penalized_lhs_in_place(
            ctx.family,
            &mut lhs_dense,
            &data_hessian_dense,
            ctx.options.ridge_floor,
        );

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
                let (delta, active_set) =
                    gam_solve::active_set::solve_quadratic_with_constraint_set(
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

/// Extension trait providing `updater()` on the relocated `gam_problem::BlockWorkingSet`.
/// `BlockWorkingSet` now lives in gam-problem (a neutral crate), so this inherent-style
/// method — which produces a `Box<dyn ParameterBlockUpdater>` (a gam-crate trait) — must be
/// an extension trait rather than an inherent impl on the foreign type.
pub(crate) trait BlockWorkingSetUpdaterExt {
    fn updater(&self) -> Box<dyn ParameterBlockUpdater + '_>;
}

impl BlockWorkingSetUpdaterExt for BlockWorkingSet {
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
    constraints: &ConstraintSet,
    tol: f64,
) -> Result<(), String> {
    if constraints.ncols() != beta.len() {
        return Err(CustomFamilyError::ConstraintViolation {
            reason: "linear constraints: shape mismatch".to_string(),
        }
        .into());
    }
    let values = constraints.values(beta.view()).map_err(|e| {
        CustomFamilyError::ConstraintViolation {
            reason: format!("linear constraints: {e}"),
        }
        .to_string()
    })?;
    let mut worst = 0.0_f64;
    let mut worst_idx = 0usize;
    for i in 0..constraints.nrows() {
        let bound = constraints.bound(i).map_err(|e| {
            CustomFamilyError::ConstraintViolation {
                reason: format!("linear constraints: {e}"),
            }
            .to_string()
        })?;
        let v = (bound - values[i]).max(0.0);
        if v > worst {
            worst = v;
            worst_idx = i;
        }
    }
    if worst > tol {
        // #1108 DIAGNOSTIC: pin down whether this infeasible β is PROJECTABLE
        // (→ a wiring bug: the seed bypassed projection) or genuinely outside a
        // hard polytope. Report the worst row's ‖a‖, the scaled violation, the
        // β magnitude, and the outcome of the exact active-set projection of
        // THIS β onto these constraints.
        let norm_worst = constraints.row_norm(worst_idx).unwrap_or(f64::NAN);
        let scaled_worst = if norm_worst > 0.0 {
            worst / norm_worst
        } else {
            worst
        };
        let beta_inf = beta.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
        let interior_outcome =
            match gam_solve::active_set::project_point_strictly_into_feasible_constraint_set(
                beta,
                constraints,
            ) {
                Ok(p) => {
                    let w = constraints
                        .max_scaled_violation(p.view())
                        .map(|(w, _)| w)
                        .unwrap_or(f64::NAN);
                    format!("strict-interior→scaled_worst={w:.3e}")
                }
                Err(e) => format!("strict-interior→refused: {e}"),
            };
        let simple_bounds_path = match extract_simple_lower_bounds(constraints, beta.len()) {
            Ok(Some(_)) => "extract_simple_lower_bounds→Some(SIMPLE-BOUNDS PATH)",
            Ok(None) => "extract_simple_lower_bounds→None(general QP)",
            Err(_) => "extract_simple_lower_bounds→Err",
        };
        return Err(CustomFamilyError::ConstraintViolation {
            reason: format!(
                "infeasible iterate: max(Aβ-b violation)={worst:.3e} at constraint row {worst_idx} \
                 [#1108 diag: rows={}, ‖a_row‖={norm_worst:.3e}, \
                 scaled_viol={scaled_worst:.3e}, |β|∞={beta_inf:.3e}, {interior_outcome}, \
                 {simple_bounds_path}]",
                constraints.nrows()
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
    if ridge_policy.accounts_for_objective() {
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
    if ridge_policy.accounts_for_objective() && ridge > 0.0 {
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
        if ridge_policy.accounts_for_objective() && ridge > 0.0 {
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
    joint_full_width: Option<&gam_problem::JointPenaltyBundle>,
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
        exact_newton_hessian_finite_check(hessian, b)?;
    }
    Ok(())
}

/// Refuse a single exact-Newton block Hessian carrying a non-finite entry,
/// using the canonical smooth-regularized logdet boundary message with the
/// offending block index appended. Shared by the family-evaluation-boundary
/// guard [`validate_block_hessians_finite`] and the per-block exact-Newton
/// updater, so a `NaN`/`Inf` in the family's analytic second derivative is
/// rejected at the same boundary with the same phrasing whether the block is
/// constrained or unconstrained — a contract violation, not a solver
/// contingency to be absorbed by a ridge or a no-op step (gam#1088).
pub(crate) fn exact_newton_hessian_finite_check(
    hessian: &SymmetricMatrix,
    block: usize,
) -> Result<(), String> {
    match hessian {
        SymmetricMatrix::Dense(matrix) => {
            smooth_regularized_logdet_hessian_finite_check(matrix, Some(block))?;
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
                            "smooth-regularized logdet Hessian contains non-finite entry at ({row}, {col}): {value} for block {block}"
                        ) }.into());
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
    let ridge = if ridge_policy.accounts_for_objective() {
        effective_solverridge(ridge_floor)
    } else {
        0.0
    };
    for i in 0..p {
        a[[i, i]] += ridge;
    }

    match ridge_policy.determinant_mode() {
        RidgeDeterminantMode::Full => match a.cholesky(Side::Lower) {
            Ok(chol) => Ok(2.0 * chol.diag().mapv(f64::ln).sum()),
            Err(_) => {
                // Cholesky failed. Separate a genuinely INDEFINITE Hessian — a real
                // SPD-contract violation, no Laplace mode exists — from a PD-but-
                // ILL-CONDITIONED one (cond > 1/ε: every true eigenvalue is
                // positive, but a rounding-negative pivot aborts the
                // factorization). The latter is exactly what a competing-risks
                // joint Hessian produces from its near-duplicate cross-cause time
                // columns (fit_orchestration comment on the K block-pairs of
                // near-identical columns): min_eig ≫ 0 yet cond > 1/ε.
                //
                // For a PD matrix the exact SPD log-determinant still exists and
                // equals `Σ log σ_j` over the symmetric spectrum — the SAME
                // estimand as `2·Σ log L_ii`, computed stably — so return it rather
                // than failing the whole ρ evaluation, which otherwise strands the
                // outer REML search at a non-stationary point (the joint Hessian
                // logdet is unavailable at that ρ, so no gradient/value is
                // produced). Only a certified-negative eigenvalue propagates the
                // error. This is byte-identical on every input where Cholesky
                // already succeeds.
                let (evals, _) = gam_linalg::faer_ndarray::FaerEigh::eigh(&a, Side::Lower)
                    .map_err(|_| {
                        format!(
                            "cholesky failed and the eigendecomposition also failed while computing full ridge-aware logdet (p={p}, ridge={ridge:.3e})"
                        )
                    })?;
                let min_eig = evals.iter().copied().fold(f64::INFINITY, f64::min);
                let max_eig = evals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                // Negative-eigenvalue tolerance, relative to the spectrum scale.
                let neg_tol = CUSTOM_FAMILY_CONDITION_RELATIVE_FLOOR * max_eig.abs().max(1.0);
                if min_eig <= -neg_tol {
                    return Err(format!(
                        "cholesky failed while computing full ridge-aware logdet and the symmetric spectrum is genuinely indefinite (p={p}, ridge={ridge:.3e}, min_eig={min_eig:.6e}, max_eig={max_eig:.6e}); an indefinite Hessian has no SPD log-determinant and defines no Laplace mode"
                    ));
                }
                // PD but ill-conditioned: exact logdet from the positive spectrum,
                // flooring round-off-nonpositive eigenvalues at the ridge-relative
                // floor so a numerically-singular direction contributes a bounded
                // (not `-inf`) term.
                let floor = ridge.max(neg_tol);
                Ok(evals.iter().map(|&e| e.max(floor).ln()).sum())
            }
        },
        RidgeDeterminantMode::PositivePartApproximation => {
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
            match gam_linalg::faer_ndarray::FaerEigh::eigh(&a, Side::Lower) {
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

pub(crate) fn symmetrize_dense_in_place(matrix: &mut Array2<f64>) {
    gam_linalg::matrix::symmetrize_in_place(matrix);
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

// `CUSTOM_FAMILY_RIDGE_FLOOR` (the initial Cholesky-escalation ridge δ) is
// single-sourced in `gam-problem` (`custom_family_blockwise`). Re-exported here
// so existing crate-local paths keep resolving after the #1521 carve instead of
// holding a second definition.
pub(crate) use gam_problem::CUSTOM_FAMILY_RIDGE_FLOOR;

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
    process_chol: impl FnOnce(&gam_linalg::faer_ndarray::FaerCholeskyFactor) -> R,
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

    let exhausted = match escalate_ridge(
        RidgeSchedule {
            initial: delta0,
            growth: STRICT_SPD_LM_RIDGE_GROWTH,
            max_escalations: STRICT_SPD_LM_MAX_ESCALATIONS,
        },
        |delta| {
            let mut ridged = sym.clone();
            for i in 0..p {
                ridged[[i, i]] += delta;
            }
            ridged.cholesky(Side::Lower).ok()
        },
    ) {
        Ok(success) => {
            return Ok((
                process_chol(&success.value),
                StrictSpdLmStats {
                    delta_used: success.ridge,
                    escalations: success.escalations,
                },
            ));
        }
        Err(exhausted) => exhausted,
    };

    // δ-ridge schedule exhausted; fall back to rank-aware eigen-floor handling.
    // Floors every eigenvalue at `eps_floor = 1e-12 · max|λ|` so well-conditioned
    // modes are resolved exactly and rank-deficient directions are handled with
    // controlled curvature, preventing the spatial-adaptive pilot from collapsing
    // to a cold full-data run.
    let max_esc = STRICT_SPD_LM_MAX_ESCALATIONS;
    let delta = exhausted.next_ridge;
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

pub(crate) struct ConstrainedHessianGeometry {
    pub(crate) matrix: Array2<f64>,
    pub(crate) nullity: usize,
    pub(crate) condition: f64,
    pub(crate) raw_min_eigenvalue: f64,
    pub(crate) stabilized_min_eigenvalue: f64,
}

/// Modified-Newton convexification and selective gauge stabilization for a
/// symmetric penalized Hessian.
///
/// Negative identified curvature is reflected to `|λ|`. Numerical-null modes
/// receive the requested Levenberg curvature, but identified modes are left at
/// their exact magnitude. Adding `μ I` to a rank-deficient Hessian makes the QP
/// unique by perturbing every weak identified direction too, reducing Newton's
/// quadratic endgame to the measured `H/(H+μ)` linear crawl (#979). Adding
/// `μ P_null` provides the same gauge uniqueness without changing the Newton
/// equation on `range(H)`.
///
/// A family may separately request the historical full-rank ill-conditioning
/// damping. That case has no numerical null projector, so it retains the
/// ambient `λ -> λ + μ` shift. Both policies are constructed from this one
/// eigendecomposition, along with their rank and condition diagnostics.
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
/// the property the dense path relies on. Eigensolver failure is a hard error:
/// silently reverting to an indefinite or differently damped QP would switch
/// algorithms after its curvature contract failed.
pub(crate) fn symmetric_constrained_hessian_geometry(
    matrix: &Array2<f64>,
    levenberg_mu: f64,
    damp_full_rank_ill_conditioned: bool,
) -> Result<ConstrainedHessianGeometry, String> {
    let p = matrix.nrows();
    if p == 0 || matrix.ncols() != p {
        return Err(format!(
            "constrained Hessian must be nonempty and square, got {}x{}",
            matrix.nrows(),
            matrix.ncols()
        ));
    }
    let mut sym = matrix.clone();
    symmetrize_dense_in_place(&mut sym);
    let (evals, evecs) = FaerEigh::eigh(&sym, Side::Lower)
        .map_err(|error| format!("constrained Hessian eigendecomposition failed: {error:?}"))?;
    let lambda_max_abs = evals.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    if !(lambda_max_abs.is_finite() && lambda_max_abs > 0.0) {
        return Err("constrained Hessian has no finite nonzero curvature scale".to_string());
    }
    let cutoff = KKT_REFUSAL_RANK_TOL * lambda_max_abs;
    let nullity = evals.iter().filter(|value| value.abs() < cutoff).count();
    let min_range = evals
        .iter()
        .map(|value| value.abs())
        .filter(|magnitude| *magnitude >= cutoff && *magnitude > 0.0)
        .fold(f64::INFINITY, f64::min);
    let condition = if min_range.is_finite() && min_range > 0.0 {
        lambda_max_abs / min_range
    } else {
        f64::INFINITY
    };
    let ambient_levenberg = nullity == 0
        && damp_full_rank_ill_conditioned
        && condition > LEVENBERG_ILL_CONDITIONING_THRESHOLD;
    let floor = lambda_max_abs * (p as f64).sqrt() * f64::EPSILON;
    let mu = if levenberg_mu.is_finite() && levenberg_mu > 0.0 {
        levenberg_mu
    } else {
        0.0
    };
    let stabilized = Array1::from_iter(evals.iter().map(|lambda| {
        if nullity > 0 && lambda.abs() < cutoff {
            mu.max(floor)
        } else if ambient_levenberg {
            (lambda + mu).abs().max(floor)
        } else {
            lambda.abs().max(floor)
        }
    }));
    let stabilized_min_eigenvalue = stabilized.iter().copied().fold(f64::INFINITY, f64::min);
    let scaled = &evecs * &stabilized.view().insert_axis(ndarray::Axis(0));
    Ok(ConstrainedHessianGeometry {
        matrix: scaled.dot(&evecs.t()),
        nullity,
        condition,
        raw_min_eigenvalue: evals.iter().copied().fold(f64::INFINITY, f64::min),
        stabilized_min_eigenvalue,
    })
}

/// Numerical nullity of a symmetric penalized Hessian at the shared
/// `KKT_REFUSAL_RANK_TOL` relative cutoff used by the spectral range and REML
/// penalty-rank machinery. Returns `None` when no finite curvature scale or
/// eigendecomposition is available.
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

#[cfg(test)]
mod constrained_hessian_geometry_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn rank_deficient_levenberg_curvature_is_confined_to_the_null_projector() {
        // Eigenpairs: range vectors (1,0,-1)/sqrt(2) at lambda=4 and
        // (0,1,0) at lambda=1; gauge vector (1,0,1)/sqrt(2) at lambda=0.
        let hessian = array![[2.0, 0.0, -2.0], [0.0, 1.0, 0.0], [-2.0, 0.0, 2.0]];
        let mu = 0.25;
        let geometry = symmetric_constrained_hessian_geometry(&hessian, mu, false)
            .expect("rank-deficient symmetric geometry");
        assert_eq!(geometry.nullity, 1);

        let inv_sqrt_two = 0.5_f64.sqrt();
        let range = array![inv_sqrt_two, 0.0, -inv_sqrt_two];
        let second_range = array![0.0, 1.0, 0.0];
        let null = array![inv_sqrt_two, 0.0, inv_sqrt_two];
        let range_image = geometry.matrix.dot(&range);
        let second_range_image = geometry.matrix.dot(&second_range);
        let null_image = geometry.matrix.dot(&null);

        for (actual, expected) in range_image.iter().zip((4.0 * &range).iter()) {
            assert!((actual - expected).abs() <= 1e-12);
        }
        for (actual, expected) in second_range_image.iter().zip(second_range.iter()) {
            assert!((actual - expected).abs() <= 1e-12);
        }
        for (actual, expected) in null_image.iter().zip((mu * &null).iter()) {
            assert!((actual - expected).abs() <= 1e-12);
        }
    }
}

#[cfg(test)]
mod penalty_logdet_unify_tests {
    use super::*;
    use gam_solve::estimate::reml::penalty_logdet::PenaltyPseudologdet;
    use ndarray::array;

    /// The penalty-logdet fallback was collapsed onto `strict_exact_pseudo_logdet`
    /// (deleting the ridge-escalating `penalty_logdet_cholesky_fallback`). This
    /// pins that the strict path computes the SAME quantity as the canonical
    /// `PenaltyPseudologdet::value()` the analytic REML gradient differentiates:
    /// the exact positive-eigenspace pseudo-logdet `Σ_{σ>tol} log σ`, so the
    /// fallback can never re-introduce a ridge-biased, gradient-inconsistent
    /// value. Covers both the rank-deficient (null space present) and the
    /// full-rank SPD case.
    #[test]
    fn strict_pseudo_logdet_matches_canonical_penalty_value() {
        // Rank-deficient PSD penalty: 2×2 active block + a structural null dim.
        // Active eigenvalues 1.5 and 2.5 ⇒ pseudo-logdet = ln 1.5 + ln 2.5.
        let s_rank_deficient = array![[2.0, 0.5, 0.0], [0.5, 2.0, 0.0], [0.0, 0.0, 0.0],];
        let expected_deficient = 1.5_f64.ln() + 2.5_f64.ln();
        let strict = strict_exact_pseudo_logdet(&s_rank_deficient, 3).expect("strict logdet");
        let canonical =
            PenaltyPseudologdet::from_components(&[s_rank_deficient.clone()], &[1.0], 0.0)
                .expect("canonical pseudo-logdet")
                .value();
        assert!(
            (strict - expected_deficient).abs() < 1e-10,
            "strict pseudo-logdet {strict} != analytic {expected_deficient}"
        );
        assert!(
            (strict - canonical).abs() < 1e-10,
            "strict pseudo-logdet {strict} != canonical PenaltyPseudologdet value {canonical}"
        );

        // Full-rank SPD penalty: pseudo-logdet = log|S| = ln(det).
        let s_spd = array![[2.0, 0.5], [0.5, 3.0]];
        let expected_spd = (2.0_f64 * 3.0 - 0.5 * 0.5).ln();
        let strict_spd = strict_exact_pseudo_logdet(&s_spd, 2).expect("strict spd logdet");
        let canonical_spd = PenaltyPseudologdet::from_components(&[s_spd.clone()], &[1.0], 0.0)
            .expect("canonical spd pseudo-logdet")
            .value();
        assert!(
            (strict_spd - expected_spd).abs() < 1e-10,
            "strict SPD logdet {strict_spd} != ln(det) {expected_spd}"
        );
        assert!(
            (strict_spd - canonical_spd).abs() < 1e-10,
            "strict SPD logdet {strict_spd} != canonical value {canonical_spd}"
        );
    }
}
