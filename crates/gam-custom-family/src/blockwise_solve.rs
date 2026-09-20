//! The inner block-coordinate solve: per-block updaters (diagonal / exact-Newton),
//! weighted normal equations, linear-constraint + lower-bound assembly and active
//! sets, block penalty/metric helpers, the total-quadratic-penalty objective, and
//! the SPD logdet / strict-solve / pseudo-inverse numeric kernels. Also the
//! labeled-rho aggregation/pullback helpers that drive the outer eval.

use super::*;

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
) -> Result<Array2<f64>, CustomFamilyError> {
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
        });
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
) -> Result<(f64, Array1<f64>, Option<Array2<f64>>), CustomFamilyError> {
    use gam_solve::rho_prior_eval::{InvalidPriorPolicy, RhoPriorError};
    match gam_solve::rho_prior_eval::evaluate(prior, rho, InvalidPriorPolicy::HardError) {
        Ok(eval) => Ok((eval.cost, eval.gradient, eval.hessian)),
        Err(RhoPriorError::DimensionMismatch { reason }) => {
            Err(CustomFamilyError::DimensionMismatch { reason })
        }
        Err(RhoPriorError::ConstraintViolation { reason }) => {
            Err(CustomFamilyError::ConstraintViolation { reason })
        }
    }
}

pub(crate) fn add_labeled_rho_prior_to_outer_eval(
    mut result: OuterObjectiveEvalResult,
    rho: &Array1<f64>,
    rho_prior: &gam_problem::RhoPrior,
    eval_mode: EvalMode,
) -> Result<OuterObjectiveEvalResult, CustomFamilyError> {
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
    result.criterion_components[0] += cost;
    if eval_mode != EvalMode::ValueOnly {
        if result.gradient.len() != gradient.len() {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "rho prior gradient length mismatch: got {}, expected {}",
                    gradient.len(),
                    result.gradient.len()
                ),
            });
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
) -> Result<OuterObjectiveEvalResult, CustomFamilyError> {
    if eval_mode == EvalMode::ValueOnly {
        result.gradient = Array1::<f64>::zeros(layout.initial_rho.len());
    } else {
        let raw = result.gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
        let raw_len = result.gradient.len();
        let raw_head: Vec<f64> = result.gradient.iter().take(6).copied().collect();
        result.gradient = aggregate_labeled_gradient(&result.gradient, layout)?;
        log::trace!(
            "[LABELED-EVAL] mode={eval_mode:?} rho0={:.4} |g_physical|={raw:.6e} len={raw_len} \
             head={raw_head:?} |g_outer|={:.6e}",
            rho[0],
            result.gradient.iter().map(|g| g * g).sum::<f64>().sqrt(),
        );
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

/// Attach the joint penalties selected by one labeled rho vector to an inner
/// problem. Per-block penalties already travel through `physical_rho`; joint
/// penalties need this full-width bundle so coefficient correction and endpoint
/// criterion assembly see exactly the same objective.
pub(crate) fn labeled_options_for_rho<'a>(
    options: &'a BlockwiseFitOptions,
    specs: &[ParameterBlockSpec],
    layout: &PenaltyLabelLayout,
    rho: &Array1<f64>,
) -> Result<std::borrow::Cow<'a, BlockwiseFitOptions>, CustomFamilyError> {
    if layout.joint_specs.is_empty() {
        return Ok(std::borrow::Cow::Borrowed(options));
    }
    let total_compiled: usize = specs.iter().map(|spec| spec.design.ncols()).sum();
    let joint_log_lambdas = layout.joint_log_lambdas(rho);
    let bundle = gam_problem::JointPenaltyBundle::from_validated_geometry(
        std::sync::Arc::clone(&layout.joint_specs),
        std::sync::Arc::clone(&layout.joint_roots),
        joint_log_lambdas,
        total_compiled,
    )?;
    let mut owned = options.clone();
    owned.joint_penalties = Some(std::sync::Arc::new(bundle));
    Ok(std::borrow::Cow::Owned(owned))
}

/// Correct one labeled-rho continuation waypoint to its certified coefficient
/// mode, without constructing a Laplace scalar that an interior waypoint would
/// discard.
pub(crate) fn correct_labeled_coefficient_mode<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    layout: &PenaltyLabelLayout,
    rho: &Array1<f64>,
    warm_start: Option<&ConstrainedWarmStart>,
) -> Result<(BlockwiseInnerResult, ConstrainedWarmStart), CustomFamilyError> {
    correct_labeled_mode(
        family,
        specs,
        options,
        layout,
        rho,
        warm_start,
        inner_blockwise_coefficient_mode::<F>,
    )
}

/// Correct a labeled-rho continuation endpoint to its certified mode with the
/// determinant artifacts an ordinary inner solve carries, so the published mode
/// is the same reusable seed a direct evaluation files (gam#2973).
pub(crate) fn correct_labeled_laplace_mode<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    layout: &PenaltyLabelLayout,
    rho: &Array1<f64>,
    warm_start: Option<&ConstrainedWarmStart>,
) -> Result<(BlockwiseInnerResult, ConstrainedWarmStart), CustomFamilyError> {
    correct_labeled_mode(
        family,
        specs,
        options,
        layout,
        rho,
        warm_start,
        inner_blockwise_fit::<F>,
    )
}

type InnerModeSolve<F> = fn(
    &F,
    &[ParameterBlockSpec],
    &[Array1<f64>],
    &BlockwiseFitOptions,
    Option<&ConstrainedWarmStart>,
) -> Result<BlockwiseInnerResult, CustomFamilyError>;

fn correct_labeled_mode<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    layout: &PenaltyLabelLayout,
    rho: &Array1<f64>,
    warm_start: Option<&ConstrainedWarmStart>,
    solve: InnerModeSolve<F>,
) -> Result<(BlockwiseInnerResult, ConstrainedWarmStart), CustomFamilyError> {
    let physical_rho = expand_labeled_log_lambdas(rho, layout)?;
    let per_block = split_log_lambdas(&physical_rho, &layout.penalty_counts)?;
    let physical_warm_start = physical_warm_start_for_labeled(warm_start, &physical_rho, layout);
    let labeled_options = labeled_options_for_rho(options, specs, layout, rho)?;
    let inner = solve(
        family,
        specs,
        &per_block,
        labeled_options.as_ref(),
        physical_warm_start.as_ref().or(warm_start),
    )?;
    if !inner.converged {
        return Err(inner_solve_not_converged_error(
            &inner,
            labeled_options.as_ref(),
            physical_rho.len(),
            0,
        ));
    }
    checked_penalizedobjective(
        inner.log_likelihood,
        inner.penalty_value,
        0.0,
        "continuation coefficient corrector",
    )?;
    let mut warm_start = constrained_warm_start_from_inner(&physical_rho, &inner);
    warm_start.rho = rho.clone();
    Ok((inner, warm_start))
}

/// Complete the endpoint criterion in `eval_mode` from a continuation-owned mode.
/// The mode is consumed, so it cannot accidentally be paired with a different
/// endpoint after the call.
pub(crate) fn outerobjective_from_coefficient_mode_labeled<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    layout: &PenaltyLabelLayout,
    rho: &Array1<f64>,
    rho_prior: &gam_problem::RhoPrior,
    inner: BlockwiseInnerResult,
    eval_mode: EvalMode,
) -> Result<OuterObjectiveEvalResult, CustomFamilyError> {
    let physical_rho = expand_labeled_log_lambdas(rho, layout)?;
    let labeled_options = labeled_options_for_rho(options, specs, layout, rho)?;
    let base = evaluate_custom_family_hyper_from_coefficient_mode(
        family,
        specs,
        labeled_options.as_ref(),
        &layout.penalty_counts,
        &physical_rho,
        gam_problem::RhoPrior::Flat,
        inner,
        eval_mode,
    )?;
    pullback_labeled_outer_eval(base, rho, layout, rho_prior, eval_mode)
        .map_err(CustomFamilyError::from)
}

pub(crate) fn split_log_lambdas(
    flat: &Array1<f64>,
    penalty_counts: &[usize],
) -> Result<Vec<Array1<f64>>, CustomFamilyError> {
    let expected: usize = penalty_counts.iter().sum();
    if flat.len() != expected {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: format!(
                "log-lambda length mismatch: got {}, expected {expected}",
                flat.len()
            ),
        });
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
) -> Result<Vec<ParameterBlockState>, CustomFamilyError> {
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
) -> Result<(), CustomFamilyError> {
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
) -> Result<(), CustomFamilyError> {
    let spec = &specs[block_idx];
    let beta = states[block_idx].beta.clone();
    states[block_idx].eta = with_block_geometry(family, states, spec, block_idx, |x, off| {
        Ok(x.matrixvectormultiply(&beta) + off)
    })?;
    Ok(())
}

pub(crate) fn weighted_normal_equations(
    x: &DesignMatrix,
    w: &Array1<f64>,
    y_star: Option<&Array1<f64>>,
) -> Result<(Array2<f64>, Option<Array1<f64>>), CustomFamilyError> {
    let n = x.nrows();
    if w.len() != n {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: "weighted normal-equation dimension mismatch".to_string(),
        });
    }
    if let Some(y) = y_star
        && y.len() != n
    {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: "weighted RHS dimension mismatch".to_string(),
        });
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
/// when no shift is needed (the matrix is already PD) or none can help (a
/// non-finite or overflowing source Hessian, which the consuming solve then
/// resolves or refuses).
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
    // Hessian (e.g. the survival marginal/slope aliasing of gam#979, where the
    // marginal and slope smooths share covariates and every row is full) the
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
    // monotone step in `δ`, so bisection between the known-indefinite `δ = 0`
    // (the fast-path Cholesky above already failed) and the known-PD Gershgorin
    // bracket squeezes `δ` onto the minimal PD shift, to the Cholesky
    // certificate's own resolution (see the stop rule below). Each step is one
    // `O(p³/3)` Cholesky and only runs on the indefinite cycles the fast path
    // did not already clear. The final `+ floor` restores the `≥ floor`
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
    // A disc bound that is not finite (a non-finite or overflowing source
    // Hessian) gives no bracket to bisect in, and no diagonal shift makes a
    // non-finite system positive definite. The matrix goes back unshifted, so the
    // solve that consumes it either resolves it or refuses it.
    if !gershgorin_min.is_finite() {
        return None;
    }
    // The margin a positive-definiteness certificate can carry is the smallest
    // pivot a computed Cholesky resolves. Pivot `k` subtracts at most `p` squares
    // whose sum is `A_kk`, so its rounding is at most `γ_{p+1}·max_k A_kk` for the
    // matrix `A` being factored. A caller's `ridge_floor` can only raise it.
    let factored_dim = cholesky_test.nrows();
    if (0..factored_dim).any(|d| !cholesky_test[[d, d]].is_finite()) {
        return None;
    }
    let max_diagonal = (0..factored_dim)
        .fold(0.0_f64, |largest, d| largest.max(cholesky_test[[d, d]].abs()));
    let floor = ridge_floor
        .max(gam_linalg::roundoff::accumulation_growth(factored_dim + 1) * max_diagonal);
    if gershgorin_min >= floor {
        // Gershgorin certifies PD-at-floor but the no-shift Cholesky failed
        // (round-off on a barely-PD matrix): a pivot-band shift suffices.
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
    // A matrix with an all-zero diagonal and no caller floor carries no pivot
    // resolution to bisect against; the guaranteed Gershgorin shift is the only
    // certified answer.
    if !(floor > 0.0) {
        return Some(bracket);
    }
    // Locate the minimal PD shift `δ* = −λ_min(cholesky_test)` in
    // `(0, bracket]`. `δ = 0` is known-indefinite (the fast path failed above)
    // and the bracket is known-PD.
    //
    // Stop rule (gam#3660). A Cholesky pass/fail decides the PD frontier only to
    // within `floor`, the pivot rounding bound computed above, so `δ*` is
    // resolved once the bracket's width is `floor`. Everything below `floor` is
    // one resolution cell, so the lower end reads `a = max(lo, floor)` and the
    // search stops at `hi − a ≤ floor`. On exit `hi ≤ δ* + 2·floor`, so the
    // returned `hi + floor` overshifts `δ*` by at most `3·floor`, whatever the
    // Gershgorin bracket's looseness. A fixed number of halvings of the bracket
    // instead leaves an error of `bracket·2⁻ⁿ`, which is relative to the
    // bracket and not to `δ*`: on a barely-indefinite Hessian (`δ*` far below
    // the O(1) Gershgorin bracket) it returned hundreds of times the minimal
    // shift and flattened every direction of curvature below it.
    //
    // Midpoints are geometric, `√(a·hi)`. Each probe halves `ln(hi/a)`, which
    // starts at `ln(bracket/floor)` (at most ~37 for a double) and must reach
    // `ln(1 + floor/a)`. That costs about `log₂ ln(bracket/floor) +
    // log₂(max(δ*, floor)/floor)` Choleskys. The near-singular gauge cycles of a
    // coupled K-block fit (gam#729/#826), where `δ*` sits at the rounding
    // level, resolve in about six.
    let mut lo = 0.0_f64; // indefinite
    let mut hi = bracket; // PD
    loop {
        let a = lo.max(floor);
        if hi - a <= floor {
            break;
        }
        let mid = (a * hi).sqrt();
        if !(mid > lo && mid < hi) {
            // The bracket is at the spacing of adjacent doubles.
            break;
        }
        if cholesky_pd_at(mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    // `hi` is the tightest shift known PD. Add `floor` to restore the strict
    // `≥ floor` margin, clamped to the guaranteed-PD Gershgorin shift.
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
) -> Result<ConstraintSet, CustomFamilyError> {
    if constraints.ncols() != beta.len() {
        return Err(CustomFamilyError::ConstraintViolation {
            reason: "linear constraints: shape mismatch".to_string(),
        });
    }
    constraints
        .shifted_to_delta(beta.view())
        .map_err(|error| CustomFamilyError::trial_point(error.to_string()))
}

pub(crate) fn collect_block_linear_constraints<F: CustomFamily + ?Sized>(
    family: &F,
    states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
) -> Result<Vec<Option<ConstraintSet>>, CustomFamilyError> {
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
) -> Result<(), CustomFamilyError> {
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
        });
    }
    if raw_beta.len() != constraints.ncols() {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: format!(
                "post-update constrained block '{}' (idx {block_idx}) width mismatch: beta={}, constraints={}",
                spec.name,
                raw_beta.len(),
                constraints.ncols(),
            ),
        });
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
        });
    }
    Ok(())
}

pub(crate) fn assemble_joint_linear_constraints(
    block_constraints: &[Option<ConstraintSet>],
    ranges: &[(usize, usize)],
    total_p: usize,
) -> Result<Option<ConstraintSet>, CustomFamilyError> {
    if block_constraints.len() != ranges.len() {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: format!(
                "joint linear constraint assembly mismatch: {} blocks but {} ranges",
                block_constraints.len(),
                ranges.len()
            ),
        });
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
            ) });
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
) -> Result<Vec<Option<Vec<usize>>>, CustomFamilyError> {
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
        let norms = constraints.all_row_norms();
        let bounds = constraints.all_bounds();
        let mut recorded = vec![false; constraints.nrows()];
        for &row in &rows {
            if row < recorded.len() {
                recorded[row] = true;
            }
        }
        for row in 0..constraints.nrows() {
            if recorded[row] {
                continue;
            }
            let norm = norms[row];
            if !(norm.is_finite() && norm > 0.0) {
                continue;
            }
            let bound = bounds[row];
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
) -> Result<Option<SimpleLowerBounds>, CustomFamilyError> {
    let constraints = match constraints {
        ConstraintSet::Dense(dense) => dense,
        // A factored cone USUALLY couples whole covariate rows, and then it is
        // not a per-coordinate lower-bound system. But that is a statement about
        // the geometry the carrier expresses, not about the carrier type, and
        // keying on the type made the box-KKT repair below UNREACHABLE for a
        // whole model class (gam#2600).
        //
        // A Khatri-Rao cone row is psi_i dot A[k,:] >= 0. When the covariate
        // factor has ONE column, psi_i is a scalar, so the row reduces to the
        // per-coordinate bound A[k] >= 0 whenever psi_i > 0 -- and on an
        // intercept-only fit every psi_i is exactly 1, so all n rows of a slot
        // are the SAME halfspace. Measured on the CTN fixture: a coefficient
        // pinned at bit-exact zero on every cycle of four independent inner
        // solves, carrying the largest raw residual in the problem, with the
        // certificate scoring that valid multiplier as a stationarity defect
        // because this function refused to see the box.
        //
        // The test below is the GEOMETRIC one, deliberately not a p_cov check:
        // every row must touch exactly one column with a positive coefficient.
        // That is the same predicate the dense arm applies, so a cone whose rows
        // are not axis-aligned-positive -- every multi-column cone, and any
        // single-column cone whose factor changes sign, where the slot is really
        // the EQUALITY A[k] = 0 -- still returns None and behaves exactly as
        // before. The row values and bounds come from the cone itself rather
        // than being re-derived from the factor, so the unit-normalization
        // convention cannot drift away from the rest of the solver.
        ConstraintSet::KhatriRaoCone(cone) => {
            if cone.factor().ncols() != 1 {
                return Ok(None);
            }
            if cone.ncols() != p {
                return Err(CustomFamilyError::ConstraintViolation {
                    reason: "linear constraints: factored cone width does not match the coefficient block".to_string(),
                });
            }
            let mut lower_bounds = Array1::from_elem(p, f64::NEG_INFINITY);
            let mut coeff_to_row = vec![None; p];
            let mut row_to_coeff = Vec::with_capacity(cone.nrows());
            for row in 0..cone.nrows() {
                let support = cone.row_column_support(row).map_err(|error| {
                    CustomFamilyError::ConstraintViolation {
                        reason: format!("factored cone row support: {error}"),
                    }
                })?;
                if support.len() != 1 {
                    return Ok(None);
                }
                let col = support[0];
                let gathered = cone.gather_rows(&[row]).map_err(|error| {
                    CustomFamilyError::ConstraintViolation {
                        reason: format!("factored cone row gather: {error}"),
                    }
                })?;
                let coeff_value = gathered.a[[0, col]];
                if !(coeff_value > 0.0) {
                    return Ok(None);
                }
                let bound = gathered.b[0] / coeff_value;
                if bound > lower_bounds[col] {
                    lower_bounds[col] = bound;
                    coeff_to_row[col] = Some(row);
                }
                row_to_coeff.push(col);
            }
            return Ok(Some(SimpleLowerBounds {
                lower_bounds,
                row_to_coeff,
                coeff_to_row,
            }));
        }
        _ => return Ok(None),
    };
    if constraints.a.ncols() != p || constraints.a.nrows() != constraints.b.len() {
        return Err(CustomFamilyError::ConstraintViolation {
            reason: "linear constraints: shape mismatch".to_string(),
        });
    }
    let mut lower_bounds = Array1::from_elem(p, f64::NEG_INFINITY);
    let mut coeff_to_row = vec![None; p];
    let mut row_to_coeff = Vec::with_capacity(constraints.a.nrows());
    for row in 0..constraints.a.nrows() {
        let mut coeff_idx = None;
        let mut coeff_value = 0.0;
        for col in 0..p {
            let value = constraints.a[[row, col]];
            // A row is a simple bound only when every other coefficient is exactly
            // zero; a small nonzero entry is still part of the constraint.
            if value == 0.0 {
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
) -> Result<(Array1<f64>, Vec<usize>), CustomFamilyError> {
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
    .map_err(|e| CustomFamilyError::trial_point(format!("lower-bound Newton solve failed: {e}")))?;
    let mut beta_new = beta_start + &delta;
    // The active-set QP leaves its final KKT set in `active_coeffs`. Each active
    // coefficient took the step `lower − β` to its bound, which one rounding can
    // leave just off it, so it is placed exactly on the bound. The reported face
    // is the solver's own, not a band re-derived from the resulting values.
    for &coeff in &active_coeffs {
        let lower = bounds.lower_bounds[coeff];
        if lower.is_finite() {
            beta_new[coeff] = lower;
        }
    }
    project_to_lower_bounds(&mut beta_new, &bounds.lower_bounds);
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
) -> Result<&Array1<f64>, CustomFamilyError> {
    if let Some((i, &wi)) = working_weights
        .iter()
        .enumerate()
        .find(|&(_, &wi)| !wi.is_finite())
    {
        return Err(CustomFamilyError::trial_point(format!(
            "invalid diagonal working weight at row {i}: {wi} (working curvature must be finite)"
        )));
    }
    Ok(working_weights)
}

pub(crate) trait ParameterBlockUpdater {
    fn compute_update_step(
        &self,
        ctx: &BlockUpdateContext<'_>,
    ) -> Result<BlockUpdateResult, CustomFamilyError>;
}

pub(crate) struct DiagonalBlockUpdater<'a> {
    pub(crate) working_response: &'a Array1<f64>,
    pub(crate) working_weights: &'a Array1<f64>,
}

pub(crate) struct NaturalDiagonalBlockUpdater<'a> {
    pub(crate) score: &'a Array1<f64>,
    pub(crate) observed_curvature: &'a Array1<f64>,
}

impl ParameterBlockUpdater for NaturalDiagonalBlockUpdater<'_> {
    fn compute_update_step(
        &self,
        ctx: &BlockUpdateContext<'_>,
    ) -> Result<BlockUpdateResult, CustomFamilyError> {
        let n = ctx.spec.solver_design().nrows();
        if self.score.len() != n || self.observed_curvature.len() != n {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "family natural-diagonal working-set size mismatch on block {} ({}): score={}, curvature={}, rows={n}",
                    ctx.block_idx,
                    ctx.spec.name,
                    self.score.len(),
                    self.observed_curvature.len(),
                ),
            });
        }
        if let Some((row, value)) = self
            .score
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(CustomFamilyError::trial_point(format!(
                "invalid natural-coordinate score at row {row}: {value}"
            )));
        }
        let curvature = certify_finite_working_weights(self.observed_curvature)?;
        with_block_geometry(ctx.family, ctx.states, ctx.spec, ctx.block_idx, |design, _| {
            let gradient = design.transpose_vector_multiply(self.score);
            let hessian = design
                .xt_diag_x_signed_op(FiniteSignedWeightsView::try_from_array(curvature)?)?;
            ExactNewtonBlockUpdater {
                gradient: &gradient,
                hessian: &SymmetricMatrix::Dense(hessian),
            }
            .compute_update_step(ctx)
        })
    }
}

impl ParameterBlockUpdater for DiagonalBlockUpdater<'_> {
    fn compute_update_step(
        &self,
        ctx: &BlockUpdateContext<'_>,
    ) -> Result<BlockUpdateResult, CustomFamilyError> {
        if self.working_response.len() != ctx.spec.design.nrows()
            || self.working_weights.len() != ctx.spec.design.nrows()
        {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "family diagonal working-set size mismatch on block {} ({})",
                    ctx.block_idx, ctx.spec.name
                ),
            });
        }

        let working_weights =
            certify_finite_working_weights(self.working_weights).map_err(|e| {
                format!(
                    "block {} ({}) diagonal solve: {e}",
                    ctx.block_idx, ctx.spec.name
                )
            })?;

        if let Some(constraints) = ctx.linear_constraints {
            check_linear_feasibility(&ctx.states[ctx.block_idx].beta, constraints).map_err(
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
                    .map_err(|error| CustomFamilyError::trial_point(error.to_string()))
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
                    .solve_system_with_ridge_floor(
                        working_weights,
                        &xtwy,
                        Some(ctx.s_lambda),
                        ctx.options.ridge_floor,
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
    ) -> Result<BlockUpdateResult, CustomFamilyError> {
        self.solve_for_rhs(ctx, self.newton_rhs(ctx)?)
            .map(|(_, step)| step)
    }
}

impl ExactNewtonBlockUpdater<'_> {
    /// The Newton update's right-hand side `gradient − S_λβ`.
    fn newton_rhs(&self, ctx: &BlockUpdateContext<'_>) -> Result<Array1<f64>, CustomFamilyError> {
        let p = ctx.spec.design.ncols();
        if self.gradient.len() != p {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "block {} exact-newton gradient length mismatch: got {}, expected {p}",
                    ctx.block_idx,
                    self.gradient.len()
                ),
            });
        }
        // Solve in delta-space for both constrained and unconstrained blocks.
        // That keeps the linear system consistent even when we add a
        // numerical ridge to stabilize an indefinite exact-Newton Hessian.
        Ok(self.gradient - &ctx.s_lambda.dot(&ctx.states[ctx.block_idx].beta))
    }

    /// The block step `δ` for the right-hand side `rhs_step`: `(H + S_λ) δ = rhs_step` on the
    /// stabilized penalized curvature and under the block's linear constraints, returned as
    /// `β + δ`. The Newton update takes it with `rhs_step = gradient − S_λβ`; a branch
    /// continuation's IFT predictor takes it with `−Σ_k Δρ_k λ_k S_k β̂` (gam#2973), so both
    /// steps are solved by one solver on one curvature.
    pub(crate) fn step_for_rhs(
        &self,
        ctx: &BlockUpdateContext<'_>,
        rhs_step: Array1<f64>,
    ) -> Result<BlockUpdateResult, CustomFamilyError> {
        self.solve_for_rhs(ctx, rhs_step).map(|(_, step)| step)
    }

    /// The one exact-Newton block step routine: the stabilized curvature and the step on it for
    /// `rhs_step`. Every block step comes from here, and the curvature is returned for a caller
    /// that also sizes the step's resolution on it.
    fn solve_for_rhs(
        &self,
        ctx: &BlockUpdateContext<'_>,
        rhs_step: Array1<f64>,
    ) -> Result<(Array2<f64>, BlockUpdateResult), CustomFamilyError> {
        let lhs_dense = self.stabilized_penalized_lhs(ctx, rhs_step.len())?;
        let step = self.step_on_lhs(ctx, &lhs_dense, rhs_step)?;
        Ok((lhs_dense, step))
    }

    /// The Newton update step at `ctx` ([`ParameterBlockUpdater::compute_update_step`], through
    /// the same routine) beside its arithmetic resolution (gam#2973), for the right-hand side
    /// `gradient − S_λβ` known to within `rhs_band` per coordinate.
    ///
    /// The resolution is the largest Euclidean norm the step can take from that rounding alone:
    /// the solve's image of the band on the face the step ends on
    /// ([`newton_step_rounding_image`]), plus the rounding of forming `β + δ`. A step within it
    /// is zero on this arithmetic, so the iterate it starts from is at its root. The band is the
    /// caller's; the rounding of the curvature, of the solve and of forming `S_λ` from its terms
    /// is not charged, so the resolution can only be too narrow, never resolve a correction the
    /// arithmetic cannot.
    pub(crate) fn update_step_with_resolution(
        &self,
        ctx: &BlockUpdateContext<'_>,
        rhs_band: &Array1<f64>,
    ) -> Result<(BlockUpdateResult, f64), CustomFamilyError> {
        let (lhs_dense, step) = self.solve_for_rhs(ctx, self.newton_rhs(ctx)?)?;
        let face = match (ctx.linear_constraints, step.active_set.as_deref()) {
            (Some(constraints), Some(active)) => {
                let rows = constraints.gather_rows(active)?;
                match active_constraint_tangent_geometry(&rows.a)? {
                    ActiveConstraintTangentGeometry::Tangent(z) => NewtonStepFace::Tangent(z),
                    ActiveConstraintTangentGeometry::FullyPinned => NewtonStepFace::Pinned,
                }
            }
            _ => NewtonStepFace::Free,
        };
        let solve = newton_step_rounding_image(&lhs_dense, &face, rhs_band)?;
        let landing = gam_linalg::roundoff::UNIT_ROUNDOFF
            * step
                .beta_new_raw
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();
        Ok((step, solve + landing))
    }

    /// `H + S_λ` stabilized as every exact-Newton block step solves on it.
    fn stabilized_penalized_lhs(
        &self,
        ctx: &BlockUpdateContext<'_>,
        rhs_len: usize,
    ) -> Result<Array2<f64>, CustomFamilyError> {
        let p = ctx.spec.design.ncols();
        if self.hessian.nrows() != p || self.hessian.ncols() != p || rhs_len != p {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "block {} exact-newton step shape mismatch: Hessian {}x{} and right-hand \
                     side {}, expected {p}",
                    ctx.block_idx,
                    self.hessian.nrows(),
                    self.hessian.ncols(),
                    rhs_len,
                ),
            });
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
        Ok(lhs_dense)
    }

    /// The block step for `rhs_step` on the stabilized curvature `lhs_dense`.
    fn step_on_lhs(
        &self,
        ctx: &BlockUpdateContext<'_>,
        lhs_dense: &Array2<f64>,
        rhs_step: Array1<f64>,
    ) -> Result<BlockUpdateResult, CustomFamilyError> {
        let p = ctx.spec.design.ncols();
        if let Some(constraints) = ctx.linear_constraints {
            check_linear_feasibility(&ctx.states[ctx.block_idx].beta, constraints).map_err(
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
                let rhs_beta = lhs_dense.dot(&ctx.states[ctx.block_idx].beta) + &rhs_step;
                solve_quadratic_with_simple_lower_bounds(
                    lhs_dense,
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
                        lhs_dense,
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
            // Solving for the step rather than `beta_new` keeps the right-hand side
            // exact when the solver acts on a projected or shifted system: solving
            // directly for `beta_new` would carry β's own component through that
            // modification, which distorts the step and can trap exact-Newton block
            // updates on nonconvex blocks such as survival `log_sigma`.
            // Every family takes the Newton step through
            // `strict_solve_spd_or_spectral_step`: the plain Cholesky step on an SPD
            // H_β + S; the Moore–Penrose step on the resolved positive eigenspace
            // when H_β + S is singular positive semidefinite; and the minimally
            // shifted step when it carries negative curvature beyond its rounding
            // band, so a saddle direction still receives a descent step. It fails
            // only when that eigendecomposition fails or the step is non-finite, and
            // the failure is returned, not traded for a diagonally scaled
            // steepest-descent step. β is recovered in the raw basis, so
            // dimensionality and identifiability are untouched.
            let delta = strict_solve_spd_or_spectral_step(lhs_dense, &rhs_step)?;
            let beta = &ctx.states[ctx.block_idx].beta + &delta;
            Ok(BlockUpdateResult {
                beta_new_raw: beta,
                active_set: None,
            })
        }
    }
}

/// The face an exact-Newton block step ends on, which its right-hand side's rounding moves it
/// along.
enum NewtonStepFace {
    /// No active constraint: every coordinate is free.
    Free,
    /// The orthonormal tangent `Z` of the active constraint rows.
    Tangent(Array2<f64>),
    /// The active rows pin every coordinate, so the step does not depend on the right-hand side.
    Pinned,
}

/// The largest Euclidean norm the step on `face`, `Z (ZᵀAZ)⁻¹ Zᵀ e` with `A = lhs`, reaches over
/// right-hand-side errors `|e_i| ≤ band_i` (gam#2973).
///
/// With `ZᵀAZ = VΓVᵀ` and `W = ZV` (`Z = I` on a free block), each coordinate of the image is
/// `|Σ_k w_jk (Σ_i w_ik e_i) / γ_k| ≤ Σ_k |w_jk| (Σ_i |w_ik| band_i) / |γ_k|`, a componentwise bound
/// that holds for every sign pattern of `e`. An exactly singular face system resolves no step, so
/// its image is unbounded.
fn newton_step_rounding_image(
    lhs: &Array2<f64>,
    face: &NewtonStepFace,
    band: &Array1<f64>,
) -> Result<f64, CustomFamilyError> {
    if band.len() != lhs.nrows() {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: format!(
                "exact-Newton step resolution: right-hand-side band has {} coordinates for a \
                 {}-coefficient block",
                band.len(),
                lhs.nrows()
            ),
        });
    }
    let mut reduced = match face {
        NewtonStepFace::Pinned => return Ok(0.0),
        NewtonStepFace::Free => lhs.clone(),
        NewtonStepFace::Tangent(z) => z.t().dot(lhs).dot(z),
    };
    symmetrize_dense_in_place(&mut reduced);
    let (eigenvalues, eigenvectors) = FaerEigh::eigh(&reduced, Side::Lower).map_err(|error| {
        CustomFamilyError::NumericalFailure {
            reason: format!("exact-Newton step resolution: eigendecomposition failed: {error}"),
        }
    })?;
    if eigenvalues.iter().any(|&gamma| gamma == 0.0) {
        return Ok(f64::INFINITY);
    }
    let w = match face {
        NewtonStepFace::Tangent(z) => z.dot(&eigenvectors),
        NewtonStepFace::Free | NewtonStepFace::Pinned => eigenvectors,
    };
    let modes: Vec<f64> = (0..w.ncols())
        .map(|k| {
            (0..w.nrows())
                .map(|i| w[[i, k]].abs() * band[i])
                .sum::<f64>()
                / eigenvalues[k].abs()
        })
        .collect();
    Ok((0..w.nrows())
        .map(|j| {
            let coordinate: f64 = (0..w.ncols()).map(|k| w[[j, k]].abs() * modes[k]).sum();
            coordinate * coordinate
        })
        .sum::<f64>()
        .sqrt())
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
            BlockWorkingSet::NaturalDiagonal {
                score,
                observed_curvature,
            } => Box::new(NaturalDiagonalBlockUpdater {
                score,
                observed_curvature,
            }),
            BlockWorkingSet::ExactNewton { gradient, hessian } => {
                Box::new(ExactNewtonBlockUpdater { gradient, hessian })
            }
        }
    }
}

/// The QP entry gate: does `beta` satisfy `constraints` to the solver's
/// primal-feasibility contract?
///
/// The verdict is [`ConstraintSet::max_scaled_violation`] against
/// [`ACTIVE_SET_PRIMAL_FEASIBILITY_TOL`] — the same quantity and the same
/// threshold the active-set solver certifies its own returned iterate against,
/// referenced rather than re-spelled.
///
/// Both halves of that changed in gam#2719. The gate used to take a `tol`
/// parameter that all four of its call sites passed as the literal `1e-8`, so
/// the contract could drift out from under them silently; and it compared the
/// RAW `b − Aβ` while the contract is stated on unit-normalized rows. The
/// second is not cosmetic. The survival time-derivative guard rows are
/// pre-normalized by `max(‖row‖, |rhs|, 1)`, so `‖a‖ ≤ 1` and
/// `scaled ≥ raw`: a point could be infeasible to the solver's own scan yet
/// pass this gate, and a step the repaired fraction-to-boundary rule now admits
/// is exactly such a point. The gate that decides whether to project has to be
/// the gate the step rule was sized against, or the projection never fires.
pub(crate) fn check_linear_feasibility(
    beta: &Array1<f64>,
    constraints: &ConstraintSet,
) -> Result<(), CustomFamilyError> {
    if constraints.ncols() != beta.len() {
        return Err(CustomFamilyError::ConstraintViolation {
            reason: "linear constraints: shape mismatch".to_string(),
        });
    }
    let (worst_scaled, worst_row) = constraints.max_scaled_violation(beta.view()).map_err(|e| {
        CustomFamilyError::ConstraintViolation {
            reason: format!("linear constraints: {e}"),
        }
        .to_string()
    })?;
    if worst_scaled > gam_solve::active_set::ACTIVE_SET_PRIMAL_FEASIBILITY_TOL {
        let worst_idx = worst_row.unwrap_or(0);
        // #1108 DIAGNOSTIC: pin down whether this infeasible β is PROJECTABLE
        // (→ a wiring bug: the seed bypassed projection) or genuinely outside a
        // hard polytope. Report the worst row's ‖a‖, the raw violation, the
        // β magnitude, and the outcome of the exact active-set projection of
        // THIS β onto these constraints.
        let norm_worst = constraints.row_norm(worst_idx).unwrap_or(f64::NAN);
        // The raw violation is the scaled one un-normalized. A vacuous row with
        // a positive bound reports an INFINITE scaled violation against a zero
        // norm, whose product is NaN; report its bound, which is the shortfall
        // no β can close.
        let raw_worst = if norm_worst > 0.0 {
            worst_scaled * norm_worst
        } else {
            constraints.bound(worst_idx).unwrap_or(f64::NAN)
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
                "infeasible iterate: scaled violation={worst_scaled:.3e} exceeds the \
                 primal-feasibility contract {:.3e} at constraint row {worst_idx} \
                 [#1108 diag: rows={}, ‖a_row‖={norm_worst:.3e}, \
                 raw_viol={raw_worst:.3e}, |β|∞={beta_inf:.3e}, {interior_outcome}, \
                 {simple_bounds_path}]",
                gam_solve::active_set::ACTIVE_SET_PRIMAL_FEASIBILITY_TOL,
                constraints.nrows()
            ),
        });
    }
    Ok(())
}

/// One additive term `½·λ·‖R·β‖²` of the custom-family penalty (#2954), with
/// `S = RᵀR` the structural root [`gam_problem::structural_penalty_root`] forms.
///
/// A block-local penalty reads its block's coefficients (`block = Some(b)`); a
/// joint-bundle penalty reads every block's coefficients concatenated in block
/// order (`block = None`), the layout `flatten_state_betas` writes. `columns`
/// are the columns of that vector `root` acts on.
#[derive(Clone, Debug)]
pub struct PenaltyRootTerm {
    pub block: Option<usize>,
    pub columns: std::ops::Range<usize>,
    pub lambda: f64,
    pub root: Arc<Array2<f64>>,
}

impl PenaltyRootTerm {
    /// `RᵀR` embedded at [`Self::columns`] of a `width × width` matrix: the penalty
    /// matrix this term's root defines, in its block's (or the joint vector's)
    /// coordinates.
    pub fn embedded_penalty(&self, width: usize) -> Array2<f64> {
        let mut matrix = Array2::<f64>::zeros((width, width));
        matrix
            .slice_mut(ndarray::s![self.columns.clone(), self.columns.clone()])
            .assign(&self.root.t().dot(self.root.as_ref()));
        matrix
    }
}

/// The penalty `½Σ_k λ_k‖R_kβ‖²` at one coefficient vector, beside the bound on
/// its rounding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PenaltyValue {
    pub value: f64,
    /// Depth `m` of the accumulation that formed `value`, for the growth factor
    /// `γ_m` (see [`BlockPenaltyRoots::value`]).
    pub depth: usize,
    /// The magnitude that accumulation carries: `|fl(value) − value| ≤
    /// γ_depth·magnitude` for the roots as given.
    pub magnitude: f64,
}

impl PenaltyValue {
    /// `γ_depth·magnitude`: the bound on the value's rounding.
    pub fn band(&self) -> f64 {
        gam_linalg::roundoff::accumulation_growth(self.depth) * self.magnitude
    }
}

/// `value(β + δ) − value(β)`, formed without differencing two values, beside
/// the bound on its rounding (#2954, for #2977's trust-region acceptance).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PenaltyIncrement {
    pub increment: f64,
    pub band: f64,
}

/// The custom-family penalty as one function (#2954): every value, increment,
/// gradient and curvature of `½Σ_k λ_k β_kᵀS_kβ_k` (plus the joint bundle's
/// full-width terms) is formed from the same structural roots `S_k = R_kᵀR_k`.
///
/// The dense `½βᵀS_λβ` this replaces was an unstable formula for a
/// well-conditioned quantity. The stored `S_k`'s null eigenvalues are
/// `O(u·‖S_k‖)`, so near the `λ → ∞` limit, where `β` lies in `S_k`'s null
/// space, the dense value carries `½λσ_null(uᵀβ)²` while the structural value
/// vanishes. On `declared_latent_law_2923` that is `λ·2.14e-16` at every point,
/// 1.8e-3 at `ρ = 29.78`, and the root form's error there is 2.8e-15. The
/// curvature `S_λ = Σ_k λ_k R_kᵀR_k` and the gradient `S_λβ` are the exact
/// Hessian and gradient of the root value, so an inner Newton model and the
/// objective it is accepted against describe one function, and the outer
/// criterion's value and its ρ-gradient read the same roots.
#[derive(Clone, Debug)]
pub struct BlockPenaltyRoots {
    terms: Vec<PenaltyRootTerm>,
    s_lambdas: Vec<Array2<f64>>,
    /// Whether any term is a joint-bundle term, which reads the concatenated
    /// coefficients.
    has_joint: bool,
}

impl BlockPenaltyRoots {
    /// Root every block penalty of `specs` at the strengths `block_log_lambdas`,
    /// and take the joint bundle's own validated roots.
    ///
    /// A block penalty's rank is [`gam_problem::structural_penalty_root`]'s: the
    /// one rank rule's resolved count, capped by a declared nullity
    /// (`nullspace_dims`, one per penalty), each of which only removes
    /// directions. Undeclared, the count is the rank; that is a named step toward
    /// declared null bases (gam#3023), not a second rule.
    pub fn new(
        specs: &[ParameterBlockSpec],
        block_log_lambdas: &[Array1<f64>],
        joint: Option<&gam_problem::JointPenaltyBundle>,
    ) -> Result<Self, CustomFamilyError> {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        if block_log_lambdas.len() != specs.len() {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "penalty roots: {} log-smoothing blocks for {} parameter blocks",
                    block_log_lambdas.len(),
                    specs.len()
                ),
            });
        }
        // Each block's roots depend only on that block's penalties, so blocks are
        // rooted on rayon workers and collected in block order.
        let per_block = (0..specs.len())
            .into_par_iter()
            .map(|b| block_penalty_roots(b, &specs[b], &block_log_lambdas[b]))
            .collect::<Result<Vec<_>, CustomFamilyError>>()?;
        let total: usize = specs.iter().map(|spec| spec.design.ncols()).sum();
        let mut s_lambdas = Vec::with_capacity(specs.len());
        let mut terms = Vec::new();
        for (s_lambda, block_terms) in per_block {
            s_lambdas.push(s_lambda);
            terms.extend(block_terms);
        }
        if let Some(bundle) = joint {
            for (root, &lambda) in bundle.roots().iter().zip(bundle.lambdas().iter()) {
                terms.push(PenaltyRootTerm {
                    block: None,
                    columns: 0..total,
                    lambda,
                    root: Arc::new(root.clone()),
                });
            }
        }
        let has_joint = terms.iter().any(|term| term.block.is_none());
        Ok(Self {
            terms,
            s_lambdas,
            has_joint,
        })
    }

    /// Each block's curvature `S_λ = Σ_k λ_k R_kᵀR_k`: the Hessian of the block
    /// terms of [`Self::value`]. The joint bundle's terms act on the full width
    /// and are applied by the bundle itself.
    pub fn s_lambdas(&self) -> &[Array2<f64>] {
        &self.s_lambdas
    }

    /// Every term of the penalty, block terms in block order, then the joint
    /// bundle's: the root accessor a caller that forms its own contraction
    /// reads, so it prices exactly the function this evaluator does.
    pub fn terms(&self) -> &[PenaltyRootTerm] {
        &self.terms
    }

    /// The depth of the accumulation [`Self::value`] charges, which the roots'
    /// shapes alone fix: the deepest term's `q + r + 2`, plus one per term summed.
    pub fn accumulation_depth(&self) -> usize {
        self.terms
            .iter()
            .map(|term| term.root.ncols() + term.root.nrows() + 2)
            .max()
            .unwrap_or(0)
            + self.terms.len()
    }

    /// The concatenated coefficients a joint term reads, formed only when a
    /// joint term exists (an empty vector otherwise, which no term reads).
    fn joint_coefficients<'a, B>(&self, block_beta: &B) -> Array1<f64>
    where
        B: Fn(usize) -> ndarray::ArrayView1<'a, f64>,
    {
        if !self.has_joint {
            return Array1::zeros(0);
        }
        (0..self.s_lambdas.len())
            .flat_map(|block| block_beta(block).to_vec())
            .collect()
    }

    fn value_with<'a, B>(&self, block_beta: B) -> PenaltyValue
    where
        B: Fn(usize) -> ndarray::ArrayView1<'a, f64>,
    {
        let joint = self.joint_coefficients(&block_beta);
        let mut value = 0.0_f64;
        let mut magnitude = 0.0_f64;
        for term in &self.terms {
            let coefficients = match term.block {
                Some(block) => block_beta(block),
                None => joint.view(),
            };
            let term_value =
                root_term_value(term, coefficients.slice(ndarray::s![term.columns.clone()]));
            value += term_value.value;
            magnitude += term_value.magnitude;
        }
        PenaltyValue {
            value,
            depth: self.accumulation_depth(),
            magnitude,
        }
    }

    /// `½Σ_k λ_k‖R_kβ‖²` over every term, with the accumulation that bounds its
    /// rounding.
    ///
    /// Error model (Higham, *ASNA* 2nd ed., §3.1 and 3.5), for the roots as given.
    /// Forming `y = R_kβ` by `q`-term dot products leaves `|ŷ_i − y_i| ≤ γ_q·m_i`
    /// with `m = |R_k||β|`, so `|ŷ_i² − y_i²| ≤ γ_q·m_i(2|ŷ_i| + γ_q·m_i)`.
    /// Squaring and summing the `r` rows, scaling by `λ_k` and halving add at most
    /// `γ_(r+2)` of `½λ_k‖ŷ‖²`. Since `γ_a + γ_b ≤ γ_(a+b)`, a term rounds by at most
    /// `γ_(q+r+2)·½λ_k(2|ŷ|ᵀm + γ_q‖m‖² + ‖ŷ‖²)`, and summing the `T` terms raises
    /// the depth by `T`. That is [`PenaltyValue`]'s `(depth, magnitude)`, whose
    /// `γ_depth·magnitude` is its [`PenaltyValue::band`].
    ///
    /// On a null direction `ŷ ≈ 0` and the magnitude is `½λγ_q‖m‖²`, so the band
    /// is `O(λu²)`: the root form carries no first-order `λ`-amplified term, which
    /// the dense `½βᵀS_λβ` could not avoid.
    pub fn value(&self, betas: &[Array1<f64>]) -> PenaltyValue {
        self.value_with(|block| betas[block].view())
    }

    /// [`Self::value`] at the coefficients `states` carry.
    pub fn value_of_states(&self, states: &[ParameterBlockState]) -> PenaltyValue {
        self.value_with(|block| states[block].beta.view())
    }

    /// Block `block`'s own terms of [`Self::value`] at `beta`, excluding the joint
    /// bundle's full-width terms. A blockwise update that changes one block
    /// replaces exactly this part of the total.
    pub fn block_value(&self, block: usize, beta: &Array1<f64>) -> f64 {
        self.block_penalty_value(block, beta).value
    }

    /// [`Self::block_value`] with the accumulation that bounds its rounding, at
    /// the depth [`Self::value`] charges.
    pub fn block_penalty_value(&self, block: usize, beta: &Array1<f64>) -> PenaltyValue {
        let mut value = 0.0_f64;
        let mut magnitude = 0.0_f64;
        for term in self.terms.iter().filter(|term| term.block == Some(block)) {
            let term_value = root_term_value(term, beta.slice(ndarray::s![term.columns.clone()]));
            value += term_value.value;
            magnitude += term_value.magnitude;
        }
        PenaltyValue {
            value,
            depth: self.accumulation_depth(),
            magnitude,
        }
    }

    /// `value(β + δ) − value(β) = Σ_k λ_k[(R_kδ)ᵀ(R_kβ) + ½‖R_kδ‖²]`, formed
    /// directly (#2954, for #2977), over the block and joint terms in the order
    /// [`Self::value`] sums them.
    ///
    /// With `a = R_kβ`, `d = R_kδ` formed with errors `e_a = γ_q|R_k||β|` and
    /// `e_d = γ_q|R_k||δ|`, the formation moves the term by at most
    /// `λ_k[|d̂|ᵀe_a + e_dᵀ|â| + e_dᵀe_a + |d̂|ᵀe_d + ½‖e_d‖²]`, and the `r`-term
    /// sums with the scaling round by `γ_(r+2)·λ_k(|d̂|ᵀ|â| + ½‖d̂‖²)`; the sum
    /// over terms adds `γ_T` of the total's magnitude.
    pub fn penalty_increment(
        &self,
        betas: &[Array1<f64>],
        deltas: &[Array1<f64>],
    ) -> PenaltyIncrement {
        let joint_betas = self.joint_coefficients(&|block| betas[block].view());
        let joint_deltas = self.joint_coefficients(&|block| deltas[block].view());
        let mut increment = 0.0_f64;
        let mut magnitude = 0.0_f64;
        let mut band = 0.0_f64;
        for term in &self.terms {
            let (beta, delta) = match term.block {
                Some(block) => (betas[block].view(), deltas[block].view()),
                None => (joint_betas.view(), joint_deltas.view()),
            };
            let beta = beta.slice(ndarray::s![term.columns.clone()]);
            let delta = delta.slice(ndarray::s![term.columns.clone()]);
            let root = term.root.as_ref();
            let growth_q = gam_linalg::roundoff::accumulation_growth(root.ncols());
            let abs_root = root.mapv(f64::abs);
            let a = root.dot(&beta);
            let d = root.dot(&delta);
            let e_a = abs_root.dot(&beta.mapv(f64::abs)) * growth_q;
            let e_d = abs_root.dot(&delta.mapv(f64::abs)) * growth_q;
            let abs_a = a.mapv(f64::abs);
            let abs_d = d.mapv(f64::abs);
            let own = 0.5 * d.dot(&d);
            increment += term.lambda * (d.dot(&a) + own);
            let term_magnitude = term.lambda * (abs_d.dot(&abs_a) + own);
            magnitude += term_magnitude;
            band += term.lambda
                * (abs_d.dot(&e_a)
                    + e_d.dot(&abs_a)
                    + e_d.dot(&e_a)
                    + abs_d.dot(&e_d)
                    + 0.5 * e_d.dot(&e_d))
                + gam_linalg::roundoff::accumulation_growth(root.nrows() + 2) * term_magnitude;
        }
        PenaltyIncrement {
            increment,
            band: band + gam_linalg::roundoff::accumulation_growth(self.terms.len()) * magnitude,
        }
    }
}

/// Block `b`'s roots at `log_lambdas` and its curvature `S_λ = Σ_k λ_k R_kᵀR_k`
/// ([`BlockPenaltyRoots::new`] for one block). Every consumer of a block's
/// assembled penalty reads this, so the curvature, the pseudo-log-determinant
/// and the value describe one function.
pub(crate) fn block_penalty_roots(
    b: usize,
    spec: &ParameterBlockSpec,
    log_lambdas: &Array1<f64>,
) -> Result<(Array2<f64>, Vec<PenaltyRootTerm>), CustomFamilyError> {
    if log_lambdas.len() != spec.penalties.len() {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: format!(
                "block {b} log-smoothing parameter length {} does not match \
                 penalties {}",
                log_lambdas.len(),
                spec.penalties.len()
            ),
        });
    }
    let lambdas =
        exact_lambdas_from_log_strengths(log_lambdas, &format!("inner block {b} log strength"))?;
    let declared = spec.nullspace_dims.len() == spec.penalties.len();
    let p = spec.design.ncols();
    let mut s_lambda = Array2::<f64>::zeros((p, p));
    let mut terms = Vec::with_capacity(spec.penalties.len());
    for (k, penalty) in spec.penalties.iter().enumerate() {
        // A block-local penalty declares the nullity of its own local matrix, the
        // convention the standard route's rank analysis reads
        // (`penalty_spec_local_matrix`), so it is rooted on those columns alone;
        // every other form is rooted as the block-wide matrix it is. No producer
        // carries its construction's formation band yet, so it is 0 here and a
        // declared nullity removes the structural zeros that rounding left
        // resolved (#2954).
        let (matrix, columns) = penalty_structure(penalty, p);
        let root = gam_problem::structural_penalty_root(
            &matrix,
            declared.then(|| spec.nullspace_dims[k]),
            0.0,
        )
        .map_err(|error| CustomFamilyError::InvalidInput {
            context: "custom-family penalty root",
            reason: format!("block {b} ({}) penalty {k}: {error}", spec.name),
        })?;
        s_lambda
            .slice_mut(ndarray::s![columns.clone(), columns.clone()])
            .scaled_add(lambdas[k], &root.t().dot(&root));
        terms.push(PenaltyRootTerm {
            block: Some(b),
            columns,
            lambda: lambdas[k],
            root: Arc::new(root),
        });
    }
    Ok((s_lambda, terms))
}

/// The matrix a penalty's declared nullity describes and the block columns it
/// acts on: a block-local penalty's own local matrix at its column range, and the
/// block-wide dense form of every other kind. Precision labels and fixed
/// strengths wrap the matrix without changing it.
fn penalty_structure(
    penalty: &PenaltyMatrix,
    width: usize,
) -> (Array2<f64>, std::ops::Range<usize>) {
    match penalty {
        PenaltyMatrix::Blockwise {
            local, col_range, ..
        } => (local.clone(), col_range.clone()),
        PenaltyMatrix::Labeled { inner, .. } | PenaltyMatrix::Fixed { inner, .. } => {
            penalty_structure(inner, width)
        }
        PenaltyMatrix::Dense(_)
        | PenaltyMatrix::Diagonal(_)
        | PenaltyMatrix::KroneckerFactored { .. } => (penalty.to_dense(), 0..width),
    }
}

/// The curvature half of [`block_penalty_roots`].
pub(crate) fn block_s_lambda(
    b: usize,
    spec: &ParameterBlockSpec,
    log_lambdas: &Array1<f64>,
) -> Result<Array2<f64>, CustomFamilyError> {
    block_penalty_roots(b, spec, log_lambdas).map(|(s_lambda, _)| s_lambda)
}

/// `½λ‖Rβ‖²` for one term, with its accumulation, by the error model at
/// [`BlockPenaltyRoots::value`].
fn root_term_value(term: &PenaltyRootTerm, beta: ndarray::ArrayView1<'_, f64>) -> PenaltyValue {
    let root = term.root.as_ref();
    let (r, q) = root.dim();
    let y = root.dot(&beta);
    // `m = |R||β|`: the magnitude each `q`-term row product accumulates.
    let m = root.mapv(f64::abs).dot(&beta.mapv(f64::abs));
    let squares = y.dot(&y);
    PenaltyValue {
        value: 0.5 * term.lambda * squares,
        depth: q + r + 2,
        magnitude: 0.5
            * term.lambda
            * (2.0 * y.mapv(f64::abs).dot(&m)
                + gam_linalg::roundoff::accumulation_growth(q) * m.dot(&m)
                + squares),
    }
}

pub(crate) fn block_penalized_hessian_vector(
    spec: &ParameterBlockSpec,
    work: &BlockWorkingSet,
    s_lambda: &Array2<f64>,
    direction: &Array1<f64>,
) -> Array1<f64> {
    let mut hpen = match work {
        BlockWorkingSet::ExactNewton { hessian, .. } => hessian.dot(direction),
        BlockWorkingSet::Diagonal {
            working_weights, ..
        }
        | BlockWorkingSet::NaturalDiagonal {
            observed_curvature: working_weights,
            ..
        } => {
            let solver_design = spec.solver_design();
            let x_direction = solver_design.matrixvectormultiply(direction);
            let wx_direction = &x_direction * working_weights;
            solver_design.transpose_vector_multiply(&wx_direction)
        }
    };
    hpen += &s_lambda.dot(direction);
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
) -> Result<Array1<f64>, CustomFamilyError> {
    let mut diagonal = match work {
        BlockWorkingSet::ExactNewton { hessian, .. } => symmetric_matrix_diagonal(hessian),
        BlockWorkingSet::Diagonal {
            working_weights, ..
        }
        | BlockWorkingSet::NaturalDiagonal {
            observed_curvature: working_weights,
            ..
        } => spec.design.diag_gram(working_weights)?,
    };
    if diagonal.len() != s_lambda.nrows() || s_lambda.nrows() != s_lambda.ncols() {
        return Err(CustomFamilyError::trial_point(format!(
            "block penalized metric diagonal shape mismatch: diag={}, S={}x{}",
            diagonal.len(),
            s_lambda.nrows(),
            s_lambda.ncols()
        )));
    }
    for j in 0..diagonal.len() {
        diagonal[j] += s_lambda[[j, j]];
        diagonal[j] = positive_joint_diagonal_entry(diagonal[j]);
    }
    Ok(diagonal)
}

pub(crate) fn block_penalized_metric_norm(
    spec: &ParameterBlockSpec,
    work: &BlockWorkingSet,
    s_lambda: &Array2<f64>,
    direction: &Array1<f64>,
) -> Result<f64, CustomFamilyError> {
    let diagonal = block_penalized_metric_diagonal(spec, work, s_lambda)?;
    if diagonal.len() != direction.len() {
        return Err(CustomFamilyError::trial_point(format!(
            "block penalized metric direction length mismatch: direction={}, diag={}",
            direction.len(),
            diagonal.len()
        )));
    }
    Ok(joint_trust_region_metric_step_norm(direction, &diagonal))
}

pub(crate) fn truncate_block_step_to_metric_radius(
    spec: &ParameterBlockSpec,
    work: &BlockWorkingSet,
    s_lambda: &Array2<f64>,
    delta: Array1<f64>,
    radius: f64,
) -> Result<(Array1<f64>, f64), CustomFamilyError> {
    let norm = block_penalized_metric_norm(spec, work, s_lambda, &delta)?;
    if norm.is_finite() && norm > radius && radius > 0.0 {
        Ok((&delta * (radius / norm), radius))
    } else {
        Ok((delta, norm))
    }
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
) -> Result<(), CustomFamilyError> {
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
    ) })
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
pub(crate) fn validate_block_hessians_finite(eval: &FamilyEvaluation) -> Result<(), CustomFamilyError> {
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
) -> Result<(), CustomFamilyError> {
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
                        ) });
                    }
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn stable_logdet(matrix: &Array2<f64>) -> Result<f64, CustomFamilyError> {
    let mut a = matrix.clone();
    symmetrize_dense_in_place(&mut a);
    let p = a.nrows();

    // #2670 — one determinant semantics, so no dispatch. The deleted
    // `RidgeDeterminantMode::PositivePartApproximation` arm evaluated a smooth
    // positive-part spectral surrogate, which is a DIFFERENT estimand from the
    // exact SPD log-determinant this function's callers consume; no production
    // path ever selected it (every construction of the policy that reached it
    // lived under `#[cfg(test)]`). The surrogate itself is untouched and still
    // live in the REML outer engine's `DenseSpectralOperator`, which is where a
    // caller that wants it can get it together with its matching gradient.
    match a.cholesky(Side::Lower) {
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
                        "cholesky failed and the eigendecomposition also failed while computing the full logdet (p={p})"
                    )
                })?;
            let min_eig = evals.iter().copied().fold(f64::INFINITY, f64::min);
            let max_eig = evals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            // Negative-eigenvalue tolerance, relative to the spectrum scale.
            let neg_tol = CUSTOM_FAMILY_CONDITION_RELATIVE_FLOOR * max_eig.abs().max(1.0);
            if min_eig <= -neg_tol {
                return Err(CustomFamilyError::trial_point(format!(
                    "cholesky failed while computing the full logdet and the symmetric spectrum is genuinely indefinite (p={p}, min_eig={min_eig:.6e}, max_eig={max_eig:.6e}); an indefinite Hessian has no SPD log-determinant and defines no Laplace mode"
                )));
            }
            // PD but ill-conditioned: exact logdet from the positive spectrum,
            // flooring round-off-nonpositive eigenvalues at the negative-eigenvalue
            // tolerance so a numerically-singular direction contributes a bounded
            // (not `-inf`) term.
            Ok(evals.iter().map(|&e| e.max(neg_tol).ln()).sum())
        }
    }
}

pub(crate) fn symmetrize_dense_in_place(matrix: &mut Array2<f64>) {
    gam_linalg::matrix::symmetrize_in_place(matrix);
}

pub(crate) fn strict_solve_spd(
    matrix: &Array2<f64>,
    rhs: &Array1<f64>,
) -> Result<Array1<f64>, CustomFamilyError> {
    let mut sym = matrix.clone();
    symmetrize_dense_in_place(&mut sym);
    let chol = sym
        .cholesky(Side::Lower)
        .map_err(|_| "strict pseudo-laplace SPD solve failed".to_string())?;
    Ok(chol.solvevec(rhs))
}

/// Relative condition guard for rejecting genuinely negative spectrum in the
/// penalty-direction projection.
pub(crate) const CUSTOM_FAMILY_CONDITION_RELATIVE_FLOOR: f64 = 1e-14;

/// The exact-Newton block step: solve `H x = b`.
///
/// A strict Cholesky factorization that certifies `H` positive definite gives the
/// plain Newton step. Otherwise the symmetrized `H` is eigendecomposed and its
/// spectrum decides, against `band = positive_eigenvalue_threshold` of that
/// spectrum:
///
/// - no eigenvalue below `−band`: `H` is positive semidefinite to its resolution
///   and singular, so the step is the Moore–Penrose solve on the resolved positive
///   eigenspace, and a direction the data and penalty do not identify takes no
///   step;
/// - an eigenvalue below `−band`: `H` has genuine negative curvature, and
///   projecting those directions away would leave the step blind to them and
///   stall at a saddle. The step uses the minimal positive-definite shift
///   `δ = band − λ_min`, the shift `stabilize_exact_newton_penalized_lhs_in_place`
///   approaches by bisection, so every negative direction takes a descent step
///   that the trust region then truncates.
///
/// No picked ridge enters either branch.
pub(crate) fn strict_solve_spd_or_spectral_step(
    matrix: &Array2<f64>,
    rhs: &Array1<f64>,
) -> Result<Array1<f64>, CustomFamilyError> {
    if let Ok(x) = strict_solve_spd(matrix, rhs)
        && x.iter().all(|value| value.is_finite())
    {
        return Ok(x);
    }
    let p = matrix.nrows();
    if p == 0 {
        return Ok(Array1::<f64>::zeros(0));
    }
    let mut sym = matrix.clone();
    symmetrize_dense_in_place(&mut sym);
    let (evals, evecs) = FaerEigh::eigh(&sym, Side::Lower).map_err(|e| {
        format!(
            "strict pseudo-laplace SPD solve: Cholesky refused the system and the \
             eigendecomposition fallback also failed: {e}"
        )
    })?;
    let band =
        positive_eigenvalue_threshold(evals.as_slice().expect("eigh returns contiguous eigenvalues"));
    let min_eigenvalue = evals.iter().copied().fold(f64::INFINITY, f64::min);
    let shift = if min_eigenvalue < -band { band - min_eigenvalue } else { 0.0 };
    let mut q_t_rhs = Array1::<f64>::zeros(p);
    for k in 0..p {
        if shift == 0.0 && !(evals[k] > band) {
            continue;
        }
        let mut acc = 0.0;
        for i in 0..p {
            acc += evecs[[i, k]] * rhs[i];
        }
        q_t_rhs[k] = acc / (evals[k] + shift);
    }
    let mut x = Array1::<f64>::zeros(p);
    for i in 0..p {
        let mut acc = 0.0;
        for k in 0..p {
            acc += evecs[[i, k]] * q_t_rhs[k];
        }
        x[i] = acc;
    }
    if !x.iter().all(|value| value.is_finite()) {
        return Err(CustomFamilyError::NumericalFailure {
            reason: format!(
                "strict pseudo-laplace SPD solve: the spectral step is non-finite \
                 (min eigenvalue {min_eigenvalue:.3e}, band {band:.3e})"
            ),
        });
    }
    Ok(x)
}

/// Eigenpairs of a Laplace precision `M = H + S_λ (+ H_Φ)` that its generalized
/// log-determinant `log|M|₊` sums and its pseudo-inverse `M⁺` spans: the top
/// [`DenseSpectralOperator::identified_rank`] eigenpairs, the rule standard REML
/// prices `log|H|₊` on (#2901 V22), returned in index order.
///
/// An eigenvalue is resolved when it exceeds `M`'s rounding band `p·ε·‖M‖₂`.
/// `M ⪰ S_λ` puts `M`'s `k`-th largest eigenvalue at or above `S_λ`'s, so while
/// that many eigenvalues are positive the kept rank never falls below
/// `penalty_rank`, the rank of `S_λ` on `M`'s geometry
/// ([`penalty_rank_at_rounding_band`]). This replaces two rules standard REML
/// does not use: a Cholesky that certified `M` positive definite kept every
/// eigenpair however deep inside the band, and otherwise the cutoff
/// `100·p·ε·max σ` had a picked factor of 100 and no penalty floor (#2695). On
/// the #1569 survival location-scale fixture (job 445401) `max σ = 7.788e16`
/// put that cutoff at `4.5e4` beside eigenvalues of `0.013`–`0.5`. The band
/// there is `450`, and the eigensolver's backward error bounds each computed
/// eigenvalue only to within it.
///
/// Every eigenvalue the rank drops must itself lie inside that band. A dropped
/// positive eigenvalue always does (every resolved positive one is kept), but a
/// negative one below `−band` is resolved curvature: `M` is indefinite, `β̂` is
/// a saddle of the penalized objective, and no Laplace approximation exists
/// there. Pricing `log|M₊|` over the positive part would be a different
/// criterion, so that precision is refused by name, as
/// `DenseSpectralOperator::from_eigenpairs` refuses an excluded eigenvalue
/// outside the band (#3303). At the survival location-scale link-wiggle modes
/// of #3303 the floor kept 8 of 10 eigenpairs and silently dropped `−4.577` and
/// `−2.217` against a band of `1.3e-11`.
pub(crate) fn laplace_precision_kept_eigenpairs(
    eigenvalues: &[f64],
    penalty_rank: usize,
) -> Result<Vec<usize>, String> {
    let rank = DenseSpectralOperator::identified_rank(eigenvalues, penalty_rank);
    let mut order: Vec<usize> = (0..eigenvalues.len()).collect();
    order.sort_by(|&a, &b| eigenvalues[b].total_cmp(&eigenvalues[a]));
    let rounding_band = gam_linalg::roundoff::symmetric_spectrum_rounding_band(eigenvalues);
    let material_negative: Vec<f64> = order[rank..]
        .iter()
        .map(|&index| eigenvalues[index])
        .filter(|&value| !(value >= -rounding_band))
        .collect();
    if !material_negative.is_empty() {
        return Err(format!(
            "Laplace precision M is indefinite: {} dropped eigenvalue(s) lie below its \
             rounding band -p*eps*||M||_2 = {:.6e} (lowest {:.6e}; kept rank {rank} of {}), so \
             the mode is a saddle of the penalized objective and no Laplace approximation \
             exists here; log|M+| over the positive part is a different criterion and is not \
             priced in its place (#3303)",
            material_negative.len(),
            -rounding_band,
            material_negative
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min),
            eigenvalues.len(),
        ));
    }
    let mut kept = order;
    kept.truncate(rank);
    kept.sort_unstable();
    Ok(kept)
}

/// Rank of a penalty `S_λ` at its own rounding band `p·ε·‖S_λ‖₂`: the
/// `penalty_rank` that [`laplace_precision_kept_eigenpairs`] floors the kept
/// rank of a precision `M ⪰ S_λ` at, taken on `M`'s geometry (#2901 V22).
pub(crate) fn penalty_rank_at_rounding_band(penalty: &Array2<f64>) -> Result<usize, String> {
    if penalty.nrows() == 0 {
        return Ok(0);
    }
    let mut sym = penalty.clone();
    symmetrize_dense_in_place(&mut sym);
    let (spectrum, _) = FaerEigh::eigh(&sym, Side::Lower)
        .map_err(|e| format!("penalty rank eigendecomposition failed: {e}"))?;
    let eigenvalues = spectrum
        .as_slice()
        .ok_or_else(|| "penalty rank: the eigenvalue array is not contiguous".to_string())?;
    Ok(gam_linalg::roundoff::resolved_eigenvalue_count(
        eigenvalues,
        0.0,
    ))
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
/// - sum `log λ` over the eigenpairs [`laplace_precision_kept_eigenpairs`]
///   keeps for `penalty_rank`, the rank of `S_λ` on this matrix's geometry:
///   the top identified rank, the rule standard REML prices `log|H|₊` on. A
///   structural null space contributes no term, matching the projected `tr`
///   derivative.
pub(crate) fn strict_exact_pseudo_logdet(
    matrix: &Array2<f64>,
    penalty_rank: usize,
    accumulation_depth: usize,
) -> Result<f64, CustomFamilyError> {
    let mut sym = matrix.clone();
    symmetrize_dense_in_place(&mut sym);
    let (evals, _) = FaerEigh::eigh(&sym, Side::Lower)
        .map_err(|e| CustomFamilyError::NumericalFailure { reason: format!("strict pseudo-laplace eigendecomposition failed: {e}") })?;
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
    // POSITIVE-eigenspace inclusion for the pseudo-logdet sum. This MUST be the
    // same kept set the analytic REML gradient's trace kernel uses
    // (`laplace_precision_kept_eigenpairs`, the `range(H+Sλ)` Moore–Penrose
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
    // (constant ‖g‖, stuck cost — gam#808). Sharing the kernel's rule here
    // removes the desync at the source.
    let evals_slice = evals.as_slice().ok_or_else(|| {
        "strict pseudo-laplace logdet: the eigenvalue array is not contiguous".to_string()
    })?;
    if evals.iter().any(|&ev| ev < -neg_tol) {
        let min_eval = evals.iter().copied().fold(f64::INFINITY, f64::min);
        let below = evals.iter().filter(|&&ev| ev < -neg_tol).count();
        return Err(CustomFamilyError::NumericalFailure {
            reason: format!(
                "strict pseudo-laplace logdet: {below} eigenvalue(s) below -neg_tol \
             (min(λ)={min_eval:.6e}, max|λ|={max_abs_eval:.6e}, neg_tol={neg_tol:.6e}, εnp={eps_np:.6e}); \
             indefinite joint coefficient Hessian rejected (no δ-ridge masking, gam#748)"
            ),
        });
    }
    Ok(laplace_precision_kept_eigenpairs(evals_slice, penalty_rank)
        .map_err(|reason| CustomFamilyError::NumericalFailure {
            reason: format!("strict pseudo-laplace logdet: {reason}"),
        })?
        .into_iter()
        .map(|index| evals[index].ln())
        .sum())
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
/// z·exp(slope) cross-coupling carry genuine negative curvature there). An
/// indefinite QP model has a direction that lowers the local quadratic objective
/// while moving AWAY from the KKT point, so the trust region — which gates on the
/// objective-reduction ratio ρ, not the stationarity residual — happily accepts
/// step after step at ρ≈1 and GROWS its radius while the stationarity residual
/// diverges (`per_block_resid[time]` 3.5e4 → 9.5e6 over 11 cycles, the gam#1040 /
/// gam#979 divergence). The self-vanishing Levenberg μ cannot rescue this: a
/// μ·I shift that is tiny relative to the most-negative eigenvalue leaves the
/// model indefinite, and a μ large enough to flip it would bias the converged β.
///
/// Reflecting (not merely clamping negative modes to zero) preserves the
/// curvature MAGNITUDE on negative modes, so the modified-Newton
/// step length matches the dense-spectral path's and the QP stays bounded (a
/// clamp-to-zero null mode would make the QP unbounded along that direction). At
/// a genuine optimum the constrained Hessian over the identified subspace is PSD,
/// so the reflection is a no-op there and the converged β is unchanged — exactly
/// the property the dense path relies on. Eigensolver failure is a hard error:
/// silently reverting to an indefinite or differently damped QP would switch
/// algorithms after its curvature contract failed.
///
/// "Numerical null" means below the eigensolver's own resolution,
/// `joint_hessian_numerical_eigenvalue_floor` (`λ_max·√p·ε`), the classification
/// `symmetric_penalized_hessian_nullity` already uses (#2690). The
/// `KKT_REFUSAL_RANK_TOL·λ_max` conditioning cutoff used here instead calls a
/// resolvable identified mode null whenever it is stiff next to a huge one, and
/// then REPLACES its curvature with μ. On the veteran frailty witness (#2714) the
/// first waypoint-0 cycle logged `lambda_min_signed_raw=1.606e0 nullity=2` with a
/// stabilized minimum of `1.499e-2`, which is μ: two identified modes at least
/// 107× flatter in the QP than in the trust-region model that judges its step.
/// The QP overshot along them every cycle, the model accepted a sliver at
/// `ρ = 1`, and the solve zig-zagged into the residual-stall guard. Below the
/// resolution a mode's curvature is not a number and μ supplies gauge
/// uniqueness; above it every mode keeps at least its exact magnitude, which the
/// ambient shift only ever raises.
///
/// The decomposition runs on the equilibrated matrix `C = D⁻¹HD⁻¹`,
/// `D = diag(√|H_jj|)`, and the result is mapped back as `D·C_stab·D`. `H` is
/// assembled entry by entry, so an entry is resolved relative to its own size,
/// and so is every entry of `C`. The eigensolver's floor `λ_max·√p·ε` on `H` is
/// set by its stiffest column instead. On the #2695 witness one direction at
/// `λ_max = 5.23e16` put that floor near 59 on a 26-wide Hessian whose other 25
/// modes had curvature 0.16 to 0.5: every one was counted null and replaced by
/// μ, so the QP crawled at about 1/116 of the Newton proposal. `C` is a
/// congruence of `H`, so it has the same inertia and nullity, and among diagonal
/// scalings its condition is within a factor `p` of the best (van der Sluis). A
/// numerical-null mode `v` of `C` is given the curvature `μ·‖D⁻¹v‖²`, so the
/// mapped direction `D⁻¹v` carries curvature μ in the frame of `H`, and the
/// ambient shift raises every mode by that same Rayleigh quotient of `μ·D⁻²`.
/// The reported condition and minimum eigenvalues are those of `C`. The ambient
/// shift is still triggered on the condition of `H`, the matrix a family's
/// request is about.
pub(crate) fn symmetric_constrained_hessian_geometry(
    matrix: &Array2<f64>,
    levenberg_mu: f64,
    damp_full_rank_ill_conditioned: bool,
) -> Result<ConstrainedHessianGeometry, CustomFamilyError> {
    let p = matrix.nrows();
    if p == 0 || matrix.ncols() != p {
        return Err(CustomFamilyError::trial_point(format!(
            "constrained Hessian must be nonempty and square, got {}x{}",
            matrix.nrows(),
            matrix.ncols()
        )));
    }
    let mut sym = matrix.clone();
    symmetrize_dense_in_place(&mut sym);
    // Column scales of the equilibration. A zero or non-finite diagonal carries
    // no scale, and its column is left as it is.
    let column_scale = Array1::from_shape_fn(p, |j| {
        let magnitude = sym[[j, j]].abs();
        if magnitude.is_finite() && magnitude > 0.0 {
            magnitude.sqrt()
        } else {
            1.0
        }
    });
    let mut equilibrated = sym.clone();
    for i in 0..p {
        for j in 0..p {
            equilibrated[[i, j]] /= column_scale[i] * column_scale[j];
        }
    }
    let (evals, evecs) = FaerEigh::eigh(&equilibrated, Side::Lower)
        .map_err(|error| CustomFamilyError::trial_point(format!("constrained Hessian eigendecomposition failed: {error:?}")))?;
    let lambda_max_abs = evals.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    if !(lambda_max_abs.is_finite() && lambda_max_abs > 0.0) {
        return Err(CustomFamilyError::trial_point("constrained Hessian has no finite nonzero curvature scale".to_string()));
    }
    let cutoff = crate::joint_newton::joint_hessian_numerical_eigenvalue_floor(lambda_max_abs, p);
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
        && assembled_hessian_condition(&sym)? > LEVENBERG_ILL_CONDITIONING_THRESHOLD;
    let mu = if levenberg_mu.is_finite() && levenberg_mu > 0.0 {
        levenberg_mu
    } else {
        0.0
    };
    let stabilized = Array1::from_iter(evals.iter().enumerate().map(|(mode, lambda)| {
        // Curvature μ along the mode's direction in the frame of H is μ·‖D⁻¹v‖² in C.
        let mu_in_mode = mu
            * evecs
                .column(mode)
                .iter()
                .zip(column_scale.iter())
                .map(|(component, scale)| (component / scale) * (component / scale))
                .sum::<f64>();
        if nullity > 0 && lambda.abs() < cutoff {
            mu_in_mode.max(cutoff)
        } else if ambient_levenberg {
            (lambda + mu_in_mode).abs().max(cutoff)
        } else {
            lambda.abs().max(cutoff)
        }
    }));
    let stabilized_min_eigenvalue = stabilized.iter().copied().fold(f64::INFINITY, f64::min);
    let scaled = &evecs * &stabilized.view().insert_axis(ndarray::Axis(0));
    let mut stabilized_matrix = scaled.dot(&evecs.t());
    for i in 0..p {
        for j in 0..p {
            stabilized_matrix[[i, j]] *= column_scale[i] * column_scale[j];
        }
    }
    Ok(ConstrainedHessianGeometry {
        matrix: stabilized_matrix,
        nullity,
        condition,
        raw_min_eigenvalue: evals.iter().copied().fold(f64::INFINITY, f64::min),
        stabilized_min_eigenvalue,
    })
}

/// Condition of an assembled symmetric Hessian over its resolvable modes: the
/// largest curvature magnitude over the smallest one at or above the
/// eigensolver's floor `λ_max·√p·ε`.
fn assembled_hessian_condition(sym: &Array2<f64>) -> Result<f64, CustomFamilyError> {
    let (evals, _) = FaerEigh::eigh(sym, Side::Lower).map_err(|error| {
        CustomFamilyError::trial_point(format!(
            "constrained Hessian eigendecomposition failed: {error:?}"
        ))
    })?;
    let lambda_max_abs = evals.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let cutoff =
        crate::joint_newton::joint_hessian_numerical_eigenvalue_floor(lambda_max_abs, sym.nrows());
    let min_range = evals
        .iter()
        .map(|value| value.abs())
        .filter(|magnitude| *magnitude >= cutoff && *magnitude > 0.0)
        .fold(f64::INFINITY, f64::min);
    Ok(if min_range.is_finite() && min_range > 0.0 {
        lambda_max_abs / min_range
    } else {
        f64::INFINITY
    })
}

/// Numerical nullity of a symmetric penalized Hessian: the number of eigenvalues
/// below the eigensolver's own resolution,
/// [`crate::joint_newton::joint_hessian_numerical_eigenvalue_floor`]
/// (`λ_max·√p·ε`, #2690). A mode above that floor carries resolvable curvature.
///
/// This used the wider `KKT_REFUSAL_RANK_TOL` conditioning cutoff, which calls
/// such modes null. The constrained fixed-point certificate, this function's
/// consumer, then refused a 3-D CTN probe at a constrained KKT point as
/// rank-deficient. There λ_min = 7.783e-4 against a 1e-10 cutoff of 3.6e-2 and a
/// resolution near 7e-7, and the flagged direction's likelihood curvature was
/// 9.3e-3 (MSI job 410061). Returns `None` when no finite curvature scale or
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
    let cutoff = crate::joint_newton::joint_hessian_numerical_eigenvalue_floor(max_abs, p);
    Some(evals.iter().filter(|x| x.abs() < cutoff).count())
}

#[cfg(test)]
mod block_penalty_roots_tests {
    use super::*;
    use ndarray::array;

    /// Error-free double-double accumulation for the references below: `TwoSum`
    /// and an FMA `TwoProd` keep each partial result's rounding as a second word,
    /// so a dot product of `q` terms is accurate to `O(q·u²)` relative, far below
    /// every band these pins compare against.
    #[derive(Clone, Copy)]
    struct DoubleDouble {
        hi: f64,
        lo: f64,
    }

    impl DoubleDouble {
        const ZERO: Self = Self { hi: 0.0, lo: 0.0 };

        fn two_sum(a: f64, b: f64) -> Self {
            let hi = a + b;
            let bb = hi - a;
            Self {
                hi,
                lo: (a - (hi - bb)) + (b - bb),
            }
        }

        fn add(self, other: Self) -> Self {
            let s = Self::two_sum(self.hi, other.hi);
            let lo = s.lo + self.lo + other.lo;
            Self::two_sum(s.hi, lo)
        }

        fn product(a: f64, b: f64) -> Self {
            let hi = a * b;
            Self {
                hi,
                lo: a.mul_add(b, -hi),
            }
        }

        fn mul(self, other: Self) -> Self {
            let p = Self::product(self.hi, other.hi);
            let lo = p.lo + self.hi * other.lo + self.lo * other.hi;
            Self::two_sum(p.hi, lo)
        }

        fn scale(self, factor: f64) -> Self {
            self.mul(Self {
                hi: factor,
                lo: 0.0,
            })
        }

        fn value(self) -> f64 {
            self.hi + self.lo
        }
    }

    fn dd_rows(root: &Array2<f64>, vector: &Array1<f64>) -> Vec<DoubleDouble> {
        root.rows()
            .into_iter()
            .map(|row| {
                row.iter()
                    .zip(vector.iter())
                    .fold(DoubleDouble::ZERO, |acc, (&r, &v)| {
                        acc.add(DoubleDouble::product(r, v))
                    })
            })
            .collect()
    }

    /// A first-difference penalty on reweighted coefficients: `A = D·W` with
    /// `W = diag(1/(j+3))` rounded to f64, stored as the formed `AᵀA`. Its null
    /// space is `W⁻¹·1` up to that rounding, and the stored matrix, like the
    /// 2923 time block's, is not exactly singular there.
    fn reweighted_difference_block(log_lambda: f64) -> (Vec<ParameterBlockSpec>, Vec<Array1<f64>>) {
        let p = 7;
        let mut a = Array2::<f64>::zeros((p - 1, p));
        for i in 0..p - 1 {
            a[[i, i]] = 1.0 / (i as f64 + 3.0);
            a[[i, i + 1]] = -1.0 / (i as f64 + 4.0);
        }
        let penalty = a.t().dot(&a);
        let spec = ParameterBlockSpec {
            name: "reweighted_difference".to_string(),
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                Array2::<f64>::eye(p),
            )),
            offset: Array1::zeros(p),
            penalties: vec![PenaltyMatrix::Dense(penalty)],
            nullspace_dims: vec![1],
            initial_log_lambdas: array![log_lambda],
            initial_beta: None,
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        };
        (vec![spec], vec![array![log_lambda]])
    }

    /// Coefficients along the stored penalty's near-null direction `W⁻¹·1`, plus
    /// a small range component: the `λ → ∞` geometry a smoothing parameter
    /// crawling to its limit face produces.
    fn near_null_beta() -> Array1<f64> {
        Array1::from_iter((0..7).map(|j| (j as f64 + 3.0) + 1.0e-3 * (j as f64 - 3.0)))
    }

    /// #2954 (supersedes gam#2959's `block_quadratic_penalty_accumulation_matches_
    /// the_blas_value_2959`): the root-form value matches a double-double reference
    /// of `½λ‖Rβ‖²` on the evaluator's own roots within the band it states, at
    /// every strength up to the `λ = e³⁰` edge. The dense form of the same stored
    /// matrix, evaluated exactly, departs from it by more than that band there:
    /// the stored null eigenvalue times `λ`.
    #[test]
    fn the_root_form_value_is_within_its_band_of_a_double_double_reference_2954() {
        let beta = near_null_beta();
        for log_lambda in [0.0, 15.0, 30.0] {
            let (specs, log_lambdas) = reweighted_difference_block(log_lambda);
            let roots = BlockPenaltyRoots::new(&specs, &log_lambdas, None).expect("roots");
            let evaluated = roots.value(std::slice::from_ref(&beta));
            let term = &roots.terms()[0];
            assert_eq!(term.root.nrows(), 6, "declared nullity 1 keeps rank 6");
            let reference = dd_rows(&term.root, &beta)
                .into_iter()
                .fold(DoubleDouble::ZERO, |acc, y| acc.add(y.mul(y)))
                .scale(0.5 * term.lambda)
                .value();
            let gap = (evaluated.value - reference).abs();
            assert!(
                gap <= evaluated.band(),
                "log λ={log_lambda}: root value {:e} against double-double {reference:e}: gap \
                 {gap:.3e} exceeds its band {:.3e}",
                evaluated.value,
                evaluated.band()
            );
            if log_lambda == 30.0 {
                let stored = specs[0].penalties[0].to_dense();
                let dense_exact = dd_rows(&stored, &beta)
                    .into_iter()
                    .zip(beta.iter())
                    .fold(DoubleDouble::ZERO, |acc, (row, &b)| acc.add(row.scale(b)))
                    .scale(0.5 * term.lambda)
                    .value();
                assert!(
                    (dense_exact - reference).abs() > evaluated.band(),
                    "the stored matrix's exact dense value {dense_exact:e} must depart from \
                     the root form {reference:e} by more than the root band {:.3e} at λ = e³⁰",
                    evaluated.band()
                );
            }
        }
    }

    /// #2954 for #2977: `penalty_increment(β, δ)` is `value(β + δ) − value(β)`
    /// against a double-double reference of `λ[(Rδ)ᵀ(Rβ) + ½‖Rδ‖²]`, which is that
    /// difference exactly, within the band it states; and it is formed without
    /// differencing two values, so its band is far below the values' own.
    #[test]
    fn the_penalty_increment_is_the_value_difference_within_its_band_2954() {
        let (specs, log_lambdas) = reweighted_difference_block(30.0);
        let roots = BlockPenaltyRoots::new(&specs, &log_lambdas, None).expect("roots");
        let beta = near_null_beta();
        let delta = Array1::from_iter((0..7).map(|j| 1.0e-4 * ((j * j) as f64 - 5.0)));
        let increment =
            roots.penalty_increment(std::slice::from_ref(&beta), std::slice::from_ref(&delta));
        let term = &roots.terms()[0];
        let a = dd_rows(&term.root, &beta);
        let d = dd_rows(&term.root, &delta);
        let reference = a
            .iter()
            .zip(d.iter())
            .fold(DoubleDouble::ZERO, |acc, (&a_i, &d_i)| {
                acc.add(d_i.mul(a_i)).add(d_i.mul(d_i).scale(0.5))
            })
            .scale(term.lambda)
            .value();
        let gap = (increment.increment - reference).abs();
        assert!(
            gap <= increment.band,
            "increment {:e} against double-double {reference:e}: gap {gap:.3e} exceeds its band \
             {:.3e}",
            increment.increment,
            increment.band
        );
        let value = roots.value(std::slice::from_ref(&beta));
        assert!(
            increment.band < value.band(),
            "the direct increment's band {:.3e} must be below the value's own band {:.3e}",
            increment.band,
            value.band()
        );
    }

    /// #2954: value, gradient and curvature are one function. For a quadratic,
    /// `value(β + δ) − value(β) = gᵀδ + ½δᵀHδ` exactly, with `g = S_λβ` and `H =
    /// S_λ` from [`BlockPenaltyRoots::s_lambdas`]. At `λ = e³⁰` the two sides agree
    /// within the increment's band plus the matrix products' own rounding,
    /// `γ_(r+p+1)·λ(|δ|ᵀ|R|ᵀ|R||β| + ½|δ|ᵀ|R|ᵀ|R||δ|)`.
    #[test]
    fn the_curvature_is_the_hessian_of_the_root_value_at_large_lambda_2954() {
        let (specs, log_lambdas) = reweighted_difference_block(30.0);
        let roots = BlockPenaltyRoots::new(&specs, &log_lambdas, None).expect("roots");
        let beta = near_null_beta();
        let delta = Array1::from_iter((0..7).map(|j| 1.0e-4 * ((j * j) as f64 - 5.0)));
        let increment =
            roots.penalty_increment(std::slice::from_ref(&beta), std::slice::from_ref(&delta));
        let s_lambda = &roots.s_lambdas()[0];
        let model = delta.dot(&s_lambda.dot(&beta)) + 0.5 * delta.dot(&s_lambda.dot(&delta));
        let term = &roots.terms()[0];
        let abs_gram = term.root.mapv(f64::abs).t().dot(&term.root.mapv(f64::abs));
        let abs_delta = delta.mapv(f64::abs);
        let product_band =
            gam_linalg::roundoff::accumulation_growth(term.root.nrows() + term.root.ncols() + 1)
                * term.lambda
                * (abs_delta.dot(&abs_gram.dot(&beta.mapv(f64::abs)))
                    + 0.5 * abs_delta.dot(&abs_gram.dot(&abs_delta)));
        let gap = (increment.increment - model).abs();
        assert!(
            gap <= increment.band + product_band,
            "increment {:e} against gᵀδ + ½δᵀHδ = {model:e}: gap {gap:.3e} exceeds {:.3e}",
            increment.increment,
            increment.band + product_band
        );
    }
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

    /// The spectrum of the refused 3-D CTN probe (MSI job 410061): a weak but
    /// resolvable mode beside a large one is not null, and an exact zero is.
    /// The weak mode sits below the old `1e-10·λ_max` cutoff (3.6e-2) and far
    /// above the resolution `λ_max·√3·ε` (1.4e-7).
    #[test]
    fn resolvable_weak_curvature_is_not_counted_as_null_979() {
        let weak = array![[3.609e8, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 7.783e-4]];
        assert_eq!(symmetric_penalized_hessian_nullity(&weak), Some(0));
        let singular = array![[3.609e8, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]];
        assert_eq!(symmetric_penalized_hessian_nullity(&singular), Some(1));
    }

    /// #2714: the veteran frailty witness's first waypoint-0 cycle logged
    /// `lambda_min_signed_raw=1.606e0 nullity=2` with a stabilized minimum of
    /// `1.499e-2`, which is μ. An identified mode that is stiff next to a huge one
    /// sat below the `1e-10·λ_max` conditioning cutoff, was counted null, and had
    /// its curvature replaced by μ, so the QP overshot it about 100× against the
    /// trust-region model. Only a mode below the eigensolver's resolution may be
    /// replaced. Here `λ_max = 2e10` puts the old cutoff at `2.0`, above the weak
    /// mode, so both arms below fail under it.
    #[test]
    fn resolvable_weak_curvature_keeps_its_magnitude_in_the_qp_geometry_2714() {
        let mu = 1.499e-2_f64;
        let weak = 1.606_f64;
        let weak_axis = array![0.0_f64, 1.0, 0.0];
        let null_axis = array![0.0_f64, 0.0, 1.0];

        let rank_deficient = array![[2.0e10, 0.0, 0.0], [0.0, weak, 0.0], [0.0, 0.0, 0.0]];
        let geometry = symmetric_constrained_hessian_geometry(&rank_deficient, mu, false)
            .expect("rank-deficient symmetric geometry");
        assert_eq!(
            geometry.nullity, 1,
            "only the exact zero is below the eigensolver's resolution"
        );
        let weak_curvature = geometry.matrix.dot(&weak_axis)[1];
        assert!(
            (weak_curvature - weak).abs() <= 1e-6 * weak,
            "a resolvable identified mode must keep its curvature {weak}, got {weak_curvature}"
        );
        let null_curvature = geometry.matrix.dot(&null_axis)[2];
        assert!(
            (null_curvature - mu).abs() <= 1e-6 * mu,
            "the numerical-null mode takes μ = {mu}, got {null_curvature}"
        );

        // Full rank and ill conditioned: the ambient shift only raises curvature.
        let full_rank = array![[2.0e10, 0.0, 0.0], [0.0, weak, 0.0], [0.0, 0.0, 3.0]];
        let damped = symmetric_constrained_hessian_geometry(&full_rank, mu, true)
            .expect("full-rank symmetric geometry");
        assert_eq!(damped.nullity, 0);
        let damped_weak = damped.matrix.dot(&weak_axis)[1];
        assert!(
            damped_weak >= weak,
            "the ambient Levenberg shift must not flatten an identified mode: {damped_weak} < {weak}"
        );
    }

    /// #2695: one column 1e8 times stiffer than the others sets the raw
    /// eigensolver floor `λ_max·√p·ε` near 3.9, above the O(1) curvature of the
    /// two weaker modes of this positive-definite Hessian. Counted null, they
    /// would take μ. Equilibrated, nothing is null, every mode keeps its exact
    /// curvature, and the QP geometry is the Hessian itself.
    #[test]
    fn a_stiff_column_does_not_flatten_the_resolvable_modes_beside_it_2695() {
        let correlation = array![[1.0, 0.3, 0.2], [0.3, 1.0, 0.4], [0.2, 0.4, 1.0]];
        let scale = array![1.0e8, 1.0, 0.7];
        let hessian =
            Array2::from_shape_fn((3, 3), |(i, j)| scale[i] * correlation[[i, j]] * scale[j]);
        let mu = 1.0e-2;
        let geometry = symmetric_constrained_hessian_geometry(&hessian, mu, false)
            .expect("stiff-column symmetric geometry");
        assert_eq!(geometry.nullity, 0, "every mode of a positive-definite Hessian is resolvable");
        for i in 0..3 {
            for j in 0..3 {
                let expected = hessian[[i, j]];
                let actual = geometry.matrix[[i, j]];
                assert!(
                    (actual - expected).abs() <= 1e-9 * expected.abs().max(1.0),
                    "entry ({i}, {j}): the QP geometry must be the Hessian, got {actual} vs {expected}"
                );
            }
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
        let strict = strict_exact_pseudo_logdet(&s_rank_deficient, 0, 3).expect("strict logdet");
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
        let strict_spd = strict_exact_pseudo_logdet(&s_spd, 0, 2).expect("strict spd logdet");
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
