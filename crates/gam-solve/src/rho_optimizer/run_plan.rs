use super::*;

/// The criterion value and iteration count a first-order run ended at, when it ended by
/// converging or stalling: the ends at which the search may cross into the stratum of a
/// trial it refused for keeping a different rank (#2765). A budget verdict or a failure
/// crosses nothing. A run whose every probe was refused stalled where it started.
fn stratum_run_end(
    outcome: &Result<Solution, BfgsError>,
    cost_stall_exit: &Mutex<Option<CostStallExit>>,
    start_cost: f64,
) -> Option<(f64, usize)> {
    match outcome {
        Ok(solution) => Some((solution.final_value, solution.iterations)),
        Err(BfgsError::LineSearchFailed { last_solution, .. }) => {
            Some((last_solution.final_value, last_solution.iterations))
        }
        Err(BfgsError::ObjectiveFailed { message }) if message == COST_STALL_CONVERGED_SENTINEL => {
            cost_stall_exit
                .lock()
                .ok()
                .and_then(|slot| slot.as_ref().map(|exit| (exit.value, exit.iterations)))
        }
        Err(BfgsError::ObjectiveFailed { message })
            if message.starts_with(PROBE_REFUSAL_FATAL_SENTINEL) =>
        {
            Some((start_cost, 0))
        }
        Err(_) => None,
    }
}

/// The non-converged checkpoint a trust-region run leaves when it stops without a
/// convergence claim but with the iterate it stopped at: its budget ran out, or its
/// region (or cubic regularisation) reached the reject floor with no accepted step.
///
/// The run measured that iterate, and it is resumable work: the checkpoint slot
/// retains it and the terminal certificate judges it from a fresh evaluation. A stop
/// is never a failure that discards what the run measured (#2953: a dense ARC run
/// whose regularisation saturated next to the prior mode, 7e4 below the point the fit
/// then returned, used to end in `RemlOptimizationFailed` and lose its iterate). When
/// the cost-stall guard saw a feasible iterate below the last one, that iterate is the
/// checkpoint instead, so a degenerate box corner the trajectory wandered to does not
/// over-shrink a supported penalty direction (#1371). Either way it is non-converged,
/// so only the certify/reseed ladder can mint it (#1476).
fn stopped_run_checkpoint(
    context: &str,
    stop: &str,
    last_solution: Solution,
    best_feasible: Option<CostStallExit>,
    the_plan: OuterPlan,
) -> OuterResult {
    match best_feasible {
        Some(best)
            if best.value.is_finite()
                && (!last_solution.final_value.is_finite()
                    || best.value < last_solution.final_value) =>
        {
            log::debug!(
                "[OUTER] {context}: {stop} last iterate (value={:.6e}) is worse than the best \
                 feasible iterate seen (value={:.6e}); substituting the best iterate so a \
                 degenerate box-corner does not over-shrink a supported penalty direction \
                 (#1371). The substituted iterate is a non-converged checkpoint, never a \
                 converged fit (#1476).",
                last_solution.final_value,
                best.value,
            );
            let mut result = outer_result_with_gradient_norm(
                best.rho,
                best.value,
                // The run spent its whole budget; `best.iterations` is only the index of
                // the iterate adopted here.
                last_solution.iterations,
                Some(best.grad_norm),
                false,
                the_plan,
            );
            result.origin = OuterResultOrigin::ArcBestIterateSubstitution;
            result
        }
        _ => solution_into_outer_result(last_solution, false, the_plan),
    }
}

/// A one-shot reseed retry returns its own outcome, and that outcome knows only
/// the reseed's iterations. Fold in what the enclosing attempt already spent, so
/// the total covers every start (#2817).
fn with_enclosing_attempt_ledger(
    outcome: PlanRunOutcome,
    spent_seed_iterations: usize,
) -> PlanRunOutcome {
    match outcome {
        PlanRunOutcome::Converged(mut result) => {
            result.iterations = result.iterations.saturating_add(spent_seed_iterations);
            PlanRunOutcome::Converged(result)
        }
        PlanRunOutcome::Exhausted(mut checkpoint) => {
            checkpoint.iterations = checkpoint.iterations.saturating_add(spent_seed_iterations);
            PlanRunOutcome::Exhausted(checkpoint)
        }
        other => other,
    }
}

/// The one point the outer search starts from.
///
/// The caller's `initial_rho` when it has the search's dimension (a cache
/// resume, a reseed retry, or a model-derived start such as the analytic
/// `initial.sp` the standard REML path computes); otherwise the caller's
/// full-length `heuristic_log_lambdas`; otherwise `ρ = 0`, which is `λ = 1` on
/// the normalized penalties every term builder emits. The result is clamped
/// into the model domain's envelope so no start sits on a face the search does
/// not have (SPEC rule 20, #2902 row 9).
///
/// There is exactly one start. A start that does not certify is continued by
/// the certify-resume loop from its own checkpoint, and a certified saddle is
/// left along its negative-curvature direction; neither re-enters the search
/// from an unrelated lattice point.
pub(crate) fn outer_start_point(
    config: &OuterConfig,
    n_params: usize,
    model_domain_bounds: &(Array1<f64>, Array1<f64>),
) -> Result<Array1<f64>, EstimationError> {
    if let Some(initial) = config.initial_rho.as_ref()
        && initial.len() == n_params
    {
        return Ok(initial.clone());
    }
    let envelope = gam_problem::OrderedRhoBounds::envelope(
        model_domain_bounds.0.iter().copied(),
        model_domain_bounds.1.iter().copied(),
    )?;
    Ok(match config.heuristic_log_lambdas.as_deref() {
        Some(heuristic) if heuristic.len() == n_params => {
            heuristic.iter().map(|&value| envelope.clamp(value)).collect()
        }
        _ => Array1::from_elem(n_params, envelope.clamp(0.0)),
    })
}

/// The typed ray a custom-family inner solve reported when it stopped
/// descending a direction with no finite minimizer in reach at this ρ, when a
/// block's penalty can close it.
fn ray_restoration_in(err: &EstimationError) -> Option<&gam_problem::RayRestoration> {
    let EstimationError::CustomFamily(error) = err else {
        return None;
    };
    match error.descending_ray_exit()? {
        gam_problem::DescendingRayExit::Closable(ray) => Some(ray),
        // No block's penalty opposes an unpenalized ray, so no raised strength
        // closes it.
        gam_problem::DescendingRayExit::Unpenalized => None,
    }
}

/// The ceiling a ray restoration may raise a log strength to (the model's own
/// upper domain) and the number of ρ coordinates, which bounds how many
/// restorations one seed may take.
#[derive(Clone, Copy)]
struct RayRestorationDomain<'a> {
    upper: &'a Array1<f64>,
    rho_dim: usize,
}

/// Evaluate a seed at full inner fidelity, restoring it when the inner solve
/// reports a ray (#2695): the inner solve names the block whose penalty is too
/// weak to close the direction it was descending and the strength ratio at
/// which it would, so the seed is not a failed seed but an under-penalized
/// one, and its named log strengths are raised by that ratio and the
/// evaluation repeated. Each restoration strictly raises the named
/// coordinates; it stops at the model's own domain ceiling, or after as many
/// restorations as there are ρ coordinates (a ray that survives that many
/// closures is not closing), and then the original refusal is returned.
fn eval_seed_restoring_rays(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    seed: &mut Array1<f64>,
    order: OuterEvalOrder,
    domain: RayRestorationDomain<'_>,
    context: &str,
    seed_idx: usize,
) -> Result<OuterEval, EstimationError> {
    let RayRestorationDomain { upper, rho_dim } = domain;
    let mut restorations = 0usize;
    loop {
        let err = match eval_seed_at_full_inner_fidelity(obj, config, seed, order) {
            Ok(eval) => return Ok(eval),
            Err(err) => err,
        };
        let Some(ray) = ray_restoration_in(&err) else {
            return Err(err);
        };
        if restorations >= rho_dim {
            log::debug!(
                "[OUTER] {context}: seed {seed_idx} still descends a ray after {restorations} \
                 restorations (one per rho coordinate); refusing it as evaluated: {ray}"
            );
            return Err(err);
        }
        let mut restored = seed.clone();
        // A ray names a BLOCK, and a block's penalty strength is spread over
        // `rho_count` coordinates. Raising every one of them by `ln r` closes
        // the ray exactly; raising the subset that still has room raises the
        // block's total strength by less, which is a step along the same
        // direction and never the wrong way. Refusing the whole seed because
        // ONE coordinate of the block sits at its ceiling threw away the others
        // (gam#2695: the 1569 pair's rays span `rho[3..5]` and refuse on
        // `rho[3]` alone), and the caller re-enters this loop, so a partial
        // raise is re-evaluated and can be raised again — up to one restoration
        // per rho coordinate.
        let mut capped: Vec<(usize, f64, f64)> = Vec::new();
        let mut raised_any = false;
        for j in ray.rho_indices() {
            let Some(current) = restored.get(j).copied() else {
                return Err(err);
            };
            let ceiling = upper.get(j).copied().unwrap_or(f64::INFINITY);
            // The closure may lie beyond the domain's ceiling: then the block
            // wants more strength than the model resolves, and the honest move
            // is to the ceiling itself (a block collapsing to its null space
            // is a result, not a wall — #2812).
            let raised = (current + ray.log_strength_ratio).min(ceiling);
            if !(raised.is_finite() && raised > current) {
                capped.push((j, current, ceiling));
                continue;
            }
            restored[j] = raised;
            raised_any = true;
        }
        if !raised_any {
            // EVERY coordinate of the block is at its ceiling: no admissible
            // strength closes this ray, and the refusal is the model's answer
            // rather than a missed opportunity.
            log::debug!(
                "[OUTER] {context}: seed {seed_idx} descends a ray whose every coordinate is \
                 already at its domain ceiling ({}); no admissible penalty strength closes it, \
                 so refusing it as evaluated: {ray}",
                capped
                    .iter()
                    .map(|(j, current, ceiling)| format!(
                        "rho[{}]={current:.4} vs ceiling {ceiling:.4}",
                        native_coordinate(config.native_coordinate_order.as_deref(), *j)
                    ))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            return Err(err);
        }
        if !capped.is_empty() {
            log::debug!(
                "[OUTER] {context}: seed {seed_idx} raises the {} coordinate(s) of this block \
                 that still have room and leaves {} at the ceiling ({}); the block gains less \
                 than the ray's full {:.4}, which is a step along it, not past it",
                ray.rho_count - capped.len(),
                capped.len(),
                capped
                    .iter()
                    .map(|(j, current, ceiling)| format!(
                        "rho[{}]={current:.4} vs ceiling {ceiling:.4}",
                        native_coordinate(config.native_coordinate_order.as_deref(), *j)
                    ))
                    .collect::<Vec<_>>()
                    .join(", "),
                ray.log_strength_ratio,
            );
        }
        log::debug!(
            "[OUTER] {context}: seed {seed_idx} is under-penalized, not failed — {ray}; \
             restoring rho{:?} from {:?} to {:?} and re-evaluating",
            native_coordinates(
                config.native_coordinate_order.as_deref(),
                &ray.rho_indices().collect::<Vec<_>>()
            ),
            ray.rho_indices().map(|j| seed[j]).collect::<Vec<_>>(),
            ray.rho_indices().map(|j| restored[j]).collect::<Vec<_>>(),
        );
        *seed = restored;
        restorations += 1;
    }
}

/// Evaluate the literal outer seed against the true profiled objective.
///
/// Adaptive inner caps are search accelerators. A capped, nonconverged inner
/// iterate is not a value or derivative of the profiled objective and therefore
/// cannot reject a seed or initialize an optimizer. Lift the shared cap only for
/// this sample, preserving any continuation/pilot warm state, then restore the
/// scheduler before search begins.
fn eval_seed_at_full_inner_fidelity(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    seed: &Array1<f64>,
    order: OuterEvalOrder,
) -> Result<OuterEval, EstimationError> {
    let full_fidelity_guard = config
        .outer_inner_cap
        .as_ref()
        .map(FullFidelityInnerCapGuard::lift);
    let result = obj.eval_with_order(seed, order);
    drop(full_fidelity_guard);
    result
}

/// Require a continuation arrival to certify the literal outer seed itself.
///
/// Only a state whose rho is bit-identical to the bounded literal seed and
/// whose real-objective value is finite may authorize the outer solver to
/// start.
pub(crate) fn reactive_arrival_postcondition(
    state: &crate::estimate::reml::continuation::ContinuationState,
    literal_seed: &Array1<f64>,
) -> Result<(), String> {
    let at_literal_seed = state.last_rho.len() == literal_seed.len()
        && state
            .last_rho
            .iter()
            .zip(literal_seed.iter())
            .all(|(actual, expected)| actual.to_bits() == expected.to_bits());
    if !at_literal_seed {
        return Err(format!(
            "reactive domain entry refused: continuation reported arrival at rho {:?}, not the literal seed {:?}",
            state.last_rho, literal_seed
        ));
    }
    if !state.last_eval.cost.is_finite() {
        return Err(format!(
            "reactive domain entry refused: continuation arrival at the literal seed retained non-finite evidence {}",
            state.last_eval.cost
        ));
    }
    Ok(())
}

/// A transferred dense outer Hessian is eligible as a BFGS seed only when the
/// current objective itself declares analytic second-order geometry. Shape and
/// finiteness are necessary but cannot establish provenance: without this gate,
/// a persistent checkpoint can inject curvature produced by an older objective
/// implementation (including the deleted SAE finite-difference path) into a
/// current Hessian-unavailable solve (#2253).
pub(crate) fn eligible_transferred_outer_hessian<'a>(
    hessian: Option<&'a Array2<f64>>,
    declared: DeclaredHessianForm,
    n_params: usize,
) -> Option<&'a Array2<f64>> {
    if !declared.is_analytic() {
        return None;
    }
    hessian.filter(|h| {
        h.nrows() == n_params && h.ncols() == n_params && h.iter().all(|v| v.is_finite())
    })
}

/// The continuation's exact start curvature, only at the start it was measured
/// at and only in this layout's dimension; see [`OuterConfig::initial_curvature`].
pub(crate) fn bound_initial_curvature<'a>(
    bound: Option<&'a BoundOuterCurvature>,
    start: &Array1<f64>,
    n_params: usize,
) -> Option<&'a Array2<f64>> {
    bound
        .filter(|bound| outer_theta_bitwise_eq(&bound.theta, start))
        .map(|bound| &bound.hessian)
        .filter(|h| {
            h.nrows() == n_params && h.ncols() == n_params && h.iter().all(|v| v.is_finite())
        })
}

/// The solver's claimed stop, once it has cleared the analytic outer certificate.
///
/// Keeping the winner slot typed this way prevents a solver status bit from
/// participating in ranking.  Raw solver iterates and exhausted checkpoints
/// remain `OuterResult`s, but only this private wrapper can enter `best`.
struct CertifiedOuterCandidate(OuterResult);

impl CertifiedOuterCandidate {
    fn from_solver_claim(
        obj: &mut dyn OuterObjective,
        config: &OuterConfig,
        context: &str,
        mut candidate: OuterResult,
    ) -> Result<Self, (OuterResult, EstimationError)> {
        // #2359: this is the first-order FILTER, not the mint. It spends
        // first-order evidence only; the single order-four curvature audit is
        // paid once, on the winner, at the `PlanRunOutcome::Converged` exit
        // below. A candidate that is stationary but sits on inadmissible
        // curvature therefore survives screening and is refused at mint, which
        // is where the one order-four evaluation lives.
        match certify_outer_optimality_with_fidelity(
            obj,
            config,
            context,
            &mut candidate,
            CertificationFidelity::Screening,
        ) {
            Ok(certificate) => {
                candidate.criterion_certificate = Some(certificate);
                Ok(Self(candidate))
            }
            Err(error) => Err((candidate, error)),
        }
    }

    fn result(&self) -> &OuterResult {
        &self.0
    }

    fn into_result(self) -> OuterResult {
        self.0
    }
}

fn retain_best_outer_checkpoint(slot: &mut Option<OuterResult>, candidate: OuterResult) {
    let improves = candidate.final_value.is_finite()
        && slot.as_ref().is_none_or(|checkpoint| {
            !checkpoint.final_value.is_finite() || candidate.final_value < checkpoint.final_value
        });
    if improves {
        *slot = Some(candidate);
    }
}

/// The runner's side of an
/// [`OuterSeedProbe`](crate::estimate::outer_eval_capture::OuterSeedProbe): the
/// real objective, the seed it is entering, and that seed's box.
///
/// Every evaluation resets the objective and installs the seed's own inner
/// start, so a probe at a displaced θ solves from the starting point the seed
/// evaluation solves from. The caller lifts the inner-iteration cap for the
/// whole observation, so every evaluation is at full inner fidelity.
struct RunnerSeedProbe<'a> {
    obj: &'a mut dyn OuterObjective,
    config: &'a OuterConfig,
    context: &'a str,
    layout: crate::estimate::outer_eval_capture::OuterSeedLayout,
}

impl crate::estimate::outer_eval_capture::OuterSeedProbe for RunnerSeedProbe<'_> {
    fn layout(&self) -> &crate::estimate::outer_eval_capture::OuterSeedLayout {
        &self.layout
    }

    fn evaluate(
        &mut self,
        theta: &Array1<f64>,
        order: crate::estimate::outer_eval_capture::OuterSeedOrder,
    ) -> Result<crate::estimate::outer_eval_capture::OuterSeedEvaluation, EstimationError> {
        if theta.len() != self.layout.seed.len() {
            return Err(EstimationError::InvalidInput(format!(
                "outer-seed probe received theta_dim={} at a seed of dim {}",
                theta.len(),
                self.layout.seed.len()
            )));
        }
        self.obj.reset();
        install_matching_initial_inner_seed(
            &mut *self.obj,
            self.config,
            &self.layout.seed,
            self.context,
        )?;
        crate::estimate::outer_eval_capture::begin_outer_seed_capture();
        let evaluated = match order {
            crate::estimate::outer_eval_capture::OuterSeedOrder::Value => {
                self.obj.eval_cost(theta).map(|cost| (cost, None, None))
            }
            crate::estimate::outer_eval_capture::OuterSeedOrder::ValueAndGradient => self
                .obj
                .eval_with_order(theta, OuterEvalOrder::ValueAndGradient)
                .map(|eval| (eval.cost, Some(eval.gradient), None)),
            crate::estimate::outer_eval_capture::OuterSeedOrder::ValueGradientHessian => self
                .obj
                .eval_with_order(theta, OuterEvalOrder::ValueGradientHessian)
                .and_then(|eval| {
                    let hessian = match eval.hessian {
                        HessianValue::Dense(hessian) => Some(hessian),
                        HessianValue::Operator(op) => {
                            Some(op.materialize_dense().map_err(|message| {
                                EstimationError::RemlOptimizationFailed(format!(
                                    "outer-seed probe Hessian operator materialization failed: \
                                     {message}"
                                ))
                            })?)
                        }
                        HessianValue::Unavailable => None,
                    };
                    Ok((eval.cost, Some(eval.gradient), hessian))
                }),
        };
        let published = crate::estimate::outer_eval_capture::end_outer_seed_capture();
        let (cost, gradient, hessian) = evaluated?;
        Ok(published.into_evaluation(cost, gradient, hessian))
    }
}

/// Execute a single plan attempt (derived start → solver loop → best result).
///
/// `allow_tail_snap_reseed` gates the one-shot #2348 Inc 2b retry from a
/// confirmed-tail snapped checkpoint (see [`OuterResult::tail_snap_reseed`]);
/// the retry pass itself runs with it `false` so a reseed can never recurse.
pub(crate) fn run_outer_with_plan(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    context: &str,
    cap: &OuterCapability,
    the_plan: &OuterPlan,
    allow_tail_snap_reseed: bool,
) -> Result<PlanRunOutcome, EstimationError> {
    // Derivative/IFT masking belongs to the model domain, never to a temporary
    // active-set search face. In particular, freezing a model-lower-rail
    // coordinate creates a singleton search interval whose "upper" endpoint
    // is still the MODEL LOWER bound; recording it as an active model upper
    // bound silently erases the feasible inward derivative (#2514).
    let model_domain_bounds = outer_model_domain_bounds_template(config, cap.n_params);
    crate::estimate::reml::outer_eval::record_current_outer_rho_model_upper_bounds_for_ift(
        &model_domain_bounds.1,
    );
    let bounds_template = outer_search_bounds_template(config, cap.n_params);
    let seed_as_generated = project_to_bounds(
        &outer_start_point(config, cap.n_params, &model_domain_bounds)?,
        Some(&bounds_template),
    );
    let seed_idx = 0usize;

    // The certified winner, when the search from the start certifies.
    let mut best: Option<CertifiedOuterCandidate> = None;
    // The evaluated state the search ended on when it did not certify: a
    // refused certification or a budget-exhausted iterate. It is the resume
    // checkpoint, and it is what a certified winner is compared against before
    // it publishes (#2596, #2627). An earlier plan attempt's lowest state starts
    // it (#2953).
    let mut best_checkpoint: Option<OuterResult> = config.carried_checkpoint.clone();
    // Confirmed-tail snapped reseed published by a refused certification
    // (#2348 Inc 2b). Consumed once, after the search, for a single polishing
    // retry pinned at the snapped rail point.
    let mut tail_snap_reseed_point: Option<Array1<f64>> = None;
    // Negative-curvature escape reseed published by a refused certification
    // whose interior reduced Hessian is a certified strict saddle (#2357).
    // Consumed once, after the search, for a single retry seeded off the saddle
    // ridge so the outer search descends to the true PSD minimum.
    let mut saddle_escape_reseed_point: Option<Array1<f64>> = None;
    // A reactive domain-entry path is created only after the objective's exact
    // start cost is non-finite. An already-feasible start therefore stays on
    // the zero-heavy-entry path.
    let reactive_domain_scalar_contract = obj.reactive_domain_scalar_contract()?;
    let reactive_domain_entry_available = reactive_domain_scalar_contract.is_some();
    // Sole owner of every start refusal. Objective failures enter this ledger
    // while their `ObjectiveEvalError` still carries the originating typed
    // `EstimationError`; there is no parallel prose ledger to reconcile later.
    let mut seed_rejections: Vec<SeedRejection> = Vec::new();
    let layout = cap.theta_layout();
    let mut started_seeds = 0usize;
    // Iterations spent by the search this attempt started (#2817).
    let mut spent_seed_iterations: usize = 0;

    'seed_attempt: {
        // The point the solver starts from: the derived start, or that point
        // with the under-penalized block's strengths raised by the ratio the
        // inner solve read off its ray (#2695).
        let mut seed_owned = seed_as_generated.clone();
        let seed = &mut seed_owned;
        log::debug!(
            "[OUTER] {context}: entering the outer search from the single derived start on \
             {the_plan}"
        );
        // Domain entry is a property of this literal start.
        let mut continuation_path: Option<crate::continuation_path::ContinuationPath> = None;
        obj.reset();
        if let Some(observer) =
            crate::estimate::outer_eval_capture::take_outer_seed_observer(cap.psi_dim)
        {
            // An observer must not decide the fit it observes.
            //
            // The finite-difference audit this hook replaced once ended in `?`.
            // When its own evaluation refused at a cold start -- common on a
            // spatial basis, where the inner solve has not converged at theta_0
            // -- the error propagated out of `run_plan` and ABORTED THE WHOLE
            // RUN, where with the audit disabled the identical start would
            // simply have been searched. Arming the instrument changed the
            // outcome it was measuring. So a refusal is warned about, the
            // objective is reset to its pristine baseline, and the real path
            // below (including any curvature homotopy) is bit-identical to a
            // run with no observer, on the refusal path as well as the success
            // path.
            let full_fidelity_guard = config
                .outer_inner_cap
                .as_ref()
                .map(FullFidelityInnerCapGuard::lift);
            let mut runner_probe = RunnerSeedProbe {
                obj: &mut *obj,
                config,
                context,
                layout: crate::estimate::outer_eval_capture::OuterSeedLayout {
                    seed: seed.clone(),
                    lower: bounds_template.0.clone(),
                    upper: bounds_template.1.clone(),
                    rho_dim: cap.theta_layout().rho_dim(),
                    psi_dim: cap.psi_dim,
                },
            };
            let probe: &mut dyn crate::estimate::outer_eval_capture::OuterSeedProbe =
                &mut runner_probe;
            if let Err(err) = observer(probe) {
                log::debug!(
                    "[OUTER] {context}: outer-seed observer refused at the start ({err}); the \
                     search proceeds unchanged"
                );
            }
            drop(full_fidelity_guard);
            obj.reset();
        }
        // Certified curvature-homotopy entry leg (#1007). When the objective
        // has a certified anchor (the SAE-manifold `η = 0` Eckart-Young
        // relaxation), run the predictor-corrector `η`-walk from it: a single
        // walk along the unique optimal branch reaches the real (`η = 1`)
        // objective, leaving the inner state warm there. The min-pivot
        // invariant + step-halving make the walk certified; a degenerate
        // anchor or a detected bifurcation returns `false` (the term is left at
        // the full basis) and the direct search below takes over — the outcome
        // is recorded on the fit payload either way, never a silent fallback.
        // The walk runs right after `reset`, so state hygiene is unchanged
        // (#1003): `reset` restores the pristine `η = 1` baseline first.
        let curvature_entry_refused = match obj.curvature_homotopy_entry(seed) {
            Some(Ok(arrived)) => {
                log::debug!("[OUTER] {context}: curvature-homotopy entry arrived={arrived}");
                !arrived
            }
            Some(Err(err)) => {
                // A hard anchor-construction failure is not a feasibility gate:
                // fall through to the direct search.
                log::debug!(
                    "[OUTER] {context}: curvature-homotopy entry errored ({err}); searching \
                     directly from the start"
                );
                obj.reset();
                false
            }
            None => false,
        };
        if curvature_entry_refused {
            // A refused walk is NEVER a feasibility gate. By contract the walk
            // leaves the term at the full `η = 1` basis (a degenerate anchor or
            // a detected branch bifurcation), so the direct search below —
            // `accept_seed_without_outer_iterations` and the solve at `seed` —
            // takes over from the pristine cold state (#1095: a periodic K=1
            // circle whose walk refuses on a small-N pivot bifurcation and which
            // does not advertise reactive domain entry).
            log::debug!(
                "[OUTER] {context}: curvature-homotopy entry refused; searching directly from \
                 the pristine baseline"
            );
            obj.reset();
        }
        install_matching_initial_inner_seed(obj, config, seed, context)?;
        // Zero-iteration acceptance, decided HERE rather than in the objective.
        //
        // Whether a start is already stationary is a question about the
        // stationarity BAND, and the band lives with `OuterConfig`
        // (`outer_gradient_tolerance`), not with the objective.
        //
        // Measured (#2363): a fit resumed from a prior fit's terminal
        // certificate is stationary where it starts. |Pg| at the resumed rho is
        // 4.225362e-9 / 6.680405e-8 / 1.369603e-7 on the three estimated-nuisance
        // fixtures, against a band of 1.61e-5 -- inside by three to four orders
        // of magnitude. It then takes one outer iteration to go nowhere, which
        // this skips along with its inner solves.
        //
        // The branch RE-CERTIFIES what it accepts through
        // `CertifiedOuterCandidate::from_solver_claim` and, when that fails,
        // runs the ordinary search from the same start, so an over-eager
        // acceptance costs nothing; and it fires only for a rho a previous
        // outer run already certified as terminal.
        let zero_iteration_cost = match obj.accept_seed_without_outer_iterations(seed)? {
            Some(cost) => Some(cost),
            None => certified_resume_is_already_stationary(
                obj,
                config,
                seed,
                &bounds_template,
                seed_idx,
                context,
            ),
        };
        if let Some(seed_cost) = zero_iteration_cost {
            let mut candidate = OuterResult::new(seed.clone(), seed_cost, 0, true, *the_plan);
            candidate.origin = OuterResultOrigin::SeedAcceptedWithoutIteration;
            match CertifiedOuterCandidate::from_solver_claim(obj, config, context, candidate) {
                Ok(candidate) => {
                    started_seeds += 1;
                    best = Some(candidate);
                    break 'seed_attempt;
                }
                Err((_, error)) => {
                    log::debug!(
                        "[OUTER] {context}: zero-iteration start claimed acceptance but failed \
                         analytic certification ({error}); searching from it"
                    );
                    obj.reset();
                    install_matching_initial_inner_seed(obj, config, seed, context)?;
                }
            }
        }
        // Typed, reactive domain entry. The literal seed is always evaluated
        // first on the real objective. A finite value keeps the converged probe
        // handoff and pays no continuation work. Only an undefined criterion
        // activates the certified heavy-smoothing path; a hard evaluation error
        // remains a seed refusal and is never converted into a pseudo-value.
        let mut reactive_domain_entry_requested = false;
        if reactive_domain_entry_available {
            match obj.eval_cost(seed) {
                Ok(cost) if cost.is_finite() => {
                    log::trace!(
                        "[OUTER] {context}: exact seed {seed_idx} is inside the objective domain; \
                         reactive continuation entry not needed"
                    );
                }
                Ok(_) => {
                    reactive_domain_entry_requested = true;
                    log::debug!(
                        "[OUTER] {context}: exact seed {seed_idx} has undefined criterion; \
                         entering through certified heavy-smoothing continuation"
                    );
                    // The failed cold probe may have left objective-owned
                    // trial state. Re-enter from the pristine baseline;
                    // successful path evaluations establish a fresh
                    // exact-seed handoff.
                    obj.reset();
                    continuation_path = Some(
                        crate::continuation_path::ContinuationPath::heavy_entry_for_rho(
                            seed.clone(),
                            bounds_template.1.clone(),
                            reactive_domain_scalar_contract
                                .clone()
                                .expect("reactive scalar contract checked above"),
                        )?,
                    );
                }
                Err(err) => {
                    log::debug!(
                        "[OUTER] {context}: rejecting seed {seed_idx}: reactive domain-entry \
                         seed probe failed before continuation: {err}"
                    );
                    seed_rejections.push(SeedRejection::from_estimation_error(
                        seed_idx,
                        "domain-entry",
                        &err,
                    ));
                    break 'seed_attempt;
                }
            }
        }
        // Reactive domain entry (SAE-manifold dense K>=2 joint fit): DRIVE the
        // coupled `ContinuationPath` homotopy explicitly. Each step installs
        // the objective-owned scalar state and evaluates its matching log-ρ
        // waypoint exactly once inside a full-state transaction. The committed
        // term/rho/loss and beta hint warm the next waypoint; arrival hands the
        // exact target state to the normal solver. A failed attempted waypoint refines the step from the last
        // successful state; representability exhaustion becomes a typed domain
        // refusal rather than a false arrival.
        //
        // The heavy-smoothing walk warms the cold inner solve after the literal
        // `eval_cost` demonstrated that its Laplace evidence is undefined (the
        // K>=2 routing-collapse failure Object 1 exists to repair).
        let mut continuation_arrived = continuation_path.is_none();
        let mut continuation_arrival_refusal: Option<String> = None;
        if continuation_path.is_some() {
            {
                let path = continuation_path
                    .as_mut()
                    .expect("reactive continuation path checked above");
                let walk_start = std::time::Instant::now();
                // Only the first path call is cold. After it commits, the path
                // and objective own the complete accepted state transactionally.
                let cold_entry_beta: Array1<f64> = Array1::zeros(0);
                let mut legs_descended = 0usize;
                // The path controls its own progress from solver evidence. It
                // can only report arrival after a successful exact-target leg;
                // inability to refine a failed leg is returned as a typed
                // refusal, so this loop needs no unrelated iteration ceiling.
                loop {
                    let step = match path.step(obj, &cold_entry_beta) {
                        Ok(step) => step,
                        Err(err) => {
                            continuation_arrival_refusal = Some(format!(
                                "reactive domain entry refused before exact-target arrival: {err}"
                            ));
                            break;
                        }
                    };
                    match step {
                        crate::continuation_path::ContinuationStep::Entered { state } => {
                            if !state.last_eval.cost.is_finite() {
                                continuation_arrival_refusal = Some(format!(
                                    "reactive domain entry committed a non-finite entry-waypoint cost {}",
                                    state.last_eval.cost
                                ));
                                break;
                            }
                            legs_descended += 1;
                        }
                        crate::continuation_path::ContinuationStep::Descended { s, state } => {
                            if !state.last_eval.cost.is_finite() {
                                continuation_arrival_refusal = Some(format!(
                                    "reactive domain entry committed a non-finite waypoint cost {} at s={s}",
                                    state.last_eval.cost
                                ));
                                break;
                            }
                            if !(s.is_finite() && s > 0.0) {
                                continuation_arrival_refusal = Some(format!(
                                    "reactive domain entry reported an invalid descended waypoint s={s}"
                                ));
                                break;
                            }
                            legs_descended += 1;
                        }
                        crate::continuation_path::ContinuationStep::Arrived { state } => {
                            // Leave the objective in the path-warmed state.
                            // The exact-value verification below owns the
                            // full-state handoff; replacing it with a copied
                            // coefficient-only seed here would discard it.
                            legs_descended += 1;
                            let scalar_at_target = path.current_scalar_targets().bitwise_eq(
                                reactive_domain_scalar_contract
                                    .as_ref()
                                    .expect("reactive scalar contract checked above")
                                    .target(),
                            );
                            if !scalar_at_target {
                                continuation_arrival_refusal = Some(
                                    "reactive domain entry reported arrival away from the literal scalar target"
                                        .to_string(),
                                );
                            } else {
                                match reactive_arrival_postcondition(&state, seed) {
                                    Ok(()) => continuation_arrived = true,
                                    Err(reason) => continuation_arrival_refusal = Some(reason),
                                }
                            }
                            break;
                        }
                        crate::continuation_path::ContinuationStep::Refined { s, reason } => {
                            use crate::continuation_path::RefinementReason;
                            // The accepted waypoint remains unchanged while the
                            // next attempted distance is refined. Consume the
                            // reason for diagnostics, then continue.
                            let RefinementReason::WaypointStruggled(failure) = reason;
                            log::debug!(
                                "[OUTER] {context}: continuation seed {seed_idx} coupled \
                                 waypoint struggled below accepted s={s:.4} ({}); refining the \
                                 next attempted distance",
                                failure.message(),
                            );
                        }
                    }
                }
                log::debug!(
                    "[OUTER] {context}: continuation-path walk seed {seed_idx} legs={legs_descended} \
                     arrived={continuation_arrived} accepted_s={:.4} elapsed={:.3}s",
                    path.s(),
                    walk_start.elapsed().as_secs_f64(),
                );
            }
        }
        if reactive_domain_entry_requested {
            if !continuation_arrived {
                let msg = continuation_arrival_refusal.take().unwrap_or_else(|| {
                    "reactive domain entry refused before a solved exact-target waypoint"
                        .to_string()
                });
                log::debug!("[OUTER] {context}: rejecting seed {seed_idx}: {msg}");
                seed_rejections.push(SeedRejection::from_message(seed_idx, "domain-entry", msg));
                break 'seed_attempt;
            }
            // Independently re-evaluate the literal target and require a finite
            // exact criterion before any optimizer can start.
            match obj.eval_cost(seed) {
                Ok(cost) if cost.is_finite() => {
                    log::debug!(
                        "[OUTER] {context}: reactive continuation seed {seed_idx} arrived with \
                         finite exact criterion {cost:.6e}"
                    );
                }
                Ok(_) => {
                    let msg = "reactive domain entry refused: exact seed criterion remained \
                               non-finite after certified continuation arrival"
                        .to_string();
                    log::debug!("[OUTER] {context}: rejecting seed {seed_idx}: {msg}");
                    seed_rejections.push(SeedRejection::from_message(
                        seed_idx,
                        "domain-entry",
                        msg,
                    ));
                    break 'seed_attempt;
                }
                Err(err) => {
                    return Err(EstimationError::fatal_outer_evaluation(
                        "reactive continuation target verification",
                        err,
                    ));
                }
            }
        }
        let t_seed_start = std::time::Instant::now();
        let result: Result<OuterResult, EstimationError> = match the_plan.solver {
            Solver::Arc => {
                let seed_eval = eval_seed_restoring_rays(
                    obj,
                    config,
                    seed,
                    OuterEvalOrder::ValueGradientHessian,
                    RayRestorationDomain { upper: &bounds_template.1, rho_dim: layout.rho_dim() },
                    context,
                    seed_idx,
                )
                    .map_err(|err| into_objective_error("outer eval failed", err));
                let seed_eval = match seed_eval {
                    Ok(seed_eval) => seed_eval,
                    Err(err) if err.is_recoverable() => {
                        log::debug!(
                            "[OUTER] {context}: rejecting seed {seed_idx} before solver start: {err}"
                        );
                        seed_rejections.push(SeedRejection::from_objective_error(
                            seed_idx,
                            "validation",
                            &err,
                        ));
                        break 'seed_attempt;
                    }
                    Err(err) => {
                        return Err(EstimationError::fatal_objective_evaluation(
                            "outer ARC seed evaluation",
                            err,
                        ));
                    }
                };
                let seed_eval = finite_outer_eval_or_error("outer eval failed", layout, seed_eval);
                let mut seed_eval = match seed_eval {
                    Ok(seed_eval) => seed_eval,
                    Err(err) if err.is_recoverable() => {
                        log::debug!(
                            "[OUTER] {context}: rejecting seed {seed_idx} before solver start: {err}"
                        );
                        seed_rejections.push(SeedRejection::from_objective_error(
                            seed_idx,
                            "validation",
                            &err,
                        ));
                        break 'seed_attempt;
                    }
                    Err(err) => {
                        return Err(EstimationError::fatal_objective_evaluation(
                            "outer ARC seed validation",
                            err,
                        ));
                    }
                };
                if let Err(err) = validate_second_order_seed_hessian(context, layout, &seed_eval) {
                    if err.is_recoverable() {
                        log::debug!(
                            "[OUTER] {context}: rejecting seed {seed_idx} before solver start: {err}"
                        );
                        seed_rejections.push(SeedRejection::from_objective_error(
                            seed_idx,
                            "validation",
                            &err,
                        ));
                        break 'seed_attempt;
                    }
                    return Err(EstimationError::fatal_objective_evaluation(
                        "outer ARC second-order seed validation",
                        err,
                    ));
                }
                started_seeds += 1;

                let cheap_materializable_operator = matches!(
                    seed_eval.hessian,
                    HessianValue::Operator(ref op) if operator_hessian_densifies(op.as_ref())
                );
                if cheap_materializable_operator {
                    // The operator already holds its dense Hessian and it fits the
                    // materialization cap; convert the seed Hessian to dense in-place.
                    // Subsequent bridge evaluations apply the same predicate.
                    if let HessianValue::Operator(op) = &seed_eval.hessian {
                        match op.materialize_dense() {
                            Ok(dense) => {
                                seed_eval.hessian = HessianValue::Dense(dense);
                            }
                            Err(message) => {
                                let err = EstimationError::RemlOptimizationFailed(format!(
                                    "outer Hessian operator materialization failed: {message}"
                                ));
                                log::debug!(
                                    "[OUTER] {context}: rejecting seed {seed_idx} before solver start: {err}"
                                );
                                // No producer verdict: a Hessian operator that
                                // cannot be densified is a statement about the
                                // operator, not about this seed's rho.
                                seed_rejections.push(SeedRejection::from_estimation_error(
                                    seed_idx,
                                    "validation",
                                    &err,
                                ));
                                break 'seed_attempt;
                            }
                        }
                    }
                }
                if matches!(seed_eval.hessian, HessianValue::Operator(_)) {
                    log::trace!(
                        "[OUTER] {context}: analytic Hessian provided as Hv operator; \
                        routing to opt::MatrixFreeTrustRegion (Steihaug-Toint CG)"
                    );
                    let (lo, hi) = &bounds_template;
                    let bounds_obj = outer_bounds(lo, hi)?;
                    // Scale-aware tolerance via opt 0.5.0:
                    // `relative_to_cost(τ)` = `τ * (1 + |f|)` resolved
                    // at run time from the seed cost and initial grad
                    // norm. Replaces the previous gam-side
                    // precomputed `outer_scaled_tolerance` hack.
                    let grad_tol = outer_gradient_tolerance(config);
                    let max_iter = outer_max_iterations(config.max_iter)?;

                    // Translate the seed_eval into an opt::OperatorSample
                    // so the matrix-free TR solver can serve its first
                    // call from cache without redoing the full outer
                    // eval. The Hessian translation goes through the
                    // gam->opt operator adapter when the seed Hessian is
                    // an Hv operator; Analytic seeds become Dense.
                    let initial_op_sample = OperatorSample {
                        value: seed_eval.cost,
                        gradient: seed_eval.gradient.clone(),
                        hessian: seed_eval.hessian.clone(),
                    };

                    // opt's matrix-free trust region has no stall stop of its
                    // own, so a boundary-limited crawl buying sub-resolution
                    // descent ended only when its iteration count ran out. The
                    // same progress certificate the dense route uses ends it
                    // instead (#2817); its floor and band are derived exactly as
                    // the ARC arm below derives them.
                    let mut cost_stall_guard = CostStallGuard::new(
                        config
                            .rel_cost_tolerance
                            .unwrap_or(config.tolerance * 1.0e-2)
                            .max(COST_STALL_REL_TOL_FLOOR),
                        ARC_COST_STALL_WINDOW,
                        config,
                        Arc::new(Mutex::new(None)),
                    );
                    cost_stall_guard.observe_seed(
                        &seed,
                        seed_eval.cost,
                        rail_projected_gradient_norm(
                            &seed,
                            &seed_eval.gradient,
                            Some(&(lo.clone(), hi.clone())),
                        ),
                    );
                    let unprogressing_stop: Arc<Mutex<Option<CostStallExit>>> =
                        Arc::new(Mutex::new(None));
                    let bridge_obj = OuterOperatorBridge {
                        obj,
                        layout,
                        outer_inner_cap: config.outer_inner_cap.clone(),
                        eval_count: 0,
                        g_norm_initial: None,
                        last_g_norm: None,
                        last_value_grad_rho: None,
                        cost_stall: Some(cost_stall_guard),
                        cost_stall_bounds: Some((lo.clone(), hi.clone())),
                        unprogressing_stop: Arc::clone(&unprogressing_stop),
                    };

                    let mut solver = MatrixFreeTrustRegion::new(seed.clone(), bridge_obj)
                        .with_bounds(bounds_obj)
                        .with_gradient_tolerance(grad_tol)
                        .with_max_iterations(max_iter)
                        .with_initial_sample(seed.clone(), initial_op_sample)
                        // Stop the search on the test that will judge it.
                        //
                        // The certificate accepts a point when the Newton
                        // decrement ½gᵀH⁻¹g at it is below the criterion's own
                        // resolution `rel_cost_floor·(1 + |V|)` — the
                        // curvature-resolvability rung, which is what mgcv
                        // means by convergence and which the certificate
                        // computes for itself at the end. The solver was
                        // driven instead to an absolute projected-gradient
                        // band, which on a flat REML valley is a far stricter
                        // and unrelated standard: measured on a gaussian
                        // n=50 000, p=93, K=11 fit, the band was 7.451e-4
                        // while the certificate accepted 2.173e-2 — 29× wider
                        // — so no seed could ever stop itself. All six runs
                        // (three seeds, then the whole sweep again through the
                        // budget-exhaustion retry) burned their 200-iteration
                        // budget, 1200 outer evaluations, for a last-100
                        // improvement of 4e-4 in a criterion of 5.3e4 (#2817).
                        //
                        // opt's Steihaug–Toint subsolve already computes the
                        // decrement: on an interior step its predicted decrease
                        // IS ½gᵀH⁻¹g. Handing it the certificate's tolerance
                        // makes the stopping rule and the acceptance rule one
                        // standard. Anchored on the seed's own cost, so the
                        // threshold is fixed for the run rather than drifting
                        // with the iterate.
                        .with_model_decrement_tolerance(
                            outer_rel_cost_floor(config) * (1.0 + seed_eval.cost.abs()),
                        );
                    // Installed unconditionally now that it also carries the
                    // trajectory census (#2735): a walk that ends on its budget
                    // has to be able to say whether it crawled or thrashed, and
                    // the inner-cap channel it used to be gated on is unrelated
                    // to that question.
                    let census = Arc::new(OuterStepCensus::default());
                    solver = solver.with_observer(OuterAcceptObserver {
                        feedback: config.outer_inner_cap.clone(),
                        accepted_steps: None,
                        census: Some(Arc::clone(&census)),
                    });
                    if let Some(r) = sanitized_operator_trust_restart_radius(
                        config.operator_initial_trust_radius,
                    ) {
                        solver = solver.with_initial_trust_radius(r);
                    }

                    let mf_start = std::time::Instant::now();
                    let report = solver.run_report();
                    let mf_elapsed = mf_start.elapsed().as_secs_f64();
                    let final_radius = report.diagnostics.final_trust_radius;
                    log::debug!(
                        "[OUTER summary] matrix-free TR finished status={:?} in {} iters \
                         elapsed={:.3}s final_value={:.6e} final_trust_radius={} | {}",
                        report.status,
                        report.solution.iterations,
                        mf_elapsed,
                        report.solution.final_value,
                        match final_radius {
                            Some(r) => format!("{:.3e}", r),
                            None => "n/a".to_string(),
                        },
                        census
                            .describe()
                            .unwrap_or_else(|| "no step observed".to_string()),
                    );
                    // Translate the structured report into an `OuterResult`.
                    // `operator_stop_reason` wiring (read by the gam-side
                    // retry orchestrator in `run_outer_with_plan`) maps
                    // directly from `OptimizationStatus`. opt 0.4.1
                    // populates `final_trust_radius` so the
                    // `operator_trust_radius` warm-start hook now works
                    // for matrix-free retries: the budget-bumped retry
                    // resumes from the geometry the previous attempt
                    // already learned instead of redoing the trust-radius
                    // adaptation from the configured initial radius.
                    match report.status {
                        OptimizationStatus::Converged
                        | OptimizationStatus::NumericallyConverged => {
                            let mut result =
                                solution_into_outer_result(report.solution, true, *the_plan);
                            result.operator_trust_radius = final_radius;
                            Ok(result)
                        }
                        OptimizationStatus::MaxIterations => {
                            log::debug!(
                                "[OUTER warning] {context}: matrix-free TR hit max_iter={} at final_value={:.6e} |g|={:.3e} final_trust_radius={}",
                                config.max_iter,
                                report.solution.final_value,
                                report.solution.final_gradient_norm.unwrap_or(f64::NAN),
                                match final_radius {
                                    Some(r) => format!("{:.3e}", r),
                                    None => "n/a".to_string(),
                                },
                            );
                            let mut result =
                                solution_into_outer_result(report.solution, false, *the_plan);
                            result.operator_trust_radius = final_radius;
                            Ok(result)
                        }
                        OptimizationStatus::TrustRegionRejectFloor => {
                            log::debug!(
                                "[OUTER warning] {context}: matrix-free TR reached trust-radius reject floor at final_value={:.6e} |g|={:.3e} final_trust_radius={}",
                                report.solution.final_value,
                                report.solution.final_gradient_norm.unwrap_or(f64::NAN),
                                match final_radius {
                                    Some(r) => format!("{:.3e}", r),
                                    None => "n/a".to_string(),
                                },
                            );
                            // The guard's own best feasible iterate is not published on
                            // this route, so the stopped iterate is the checkpoint.
                            let mut result = stopped_run_checkpoint(
                                context,
                                "matrix-free TR reject-floor",
                                report.solution,
                                None,
                                *the_plan,
                            );
                            result.operator_trust_radius = final_radius;
                            Ok(result)
                        }
                        // opt 0.5.13 native cost-stall exits: `CostStallConverged`
                        // means the cost flatlined AND the bound-projected
                        // gradient at the best iterate cleared the outer
                        // tolerance — a KKT-stationary success, same verdict as
                        // `Converged`. `CostStallFloor` is the flat-valley floor
                        // with residual non-stationarity: halt is correct but
                        // NOT a success; map it to `CostStallFlatValley` so the
                        // retry orchestrator (run.rs) skips the wasted replay
                        // and the shipped-β gradient reconciliation
                        // (estimate/optimizer.rs) can still upgrade a
                        // score-relative near-stationary floor.
                        OptimizationStatus::CostStallConverged => {
                            let mut result =
                                solution_into_outer_result(report.solution, true, *the_plan);
                            result.operator_trust_radius = final_radius;
                            Ok(result)
                        }
                        OptimizationStatus::CostStallFloor => {
                            log::debug!(
                                "[OUTER warning] {context}: matrix-free TR stopped on a cost stall \
                                 with non-stationary projected gradient at final_value={:.6e} |g|={:.3e}",
                                report.solution.final_value,
                                report.solution.final_gradient_norm.unwrap_or(f64::NAN),
                            );
                            let mut result =
                                solution_into_outer_result(report.solution, false, *the_plan);
                            result.operator_trust_radius = final_radius;
                            Ok(result)
                        }
                        // `opt` reports an objective failure without its
                        // message, so the bridge's stop slot is what tells the
                        // guard's unprogressing stop from a genuine failure.
                        OptimizationStatus::ObjectiveFailed => {
                            match unprogressing_stop.lock().ok().and_then(|mut slot| slot.take()) {
                                Some(exit) => {
                                    let mut result = outer_result_with_gradient_norm(
                                        exit.rho,
                                        exit.value,
                                        exit.iterations,
                                        Some(exit.grad_norm),
                                        false,
                                        *the_plan,
                                    );
                                    result.origin =
                                        OuterResultOrigin::OperatorUnprogressingStallCheckpoint;
                                    result.operator_trust_radius = final_radius;
                                    Ok(result)
                                }
                                None => Err(EstimationError::fatal_outer_evaluation(
                                    "matrix-free trust-region evaluation",
                                    EstimationError::RemlOptimizationFailed(
                                        "matrix-free trust-region objective evaluation failed"
                                            .to_string(),
                                    ),
                                )),
                            }
                        }
                        OptimizationStatus::NumericalFailure
                        | OptimizationStatus::LineSearchFailed => {
                            Err(EstimationError::RemlOptimizationFailed(format!(
                                "matrix-free TR solver failed with status={:?}", report.status
                            )))
                        }
                    }
                } else {
                    let hessian_source = the_plan.hessian_source;
                    let (lo, hi) = &bounds_template;
                    let bounds = outer_bounds(lo, hi)?;
                    let grad_tol = outer_gradient_tolerance(config);
                    let max_iter = outer_max_iterations(config.max_iter)?;

                    // Cost-stall convergence guard for the ARC outer loop
                    // (#1089/#1237). Identical wiring to the BFGS branch below:
                    // a near-separable multinomial REML criterion decreases
                    // monotonically as λ→0, so several log-λ directions slam to
                    // the lower bound and bounce and ARC otherwise cycles to its
                    // `max_iter` cap (the #1082 multinomial timeout) without
                    // certifying a stationary point. The guard halts ARC at the
                    // best iterate; the bound-PROJECTED gradient norm decides the
                    // converged verdict (a bound-pinned separating direction is
                    // KKT-stationary even though its raw ∂V/∂ρ never vanishes).
                    let cost_stall_exit: Arc<Mutex<Option<CostStallExit>>> =
                        Arc::new(Mutex::new(None));
                    let cost_stall_rel_tol = config
                        .rel_cost_tolerance
                        .unwrap_or(config.tolerance * 1.0e-2)
                        .max(COST_STALL_REL_TOL_FLOOR);

                    // Build the exact seed Hessian before enrolling the seed in
                    // the stall guard. The guard must know whether its incumbent
                    // is a second-order point: repeated infeasible trials cannot
                    // justify halting at a certified strict saddle.
                    let seed_hessian = build_bridge_hessian_for_source(
                        hessian_source,
                        seed_eval.hessian,
                    )
                    .map_err(|err| {
                        EstimationError::fatal_objective_evaluation(
                            "outer ARC seed Hessian preparation",
                            err,
                        )
                    })?;
                    // Same rail-relaxed box the guard's later curvature reads
                    // use (#2412); the seed must not be judged against a
                    // different critical cone than the iterates that follow it.
                    let seed_rail_bounds = rail_relaxed_bounds(&(lo.clone(), hi.clone()));
                    // Judged at the same criterion curvature resolution the
                    // bridge's later verdicts use (#1082), so the seed is not a
                    // strict saddle by a standard the iterates never face.
                    let seed_curvature_resolution = super::run::criterion_curvature_resolution(
                        outer_rel_cost_floor(config),
                        seed_eval.cost,
                    );
                    let seed_hessian_psd = seed_hessian.as_ref().and_then(|dense| {
                        reduced_hessian_psd_at_point(
                            &seed,
                            &seed_eval.gradient,
                            dense,
                            Some((&seed_rail_bounds.0, &seed_rail_bounds.1)),
                            seed_curvature_resolution,
                        )
                    });

                    let mut cost_stall_guard = CostStallGuard::new(
                        cost_stall_rel_tol,
                        ARC_COST_STALL_WINDOW,
                        config,
                        cost_stall_exit.clone(),
                    );
                    cost_stall_guard.observe_second_order_seed(
                        &seed,
                        seed_eval.cost,
                        // Same rail-relaxed box the guard's later observations
                        // and the terminal certificate use (#2412); a seed that
                        // starts on a rail must not be scored against a
                        // different box than the iterates that follow it.
                        rail_projected_gradient_norm(
                            &seed,
                            &seed_eval.gradient,
                            Some(&(lo.clone(), hi.clone())),
                        ),
                        seed_hessian_psd,
                    );

                    let last_objective_error: Arc<Mutex<Option<ObjectiveEvalError>>> =
                        Arc::new(Mutex::new(None));
                    let objective = RetainingObjective::new(
                        OuterSecondOrderBridge {
                        obj,
                        layout,
                        hessian_source,
                        eval_count: 0,
                        outer_inner_cap: config.outer_inner_cap.clone(),
                        g_norm_initial: None,
                        last_g_norm: None,
                        last_value_grad_rho: None,
                        cost_stall: Some(cost_stall_guard),
                        cost_stall_bounds: Some((lo.clone(), hi.clone())),
                        // #2817 — the search stops on the test that judges it.
                        // The certificate accepts a point whose Newton
                        // decrement ½gᵀH⁻¹g is at or below the criterion's own
                        // resolution `rel_cost_floor·(1 + |V|)`; the dense ARC
                        // route was driven instead to an absolute
                        // projected-gradient band, which on a flat REML valley
                        // is a far stricter and unrelated standard, so no seed
                        // could stop itself and every fit ran to its iteration
                        // cap. Handing the bridge the same floor the
                        // certificate uses makes the stopping rule and the
                        // acceptance rule one standard. The matrix-free route
                        // already does this through opt's own decrement rung
                        // (`with_model_decrement_tolerance` above); this is the
                        // dense route's half of the same repair.
                        curvature_stationary_floor: Some(outer_rel_cost_floor(config)),
                        // #2954 — and on the rung it judges on. Where the route
                        // declares its size the certificate decides on the
                        // Newton-decrement verdict on rounding bands, not on
                        // `floor·(1 + |V|)`; without the config the loop kept
                        // stopping on the older rung at points the certificate
                        // then refused.
                        decrement_verdict_config: Some(config),
                        },
                        Arc::clone(&last_objective_error),
                    );

                    let initial_sample = SecondOrderSample {
                        value: seed_eval.cost,
                        gradient: seed_eval.gradient,
                        hessian: seed_hessian,
                        decrement_bands: None,
                    };

                    let mut optimizer = ArcOptimizer::new(seed.clone(), objective)
                        .with_bounds(bounds)
                        .with_gradient_tolerance(grad_tol)
                        .with_max_iterations(max_iter)
                        .with_initial_sample(seed.clone(), initial_sample);
                    if let Some(sigma) = config.arc_initial_regularization {
                        optimizer = optimizer.with_initial_regularization(sigma);
                    }
                    // Same reason as the matrix-free route above: the census
                    // is what a budget-exhausted ARC run needs to report (#2735).
                    let arc_census = Arc::new(OuterStepCensus::default());
                    optimizer = optimizer.with_observer(OuterAcceptObserver {
                        feedback: config.outer_inner_cap.clone(),
                        accepted_steps: None,
                        census: Some(Arc::clone(&arc_census)),
                    });
                    // On the exact-Hessian ARC route, forbid `opt`'s
                    // internal AutoBfgs demotion on step failure, so the
                    // run stays inside analytic-Hessian geometry and
                    // surfaces mismatches loudly. opt itself refuses a
                    // missing Hessian (`SecondOrderSample { hessian: None }`)
                    // as a fatal evaluation error; it never estimates one.
                    if matches!(hessian_source, HessianSource::Analytic) {
                        optimizer = optimizer.with_fallback_policy(OptFallbackPolicy::Never);
                    }
                    match optimizer.run() {
                        Ok(sol) => Ok(solution_into_outer_result(sol, true, *the_plan)),
                        Err(ArcError::MaxIterationsReached { last_solution, .. }) => {
                            log::debug!(
                                "[OUTER warning] {context}: ARC hit max_iter={} at final_value={:.6e} |g|={:.3e} | {}",
                                config.max_iter,
                                last_solution.final_value,
                                last_solution.final_gradient_norm.unwrap_or(f64::NAN),
                                arc_census
                                    .describe()
                                    .unwrap_or_else(|| "no step observed".to_string()),
                            );
                            // Budget exhaustion (#1371): the optimizer hands back
                            // its LAST iterate, which on a flat REML valley can be
                            // a degenerate box corner the trajectory wandered to
                            // on an indefinite ρ-Hessian step — e.g. `ρ_nullspace
                            // → +∞` on a `bs="ps"` double-penalty smooth, which
                            // shrinks the null-space ridge `Z Zᵀ` so hard that a
                            // genuine, strongly-supported linear trend is
                            // annihilated and the fit collapses to a flat constant
                            // (edf_total→1). The cost-stall guard tracked the best
                            // FEASIBLE iterate the trajectory actually evaluated
                            // and published it to `cost_stall_exit`; never return
                            // an iterate whose REML objective is worse than one the
                            // optimizer already passed through. Mirrors the
                            // separation-corner regression guard in
                            // `CostStallGuard::observe_constrained_stationary`
                            // (#1355); here it covers the budget-exhaustion exit.
                            let best_exit =
                                cost_stall_exit.lock().ok().and_then(|slot| slot.clone());
                            // The best-feasible-iterate substitution produces this
                            // run's non-converged `result`; it never counts as a
                            // converged fit, so it flows into the checkpoint slot
                            // and the post-run certify/reseed ladder decides
                            // whether it can be minted (#1371, #1476).
                            Ok(stopped_run_checkpoint(
                                context,
                                "ARC budget-exhaustion",
                                *last_solution,
                                best_exit,
                                *the_plan,
                            ))
                        }
                        // The cubic regularisation reached its ceiling with no accepted step
                        // (#2953). `opt` hands back the iterate the run stopped at, so it is a
                        // checkpoint like a budget exit, not a failure that loses it.
                        Err(ArcError::TrustRegionRejectFloor { last_solution }) => {
                            log::debug!(
                                "[OUTER warning] {context}: ARC regularization reached its ceiling \
                                 with no accepted step at final_value={:.6e} |g|={:.3e} | {}; the \
                                 iterate is kept as a checkpoint for the terminal certificate \
                                 (#2953)",
                                last_solution.final_value,
                                last_solution.final_gradient_norm.unwrap_or(f64::NAN),
                                arc_census
                                    .describe()
                                    .unwrap_or_else(|| "no step observed".to_string()),
                            );
                            let best_exit =
                                cost_stall_exit.lock().ok().and_then(|slot| slot.clone());
                            Ok(stopped_run_checkpoint(
                                context,
                                "ARC reject-floor",
                                *last_solution,
                                best_exit,
                                *the_plan,
                            ))
                        }
                        Err(ArcError::ObjectiveFailed { message })
                            if message == ARC_INFEASIBLE_STALL_SENTINEL =>
                        {
                            // ARC received a consecutive run of non-finite
                            // probes, so there was no current Hessian with which
                            // to certify the stored best. Rebuild a checkpoint
                            // from that best, but never report bridge-level
                            // convergence: only ARC's synchronized projected-
                            // gradient + reduced-Hessian gate can own a finite
                            // second-order convergence verdict (#979).
                            let exit = cost_stall_exit.lock().ok().and_then(|mut slot| slot.take());
                            match exit {
                                Some(exit) => {
                                    let mut result = outer_result_with_gradient_norm(
                                        exit.rho,
                                        exit.value,
                                        // The exit counts accepted iterates only; the
                                        // run also spent its rejected steps (#2817).
                                        exit.iterations.max(arc_census.steps_taken()),
                                        Some(exit.grad_norm),
                                        false,
                                        *the_plan,
                                    );
                                    result.origin =
                                        OuterResultOrigin::ArcInfeasibleStallCheckpoint;
                                    // The stall window's evidence travels with the
                                    // checkpoint as reported text only (#2817).
                                    result.cost_stall_probe_scale = exit.probe_scale;
                                    // Preserve HOW ARC stopped so the mandatory
                                    // final analytic certificate can report the
                                    // checkpoint provenance without confusing it
                                    // with an optimizer convergence result.
                                    Ok(result)
                                }
                                None => Err(EstimationError::RemlOptimizationFailed(format!(
                                    "ARC infeasible-stall sentinel fired without a published best \
                                     iterate ({context})"
                                ))),
                            }
                        }
                        Err(ArcError::ObjectiveFailed { message })
                            if message == ARC_UNPROGRESSING_STALL_SENTINEL =>
                        {
                            // A stall window carried no progress since the last
                            // one, so the run stopped at its incumbent without a
                            // convergence claim; the terminal certificate judges
                            // that point from a fresh evaluation (#2817).
                            let exit = cost_stall_exit.lock().ok().and_then(|mut slot| slot.take());
                            match exit {
                                Some(exit) => {
                                    let mut result = outer_result_with_gradient_norm(
                                        exit.rho,
                                        exit.value,
                                        exit.iterations.max(arc_census.steps_taken()),
                                        Some(exit.grad_norm),
                                        false,
                                        *the_plan,
                                    );
                                    result.origin =
                                        OuterResultOrigin::ArcUnprogressingStallCheckpoint;
                                    Ok(result)
                                }
                                None => Err(EstimationError::RemlOptimizationFailed(format!(
                                    "ARC unprogressing-stall sentinel fired without a published \
                                     best iterate ({context})"
                                ))),
                            }
                        }
                        Err(ArcError::ObjectiveFailed { message })
                            if message == ARC_CURVATURE_STATIONARY_SENTINEL =>
                        {
                            // #2817 — the bridge stopped ARC at a point its own
                            // terminal certificate accepts: PSD reduced Hessian,
                            // and a Newton decrement at or below the criterion's
                            // resolution, or (#1082) a strict-saddle incumbent
                            // inside the solver band whose negative curvature the
                            // criterion contradicts. Unlike the infeasible-stall
                            // sentinel above, the bridge held a synchronized analytic
                            // Hessian AT this point and evaluated the
                            // certificate's rung on it, so the rebuilt result is
                            // reported converged. The mandatory final analytic
                            // certificate still re-derives its verdict from a
                            // fresh evaluation, so nothing here exempts the
                            // point from being judged.
                            let exit = cost_stall_exit.lock().ok().and_then(|mut slot| slot.take());
                            match exit {
                                Some(exit) => {
                                    let mut result = outer_result_with_gradient_norm(
                                        exit.rho,
                                        exit.value,
                                        exit.iterations.max(arc_census.steps_taken()),
                                        Some(exit.grad_norm),
                                        exit.converged,
                                        *the_plan,
                                    );
                                    result.origin =
                                        OuterResultOrigin::ArcCurvatureStationaryStop;
                                    Ok(result)
                                }
                                None => Err(EstimationError::RemlOptimizationFailed(format!(
                                    "ARC curvature-stationary sentinel fired without a published \
                                     iterate ({context})"
                                ))),
                            }
                        }
                        Err(ArcError::ObjectiveFailed { message }) => {
                            Err(objective_failure_from_publication(
                                "outer ARC evaluation",
                                message,
                                &last_objective_error,
                            ))
                        }
                        Err(e) => Err(EstimationError::RemlOptimizationFailed(format!(
                            "Arc solver failed: {e:?}"
                        ))),
                    }
                }
            }
            Solver::Bfgs => {
                // Production invariant: the outer BFGS runner requires an
                // analytic gradient capability. Fail loudly at the top of the
                // seed loop so the caller surfaces the underlying
                // capability/plan mismatch instead of degrading correctness
                // behind the scenes.
                if cap.gradient != Derivative::Analytic {
                    return Err(EstimationError::RemlOptimizationFailed(format!(
                        "{context}: outer BFGS requires an analytic gradient capability; \
                         no non-analytic fallback is available (plan={the_plan}, \
                         declared gradient={:?})",
                        cap.gradient,
                    )));
                }
                // Device-resident outer-BFGS dispatch branch.
                //
                // Consult the REML objective's `outer_device_admission()`
                // hook — the only call site that consumes
                // `RemlOuterAdmission` — and route to
                // `solver::gpu::reml_outer::run_reml_outer_on_device` when
                // the (family, n, p, num_rho, gpu_available) admission
                // accepts. The driver keeps the BFGS state (ρ, gradient,
                // inverse-Hessian approx, line search) tied to the inner
                // device session pool and only downloads the per-step
                // scalar objective for the Armijo check. The per-step
                // (objective, gradient) pair is computed end-to-end on
                // device through the already-resident PIRLS loop +
                // Hutchinson trace + arrow-Schur Cholesky kernels — the
                // host hop count per outer iteration is exactly one
                // scalar download.
                //
                // The dispatch is magic-by-default: nothing the caller
                // sees changes, the host BFGS branch below remains the
                // unconditional fallback when admission declines (small
                // fit, custom inverse-link family, num_rho < 2, no GPU
                // runtime, or the objective is not a REML evaluator).
                if let Some(admission) = obj.outer_device_admission() {
                    let (lo_dev, hi_dev) = &bounds_template;
                    let bounds_dev = (lo_dev.clone(), hi_dev.clone());
                    let grad_tol_dev = outer_gradient_tolerance(config);
                    // Validate the iteration count via the same `MaxIterations`
                    // wrapper the host BFGS / ARC / matrix-free TR branches use;
                    // the device input below carries it as a raw `usize`, so we
                    // only need the wrapper for its bail-on-invalid behaviour.
                    outer_max_iterations(config.max_iter)?;
                    let axis_caps_dev = bfgs_axis_step_caps(config, layout);
                    let seed_eval_dev = match eval_seed_restoring_rays(
                            obj,
                            config,
                            seed,
                            OuterEvalOrder::ValueAndGradient,
                            RayRestorationDomain { upper: &bounds_template.1, rho_dim: layout.rho_dim() },
                            context,
                            seed_idx,
                        )
                        .map_err(|err| into_objective_error("outer eval failed", err))
                    {
                        Ok(e) => e,
                        Err(err) if err.is_recoverable() => {
                            log::debug!(
                                "[OUTER] {context}: rejecting seed {seed_idx} before device-BFGS start: {err}"
                            );
                            seed_rejections.push(SeedRejection::from_objective_error(
                                seed_idx,
                                "validation",
                                &err,
                            ));
                            break 'seed_attempt;
                        }
                        Err(err) => {
                            return Err(EstimationError::fatal_objective_evaluation(
                                "outer device-BFGS seed evaluation",
                                err,
                            ));
                        }
                    };
                    started_seeds += 1;
                    let device_input = crate::gpu::reml_outer::RemlOuterGpuInput {
                        seed_rho: seed.clone(),
                        bounds: bounds_dev,
                        gradient_tolerance: grad_tol_dev,
                        max_iterations: config.max_iter,
                        // The host BFGS arm's cost-stall floor and band, derived the
                        // same way, so the device walk ends on the same progress
                        // test instead of its iteration count (#2817).
                        cost_stall_rel_tol: config
                            .rel_cost_tolerance
                            .unwrap_or(config.tolerance * 1.0e-2)
                            .max(COST_STALL_REL_TOL_FLOOR),
                        // opt's cost stall takes one number before the walk
                        // starts, so it gets the solver band this walk was driven
                        // to; the 1e-3 floor it used to add had no derivation, and
                        // the terminal certificate judges the point it stops at
                        // regardless (#2817).
                        cost_stall_projected_grad_tol: grad_tol_dev.abs,
                        axis_step_caps: axis_caps_dev,
                        admission,
                        seed_objective: seed_eval_dev.cost,
                        seed_gradient: seed_eval_dev.gradient.clone(),
                    };
                    // The per-step evaluator routes the on-device
                    // (cost, gradient) assembly through the same
                    // `OuterObjective::eval_with_order` hook the host
                    // branch uses: the REML evaluator's inner kernels
                    // are device-resident already, so the gradient
                    // computed here lands on the host as a length-
                    // `num_rho` vector with all heavy work having
                    // happened on the device.
                    let device_outcome = {
                        let obj_cell = std::cell::RefCell::new(&mut *obj);
                        let evaluator = |rho_trial: &Array1<f64>| {
                            let mut obj_ref = obj_cell.borrow_mut();
                            let eval = obj_ref
                                .eval_with_order(rho_trial, OuterEvalOrder::ValueAndGradient)?;
                            Ok(crate::gpu::reml_outer::RemlOuterDeviceEval {
                                objective: eval.cost,
                                gradient: eval.gradient,
                            })
                        };
                        crate::gpu::reml_outer::run_reml_outer_on_device(device_input, evaluator)
                    };
                    match device_outcome {
                        Ok(outcome) => {
                            log::debug!(
                                "[OUTER summary] device-BFGS finished in {} iters \
                                 final_value={:.6e} |g|∞={:.3e} converged={}",
                                outcome.iterations,
                                outcome.objective,
                                outcome.final_grad_norm.unwrap_or(f64::NAN),
                                outcome.converged,
                            );
                            let result = outer_result_with_gradient(
                                outcome.rho,
                                outcome.objective,
                                outcome.iterations,
                                outcome.final_grad_norm,
                                outcome.final_gradient,
                                outcome.converged,
                                *the_plan,
                            );
                            Ok::<OuterResult, EstimationError>(result)
                        }
                        Err(err) => {
                            if err.is_fatal_outer_evaluation() {
                                return Err(err);
                            }
                            log::debug!(
                                "[OUTER] {context}: device-BFGS failed at seed {seed_idx}: {err}; falling back to host BFGS"
                            );
                            // Fall through to the host BFGS path below by
                            // re-running the seed evaluation; the
                            // existing branch will re-validate it and
                            // proceed.
                            let seed_eval = eval_seed_restoring_rays(
                            obj,
                            config,
                            seed,
                            OuterEvalOrder::ValueAndGradient,
                            RayRestorationDomain { upper: &bounds_template.1, rho_dim: layout.rho_dim() },
                            context,
                            seed_idx,
                        )
                                .map_err(|err| into_objective_error("outer eval failed", err));
                            let seed_eval = match seed_eval {
                                Ok(eval) => eval,
                                Err(eval_error) if eval_error.is_recoverable() => {
                                    seed_rejections.push(SeedRejection::from_objective_error(
                                        seed_idx,
                                        "validation",
                                        &eval_error,
                                    ));
                                    break 'seed_attempt;
                                }
                                Err(eval_error) => {
                                    return Err(EstimationError::fatal_objective_evaluation(
                                        "outer host-BFGS fallback seed evaluation",
                                        eval_error,
                                    ));
                                }
                            };
                            match finite_outer_first_order_eval_or_error(
                                "outer eval failed",
                                layout,
                                seed_eval,
                            ) {
                                Ok(_) => Err(err),
                                Err(eval_error) if eval_error.is_recoverable() => {
                                    seed_rejections.push(SeedRejection::from_objective_error(
                                        seed_idx,
                                        "validation",
                                        &eval_error,
                                    ));
                                    break 'seed_attempt;
                                }
                                Err(eval_error) => {
                                    return Err(EstimationError::fatal_objective_evaluation(
                                        "outer host-BFGS fallback seed validation",
                                        eval_error,
                                    ));
                                }
                            }
                        }
                    }
                } else {
                    let seed_eval = eval_seed_restoring_rays(
                            obj,
                            config,
                            seed,
                            OuterEvalOrder::ValueAndGradient,
                            RayRestorationDomain { upper: &bounds_template.1, rho_dim: layout.rho_dim() },
                            context,
                            seed_idx,
                        )
                        .map_err(|err| into_objective_error("outer eval failed", err));
                    let seed_eval = match seed_eval {
                        Ok(seed_eval) => seed_eval,
                        Err(err) if err.is_recoverable() => {
                            log::debug!(
                                "[OUTER] {context}: rejecting seed {seed_idx} before solver start: {err}"
                            );
                            seed_rejections.push(SeedRejection::from_objective_error(
                                seed_idx,
                                "validation",
                                &err,
                            ));
                            break 'seed_attempt;
                        }
                        Err(err) => {
                            return Err(EstimationError::fatal_objective_evaluation(
                                "outer BFGS seed evaluation",
                                err,
                            ));
                        }
                    };
                    let seed_eval = match finite_outer_first_order_eval_or_error(
                        "outer eval failed",
                        layout,
                        seed_eval,
                    ) {
                        Ok(eval) => eval,
                        Err(err) if err.is_recoverable() => {
                            log::debug!(
                                "[OUTER] {context}: rejecting seed {seed_idx} before solver start: {err}"
                            );
                            seed_rejections.push(SeedRejection::from_objective_error(
                                seed_idx,
                                "validation",
                                &err,
                            ));
                            break 'seed_attempt;
                        }
                        Err(err) => {
                            return Err(EstimationError::fatal_objective_evaluation(
                                "outer BFGS seed validation",
                                err,
                            ));
                        }
                    };
                    started_seeds += 1;
                    // The seed a BFGS run is handed, and the (cost, gradient)
                    // it is handed WITH it. `with_initial_sample` below means
                    // `opt::Bfgs` never re-evaluates here, so a zero gradient
                    // in this sample is a zero-iteration "convergence" at
                    // whatever rho this seed happens to be.
                    log::debug!(
                        "[OUTER] {context}: BFGS start cost={:.6e} \
                         |g|={:.6e} rho={:?}",
                        seed_eval.cost,
                        seed_eval.gradient.iter().map(|g| g * g).sum::<f64>().sqrt(),
                        seed.iter().map(|r| (r * 1e6).round() / 1e6).collect::<Vec<_>>(),
                    );
                    // #2765: the criterion prices `½·log|ZᵀMZ|₊` over a kept rank that moves
                    // where the inner mode changes face, and two ranks price two criteria. So
                    // one BFGS run searches one stratum: the bridge refuses a trial whose kept
                    // rank differs from the run's start and keeps the lowest trial it refused.
                    // A run that converges or stalls above that trial by more than the
                    // criterion's roundoff restarts there, on that trial's rank, and logs the
                    // crossing. Every crossing lowers the criterion by more than its roundoff,
                    // so crossings cannot cycle.
                    let bfgs_start = std::time::Instant::now();
                    let mut stratum_start = seed.clone();
                    let mut stratum_eval = seed_eval;
                    let mut crossed_iterations = 0usize;
                    let (outcome, cost_stall_exit, last_objective_error) = loop {
                        let stratum_rank = obj.criterion_rank();
                        let stratum_probe: Arc<Mutex<Option<StratumProbe>>> = Arc::default();
                        let (lo, hi) = &bounds_template;
                        let bounds = outer_bounds(lo, hi)?;
                        let grad_tol = outer_gradient_tolerance(config);
                        let max_iter = outer_max_iterations(config.max_iter)?;
                        // Cost-stall convergence shared cell (#1089). The bridge is
                        // moved into `opt::Bfgs`, so the best iterate it captures on
                        // a flat-valley stall is handed back through this `Arc`.
                        // Relative score-change floor is derived from the outer
                        // tolerance but has a numerical floor so very tight user
                        // tolerances do not disable the mgcv-style flat-valley stop.
                        let cost_stall_exit: Arc<Mutex<Option<CostStallExit>>> =
                            Arc::new(Mutex::new(None));
                        // Accepted-outer-step channel from the observer back into
                        // the bridge's cost-stall guard (#2613). Same shape as the
                        // exit cell above and for the same reason: the observer and
                        // the objective are two values both moved into `opt::Bfgs`.
                        let accepted_steps: Arc<AcceptedStepLedger> = Arc::default();
                        let cost_stall_rel_tol = config
                            .rel_cost_tolerance
                            .unwrap_or(config.tolerance * 1.0e-2)
                            .max(COST_STALL_REL_TOL_FLOOR);
                        // Convergence must mean stationarity, not cost-flatness: a
                        // cost stall claims a converged optimum only when the
                        // projected gradient at its best iterate is inside the
                        // band the terminal certificate applies at that value
                        // (`CostStallGuard::stationarity_band`, #2817).
                        let seed_grad_norm =
                            stratum_eval.gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
                        let mut cost_stall_guard = CostStallGuard::new(
                            cost_stall_rel_tol,
                            COST_STALL_WINDOW,
                            config,
                            cost_stall_exit.clone(),
                        );
                        cost_stall_guard.observe_seed(&stratum_start, stratum_eval.cost, seed_grad_norm);
                        let last_objective_error: Arc<Mutex<Option<ObjectiveEvalError>>> =
                            Arc::new(Mutex::new(None));
                        let objective = RetainingObjective::new(
                            OuterFirstOrderBridge {
                                obj,
                                layout,
                                outer_inner_cap: config.outer_inner_cap.clone(),
                                first_order_evals: 0,
                                g_norm_initial: None,
                                last_g_norm: None,
                                last_value_grad_rho: None,
                                value_probe_cache: Vec::new(),
                                cost_stall: Some(cost_stall_guard),
                                cost_stall_bounds: Some((lo.clone(), hi.clone())),
                                consecutive_probe_refusals: 0,
                                accepted_steps: Some(Arc::clone(&accepted_steps)),
                                pending_first_order: Vec::new(),
                                incumbent: Some((stratum_start.clone(), stratum_eval.cost)),
                                stratum_rank,
                                stratum_probe: Some(Arc::clone(&stratum_probe)),
                            },
                            Arc::clone(&last_objective_error),
                        );
                        // Hand the precomputed (cost, gradient) seed eval to
                        // `opt::Bfgs` so its first internal `eval_grad` call is
                        // served from cache instead of re-running the outer
                        // objective. Inner P-IRLS solves dominate outer cost
                        // at large scale; skipping one re-eval at the seed
                        // is one of the cheapest wins available. (opt 0.3.0
                        // API; before that this was implemented via a
                        // gam-side cache on the bridge.)
                        let initial_sample = FirstOrderSample {
                            value: stratum_eval.cost,
                            gradient: stratum_eval.gradient.clone(),
                        };
                        let mut optimizer = Bfgs::new(stratum_start.clone(), objective)
                            .with_initial_sample(stratum_start.clone(), initial_sample)
                            .with_bounds(bounds)
                            .with_gradient_tolerance(grad_tol)
                            .with_max_iterations(max_iter)
                            // GAM owns the authoritative six-iterate cost-stall
                            // guard in `OuterFirstOrderBridge` and independently
                            // certifies the terminal KKT residual. `opt`'s generic
                            // three-iterate relative-stall gate multiplies the
                            // gradient tolerance by `(1 + ||rho||_inf)`; a railed
                            // smoothing parameter can therefore make that duplicate
                            // gate claim convergence while a different interior
                            // coordinate still has measurable descent (#2524).
                            .without_relative_stall();
                        // First-step scaling. `opt::Bfgs` begins with an
                        // UNSCALED identity inverse-Hessian (`B_inv = I`) on iter 0:
                        // the search direction is the raw `d = -g`, so the unit
                        // line-search step (`α = 1`) is `-g` in ρ-space. The
                        // optimizer's Barzilai-Borwein self-scaling (`γ = sᵀy/yᵀy`)
                        // only fires AFTER the first line search completes. When a
                        // seed's residual gradient has a large component along a
                        // weakly-curved (heavily penalized) log-lambda direction, the
                        // raw `-g` step overshoots and the StrongWolfe search has to
                        // bracket/zoom; in the SAE manifold objective each bracketing
                        // probe is a full inner joint-Newton re-solve. K=1 circle
                        // fits hit this especially hard because the saturated single
                        // assignment gate leaves the outer objective nearly flat in
                        // one direction but still returns a large scale gradient at
                        // the seed.
                        //
                        // Seed the iter-0 metric with the one-point magnitude estimate
                        // the `InitialMetric::Scalar` API is designed for ("a previous
                        // run's gradient norm"): `H₀⁻¹ = (1/‖g₀‖)·I` makes the first
                        // direction `d = -g₀/‖g₀‖` a unit-ℓ²-norm ρ step — bounded,
                        // still exactly steepest-descent (so still a descent
                        // direction), and almost always Wolfe-acceptable at `α = 1`.
                        // This changes only the LINE-SEARCH PATH, never the accepted
                        // optimum: BFGS converges to the same stationary point
                        // `∇_ρ V(ρ*) = 0` under any symmetric-positive-definite initial
                        // metric, and the gradient/KKT convergence tests are unchanged.
                        // This scalar normalization is safe for every finite seed:
                        // it changes only the line-search path, not the stationary
                        // point. Dense transferred curvature stays gated on true warm
                        // starts, because it is local to the parent fit. Every
                        // warm-start mechanism pins `initial_rho`, so seed identity
                        // is the complete authority for transferred curvature. The
                        // scalar scale is clamped
                        // to the same `[1e-3, 1e3]` band the optimizer applies to its
                        // own BB estimate so a pathological seed gradient cannot
                        // produce a degenerate metric.
                        let is_warm_seed = config
                            .initial_rho
                            .as_ref()
                            .is_some_and(|initial| outer_theta_bitwise_eq(initial, &stratum_start));
                        let mut installed_initial_metric = false;
                        if is_warm_seed {
                            // Prefer the converged outer curvature transferred from
                            // the prior structurally-matching fit (`H(θ̂)_parent`):
                            // its inverse is the ideal BFGS iter-0 metric, making the
                            // first outer direction a quasi-Newton step `d = -H⁻¹g₀`
                            // rather than the unscaled `-g₀`. Across LOSO folds the
                            // curvature differs by one held-out row, so the parent's
                            // anisotropic Hessian is a far better local model than the
                            // single-magnitude scalar — it eliminates most of the
                            // StrongWolfe bracketing whose every probe is a full inner
                            // joint-Newton re-solve. Only an exact certified SPD
                            // transferred Hessian can seed this metric; an indefinite
                            // or singular parent curvature is rejected without
                            // perturbing it and the scalar metric is selected. Either
                            // way the converged
                            // optimum is unchanged: BFGS reaches ∇V=0 under any SPD
                            // initial metric, and the gradient/KKT tests are identical.
                            let dense_metric = eligible_transferred_outer_hessian(
                                config.warm_start_outer_hessian.as_ref(),
                                cap.hessian,
                                layout.n_params,
                            )
                                .or_else(|| {
                                    bound_initial_curvature(
                                        config.initial_curvature.as_ref(),
                                        &stratum_start,
                                        layout.n_params,
                                    )
                                })
                                .and_then(|h| {
                                    match gam_linalg::utils::certified_spd_inverse(
                                        h,
                                        "transferred outer-Hessian BFGS metric",
                                    ) {
                                        Ok(inverse) => Some(inverse.into_inverse()),
                                        Err(error) => {
                                            log::debug!(
                                                "[OUTER] {context}: rejected transferred BFGS metric: {error}"
                                            );
                                            None
                                        }
                                    }
                                });
                            if let Some(h_inv) = dense_metric {
                                log::debug!(
                                    "[OUTER] {context}: warm-start BFGS metric = transferred \
                                     H(θ̂)⁻¹ (dim={}); quasi-Newton first step",
                                    layout.n_params,
                                );
                                optimizer = optimizer
                                    .with_initial_metric(InitialMetric::DenseInverseHessian(h_inv));
                                installed_initial_metric = true;
                            }
                        }
                        if !installed_initial_metric {
                            let g0_norm = stratum_eval.gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
                            // `H_0^{-1} = I/‖g₀‖` makes the first trial step unit length in
                            // ρ; a norm with no finite positive reciprocal keeps opt's default.
                            let scale = 1.0 / g0_norm;
                            if scale.is_finite() && scale > 0.0 {
                                optimizer = optimizer.with_initial_metric(InitialMetric::Scalar(scale));
                            }
                        }
                        if let Some(caps) = bfgs_axis_step_caps(config, layout) {
                            optimizer = optimizer.with_axis_step_caps(caps);
                        }
                        // The observer is installed UNCONDITIONALLY on this route
                        // (#2613). It used to be gated on `outer_inner_cap`, the
                        // only consumer at the time; the cost-stall guard now
                        // depends on the same accepted-step signal to tell an
                        // accepted outer iterate from a line-search trial, and that
                        // guard is present on every BFGS seed.
                        optimizer = optimizer.with_observer(OuterAcceptObserver {
                            feedback: config.outer_inner_cap.clone(),
                            accepted_steps: Some(Arc::clone(&accepted_steps)),
                            // BFGS reports no trust radius, so a region census would
                            // be a column of `None`s; its own non-convergence
                            // reporting is the line-search failure path.
                            census: None,
                        });
                        let outcome = optimizer.run();
                        drop(optimizer);
                        let probe = stratum_probe.lock().ok().and_then(|mut slot| slot.take());
                        let run_end = stratum_run_end(&outcome, &cost_stall_exit, stratum_eval.cost);
                        let (Some(from_rank), Some(probe), Some((final_value, run_iterations))) =
                            (stratum_rank, probe, run_end)
                        else {
                            break (outcome, cost_stall_exit, last_objective_error);
                        };
                        // The criterion value's own roundoff: a value lower by no more than
                        // this is not a lower criterion.
                        let resolution = f64::EPSILON * (1.0 + final_value.abs());
                        if !(probe.cost < final_value - resolution) {
                            break (outcome, cost_stall_exit, last_objective_error);
                        }
                        let crossing_eval = eval_seed_at_full_inner_fidelity(
                            obj,
                            config,
                            &probe.rho,
                            OuterEvalOrder::ValueAndGradient,
                        )
                        .map_err(|err| into_objective_error("outer eval failed", err))
                        .and_then(|eval| {
                            finite_outer_first_order_eval_or_error("outer eval failed", layout, eval)
                        });
                        match crossing_eval {
                            Ok(eval) if eval.cost < final_value - resolution => {
                                log::debug!(
                                    "[OUTER] {context}: seed {seed_idx} crosses from kept rank \
                                     {from_rank} to {} at criterion {:.6e} -> {:.6e} (delta {:.3e}) \
                                     and restarts BFGS there (#2765)",
                                    obj.criterion_rank()
                                        .map_or_else(|| "none".to_string(), |rank| rank.to_string()),
                                    final_value,
                                    eval.cost,
                                    eval.cost - final_value,
                                );
                                crossed_iterations = crossed_iterations.saturating_add(run_iterations);
                                stratum_start = probe.rho;
                                stratum_eval = eval;
                            }
                            Ok(eval) => {
                                log::debug!(
                                    "[OUTER] {context}: seed {seed_idx} stays on kept rank {from_rank}: \
                                     the refused rank-{} trial re-evaluates at {:.6e}, not below the \
                                     run's {:.6e} by more than {:.3e} (#2765)",
                                    probe.rank,
                                    eval.cost,
                                    final_value,
                                    resolution,
                                );
                                break (outcome, cost_stall_exit, last_objective_error);
                            }
                            Err(err) => {
                                log::debug!(
                                    "[OUTER] {context}: seed {seed_idx} stays on kept rank {from_rank}: \
                                     the refused rank-{} trial did not re-evaluate: {err} (#2765)",
                                    probe.rank,
                                );
                                break (outcome, cost_stall_exit, last_objective_error);
                            }
                        }
                    };
                    let bfgs_elapsed = bfgs_start.elapsed().as_secs_f64();
                    match &outcome {
                        Ok(sol) => log::debug!(
                            "[OUTER summary] BFGS converged in {} iters elapsed={:.3}s final_value={:.6e}",
                            sol.iterations,
                            bfgs_elapsed,
                            sol.final_value
                        ),
                        Err(BfgsError::MaxIterationsReached { last_solution }) => log::debug!(
                            // Include `in N iters` for symmetry with the
                            // converged log line — the runner aggregator
                            // (commit afd66d6a) reads the optional iters
                            // group to build `bfgs_iters_p50/_max` across
                            // both successful and cap-hit runs. Without
                            // this, the iter-count distribution would be
                            // biased toward fast-converged runs.
                            "[OUTER summary] BFGS hit max_iter in {} iters elapsed={:.3}s final_value={:.6e}",
                            last_solution.iterations,
                            bfgs_elapsed,
                            last_solution.final_value
                        ),
                        Err(BfgsError::LineSearchFailed {
                            last_solution,
                            max_attempts,
                            failure_reason,
                        }) => log::debug!(
                            // Same rationale as the MaxIterationsReached
                            // arm: surface `in N iters` so the runner can
                            // include line-search-failed runs in the
                            // iter-count distribution. A line-search
                            // failure at iter 1 (cold start collapses
                            // immediately) is a different signal from
                            // failure at iter 50 (the optimizer made
                            // substantial progress before stalling).
                            "[OUTER summary] BFGS line-search failed in {} iters elapsed={:.3}s final_value={:.6e} reason={:?} max_attempts={} |g|={:.3e}",
                            last_solution.iterations,
                            bfgs_elapsed,
                            last_solution.final_value,
                            failure_reason,
                            max_attempts,
                            last_solution.final_gradient_norm.unwrap_or(f64::NAN),
                        ),
                        Err(e) => log::debug!(
                            "[OUTER summary] BFGS failed elapsed={:.3}s err={:?}",
                            bfgs_elapsed,
                            e
                        ),
                    }
                    let stratum_result = match outcome {
                        Ok(sol) => Ok(solution_into_outer_result(sol, true, *the_plan)),
                        Err(BfgsError::MaxIterationsReached { last_solution }) => {
                            Ok(solution_into_outer_result(*last_solution, false, *the_plan))
                        }
                        Err(BfgsError::LineSearchFailed {
                            last_solution,
                            max_attempts,
                            failure_reason,
                        }) => {
                            if last_solution.final_value.is_finite()
                                && last_solution.final_point.iter().all(|v| v.is_finite())
                                && last_solution
                                    .final_gradient
                                    .as_ref()
                                    .is_none_or(|g| g.iter().all(|v| v.is_finite()))
                            {
                                // Carry the line search's own verdict (#2465).
                                //
                                // This arm turns a line-search failure with a
                                // finite last iterate into `Ok(non-converged)`,
                                // which is right -- the iterate is usable as a
                                // checkpoint -- but it means the caller NEVER
                                // sees the `Err` that holds `failure_reason` and
                                // `max_attempts`. Downstream the certificate
                                // could say only `termination=line_search_failed`,
                                // and `StepSizeTooSmall` (the direction descended
                                // but nothing improved the objective) and
                                // `MaxAttempts` (the bracket never closed) are
                                // different defects with different repairs.
                                let mut outer_result =
                                    solution_into_outer_result(*last_solution, false, *the_plan);
                                outer_result.line_search_failure =
                                    Some((failure_reason, max_attempts));
                                Ok(outer_result)
                            } else {
                                Err(EstimationError::RemlOptimizationFailed(
                                    bfgs_line_search_failure_message(
                                        context,
                                        &last_solution,
                                        max_attempts,
                                        failure_reason,
                                    ),
                                ))
                            }
                        }
                        Err(BfgsError::ObjectiveFailed { message })
                            if message == COST_STALL_CONVERGED_SENTINEL =>
                        {
                            // The bridge's cost-stall guard halted BFGS because
                            // the REML score stopped decreasing (#1089). Rebuild
                            // the outer result from the best iterate it
                            // published. Whether the run is CONVERGED is decided
                            // by the guard's stationarity test and rides on
                            // `exit.converged`: `true` only when the projected
                            // gradient at the best iterate cleared the outer
                            // gradient tolerance (a stationary optimum on a flat
                            // surface); `false` for a flat-valley floor with
                            // residual non-stationarity. A non-converged
                            // cost-stall flows into the same non-convergence
                            // reporting as MaxIterations / line-search-failed
                            // (best-so-far returned, `converged = false`), not a
                            // panic and not a silently-relabeled optimum.
                            let exit = cost_stall_exit.lock().ok().and_then(|mut slot| slot.take());
                            match exit {
                                Some(exit) => {
                                    let mut result = outer_result_with_gradient_norm(
                                        exit.rho,
                                        exit.value,
                                        exit.iterations,
                                        Some(exit.grad_norm),
                                        exit.converged,
                                        *the_plan,
                                    );
                                    result.origin = OuterResultOrigin::BfgsCostStallExit;
                                    // The stall window's evidence travels with the
                                    // result as reported text only (#2817).
                                    result.cost_stall_probe_scale = exit.probe_scale;
                                    // So does a halt where the search's kept rank
                                    // ends (#2939).
                                    result.rank_boundary_stall = exit.rank_boundary;
                                    // The mandatory final analytic certificate
                                    // judges this point by its own ladder, and
                                    // the guard claimed it only inside that
                                    // ladder's band at the incumbent (#2817).
                                    // The two used to disagree: the guard
                                    // claimed through a score-relative band the
                                    // certificate never applied (#1689 in ARC;
                                    // the GPT-2 E1 structured pass on the BFGS
                                    // route: guard accepted |g|=4.97e-1 on a
                                    // score of 2.7e3, certificate refused at
                                    // its 4.4e-2 bound and the fit died with
                                    // RemlConvergenceError).
                                    Ok(result)
                                }
                                None => Err(EstimationError::RemlOptimizationFailed(format!(
                                    "BFGS cost-stall sentinel fired without a published best \
                                     iterate ({context})"
                                ))),
                            }
                        }
                        Err(BfgsError::ObjectiveFailed { message })
                            if message.starts_with(PROBE_REFUSAL_FATAL_SENTINEL) =>
                        {
                            // The bridge's probe-refusal non-termination guard
                            // (#NaN-outer-loop): every line-search cost probe at
                            // this seed was infeasible, so BFGS would have spent
                            // its entire max_iterations budget on inner solves
                            // that all fail. Route as a seed rejection so the
                            // cascade tries the next seed instead of propagating
                            // a fatal error.
                            Err(EstimationError::RemlOptimizationFailed(format!(
                                "BFGS aborted: globally infeasible neighbourhood \
                                 at seed (probe-refusal guard): {message}"
                            )))
                        }
                        Err(BfgsError::ObjectiveFailed { message }) => {
                            Err(objective_failure_from_publication(
                                "outer BFGS evaluation",
                                message,
                                &last_objective_error,
                            ))
                        }
                        Err(e) => Err(EstimationError::RemlOptimizationFailed(format!(
                            "BFGS solver failed: {e:?}"
                        ))),
                    };
                    // A crossing ends one run and starts another; the seed spent both (#2817).
                    stratum_result.map(|mut result| {
                        result.iterations = result.iterations.saturating_add(crossed_iterations);
                        result
                    })
                }
            }
            Solver::Efs => {
                match run_fixed_point_outer_solver(
                    obj,
                    layout,
                    cap.barrier_config.clone(),
                    config,
                    context,
                    seed,
                    *the_plan,
                    "EFS",
                    "fixed-point solver failed",
                ) {
                    Ok(result) => {
                        started_seeds += 1;
                        Ok(result)
                    }
                    Err(FixedPointOuterRunError::SeedRejected(err)) => {
                        log::debug!(
                            "[OUTER] {context}: rejecting seed {seed_idx} before solver start: {err}"
                        );
                        seed_rejections.push(SeedRejection::from_objective_error(
                            seed_idx,
                            "validation",
                            &err,
                        ));
                        break 'seed_attempt;
                    }
                    Err(FixedPointOuterRunError::IterationRejected(mut request)) => {
                        log::debug!(
                            "[OUTER] {context}: EFS trial refused after {} finite iteration(s) \
                             at cost={:.6e}; continuing the exact incumbent with the \
                             analytic-gradient fallback: {}",
                            request.checkpoint.iterations,
                            request.checkpoint.sample.value,
                            request.refusal,
                        );
                        // The attempt's earlier seeds spent iterations too (#2817).
                        request.checkpoint.iterations =
                            request.checkpoint.iterations.saturating_add(spent_seed_iterations);
                        return Ok(PlanRunOutcome::FixedPointContinuationRequested(request));
                    }
                    Err(FixedPointOuterRunError::ImmediateFallback(request)) => {
                        // This seed's own evaluation asked, but the attempt's earlier
                        // seeds already spent iterations (#2817).
                        return Ok(PlanRunOutcome::FirstOrderFallbackRequested(
                            request.with_spent_iterations(spent_seed_iterations),
                        ));
                    }
                    Err(FixedPointOuterRunError::Failed(err)) => {
                        started_seeds += 1;
                        Err(err)
                    }
                }
            }
            Solver::HybridEfs => {
                match run_fixed_point_outer_solver(
                    obj,
                    layout,
                    cap.barrier_config.clone(),
                    config,
                    context,
                    seed,
                    *the_plan,
                    "HybridEFS",
                    "hybrid EFS solver failed",
                ) {
                    Ok(result) => {
                        started_seeds += 1;
                        Ok(result)
                    }
                    Err(FixedPointOuterRunError::SeedRejected(err)) => {
                        log::debug!(
                            "[OUTER] {context}: rejecting seed {seed_idx} before solver start: {err}"
                        );
                        seed_rejections.push(SeedRejection::from_objective_error(
                            seed_idx,
                            "validation",
                            &err,
                        ));
                        break 'seed_attempt;
                    }
                    Err(FixedPointOuterRunError::IterationRejected(mut request)) => {
                        log::debug!(
                            "[OUTER] {context}: HybridEFS trial refused after {} finite \
                             iteration(s) at cost={:.6e}; continuing the exact incumbent \
                             with the analytic-gradient fallback: {}",
                            request.checkpoint.iterations,
                            request.checkpoint.sample.value,
                            request.refusal,
                        );
                        // The attempt's earlier seeds spent iterations too (#2817).
                        request.checkpoint.iterations =
                            request.checkpoint.iterations.saturating_add(spent_seed_iterations);
                        return Ok(PlanRunOutcome::FixedPointContinuationRequested(request));
                    }
                    Err(FixedPointOuterRunError::ImmediateFallback(request)) => {
                        // This seed's own evaluation asked, but the attempt's earlier
                        // seeds already spent iterations (#2817).
                        return Ok(PlanRunOutcome::FirstOrderFallbackRequested(
                            request.with_spent_iterations(spent_seed_iterations),
                        ));
                    }
                    Err(FixedPointOuterRunError::Failed(err)) => {
                        started_seeds += 1;
                        Err(err)
                    }
                }
            }
        };

        let seed_elapsed = t_seed_start.elapsed().as_secs_f64();
        match result {
            Ok(candidate) => {
                log::trace!(
                    "[outer-timing] start ({:?}): {:.3}s  cost={:.6e}  converged={}",
                    the_plan.solver,
                    seed_elapsed,
                    candidate.final_value,
                    candidate.solver_claimed_convergence(),
                );
                spent_seed_iterations = spent_seed_iterations.saturating_add(candidate.iterations);
                if !candidate.solver_claimed_convergence() {
                    // An exhausted iterate is resumable work, not a fit
                    // candidate: it is the resume checkpoint and never
                    // populates the certified winner slot.
                    retain_best_outer_checkpoint(&mut best_checkpoint, candidate);
                    break 'seed_attempt;
                }
                match CertifiedOuterCandidate::from_solver_claim(obj, config, context, candidate) {
                    Ok(candidate) => best = Some(candidate),
                    Err((checkpoint, error)) => {
                        log::debug!(
                            "[OUTER] {context}: solver convergence claim failed analytic \
                             certification: {error}; retaining only a resume checkpoint"
                        );
                        tail_snap_reseed_point = checkpoint.tail_snap_reseed.clone();
                        saddle_escape_reseed_point = checkpoint.saddle_escape_reseed.clone();
                        retain_best_outer_checkpoint(&mut best_checkpoint, checkpoint);
                        seed_rejections.push(SeedRejection::from_estimation_error(
                            seed_idx,
                            "certificate",
                            &error,
                        ));
                    }
                }
            }
            Err(e) => {
                if e.is_fatal_outer_evaluation() {
                    return Err(e);
                }
                log::trace!(
                    "[outer-timing] start ({:?}): {:.3}s  FAILED: {}",
                    the_plan.solver,
                    seed_elapsed,
                    e,
                );
                seed_rejections.push(SeedRejection::from_estimation_error(seed_idx, "solver", &e));
            }
        }
    }

    // #2596, #2627 — a certified winner that an evaluated state of this attempt
    // beats does not publish.
    //
    // A certificate answers "is this ρ stationary?". It does not answer "is this
    // the best ρ we found?". The two come apart on a face of the declared domain:
    // the criterion is flat there and the box-KKT projection leaves |Pg|
    // negligible whatever the criterion scores, so a face seed certifies in zero
    // iterations while the interior searches that already reached far lower
    // values end on a refused certificate or an exhausted budget. Publishing the
    // face shipped an intercept-only fit on #2596 (110.94 over a refused 4.19),
    // #561 seed 201 (315.15 over 230.68) and the penguins species fit (272.66
    // over an evaluated 21.44). The warning that named each inversion published
    // anyway.
    //
    // The incumbent is the lowest checkpoint the attempt kept. Its stored value is
    // where a search stopped, so it is re-evaluated at its own ρ before it can
    // outrank anything. The gap is judged at the criterion's own rounding
    // envelope, [`outer_value_agreement_bound`], because two values of one
    // criterion closer than that cannot be ranked. Beyond it the winner loses.
    // The search continues once from the incumbent, with the same one-shot reseed
    // the tail-snap and saddle-escape retries use. If that does not certify, the
    // attempt returns the typed [`PlanRunOutcome::DominatedPlateau`], and the
    // incumbent is the resume checkpoint. When the objective refuses to
    // re-evaluate the incumbent, its stored value, the criterion's own evaluation
    // at that ρ, decides the gap, and no search continues from a point the
    // objective refuses (#2953).
    let mut dominance: Option<(f64, f64)> = None;
    // Why the incumbent could not be re-evaluated at its own ρ, when it could not.
    let mut reevaluation_refusal: Option<EstimationError> = None;
    if let (Some(certified), Some(incumbent)) = (best.as_ref(), best_checkpoint.as_ref()) {
        let winner_value = certified.result().final_value;
        let cached_band =
            crate::rho_optimizer::outer_value_agreement_bound(winner_value, incumbent.final_value);
        if winner_value.is_finite()
            && incumbent.final_value.is_finite()
            && winner_value - incumbent.final_value > cached_band
        {
            let incumbent_rho = incumbent.rho.clone();
            obj.reset();
            install_matching_initial_inner_seed(obj, config, &incumbent_rho, context)?;
            let incumbent_value = match obj.eval_cost(&incumbent_rho) {
                Ok(value) => value,
                // The stored checkpoint cannot be re-evaluated at its own ρ. Its stored
                // value is the criterion's evaluation there and beats the winner beyond
                // the envelope, so the winner is declined on it (#2953).
                Err(error) if error.is_trial_point_infeasible() => {
                    log::debug!(
                        "[OUTER] {context}: certified winner rho={:?} cost={:.6e} sits above a stored \
                         checkpoint rho={:?} cost={:.6e} by more than the criterion's rounding envelope \
                         {:.3e}, and re-evaluating that checkpoint was refused ({error}); the winner is \
                         declined on the stored value, and no search continues from the checkpoint \
                         (#2953)",
                        certified.result().rho.to_vec(),
                        winner_value,
                        incumbent_rho.to_vec(),
                        incumbent.final_value,
                        cached_band,
                    );
                    reevaluation_refusal = Some(error);
                    incumbent.final_value
                }
                Err(error) => return Err(error),
            };
            obj.reset();
            let band =
                crate::rho_optimizer::outer_value_agreement_bound(winner_value, incumbent_value);
            if incumbent_value.is_finite() && winner_value - incumbent_value > band {
                dominance = Some((incumbent_value, band));
            }
        }
    }
    if let Some((incumbent_value, band)) = dominance
        && let Some(certified) = best.take()
        && let Some(mut incumbent) = best_checkpoint.take()
    {
        let plateau = certified.into_result();
        incumbent.final_value = incumbent_value;
        let gap = plateau.final_value - incumbent.final_value;
        let reevaluated = reevaluation_refusal.is_none();
        let mut continuation = match reevaluation_refusal {
            // No search can start from a point the objective refuses.
            Some(refusal) => DominanceContinuationStop::Failed {
                error: format!(
                    "re-evaluating the checkpoint at its own rho was refused: {refusal}"
                ),
            },
            None => {
                log::debug!(
                    "[OUTER] {context}: certified winner rho={:?} cost={:.6e} is dominated by an \
                     evaluated state rho={:?} cost={:.6e} (gap {:.3e} > the criterion's rounding \
                     envelope {:.3e}); it is not published, and the search continues from that \
                     state (#2596, #2627)",
                    plateau.rho.to_vec(),
                    plateau.final_value,
                    incumbent.rho.to_vec(),
                    incumbent.final_value,
                    gap,
                    band,
                );
                DominanceContinuationStop::NotRun
            }
        };
        if allow_tail_snap_reseed && reevaluated {
            let mut retry_config = config.clone();
            // The continuation judges what it certifies against the state it starts from, so it
            // cannot publish the optimum that state just beat (#2953).
            retry_config.carried_checkpoint = Some(carried_checkpoint_of(&incumbent));
            retry_config.initial_rho = Some(incumbent.rho.clone());
            match run_outer_with_plan(obj, &retry_config, context, cap, the_plan, false) {
                Ok(PlanRunOutcome::Exhausted(retry_checkpoint)) => {
                    log::debug!(
                        "[OUTER] {context}: the retry from the dominating incumbent exhausted \
                         at cost {:.6e} without certifying (#2627)",
                        retry_checkpoint.final_value,
                    );
                    continuation = DominanceContinuationStop::Exhausted {
                        final_value: retry_checkpoint.final_value,
                    };
                    // The whole retry result, not its point and value alone: every
                    // other field describes the point it stopped at, and a gradient
                    // left from the incumbent's own point is how the #2953 floor
                    // certified a non-stationary ρ.
                    if retry_checkpoint.final_value < incumbent.final_value {
                        incumbent = retry_checkpoint;
                    }
                }
                Ok(PlanRunOutcome::DominatedPlateau(retry)) => {
                    log::debug!(
                        "[OUTER] {context}: the retry from the dominating incumbent ended on \
                         another dominated plateau at cost {:.6e} (#2627)",
                        retry.plateau.final_value,
                    );
                    continuation = DominanceContinuationStop::DominatedAgain {
                        plateau_value: retry.plateau.final_value,
                    };
                    if retry.incumbent.final_value < incumbent.final_value {
                        incumbent = retry.incumbent;
                    }
                }
                // The continuation certified under the screening certificate. It publishes
                // only if the terminal certificate agrees, so the declined optimum rides on
                // it for the refusal that follows otherwise (#2953).
                Ok(PlanRunOutcome::Converged(mut retry_result)) => {
                    retry_result.dominated_plateau = lowest_dominated_plateau(
                        retry_result.dominated_plateau.take(),
                        Some(DominatedPlateauRecord {
                            plateau_rho: plateau.rho,
                            plateau_value: plateau.final_value,
                            gap,
                            band,
                            continuation: DominanceContinuationStop::Certified {
                                final_value: retry_result.final_value,
                            },
                        }),
                    );
                    return Ok(with_enclosing_attempt_ledger(
                        PlanRunOutcome::Converged(retry_result),
                        spent_seed_iterations,
                    ));
                }
                Ok(outcome) => {
                    return Ok(with_enclosing_attempt_ledger(outcome, spent_seed_iterations));
                }
                Err(retry_error) => {
                    log::debug!(
                        "[OUTER] {context}: the retry from the dominating incumbent failed \
                         ({retry_error}); returning the dominated plateau with the incumbent as \
                         the resume checkpoint (#2627)"
                    );
                    continuation = DominanceContinuationStop::Failed {
                        error: retry_error.to_string(),
                    };
                }
            }
        }
        incumbent.iterations = spent_seed_iterations;
        return Ok(PlanRunOutcome::DominatedPlateau(DominatedPlateau {
            plateau,
            incumbent,
            gap,
            band,
            continuation,
        }));
    }

    if let Some(certified) = best {
        let mut result = certified.into_result();
        // Certification attaches a certificate and changes no count, so the
        // winner's total is the ledger, its own run included.
        result.iterations = spent_seed_iterations;
        // NO mint audit here (#2359). Every candidate above was screened at
        // order three, and the winner's order-four audit is paid EXACTLY ONCE —
        // but it is paid by `run_outer`, not here.
        //
        // The three production paths into this function
        // (`run.rs:5034`, and the two retries at `:1962`/`:1993` below) are all
        // reached from `run_outer_uncertified`, whose only callers are inside
        // `run_outer` — and `run_outer` finishes with
        // `certify_diagnose_and_install`, a `CertificationFidelity::Mint`
        // certification that must run LAST so the certificate's own evaluation
        // is the final objective-state installer (that is what makes the sealed
        // terminal fit bind bitwise on a bimodal inner solve). Minting here as
        // well spent the order-four derivative tower TWICE on every
        // analytic-Hessian fit, and on the sampled-pilot path three times —
        // which is exactly what #2359 exists to prevent. Measured on
        // `optimize_three_certify_four_exactly_once_at_mint_2359`:
        // `ValueGradientHessian` calls 2 → 1, and the last order seen is the
        // mint's, by construction.
        //
        // The winner therefore leaves this function carrying its SCREENING
        // certificate, which is first-order evidence only. That is the correct
        // provenance: nothing downstream of here treats a plan result as minted
        // until `run_outer` replaces the certificate with its own.
        // The finalize evaluation re-installs the selected outer result by
        // re-running the inner P-IRLS at θ̂. During the outer search the ARC /
        // BFGS bridge schedule throttles `RemlState::outer_inner_cap` down to a
        // small adaptive cap (e.g. 3 iters) so early, far-from-converged outer
        // steps spend a coarse inner solve. That cap MUST NOT leak into the
        // finalize solve at the optimum: the inner Newton there can need many
        // iterations (SAS link drives η to extreme magnitudes mid-search,
        // #1572), and a capped `MaxIterationsReached` is escalated to a fatal
        // `PirlsDidNotConverge` ("did not converge within 3 iterations"),
        // aborting the whole fit. Lift the cap to 0 (no cap) for the finalize,
        // mirroring the post-run `run_outer_inner_cap_guard`
        // (optimizer.rs:135) and the accept-fit's "full inner budget" intent
        // (gradient_hessian.rs:6469), then restore the prior cap so any later
        // schedule-driven evaluation sees the value it expects.
        // Held in a named binding and dropped explicitly after the finalize
        // (which restores the prior cap), rather than `let _guard`: the
        // workspace ban-scanner (build.rs) forbids every underscore-leading
        // `let` pattern, and a plain `let guard` would trip `unused_variables`
        // under `warnings = "deny"`. The explicit `drop(...)` is the idiomatic
        // "use" (see e.g. `hessian_scope_guard` in custom_family). The guard's
        // Drop runs before `?` propagates a finalize error, so the cap is
        // restored on both the success and the abort path.
        let finalize_cap_guard = config
            .outer_inner_cap
            .as_ref()
            .map(FullFidelityInnerCapGuard::lift);
        if finalize_cap_guard.is_some() {
            // Certification evaluated trial points after the search ended.
            // Clear every search-state cache before installing the selected
            // point so a rho-only hit cannot leave the objective owning the
            // last trial's inner mode.
            obj.reset();
        }
        let finalize_outcome = obj.finalize_outer_result(&result.rho, the_plan);
        drop(finalize_cap_guard);
        finalize_outcome?;
        return Ok(PlanRunOutcome::Converged(result));
    }

    // #2348 Inc 2b: a refused certification CONFIRMED an exponential tail
    // (probing passed) but the interior was still unpolished — the budget died
    // mid-crawl while the interior tracked the crawling tail coordinate.
    // Retry ONCE seeded at the snapped rail point: the box projection pins the
    // tail coordinate at its bound while the interior converges in its few
    // remaining Newton steps, and the Inc 1 railed mint then certifies through
    // the natural path. The retry pass runs with the reseed gate closed, so
    // this can never recurse; a failed retry falls back to the original
    // exhaustion accounting.
    if allow_tail_snap_reseed && let Some(reseed) = tail_snap_reseed_point {
        log::debug!(
            "[OUTER] {context}: retrying once from the confirmed-tail snapped \
             reseed {reseed} (#2348 Inc 2b)"
        );
        let mut retry_config = config.clone();
        retry_config.initial_rho = Some(reseed);
        obj.reset();
        match run_outer_with_plan(obj, &retry_config, context, cap, the_plan, false) {
            Ok(outcome) => {
                return Ok(with_enclosing_attempt_ledger(outcome, spent_seed_iterations));
            }
            Err(retry_error) => {
                log::debug!(
                    "[OUTER] {context}: confirmed-tail reseed retry failed ({retry_error}); \
                     falling through to the original exhaustion accounting"
                );
            }
        }
    }

    // #2357 — saddle escape. A refused certification identified an interior
    // strict saddle (first-order stationary, indefinite curvature, no rail) and
    // published a negative-curvature escape point strictly below it. Retry ONCE
    // seeded there: the outer search resumes off the saddle ridge and descends
    // to the true PSD minimum — the deterministic form of the identical
    // warm-started resume that converges where the cold run refuses. The retry
    // pass runs with the reseed gate closed (`allow_tail_snap_reseed = false`),
    // so it can never recurse; a failed retry falls back to the original
    // exhaustion accounting.
    if allow_tail_snap_reseed && let Some(reseed) = saddle_escape_reseed_point {
        log::debug!(
            "[OUTER] {context}: retrying once from the negative-curvature saddle-escape \
             reseed {reseed} (#2357)"
        );
        let mut retry_config = config.clone();
        retry_config.initial_rho = Some(reseed);
        obj.reset();
        match run_outer_with_plan(obj, &retry_config, context, cap, the_plan, false) {
            Ok(outcome) => {
                return Ok(with_enclosing_attempt_ledger(outcome, spent_seed_iterations));
            }
            Err(retry_error) => {
                log::debug!(
                    "[OUTER] {context}: saddle-escape reseed retry failed ({retry_error}); \
                     falling through to the original exhaustion accounting"
                );
            }
        }
    }

    if let Some(mut checkpoint) = best_checkpoint {
        // Every start's iterations, this checkpoint's own included.
        checkpoint.iterations = spent_seed_iterations;
        return Ok(PlanRunOutcome::Exhausted(checkpoint));
    }

    Err({
        // One start was generated and screened; it either reached a rejection
        // site in this attempt or started the solver.
        let n_exact_validated = seed_rejections.len() + started_seeds;
        let stats = StartupStats::from_rejections(
            1,
            1,
            n_exact_validated,
            started_seeds,
            &seed_rejections,
        );
        let structural = uniform_structural_key(&seed_rejections, 1);
        if started_seeds == 0 {
            EstimationError::StartupSeedsRefused(format_no_seeds_passed(
                context,
                &stats,
                &seed_rejections,
                structural.as_ref(),
                "",
            ))
        } else {
            // The start reached the outer optimiser but did not converge. Keep
            // the structured payload so the caller sees the per-rejection
            // breakdown.
            let header = format!(
                "the outer start failed ({context}); generated={}, screened={}, \
                 exact_validated={}, solver_started={}",
                stats.generated, stats.screened, stats.exact_validated, stats.solver_started,
            );
            let body = format_no_seeds_passed(
                context,
                &stats,
                &seed_rejections,
                structural.as_ref(),
                "",
            );
            EstimationError::RemlOptimizationFailed(format!("{header}\n{body}"))
        }
    })
}

/// An outer solver's `ObjectiveFailed`, read against the objective error its bridge published.
///
/// `RetainingObjective` writes the publication slot on every evaluation and clears it on
/// success, so a solver that stops on a failed evaluation hands back exactly the error the
/// producer classified, and that maps as a fatal objective evaluation. A solver can also report
/// `ObjectiveFailed` without having evaluated anything: a refused seed or initial metric, or a
/// solver-internal exit. The slot can also disagree with the solver's message, hold a recoverable
/// verdict, or be poisoned. Each of those is a typed fatal outer-evaluation failure naming the
/// solver context and both messages, never a panic (#1561).
fn objective_failure_from_publication(
    context: &str,
    solver_message: String,
    publication: &Mutex<Option<ObjectiveEvalError>>,
) -> EstimationError {
    match publication.lock().ok().and_then(|mut slot| slot.take()) {
        Some(published) if published.is_fatal() && published.message() == solver_message => {
            EstimationError::fatal_objective_evaluation(context, published)
        }
        Some(published) => EstimationError::fatal_outer_evaluation(
            context,
            EstimationError::RemlOptimizationFailed(format!(
                "the solver reported `{solver_message}`, but the objective bridge published {} error `{}`",
                if published.is_fatal() {
                    "a fatal"
                } else {
                    "a recoverable"
                },
                published.message()
            )),
        ),
        None => EstimationError::fatal_outer_evaluation(
            context,
            EstimationError::RemlOptimizationFailed(format!(
                "the solver reported `{solver_message}` with no published objective evaluation"
            )),
        ),
    }
}

#[cfg(test)]
#[path = "zz_ray_restoration_reasons_2695_tests.rs"]
mod zz_ray_restoration_reasons_2695_tests;

#[cfg(test)]
#[path = "run_plan_tests.rs"]
mod run_plan_tests;

// The #2568 caller-requirement tests live in their own file: `run_plan_tests.rs`
// reached 10,063 lines and tripped the 10,000-line ban gate (#780), which aborts
// the ROOT crate's build and with it every root-crate target. Same `#[path]`
// form as its sibling, so `use super::*` resolves to this module either way.
#[cfg(test)]
#[path = "run_plan_caller_requirement_tests_2568.rs"]
mod run_plan_caller_requirement_2568_tests;

#[cfg(test)]
#[path = "run_fixed_point_continuation_tests.rs"]
mod run_fixed_point_continuation_tests;

#[cfg(test)]
#[path = "run_trial_inner_nonconvergence_retreat_2943_tests.rs"]
mod run_trial_inner_nonconvergence_retreat_2943_tests;

/// Is `seed` a prior fit's terminal certificate that is STILL stationary here?
///
/// `Some(cost)` only when all of: the seed is the resumed rho itself; a first
/// order evaluation succeeds and is finite; on a resume attempt
/// (`OuterConfig::resume_value`) its value agrees with the recorded one; and the
/// rail-projected gradient sits inside the band the outer certificate demands. Anything else is `None` and
/// the ordinary seed cascade runs. This refuses by default and never turns an
/// evaluation failure into an acceptance.
fn certified_resume_is_already_stationary(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    seed: &Array1<f64>,
    bounds_template: &(Array1<f64>, Array1<f64>),
    seed_idx: usize,
    context: &str,
) -> Option<f64> {
    if !config.initial_rho_is_prior_terminal_certificate {
        return None;
    }
    if config.initial_rho.as_ref() != Some(seed) {
        return None;
    }
    let eval = eval_seed_at_full_inner_fidelity(
                            obj,
                            config,
                            seed,
                            OuterEvalOrder::ValueAndGradient,
                        )
        .ok()?;
    if !eval.cost.is_finite() || eval.gradient.iter().any(|value| !value.is_finite()) {
        return None;
    }
    // A certificate belongs to a point AND a criterion (gam#3002). A resume
    // records the value its criterion took at the point; another criterion (a
    // pilot's, an unarmed evidence fit's, an earlier alternation round's) takes
    // another value there, and a small projected gradient under it, which a
    // point railed on its box faces has for many criteria, certifies nothing.
    if let Some(recorded) = config.resume_value
        && (eval.cost - recorded).abs()
            > crate::rho_optimizer::outer_value_agreement_bound(recorded, eval.cost)
    {
        log::debug!(
            "[OUTER] {context}: resumed certificate seed {seed_idx} was certified at \
             value {recorded:.12e}, this search's criterion is {:.12e} there; declining",
            eval.cost
        );
        return None;
    }
    let projected = rail_projected_gradient_norm(seed, &eval.gradient, Some(bounds_template));
    let band = outer_gradient_tolerance(config).threshold(eval.cost, projected);
    if projected > band {
        log::trace!(
            "[OUTER] {context}: resumed terminal certificate seed {seed_idx} is not stationary \
             here (|Pg|={projected:.6e} > band {band:.6e}); running the ordinary cascade"
        );
        return None;
    }
    log::debug!(
        "[OUTER] {context}: seed {seed_idx} is a prior fit's terminal certificate and is still \
         stationary (|Pg|={projected:.6e} <= band {band:.6e}); accepting with zero outer iterations"
    );
    Some(eval.cost)
}

/// A resume attempt (gam#3002, `OuterConfig::resume_value`): accept the prior
/// fit's certified point `initial_rho` where it stands, or decline.
///
/// The point is accepted exactly as the seed loop accepts a still-stationary
/// terminal certificate, with no outer iteration: certified for this search's
/// criterion (`certified_resume_is_already_stationary`), then screened by the
/// analytic certificate and installed as the terminal state, and `run_outer`
/// mints it as it mints every plan's winner. Anything else declines with an
/// error. No plan runs, so no reseed, fallback or retry can search from the
/// point: a declined attempt costs one evaluation, and the caller runs the cold
/// search from a reset objective.
pub(crate) fn resume_prior_certificate(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    cap: &OuterCapability,
    context: &str,
) -> Result<OuterResult, EstimationError> {
    let declined = |reason: String| {
        EstimationError::RemlOptimizationFailed(format!(
            "{context}: the prior certificate is declined: {reason}"
        ))
    };
    let seed = config
        .initial_rho
        .clone()
        .ok_or_else(|| declined("the resume attempt carries no point".to_string()))?;
    let the_plan = plan(cap);
    let bounds_template = outer_search_bounds_template(config, cap.n_params);
    obj.reset();
    install_matching_initial_inner_seed(obj, config, &seed, context)?;
    let cost =
        certified_resume_is_already_stationary(obj, config, &seed, &bounds_template, 0, context)
            .ok_or_else(|| {
                declined("the point is not certified for this search's criterion".to_string())
            })?;
    let mut candidate = OuterResult::new(seed, cost, 0, true, the_plan);
    candidate.origin = OuterResultOrigin::SeedAcceptedWithoutIteration;
    let result = CertifiedOuterCandidate::from_solver_claim(obj, config, context, candidate)
        .map_err(|(_, error)| declined(format!("the analytic certificate refused it: {error}")))?
        .into_result();
    // Install the accepted point at full inner fidelity, as the seed loop's
    // winner is installed.
    let finalize_cap_guard = config
        .outer_inner_cap
        .as_ref()
        .map(FullFidelityInnerCapGuard::lift);
    if finalize_cap_guard.is_some() {
        obj.reset();
    }
    let finalize_outcome = obj.finalize_outer_result(&result.rho, &the_plan);
    drop(finalize_cap_guard);
    finalize_outcome?;
    Ok(result)
}
