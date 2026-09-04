use super::*;

#[inline]
fn should_start_next_seed(
    started_seeds: usize,
    seed_budget: usize,
    has_certified_candidate: bool,
) -> bool {
    started_seeds < seed_budget || !has_certified_candidate
}

/// Drop from `seeds` every point an earlier certify-resume round already
/// STARTED and already had REFUSED (#2569).
///
/// Slot 0 is never dropped: it is the caller's `initial_rho`, which on a resume
/// is the reseed point the retry exists to explore. Everything else in the
/// cascade is regenerated identically each round from a state `obj.reset()` has
/// restored, so a seed already in the record terminates exactly where it
/// terminated before — measured on #2569 as one cold seed re-run 17 times to
/// the identical `|g|` after the identical 42 outer iterations. Returns the
/// surviving cascade and how many seeds were replayed rather than re-run.
fn seeds_without_recorded_refusals(
    seeds: Vec<Array1<f64>>,
    previously_refused: &[Array1<f64>],
) -> (Vec<Array1<f64>>, usize) {
    if previously_refused.is_empty() || seeds.len() < 2 {
        return (seeds, 0);
    }
    let mut kept: Vec<Array1<f64>> = Vec::with_capacity(seeds.len());
    let mut replayed = 0usize;
    for (seed_idx, seed) in seeds.into_iter().enumerate() {
        if seed_idx > 0 && previously_refused.iter().any(|refused| refused == &seed) {
            replayed += 1;
            continue;
        }
        kept.push(seed);
    }
    (kept, replayed)
}

/// Seed start points this plan run STARTED and whose mandatory analytic
/// certificate then REFUSED (#2569).
///
/// Derived in one place from the rejection ledger the seed loop already keeps,
/// rather than recorded at each of its refusal sites, so the set cannot drift
/// from what the ledger reports. Only the `"certificate"` phase qualifies: a
/// seed rejected at screening, domain entry or validation never reached a
/// solver, and its refusal carries no statement about where the search would
/// have terminated.
fn certificate_refused_seed_points(
    seed_rejections: &[SeedRejection],
    seeds: &[Array1<f64>],
) -> Vec<Array1<f64>> {
    let mut points: Vec<Array1<f64>> = Vec::new();
    for rejection in seed_rejections
        .iter()
        .filter(|rejection| rejection.phase == "certificate")
    {
        if let Some(seed) = seeds.get(rejection.seed_idx)
            && !points.contains(seed)
        {
            points.push(seed.clone());
        }
    }
    points
}

/// Parsimonious screening has exactly two roles: the flexible slot-0 basin and
/// the deliberately promoted, more-smoothed slot-1 basin. The remaining budget
/// is failure recovery, not an instruction to solve extra certified basins.
const PARSIMONY_COMPARISON_SEED_COUNT: usize = 2;

#[inline]
fn should_await_promoted_parsimony_seed(
    seed_budget: usize,
    started_seeds: usize,
    promoted_seed_is_redundant: bool,
) -> bool {
    seed_budget >= PARSIMONY_COMPARISON_SEED_COUNT
        && started_seeds < PARSIMONY_COMPARISON_SEED_COUNT
        && !promoted_seed_is_redundant
}

/// Evaluate the literal outer seed against the true profiled objective.
///
/// Adaptive inner caps are search accelerators. A capped, nonconverged inner
/// iterate is not a value or derivative of the profiled objective and therefore
/// cannot reject a seed or initialize an optimizer. Lift the shared cap only for
/// this sample, preserving any continuation/pilot warm state, then restore the
/// scheduler before search begins.
/// The typed ray a custom-family inner solve reported when it stopped
/// descending a direction with no finite minimizer in reach at this ρ.
fn ray_restoration_in(err: &EstimationError) -> Option<&gam_problem::RayRestoration> {
    match err {
        EstimationError::CustomFamily(gam_problem::CustomFamilyError::InnerSolveNotConverged {
            terminal:
                Some(gam_problem::InnerConvergenceTerminalState::JointNewton {
                    termination_reason,
                    ..
                }),
            ..
        }) => match termination_reason {
            // A ray is a property of the accepted step, so every terminal
            // reason that can carry one is read here (gam#2695): the
            // slow-rate exit, and the residual-stall / divergence exits that
            // used to drop it.
            gam_problem::JointNewtonTerminalReason::SlowGeometricRate { ray, .. } => ray.as_ref(),
            gam_problem::JointNewtonTerminalReason::StalledOnDescendingRay { ray, .. } => Some(ray),
            _ => None,
        },
        _ => None,
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
            log::warn!(
                "[OUTER] {context}: seed {seed_idx} still descends a ray after {restorations} \
                 restorations (one per rho coordinate); refusing it as evaluated: {ray}"
            );
            return Err(err);
        }
        let mut restored = seed.clone();
        for j in ray.rho_indices() {
            let Some(current) = restored.get(j).copied() else {
                return Err(err);
            };
            let ceiling = upper.get(j).copied().unwrap_or(f64::INFINITY);
            // The closure may lie beyond the domain's ceiling: then the block
            // wants more strength than the model resolves, and the honest move
            // is to the ceiling itself (a block collapsing to its null space
            // is a result, not a wall — #2812). A coordinate already at its
            // ceiling cannot be raised, and the refusal stands.
            let raised = (current + ray.log_strength_ratio).min(ceiling);
            if !(raised.is_finite() && raised > current) {
                log::warn!(
                    "[OUTER] {context}: seed {seed_idx} descends a ray on a coordinate already \
                     at its domain ceiling (rho[{j}]={current:.4}, ceiling {ceiling:.4}); \
                     refusing it as evaluated: {ray}"
                );
                return Err(err);
            }
            restored[j] = raised;
        }
        log::warn!(
            "[OUTER] {context}: seed {seed_idx} is under-penalized, not failed — {ray}; \
             restoring rho[{}..{}] from {:?} to {:?} and re-evaluating",
            ray.rho_first,
            ray.rho_first + ray.rho_count,
            ray.rho_indices().map(|j| seed[j]).collect::<Vec<_>>(),
            ray.rho_indices().map(|j| restored[j]).collect::<Vec<_>>(),
        );
        *seed = restored;
        restorations += 1;
    }
}

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

/// A multistart candidate that has cleared the analytic outer certificate.
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
        // #2359: this is the multi-start's FILTER, not the mint. It spends
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

/// One finite-difference probe: the criterion value, and its atom breakdown
/// where the evaluator publishes one.
///
/// `decompose` carries the verdict reached at the analytic seed. Once an
/// evaluator has shown it decomposes, every probe of the same objective must
/// decompose too, and a probe that stops publishing is a defect rather than a
/// route difference — so the demand is unchanged there. Where the seed showed
/// the criterion is not assembled from atoms (#2460), the probe supplies the
/// value alone and the atom stencils are not formed.
fn evaluate_fd_cost_with_criterion_components(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    context: &str,
    inner_seed: &Array1<f64>,
    theta: &Array1<f64>,
    decompose: bool,
) -> Result<(f64, Option<([f64; 4], Array1<f64>)>), EstimationError> {
    evaluate_fd_cost_with_criterion_components_and_curvature(
        obj, config, context, inner_seed, theta, decompose,
    )
    .map(|(cost, parts, _)| (cost, parts))
}

/// [`evaluate_fd_cost_with_criterion_components`] plus the dense curvature the
/// `logdet_h` atom was taken on at this displaced θ (#2765).
///
/// Separate entry point because only the coordinate that differences the
/// CURVATURE needs the `O(p²)` matrix; every other stencil consumer wants the
/// scalars alone.
fn evaluate_fd_cost_with_criterion_components_and_curvature(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    context: &str,
    inner_seed: &Array1<f64>,
    theta: &Array1<f64>,
    decompose: bool,
) -> Result<
    (
        f64,
        Option<([f64; 4], Array1<f64>)>,
        Option<crate::estimate::outer_eval_capture::OuterCurvatureSnapshot>,
    ),
    EstimationError,
> {
    obj.reset();
    install_matching_initial_inner_seed(obj, config, inner_seed, context)?;
    crate::estimate::outer_eval_capture::begin_outer_criterion_component_capture();
    let cost = obj.eval_cost(theta)?;
    let curvature = crate::estimate::outer_eval_capture::take_outer_curvature_snapshot();
    if !decompose {
        return Ok((cost, None, curvature));
    }
    let (component_cost, components) =
        crate::estimate::outer_eval_capture::take_outer_criterion_components().ok_or_else(|| {
            EstimationError::InvalidInput(
                "outer-gradient FD capture received no scalar criterion components".to_string(),
            )
        })?;
    if component_cost.to_bits() != cost.to_bits() {
        return Err(EstimationError::InvalidInput(format!(
            "outer-gradient FD scalar-component cost mismatch: objective={cost:.17e} \
             components={component_cost:.17e}"
        )));
    }
    let (beta, _) =
        crate::estimate::outer_eval_capture::take_outer_selected_mode().ok_or_else(|| {
            EstimationError::InvalidInput(
                "outer-gradient FD capture received no selected coefficient mode".to_string(),
            )
        })?;
    Ok((cost, Some((components, beta)), curvature))
}

/// The seed and the box it must stay inside, which is one fact about one point
/// rather than three parallel vectors a caller could pass out of order.
#[derive(Clone, Copy)]
struct SeedBox<'a> {
    seed: &'a Array1<f64>,
    lower: &'a Array1<f64>,
    upper: &'a Array1<f64>,
}

/// The analytic evaluation at the seed, as the one-sided stencils consume it.
///
/// The three travel together — a one-sided rule differences the criterion, its
/// atoms and its coefficient mode against the SAME base point — so they are one
/// argument.
#[derive(Clone, Copy)]
struct AnalyticSeedBase<'a> {
    cost: f64,
    components: &'a [f64; 4],
    beta: &'a Array1<f64>,
}

/// One θ coordinate's Ridders-certified finite difference of the criterion,
/// together with the per-atom stencils and the coefficient mode response taken
/// at the step the ladder accepted.
///
/// Extracted from the ψ loop so the ρ block is differenced by the SAME ladder,
/// the same box-room policy and the same one-sided fallbacks rather than by a
/// second copy that could drift from it (#2765).
struct CoordinateDifference {
    value: f64,
    uncertainty: f64,
    order: usize,
    step: f64,
    /// `[fixed_beta, logdet_h, logdet_s, kkt]` differenced at `step`, when the
    /// criterion decomposes into atoms at all.
    component_gradients: Option<[f64; 4]>,
    /// `dβ̂/dθ_j` differenced at the same `step`, when the criterion decomposes.
    beta_dot: Option<Array1<f64>>,
}

/// The finite difference is SELF-CERTIFYING (#2461).
///
/// This audit used to difference the criterion once at
/// `eps^0.25 · (1 + |θ|) ≈ 1.2e-4` and report the result as fact. A central
/// difference's error is `ν/h + h²·V‴/6`, and on a REML criterion evaluated
/// through an inner profile NEITHER coefficient is known: at the ψ-saturated
/// rungs measured in #2461 the criterion's third derivative is `~9e7`, which
/// costs a fixed `1.2e-4` step a relative error of `(h/s)²/6` — larger than
/// any tolerance a gradient gate would want to set, and constant in every
/// parameter except `h`, so it reads as a formula error. That is why the
/// consumers of this record grade it at `5e-2`: the oracle's own error, not
/// the gradient's, set the floor.
///
/// `ridders_from_stencil` runs a shrinking ladder of the SAME stencil a single
/// difference would have used once, Neville-extrapolates across it, and reports
/// the extrapolant with an estimate of its own error. The record then carries
/// the uncertainty, so a gate can judge `|analytic − fd|` against a band
/// widened by what the measurement actually knows, and can decline to judge a
/// component the ladder could not resolve.
///
/// The ladder is kept strictly inside the box, and the per-atom component
/// stencils and the coefficient mode response are then evaluated ONCE at the
/// step the ladder accepted, so every number in the record is taken at a step
/// something justified rather than at a step something guessed.
fn difference_theta_coordinate(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    context: &str,
    point: SeedBox<'_>,
    j: usize,
    decompose: bool,
    base: AnalyticSeedBase<'_>,
) -> Result<CoordinateDifference, EstimationError> {
    let SeedBox { seed, lower, upper } = point;
    let AnalyticSeedBase {
        cost: analytic_cost,
        components: analytic_cost_components,
        beta: analytic_beta,
    } = base;
    let ladder_rungs = 10usize;
    let ladder_shrink = 2.0_f64;
    let nominal_step = f64::EPSILON.powf(0.25) * (1.0 + seed[j].abs());
    let left_room = (seed[j] - lower[j]).max(0.0);
    let right_room = (upper[j] - seed[j]).max(0.0);
    // A ladder rung must be evaluable, so the coarsest step has to fit in
    // the room the box leaves. `1e-2 · (1 + |θ|)` is the nominal start; the
    // box shrinks it when it must, and the ladder still spans
    // `shrink^(rungs-1) = 512` from there.
    let ladder_start = |room: f64| (1.0e-2 * (1.0 + seed[j].abs())).min(0.5 * room);
    let mut fd_error: Option<EstimationError> = None;
    let probe = |offset: f64,
                 obj: &mut dyn OuterObjective,
                 fd_error: &mut Option<EstimationError>|
     -> f64 {
        if fd_error.is_some() {
            return f64::NAN;
        }
        let mut theta = seed.clone();
        theta[j] += offset;
        match evaluate_fd_cost_with_criterion_components(
            obj, config, context, seed, &theta, decompose,
        ) {
            Ok((cost, _)) => cost,
            Err(error) => {
                *fd_error = Some(error);
                f64::NAN
            }
        }
    };
    if left_room >= nominal_step && right_room >= nominal_step {
        let measured = gam_linalg::numeric_derivative::ridders_from_stencil(
            |h| {
                (probe(h, &mut *obj, &mut fd_error) - probe(-h, &mut *obj, &mut fd_error))
                    / (2.0 * h)
            },
            gam_linalg::numeric_derivative::RiddersConfig {
                initial_step: ladder_start(left_room.min(right_room)),
                shrink: ladder_shrink,
                rungs: ladder_rungs,
            },
        );
        if let Some(error) = fd_error.take() {
            return Err(error);
        }
        let step = if measured.step.is_finite() && measured.step > 0.0 {
            measured.step
        } else {
            nominal_step
        };
        let mut plus = seed.clone();
        let mut minus = seed.clone();
        plus[j] += step;
        minus[j] -= step;
        let (_cost_plus, parts_plus) = evaluate_fd_cost_with_criterion_components(
            obj, config, context, seed, &plus, decompose,
        )?;
        let (_cost_minus, parts_minus) = evaluate_fd_cost_with_criterion_components(
            obj, config, context, seed, &minus, decompose,
        )?;
        let (component_gradients, beta_dot) = if let (
            Some((components_plus, beta_plus)),
            Some((components_minus, beta_minus)),
        ) = (parts_plus, parts_minus)
        {
            (
                Some(std::array::from_fn(|atom| {
                    (components_plus[atom] - components_minus[atom]) / (2.0 * step)
                })),
                Some(Array1::from_iter(
                    beta_plus.iter().zip(beta_minus.iter()).map(
                        |(&plus_value, &minus_value)| (plus_value - minus_value) / (2.0 * step),
                    ),
                )),
            )
        } else {
            (None, None)
        };
        Ok(CoordinateDifference {
            value: measured.value,
            uncertainty: measured.uncertainty,
            order: measured.order,
            step,
            component_gradients,
            beta_dot,
        })
    } else if right_room >= left_room && right_room > 0.0 {
        // Pinned against the LOWER face: only the forward three-point rule
        // is evaluable. Its error is `O(h²)` like the central difference,
        // so the same ladder certifies it (`ridders_certifies_a_one_sided_
        // stencil`). The coarsest rung must fit `2h` inside the room.
        let measured = gam_linalg::numeric_derivative::ridders_from_stencil(
            |h| {
                (-3.0 * analytic_cost + 4.0 * probe(h, &mut *obj, &mut fd_error)
                    - probe(2.0 * h, &mut *obj, &mut fd_error))
                    / (2.0 * h)
            },
            gam_linalg::numeric_derivative::RiddersConfig {
                initial_step: ladder_start(0.5 * right_room),
                shrink: ladder_shrink,
                rungs: ladder_rungs,
            },
        );
        if let Some(error) = fd_error.take() {
            return Err(error);
        }
        let step = if measured.step.is_finite() && measured.step > 0.0 {
            measured.step
        } else {
            nominal_step.min(0.5 * right_room)
        };
        let mut one = seed.clone();
        let mut two = seed.clone();
        one[j] += step;
        two[j] += 2.0 * step;
        let (_cost_one, parts_one) = evaluate_fd_cost_with_criterion_components(
            obj, config, context, seed, &one, decompose,
        )?;
        let (_cost_two, parts_two) = evaluate_fd_cost_with_criterion_components(
            obj, config, context, seed, &two, decompose,
        )?;
        let (component_gradients, beta_dot) =
            if let (Some((components_one, beta_one)), Some((components_two, beta_two))) =
                (parts_one, parts_two)
            {
                (
                    Some(std::array::from_fn(|atom| {
                        (-3.0 * analytic_cost_components[atom] + 4.0 * components_one[atom]
                            - components_two[atom])
                            / (2.0 * step)
                    })),
                    Some(Array1::from_iter(
                        analytic_beta
                            .iter()
                            .zip(beta_one.iter())
                            .zip(beta_two.iter())
                            .map(|((&base_value, &one_value), &two_value)| {
                                (-3.0 * base_value + 4.0 * one_value - two_value) / (2.0 * step)
                            }),
                    )),
                )
            } else {
                (None, None)
            };
        Ok(CoordinateDifference {
            value: measured.value,
            uncertainty: measured.uncertainty,
            order: measured.order,
            step,
            component_gradients,
            beta_dot,
        })
    } else if left_room > 0.0 {
        // Pinned against the UPPER face: the backward three-point mirror of
        // the branch above, certified by the same ladder.
        let measured = gam_linalg::numeric_derivative::ridders_from_stencil(
            |h| {
                (3.0 * analytic_cost - 4.0 * probe(-h, &mut *obj, &mut fd_error)
                    + probe(-2.0 * h, &mut *obj, &mut fd_error))
                    / (2.0 * h)
            },
            gam_linalg::numeric_derivative::RiddersConfig {
                initial_step: ladder_start(0.5 * left_room),
                shrink: ladder_shrink,
                rungs: ladder_rungs,
            },
        );
        if let Some(error) = fd_error.take() {
            return Err(error);
        }
        let step = if measured.step.is_finite() && measured.step > 0.0 {
            measured.step
        } else {
            nominal_step.min(0.5 * left_room)
        };
        let mut one = seed.clone();
        let mut two = seed.clone();
        one[j] -= step;
        two[j] -= 2.0 * step;
        let (_cost_one, parts_one) = evaluate_fd_cost_with_criterion_components(
            obj, config, context, seed, &one, decompose,
        )?;
        let (_cost_two, parts_two) = evaluate_fd_cost_with_criterion_components(
            obj, config, context, seed, &two, decompose,
        )?;
        let (component_gradients, beta_dot) =
            if let (Some((components_one, beta_one)), Some((components_two, beta_two))) =
                (parts_one, parts_two)
            {
                (
                    Some(std::array::from_fn(|atom| {
                        (3.0 * analytic_cost_components[atom] - 4.0 * components_one[atom]
                            + components_two[atom])
                            / (2.0 * step)
                    })),
                    Some(Array1::from_iter(
                        analytic_beta
                            .iter()
                            .zip(beta_one.iter())
                            .zip(beta_two.iter())
                            .map(|((&base_value, &one_value), &two_value)| {
                                (3.0 * base_value - 4.0 * one_value + two_value) / (2.0 * step)
                            }),
                    )),
                )
            } else {
                (None, None)
            };
        Ok(CoordinateDifference {
            value: measured.value,
            uncertainty: measured.uncertainty,
            order: measured.order,
            step,
            component_gradients,
            beta_dot,
        })
    } else {
        Err(EstimationError::InvalidInput(format!(
            "outer-gradient FD capture cannot perturb collapsed coordinate {j}"
        )))
    }
}

fn capture_outer_gradient_fd_at_seed(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    context: &str,
    seed: &Array1<f64>,
    rho_dim: usize,
    psi_dim: usize,
    lower: &Array1<f64>,
    upper: &Array1<f64>,
) -> Result<(), EstimationError> {
    if !crate::estimate::outer_eval_capture::outer_gradient_fd_capture_enabled(psi_dim) {
        return Ok(());
    }
    if rho_dim.checked_add(psi_dim) != Some(seed.len())
        || lower.len() != seed.len()
        || upper.len() != seed.len()
    {
        return Err(EstimationError::InvalidInput(format!(
            "outer-gradient FD capture received inconsistent layout: \
             rho_dim={rho_dim} psi_dim={psi_dim} theta_dim={} lower_dim={} upper_dim={}",
            seed.len(),
            lower.len(),
            upper.len()
        )));
    }
    let full_fidelity_guard = config
        .outer_inner_cap
        .as_ref()
        .map(FullFidelityInnerCapGuard::lift);
    // Grading ρ needs the analytic ρ-block atoms, which the evaluator publishes
    // on a SEPARATE channel (#2454) that only emits while armed. Arm it here
    // rather than asking the caller to, so the two halves of one comparison
    // cannot be requested independently; restore whatever the caller had after
    // the analytic evaluation so an outer ρ audit already in flight keeps its
    // window.
    let grade_rho =
        rho_dim > 0 && crate::estimate::outer_eval_capture::outer_gradient_fd_capture_grades_rho();
    let caller_rho_audit = if grade_rho {
        let existing = crate::estimate::outer_eval_capture::take_rho_outer_audit();
        crate::estimate::outer_eval_capture::enable_rho_outer_audit();
        existing
    } else {
        None
    };
    obj.reset();
    install_matching_initial_inner_seed(obj, config, seed, context)?;
    crate::estimate::outer_eval_capture::begin_outer_gradient_component_capture();
    crate::estimate::outer_eval_capture::begin_outer_criterion_component_capture();
    let analytic = eval_seed_at_full_inner_fidelity(
                            obj,
                            config,
                            seed,
                            OuterEvalOrder::ValueAndGradient,
                        )?;
    if analytic.gradient.len() != seed.len() || !analytic.cost.is_finite() {
        return Err(EstimationError::InvalidInput(format!(
            "outer-gradient FD capture received invalid analytic evidence: \
             theta_dim={} gradient_dim={} cost={}",
            seed.len(),
            analytic.gradient.len(),
            analytic.cost
        )));
    }
    // #2765: the curvature and its analytic drifts, taken from the SAME
    // evaluation the analytic gradient came from. Drained here rather than at
    // the end because every finite-difference probe below republishes the
    // channel with its own displaced curvature.
    let analytic_curvature = crate::estimate::outer_eval_capture::take_outer_curvature_snapshot();
    let analytic_rho_parts = if grade_rho {
        let audit = crate::estimate::outer_eval_capture::take_rho_outer_audit();
        if let Some(restored) = caller_rho_audit {
            crate::estimate::outer_eval_capture::restore_rho_outer_audit(restored);
        }
        audit.map(|audit| audit.parts).unwrap_or_default()
    } else {
        Vec::new()
    };
    let analytic_psi_gradient =
        Array1::from_iter((0..psi_dim).map(|psi_j| analytic.gradient[rho_dim + psi_j]));
    let components = crate::estimate::outer_eval_capture::take_outer_gradient_components();
    let mut criterion_components =
        crate::estimate::outer_eval_capture::take_outer_criterion_components();
    let mut selected_mode = crate::estimate::outer_eval_capture::take_outer_selected_mode();
    // #2460: does this objective's criterion decompose into REML atoms at all?
    //
    // The three publications are one fact, so the question has only two honest
    // answers. All three present is a decomposing evaluator; NONE present is a
    // route that computes its criterion directly — the constant-curvature fair
    // profile returns `(value, derivative)` in closed form and never reaches a
    // REML assembly, so it has no atoms, no scalar components, and no selected
    // coefficient mode to difference.
    //
    // Anything in between is an evaluator that MEANS to decompose and stopped
    // partway, which is a defect and must not be laundered into a decline. It
    // falls through to the same demands as before, and fails naming the half
    // that went missing.
    let decompose =
        !components.is_empty() || criterion_components.is_some() || selected_mode.is_some();
    let zeros = || Array1::<f64>::zeros(psi_dim);
    let (
        fixed_beta_psi_gradient,
        logdet_h_psi_gradient,
        frozen_logdet_h_psi_gradient,
        mode_response_logdet_h_psi_gradient,
        logdet_s_psi_gradient,
        kkt_psi_gradient,
    ) = if decompose {
        if components.len() != psi_dim {
            return Err(EstimationError::InvalidInput(format!(
                "outer-gradient FD capture received {} ψ component rows, expected {psi_dim}",
                components.len()
            )));
        }
        (
            Array1::from_iter(components.iter().map(|component| component.0)),
            Array1::from_iter(components.iter().map(|component| component.1)),
            Array1::from_iter(components.iter().map(|component| component.2)),
            Array1::from_iter(components.iter().map(|component| component.3)),
            Array1::from_iter(components.iter().map(|component| component.4)),
            Array1::from_iter(components.iter().map(|component| component.5)),
        )
    } else {
        (zeros(), zeros(), zeros(), zeros(), zeros(), zeros())
    };
    let analytic_cost_components: [f64; 4] = if decompose {
        let (analytic_component_cost, analytic_cost_components) =
            criterion_components.take().ok_or_else(|| {
                EstimationError::InvalidInput(
                    "outer-gradient FD capture received no analytic scalar criterion components"
                        .to_string(),
                )
            })?;
        if analytic_component_cost.to_bits() != analytic.cost.to_bits() {
            return Err(EstimationError::InvalidInput(format!(
                "outer-gradient FD analytic scalar-component cost mismatch: objective={:.17e} \
                 components={analytic_component_cost:.17e}",
                analytic.cost
            )));
        }
        analytic_cost_components
    } else {
        [0.0; 4]
    };
    let (analytic_beta, analytic_ext_mode_response_cols) = if decompose {
        let (analytic_beta, analytic_ext_mode_response_cols) =
            selected_mode.take().ok_or_else(|| {
                EstimationError::InvalidInput(
                    "outer-gradient FD capture received no analytic selected coefficient mode"
                        .to_string(),
                )
            })?;
        let analytic_ext_mode_response_cols = analytic_ext_mode_response_cols.ok_or_else(|| {
            EstimationError::InvalidInput(
                "outer-gradient FD capture received no analytic extended-coordinate mode responses"
                    .to_string(),
            )
        })?;
        if analytic_ext_mode_response_cols.nrows() != analytic_beta.len()
            || analytic_ext_mode_response_cols.ncols() != psi_dim
        {
            return Err(EstimationError::InvalidInput(format!(
                "outer-gradient FD mode-response layout mismatch: beta_dim={} response_shape={}x{} psi_dim={psi_dim}",
                analytic_beta.len(),
                analytic_ext_mode_response_cols.nrows(),
                analytic_ext_mode_response_cols.ncols(),
            )));
        }
        (analytic_beta, Some(analytic_ext_mode_response_cols))
    } else {
        (Array1::<f64>::zeros(0), None)
    };
    let mut finite_difference_psi_gradient = Array1::<f64>::zeros(psi_dim);
    let mut finite_difference_component_psi_gradients: [Array1<f64>; 4] =
        std::array::from_fn(|_| Array1::<f64>::zeros(psi_dim));
    let mut psi_steps = Array1::<f64>::zeros(psi_dim);
    let mut psi_fd_uncertainty = Array1::<f64>::from_elem(psi_dim, f64::INFINITY);
    let mut psi_fd_orders = vec![0usize; psi_dim];
    let mut analytic_mode_response_norm = Array1::<f64>::zeros(psi_dim);
    let mut finite_difference_mode_response_norm = Array1::<f64>::zeros(psi_dim);
    let mut mode_response_relative_error = Array1::<f64>::zeros(psi_dim);
    let mut mode_response_max_abs_error = Array1::<f64>::zeros(psi_dim);
    let mut record_mode_response =
        |psi_j: usize,
         finite_difference_beta_dot: &Array1<f64>|
         -> Result<(), EstimationError> {
            // Only a decomposing evaluator publishes response columns; without
            // them there is no analytic mode response to compare against, and
            // the caller never forms a `beta_dot` to pass here (#2460).
            let Some(analytic_ext_mode_response_cols) = analytic_ext_mode_response_cols.as_ref()
            else {
                return Ok(());
            };
            if finite_difference_beta_dot.len() != analytic_beta.len() {
                return Err(EstimationError::InvalidInput(format!(
                    "outer-gradient FD coefficient-response length mismatch: analytic={} finite_difference={}",
                    analytic_beta.len(),
                    finite_difference_beta_dot.len(),
                )));
            }
            let analytic_beta_dot = analytic_ext_mode_response_cols
                .column(psi_j)
                .mapv(|value| -value);
            let analytic_norm = analytic_beta_dot.dot(&analytic_beta_dot).sqrt();
            let finite_difference_norm = finite_difference_beta_dot
                .dot(finite_difference_beta_dot)
                .sqrt();
            let mut squared_error = 0.0_f64;
            let mut max_abs_error = 0.0_f64;
            for (&analytic_value, &fd_value) in
                analytic_beta_dot.iter().zip(finite_difference_beta_dot.iter())
            {
                let error = analytic_value - fd_value;
                squared_error += error * error;
                max_abs_error = max_abs_error.max(error.abs());
            }
            analytic_mode_response_norm[psi_j] = analytic_norm;
            finite_difference_mode_response_norm[psi_j] = finite_difference_norm;
            mode_response_relative_error[psi_j] =
                squared_error.sqrt() / analytic_norm.max(finite_difference_norm).max(1e-12);
            mode_response_max_abs_error[psi_j] = max_abs_error;
            Ok(())
        };
    // Every coordinate is differenced by `difference_theta_coordinate`, whose
    // doc comment carries the self-certifying-ladder rationale (#2461).
    for psi_j in 0..psi_dim {
        let measured = difference_theta_coordinate(
            obj,
            config,
            context,
            SeedBox { seed, lower, upper },
            rho_dim + psi_j,
            decompose,
            AnalyticSeedBase {
                cost: analytic.cost,
                components: &analytic_cost_components,
                beta: &analytic_beta,
            },
        )?;
        finite_difference_psi_gradient[psi_j] = measured.value;
        psi_fd_uncertainty[psi_j] = measured.uncertainty;
        psi_fd_orders[psi_j] = measured.order;
        psi_steps[psi_j] = measured.step;
        if let Some(components) = measured.component_gradients {
            for atom in 0..4 {
                finite_difference_component_psi_gradients[atom][psi_j] = components[atom];
            }
        }
        if let Some(beta_dot) = measured.beta_dot.as_ref() {
            record_mode_response(psi_j, beta_dot)?;
        }
    }
    // The ρ block, by the SAME ladder, when the caller opted in. Its analytic
    // atoms come from the ρ-block audit channel; a run whose evaluator did not
    // publish them (the channel emits only from the REML/LAML assembly) records
    // the gradient comparison alone rather than fabricating zeros.
    let rho_block = if grade_rho {
        let mut analytic_gradient = Array1::<f64>::zeros(rho_dim);
        let mut finite_difference_gradient = Array1::<f64>::zeros(rho_dim);
        let mut steps = Array1::<f64>::zeros(rho_dim);
        let mut fd_uncertainty = Array1::<f64>::from_elem(rho_dim, f64::INFINITY);
        let mut fd_orders = vec![0usize; rho_dim];
        let mut finite_difference_components: [Array1<f64>; 4] =
            std::array::from_fn(|_| Array1::<f64>::zeros(rho_dim));
        for rho_j in 0..rho_dim {
            analytic_gradient[rho_j] = analytic.gradient[rho_j];
            let measured = difference_theta_coordinate(
                obj,
                config,
                context,
                SeedBox { seed, lower, upper },
                rho_j,
                decompose,
                AnalyticSeedBase {
                    cost: analytic.cost,
                    components: &analytic_cost_components,
                    beta: &analytic_beta,
                },
            )?;
            finite_difference_gradient[rho_j] = measured.value;
            fd_uncertainty[rho_j] = measured.uncertainty;
            fd_orders[rho_j] = measured.order;
            steps[rho_j] = measured.step;
            if let Some(components) = measured.component_gradients {
                for atom in 0..4 {
                    finite_difference_components[atom][rho_j] = components[atom];
                }
            }
        }
        let mut analytic_audit_total = Array1::<f64>::from_elem(rho_dim, f64::NAN);
        let mut analytic_fixed_beta = Array1::<f64>::from_elem(rho_dim, f64::NAN);
        let mut analytic_logdet_h = Array1::<f64>::from_elem(rho_dim, f64::NAN);
        let mut analytic_frozen_logdet_h = Array1::<f64>::from_elem(rho_dim, f64::NAN);
        let mut analytic_mode_response_logdet_h = Array1::<f64>::from_elem(rho_dim, f64::NAN);
        let mut analytic_logdet_s = Array1::<f64>::from_elem(rho_dim, f64::NAN);
        let mut analytic_kkt = Array1::<f64>::from_elem(rho_dim, f64::NAN);
        for part in analytic_rho_parts.iter().filter(|part| part.index < rho_dim) {
            analytic_audit_total[part.index] = part.total;
            analytic_fixed_beta[part.index] = part.fixed_beta;
            analytic_logdet_h[part.index] = part.logdet_h;
            analytic_frozen_logdet_h[part.index] = part.frozen_logdet_h;
            analytic_mode_response_logdet_h[part.index] = part.mode_response_logdet_h;
            analytic_logdet_s[part.index] = part.logdet_s;
            analytic_kkt[part.index] =
                part.total - (part.fixed_beta + part.logdet_h + part.logdet_s);
        }
        Some(crate::estimate::OuterGradientFdRhoBlock {
            analytic_gradient,
            finite_difference_gradient,
            steps,
            fd_uncertainty,
            fd_orders,
            analytic_audit_total,
            analytic_fixed_beta,
            analytic_logdet_h,
            analytic_frozen_logdet_h,
            analytic_mode_response_logdet_h,
            analytic_logdet_s,
            analytic_kkt,
            finite_difference_fixed_beta: finite_difference_components[0].clone(),
            finite_difference_logdet_h: finite_difference_components[1].clone(),
            finite_difference_logdet_s: finite_difference_components[2].clone(),
            finite_difference_kkt: finite_difference_components[3].clone(),
        })
    } else {
        None
    };
    // #2765: difference the CURVATURE, not only the contracted scalar.
    //
    // `½ tr(K·Ḣ_i)` disagreeing with a finite difference of `½ log|H|` has
    // three possible owners and the scalar names none of them. This pass
    // differences `H` itself at the step each coordinate's Ridders ladder
    // accepted and compares it entrywise to the analytic `Ḣ_i` the same
    // evaluation published, which settles the drift outright and localizes a
    // failure to a coefficient block. It costs two extra inner profiles per θ
    // coordinate and runs only inside an armed audit window.
    if analytic_curvature.is_none() {
        log::info!(
            "[2765-AUDIT] no analytic curvature snapshot at the audited seed; the criterion's \
             log-determinant operator did not publish one"
        );
    }
    let curvature = analytic_curvature.as_ref().and_then(|base| {
        let theta_dim = rho_dim + psi_dim;
        if base.drifts.len() != theta_dim {
            log::info!(
                "[2765-AUDIT] curvature snapshot carries {} drifts against theta_dim={theta_dim}; \
                 the last publication was a value-only evaluation",
                base.drifts.len()
            );
            return None;
        }
        let mut drift_max_abs_error = Array1::<f64>::from_elem(theta_dim, f64::NAN);
        let mut drift_relative_error = Array1::<f64>::from_elem(theta_dim, f64::NAN);
        let mut drift_worst_entry = vec![(0usize, 0usize); theta_dim];
        let mut analytic_drift_max_abs = Array1::<f64>::zeros(theta_dim);
        let mut face_drift_max_abs = Array1::<f64>::from_elem(theta_dim, f64::NAN);
        let mut displaced_tangent_dim = vec![(None, None); theta_dim];
        let mut analytic_face_drift: Vec<Array2<f64>> = Vec::with_capacity(theta_dim);
        let mut measured_face_drift: Vec<Array2<f64>> = Vec::with_capacity(theta_dim);
        // On an active inequality face the criterion is `½log|ZᵀHZ|`, so the
        // object a `p`-space drift has to be differenced against is `ZᵀḢZ`
        // against `d(ZᵀHZ)/dθ` — with the BASE face on both sides, because a
        // comparison in two different bases measures the basis, not the drift.
        let base_basis = base.tangent_basis.clone();
        let to_face = |matrix: &Array2<f64>| -> Array2<f64> {
            match base_basis.as_ref() {
                Some(z) if z.nrows() == matrix.nrows() => z.t().dot(matrix).dot(z),
                _ => matrix.clone(),
            }
        };
        let projector = |z: Option<&Array2<f64>>| z.map(|basis| basis.dot(&basis.t()));
        let base_projector = projector(base.tangent_basis.as_ref());
        for j in 0..theta_dim {
            let step = if j < rho_dim {
                rho_block.as_ref().map_or(f64::NAN, |block| block.steps[j])
            } else {
                psi_steps[j - rho_dim]
            };
            let analytic_drift = to_face(&base.drifts[j]);
            analytic_drift_max_abs[j] = analytic_drift
                .iter()
                .fold(0.0_f64, |acc, value| acc.max(value.abs()));
            let empty = Array2::<f64>::zeros((0, 0));
            analytic_face_drift.push(analytic_drift.clone());
            measured_face_drift.push(empty);
            if !(step.is_finite() && step > 0.0) {
                continue;
            }
            if seed[j] - step < lower[j] || seed[j] + step > upper[j] {
                continue;
            }
            let mut plus = seed.clone();
            let mut minus = seed.clone();
            plus[j] += step;
            minus[j] -= step;
            let curvature_at = |theta: &Array1<f64>, obj: &mut dyn OuterObjective| {
                evaluate_fd_cost_with_criterion_components_and_curvature(
                    obj, config, context, seed, theta, false,
                )
                .ok()
                .and_then(|(_, _, snapshot)| snapshot)
            };
            let (Some(snapshot_plus), Some(snapshot_minus)) = (
                curvature_at(&plus, &mut *obj),
                curvature_at(&minus, &mut *obj),
            ) else {
                continue;
            };
            displaced_tangent_dim[j] = (
                snapshot_plus.tangent_basis.as_ref().map(Array2::ncols),
                snapshot_minus.tangent_basis.as_ref().map(Array2::ncols),
            );
            // Does the face itself move? Compared as PROJECTORS `ZZᵀ`, which are
            // basis-rotation invariant, so an orthogonal re-parameterisation of
            // the SAME face reads as zero rather than as motion.
            let plus_projector = projector(snapshot_plus.tangent_basis.as_ref());
            let minus_projector = projector(snapshot_minus.tangent_basis.as_ref());
            face_drift_max_abs[j] = match (
                base_projector.as_ref(),
                plus_projector.as_ref(),
                minus_projector.as_ref(),
            ) {
                (Some(base_p), Some(plus_p), Some(minus_p))
                    if base_p.dim() == plus_p.dim() && base_p.dim() == minus_p.dim() =>
                {
                    let mut worst = 0.0_f64;
                    for row in 0..base_p.nrows() {
                        for col in 0..base_p.ncols() {
                            worst = worst
                                .max((plus_p[[row, col]] - base_p[[row, col]]).abs())
                                .max((minus_p[[row, col]] - base_p[[row, col]]).abs());
                        }
                    }
                    worst
                }
                (None, None, None) => 0.0,
                _ => f64::INFINITY,
            };
            let measured_plus = to_face(&snapshot_plus.hessian);
            let measured_minus = to_face(&snapshot_minus.hessian);
            if measured_plus.dim() != analytic_drift.dim()
                || measured_minus.dim() != analytic_drift.dim()
            {
                continue;
            }
            let mut worst = 0.0_f64;
            let mut worst_entry = (0usize, 0usize);
            let mut scale = 0.0_f64;
            for row in 0..analytic_drift.nrows() {
                for col in 0..analytic_drift.ncols() {
                    let measured =
                        (measured_plus[[row, col]] - measured_minus[[row, col]]) / (2.0 * step);
                    let error = (analytic_drift[[row, col]] - measured).abs();
                    scale = scale.max(measured.abs());
                    if error > worst {
                        worst = error;
                        worst_entry = (row, col);
                    }
                }
            }
            drift_max_abs_error[j] = worst;
            // Relative to the larger of the measured and analytic scales; with no
            // scale at all the error is either exactly zero or unbounded, never a
            // number manufactured by a floored denominator.
            let drift_scale = scale.max(analytic_drift_max_abs[j]);
            drift_relative_error[j] = if drift_scale > 0.0 {
                worst / drift_scale
            } else if worst == 0.0 {
                0.0
            } else {
                f64::INFINITY
            };
            drift_worst_entry[j] = worst_entry;
            measured_face_drift[j] = (&measured_plus - &measured_minus).mapv(|v| v / (2.0 * step));
        }
        // The plain `½ Σ log σ_j` of the captured matrix, against the `½log|H|`
        // the criterion actually consumed. A gap says the operator's
        // log-determinant is not this matrix's — a spectral floor, a rank mask
        // or a uniform rescale is in play — which changes what kernel the
        // trace has to contract, and is therefore worth separating from a
        // derivative defect rather than folding into one.
        let dense_half_logdet = {
            use gam_linalg::faer_ndarray::FaerEigh;
            base.hessian
                .eigh(faer::Side::Lower)
                .ok()
                .filter(|(values, _)| values.iter().all(|&value| value > 0.0))
                .map_or(f64::NAN, |(values, _)| {
                    0.5 * values.iter().map(|value| value.ln()).sum::<f64>()
                })
        };
        Some(crate::estimate::OuterCurvatureDriftAudit {
            drift_max_abs_error,
            drift_relative_error,
            drift_worst_entry,
            face_drift_max_abs,
            tangent_dim: base.tangent_basis.as_ref().map(Array2::ncols),
            displaced_tangent_dim,
            analytic_face_drift,
            measured_face_drift,
            face_curvature: to_face(&base.hessian),
            analytic_drift_max_abs,
            dense_half_logdet,
            criterion_half_logdet: 0.5 * base.logdet,
        })
    });
    obj.reset();
    crate::estimate::outer_eval_capture::record_outer_gradient_fd(
        crate::estimate::OuterGradientFdRecord {
            theta: seed.clone(),
            rho_dim,
            psi_dim,
            cost: analytic.cost,
            analytic_psi_gradient,
            finite_difference_psi_gradient,
            psi_steps,
            psi_fd_uncertainty,
            psi_fd_orders,
            psi_gram_anchor_deltas:
                crate::estimate::outer_eval_capture::take_psi_gram_anchor_deltas(),
            decomposition: if decompose {
                crate::estimate::OuterGradientFdDecomposition::Decomposed(Box::new(
                    crate::estimate::OuterGradientFdAtoms {
                        fixed_beta_psi_gradient,
                        logdet_h_psi_gradient,
                        frozen_logdet_h_psi_gradient,
                        mode_response_logdet_h_psi_gradient,
                        analytic_mode_response_norm,
                        finite_difference_mode_response_norm,
                        mode_response_relative_error,
                        mode_response_max_abs_error,
                        logdet_s_psi_gradient,
                        kkt_psi_gradient,
                        finite_difference_fixed_beta_psi_gradient:
                            finite_difference_component_psi_gradients[0].clone(),
                        finite_difference_logdet_h_psi_gradient:
                            finite_difference_component_psi_gradients[1].clone(),
                        finite_difference_logdet_s_psi_gradient:
                            finite_difference_component_psi_gradients[2].clone(),
                        finite_difference_kkt_psi_gradient:
                            finite_difference_component_psi_gradients[3].clone(),
                    },
                ))
            } else {
                crate::estimate::OuterGradientFdDecomposition::NotDecomposed {
                    reason: format!(
                        "{context}: this outer objective published no ψ gradient atoms, no scalar \
                         criterion components and no selected coefficient mode, so its criterion \
                         is not assembled from REML atoms; the ψ gradient and its \
                         Ridders-certified finite difference above are the whole audit"
                    ),
                }
            },
            rho: rho_block,
            curvature,
        },
    );
    drop(full_fidelity_guard);
    Ok(())
}

/// Execute a single plan attempt (seed generation → solver loop → best result).
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
    let mut seeds = {
        let generated = crate::seeding::generate_rho_candidates(
            cap.n_params,
            config.heuristic_lambdas.as_deref(),
            &config.seed_config,
        )?;
        if generated.is_empty() {
            Vec::new()
        } else {
            generated
        }
    };
    // Explicit model-derived candidates precede the generic generator and are
    // not truncated by `SeedConfig::max_seeds`. Insert in reverse so the
    // primary `initial_rho` remains slot zero and the caller's candidate order
    // is preserved. Bounds projection and exact deduplication happen below.
    for candidate in config.initial_rho_candidates.iter().rev() {
        if !seeds.iter().any(|seed| seed == candidate) {
            seeds.insert(0, candidate.clone());
        }
    }
    if let Some(initial_rho) = config.initial_rho.as_ref() {
        if let Some(position) = seeds.iter().position(|seed| seed == initial_rho) {
            let initial = seeds.remove(position);
            seeds.insert(0, initial);
        } else {
            seeds.insert(0, initial_rho.clone());
        }
    }
    if seeds.is_empty() {
        return Err(EstimationError::RemlOptimizationFailed(format!(
            "no seeds generated for outer optimization ({context})"
        )));
    }

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
    let mut projected_seeds = Vec::with_capacity(seeds.len());
    for seed in seeds {
        let projected = project_to_bounds(&seed, Some(&bounds_template));
        if !projected_seeds.contains(&projected) {
            projected_seeds.push(projected);
        }
    }
    seeds = projected_seeds;
    if seeds.is_empty() {
        return Err(EstimationError::RemlOptimizationFailed(format!(
            "no bounded seeds generated for outer optimization ({context})"
        )));
    }

    // #2569 — replay, rather than re-derive, a seed verdict this outer call
    // already recorded. `config.previously_refused_seed_points` is non-empty
    // only on a certify-resume round, and holds seeds an earlier round started
    // and whose analytic certificate refused. The resume reseeds `initial_rho`
    // and resets the objective; every OTHER seed in this cascade is therefore
    // re-entered from the identical state that already refused it, and the
    // trace shows it terminating at the identical gradient after the identical
    // iteration count. Slot 0 is the caller's reseed point and is never
    // dropped — the resume exists to explore it — and a seed that was never
    // started is never in the list, so the `should_start_next_seed`
    // fall-through keeps every rescue it could previously perform.
    let generated_seed_count = seeds.len();
    let (kept_seeds, replayed_seeds) =
        seeds_without_recorded_refusals(seeds, &config.previously_refused_seed_points);
    seeds = kept_seeds;
    if replayed_seeds > 0 {
        log::info!(
            "[OUTER] {context}: replaying {replayed_seeds} recorded seed refusal(s) instead of \
             re-running them ({}/{generated_seed_count} seeds kept); a resume re-enters a \
             non-initial seed from the state that already refused it (#2569/#2080)",
            seeds.len(),
        );
    }

    let seed_budget =
        effective_seed_budget_for_config(&config.seed_config, the_plan.solver).min(seeds.len());
    // Who owns the one budgeted slot: the caller, or the heuristics?
    //
    // `config.initial_rho` with `screen_initial_rho == false` is the caller
    // saying "start HERE and do not re-rank it". Two independent things
    // downstream can displace that seed — the screening ranker below, and the
    // neutral-baseline promotion after it — so both consult this one predicate.
    // They disagreed before: the ranker honoured the caller and the promotion
    // then moved the always-injected `[0.0]` baseline in front of it, which at
    // `seed_budget == 1` does not reorder the cascade, it REPLACES the seed.
    //
    // The seed it replaced is not always a heuristic guess. A cache resume
    // installs its checkpoint ρ through exactly this field (see the
    // `CacheSeedDecision::{Seed, ExactFinal}` arms in `OuterProblem::run`,
    // which set `initial_rho` and clear `screen_initial_rho`), so the promotion
    // was discarding resumed work and re-starting the fit from λ=1 —
    // `all_saturated_cached_rho_is_honored_as_seed` measured it doing that with
    // a checkpoint at ρ=[10,−10] and every evaluation landing at [0,0].
    let explicit_initial_rho_owns_single_seed_budget = config.initial_rho.is_some()
        && seed_budget == 1
        && seeds.len() > 1
        && !config.screen_initial_rho;
    if !explicit_initial_rho_owns_single_seed_budget
        && should_screen_seeds(config, the_plan.solver, seeds.len(), seed_budget)
    {
        // Screening RANKS the seeds; it does not decide whether they can be
        // used. `rank_seeds_with_screening` says so itself — with no screening
        // cap configured it returns the seeds unranked and the run proceeds —
        // so "unranked" is an outcome this code already supports.
        //
        // `fatal_outer_evaluation` here overrode that unconditionally, and it
        // overrode the producer's classification with it. Measured on the
        // coxph-frailty arm: the refusal reached this line already typed and
        // already correct (`Custom-family fit failed: inner solve refused this
        // trial point: …`), `is_trial_point_infeasible()` answered true for it,
        // and the whole fit died anyway — over a ranking.
        //
        // A trial-point refusal at a screening probe means that seed could not
        // be scored, not that the problem is unfittable. Keep the generated
        // order and let the seeds be evaluated for real; if the cause is
        // structural it recurs there and is reported with its own context.
        // Anything the producer did NOT call rho-local still escalates.
        let screened = rank_seeds_with_screening(obj, config, context, &seeds);
        seeds = match screened {
            Ok(ranked) => ranked,
            Err(error) if error.is_trial_point_infeasible() => {
                log::warn!(
                    "[OUTER] {context}: seed screening could not rank the seeds \
                     ({error}); continuing with the generated order",
                );
                seeds
            }
            Err(error) => {
                return Err(EstimationError::fatal_outer_evaluation(
                    "outer seed screening",
                    error,
                ));
            }
        };
    }
    if !explicit_initial_rho_owns_single_seed_budget {
        prioritize_neutral_bfgs_glm_seed(
            &mut seeds,
            &config.seed_config,
            the_plan.solver,
            seed_budget,
        );
    }
    log::debug!(
        "[OUTER] {context}: trying generated seeds directly (generated={}, budget={})",
        seeds.len(),
        seed_budget,
    );
    if seed_budget < config.seed_config.seed_budget.max(1) {
        log::debug!(
            "[OUTER] {context}: capped requested seed budget {} -> {} for {:?} ({:?})",
            config.seed_config.seed_budget.max(1),
            seed_budget,
            the_plan.solver,
            config.seed_config.risk_profile,
        );
    }
    if seeds.len() > seed_budget {
        log::debug!(
            "[OUTER] {context}: trying up to {seed_budget}/{} generated seeds in heuristic order",
            seeds.len(),
        );
    }

    let mut best: Option<CertifiedOuterCandidate> = None;
    let mut best_checkpoint: Option<OuterResult> = None;
    // Best-scoring point whose SOLVER claimed convergence and whose analytic
    // certification then refused it (#2596). Distinct from `best_checkpoint`,
    // which also collects budget-exhausted iterates: this slot holds only
    // points the optimizer itself believes are optima, so comparing the
    // published winner against it is a like-for-like comparison of optima and
    // not a comparison against a half-finished search.
    //
    // A certificate is an ADMISSIBILITY filter, not an ordering. When the best
    // admissible candidate scores materially worse than a refused optimum the
    // run already measured, the multistart has not found the optimum — it has
    // fallen back to one — and that has to be visible. It was not: on #2596 a
    // ρ-box corner (where the box-KKT projection makes |Pg| identically zero,
    // so it certifies whatever its criterion says) was published at cost
    // 110.94 while a refused interior optimum at cost 4.19 sat in this slot,
    // with nothing in the log naming the inversion.
    let mut best_refused_optimum: Option<OuterResult> = None;
    // First confirmed-tail snapped reseed published by a refused certification
    // (#2348 Inc 2b). Consumed once, after the seed cascade, for a single
    // polishing retry pinned at the snapped rail point.
    let mut tail_snap_reseed_point: Option<Array1<f64>> = None;
    // First negative-curvature escape reseed published by a refused
    // certification whose interior reduced Hessian is a certified strict saddle
    // (#2357). Consumed once, after the seed cascade, for a single retry seeded
    // off the saddle ridge so the outer search descends to the true PSD minimum.
    let mut saddle_escape_reseed_point: Option<Array1<f64>> = None;
    // A reactive domain-entry path is created inside a seed attempt only after
    // that objective's exact seed cost is non-finite. Already-feasible seeds
    // therefore stay on the zero-heavy-entry path.
    let reactive_domain_scalar_contract = obj.reactive_domain_scalar_contract()?;
    let reactive_domain_entry_available = reactive_domain_scalar_contract.is_some();
    // Sole owner of every seed refusal. Objective failures enter this ledger
    // while their `ObjectiveEvalError` still carries the originating typed
    // `EstimationError`; there is no parallel prose ledger to reconcile later.
    let mut seed_rejections: Vec<SeedRejection> = Vec::new();
    let layout = cap.theta_layout();
    // Number of smoothing (ρ) coordinates, used to break a near-LAML-tie toward
    // the more-penalized basin in the non-Gaussian multi-start keep-best.
    let rho_dim = layout.rho_dim();
    let mut started_seeds = 0usize;
    // Set to `Some(key)` when every observed rejection so far carries
    // the same genuinely structural `(KktRefusalDiagnosis,
    // carrying_block)` pair AND we've seen at least
    // `STRUCTURAL_EARLY_EXIT_MIN_COUNT` consistent failures. Once set,
    // the remaining ρ candidates are skipped.
    let mut structural_early_exit_key: Option<(
        gam_problem::diagnostics::KktRefusalDiagnosis,
        Option<String>,
    )> = None;
    // Two matching structural observations are enough to break the
    // loop. A single observation could be transient noise — an
    // exploration seed in a degenerate ρ corner, a one-off domain
    // excursion that happens to surface at the cert site. Requiring
    // k=2 across DIFFERENT seeds is the smallest sample size that
    // distinguishes noise from a structural rank/alias/active-set
    // defect; recoverable cert refusals such as phantom multipliers are
    // not eligible for this key.
    const STRUCTURAL_EARLY_EXIT_MIN_COUNT: usize = 2;
    // Generic cross-seed structural-failure bail (#1036). The structural
    // early-exit above only fires for genuinely structural `CertRefused`
    // diagnoses; it never sees the `RemlConvergenceError` / non-PD per-row
    // H_tt / KKT-stuck class, which classifies as Budget/TrustRegion/Other and
    // burned all 12 seeds (sphere: 3.5h for one failed candidate). This
    // detector keys on the generic `(variant, signed-order-of-magnitude
    // pivot/KKT bucket)` signature: when the LAST `n_struct` seeds reject with
    // an identical *quantified* signature, the blocker is the design, not the
    // warm-start, so we bail and skip the remaining seeds. A single deviating
    // signature breaks the trailing run, so genuine seed-luck still runs the
    // full cascade.
    const GENERIC_STRUCTURAL_BAIL_MIN_RUN: usize = 3;
    // `Some((signature, run_len))` once the generic detector has fired on a
    // trailing run of identical quantified signatures. Drives the aggregated
    // "structural: <signature> on seeds a..b; remaining N seeds skipped" note.
    let mut generic_structural_bail: Option<(
        crate::startup_stats::GenericFailureSignature,
        usize,
        usize,
    )> = None;
    // #2080 — the reactive continuation's COLD ENTRY leg is seed-independent, so
    // its refusal is recorded once and replayed rather than re-derived per seed.
    //
    // `ContinuationPath::step` takes `entering = self.warm.is_none()` and pins
    // that leg at `s_next = 1.0`; `continuation_path.rs`'s own
    // `literal_endpoint_bits_survive_without_affine_rounding` pins
    // `rho_target_at(1.0)` as BITWISE `rho_entry`, which is `bounds_template.1`.
    // The scalar leg is `contract.at(1.0)`, likewise the contract's literal entry
    // state, and the walk is opened with an EMPTY warm start
    // (`cold_entry_beta = Array1::zeros(0)`). All three inputs are loop
    // invariants: `bounds_template` and `reactive_domain_scalar_contract` are
    // built above this cascade, and `obj.reset()` runs immediately before the
    // path is constructed. So every seed submits the SAME problem to the same
    // cold solver on that first leg, and a refusal there is a property of the
    // FIXTURE, not of the seed. Only the seed's own `eval_cost` probe above and
    // the LATER legs — which descend toward the seed's literal rho — are
    // seed-dependent, and those are untouched.
    //
    // Measured (#2599's `test_sae_fit_is_deterministic_for_fixed_seed`, K=2,
    // n=400/p=64, private TMPDIR, no timeout): four attempts, 247 of 248 log
    // lines byte-identical across the seed blocks with the sole difference being
    // the seed index in the refusal message, 1650 inner iterations reported
    // identically in each. Re-deriving a constant of the problem once per seed
    // buys nothing; `max_seeds` / `seed_budget` are inert against it.
    let mut cold_entry_leg_refusal: Option<String> = None;

    'seed_attempts: for (seed_idx, seed_as_generated) in seeds.iter().enumerate() {
        // The seed the solver starts from: the generated point, or that point
        // with the under-penalized block's strengths raised by the ratio the
        // inner solve read off its ray (#2695).
        let mut seed_owned = seed_as_generated.clone();
        let seed = &mut seed_owned;
        if !should_start_next_seed(started_seeds, seed_budget, best.is_some()) {
            break;
        }
        // Domain entry is a property of this literal seed. A loop-local path
        // cannot leak its state or regime into another candidate.
        let mut continuation_path: Option<crate::continuation_path::ContinuationPath> = None;
        // Probe whether the seed cascade has slipped into a uniform structural
        // failure mode that the remaining candidates cannot escape.
        if structural_early_exit_key.is_none() {
            if let Some(key) =
                uniform_structural_key(&seed_rejections, STRUCTURAL_EARLY_EXIT_MIN_COUNT)
            {
                log::warn!(
                    "[OUTER] {context}: structural early-exit after {} uniform structural \
                     rejections (diagnosis={}, carrying-block={}); skipping remaining {} seed(s)",
                    seed_rejections.len(),
                    key.0.as_str(),
                    key.1.as_deref().unwrap_or("<unknown>"),
                    seeds.len().saturating_sub(seed_idx),
                );
                structural_early_exit_key = Some(key);
                break;
            }
        }
        // Generic cross-seed structural bail (#1036). Reactive domain entry is
        // only a repair for an undefined literal seed value; it does not turn
        // later, repeated structural solver failures into path re-entry.
        if structural_early_exit_key.is_none() && generic_structural_bail.is_none() {
            if let Some((sig, run_len)) = crate::startup_stats::consecutive_generic_signature(
                &seed_rejections,
                GENERIC_STRUCTURAL_BAIL_MIN_RUN,
            ) {
                let first_seed = seed_rejections[seed_rejections.len() - run_len].seed_idx;
                let last_seed = seed_rejections[seed_rejections.len() - 1].seed_idx;
                let label = crate::startup_stats::generic_signature_label(&sig);
                log::warn!(
                    "[OUTER] {context}: generic structural bail after {run_len} consecutive \
                     identical failure signatures ({label}) on seeds {first_seed}..{last_seed}; \
                     skipping remaining {} seed(s)",
                    seeds.len().saturating_sub(seed_idx),
                );
                generic_structural_bail = Some((sig, first_seed, last_seed));
                break;
            }
        }
        obj.reset();
        if crate::estimate::outer_eval_capture::outer_gradient_fd_capture_enabled(cap.psi_dim) {
            // The audit is an OBSERVER. Its failure must not decide the fit.
            //
            // This call used to end in `?`. That made the claim in the comment
            // below false in exactly the case that matters: when the audit's own
            // `obj.eval_with_order(seed, ..)` refuses -- which is common at a
            // cold seed on a spatial basis, where the inner solve has not
            // converged at theta_0 -- the `?` propagated that error out of
            // `run_plan` and ABORTED THE WHOLE RUN. With the audit disabled the
            // identical seed would simply have been rejected, recorded in
            // `seed_rejections`, and the cascade would have moved on to the next
            // one, which may well succeed.
            //
            // So arming the audit CHANGED THE OUTCOME IT WAS MEASURING: a
            // recoverable per-seed rejection became a hard run abort, and the
            // capture window closed empty, so `take_outer_gradient_fd_capture()`
            // returned `None` and the consuming test reported "no evidence"
            // rather than the gradient disagreement it was written to detect.
            // An instrument that perturbs its subject reports about itself.
            //
            // Handled exactly like the curvature-homotopy entry error directly
            // below -- warn, reset to the pristine baseline, fall through to the
            // ordinary seed cascade -- because it is the same situation: a
            // non-feasibility-gating failure on an optional entry path.
            if let Err(err) = capture_outer_gradient_fd_at_seed(
                obj,
                config,
                context,
                seed,
                cap.theta_layout().rho_dim(),
                cap.psi_dim,
                &bounds_template.0,
                &bounds_template.1,
            ) {
                log::warn!(
                    "[OUTER] {context}: outer-gradient FD audit refused at seed {seed_idx} \
                     ({err}); the audit records nothing for this seed and the seed cascade \
                     proceeds unchanged"
                );
                obj.reset();
            }
            // The audit leaves the objective pristine, so the real seed path
            // below (including any curvature homotopy) is bit-identical to a
            // run with capture disabled -- which is now true on the refusal
            // path as well, not only on the success path.
        }
        // Certified curvature-homotopy entry leg (#1007). When the objective
        // has a certified anchor (the SAE-manifold `η = 0` Eckart-Young
        // relaxation), run the predictor-corrector `η`-walk from it INSTEAD of
        // relying on the blind multi-seed multistart: a single walk along the
        // unique optimal branch reaches the real (`η = 1`) objective, leaving
        // the inner state warm there. The min-pivot invariant + step-halving
        // make the walk certified; a degenerate anchor or a detected
        // bifurcation returns `false` (the term is left at the full basis) and
        // the seed cascade below takes over — the outcome is recorded on the
        // fit payload either way, never a silent fallback. The walk runs once
        // per accepted seed entry right after `reset`, so cross-seed state
        // hygiene is unchanged (#1003): `reset` restores the pristine `η = 1`
        // baseline before each walk.
        let curvature_entry_refused = match obj.curvature_homotopy_entry(seed) {
            Some(Ok(arrived)) => {
                log::info!(
                    "[OUTER] {context}: curvature-homotopy entry seed {seed_idx} arrived={arrived}"
                );
                !arrived
            }
            Some(Err(err)) => {
                // A hard anchor-construction failure is not a feasibility gate:
                // fall through to the ordinary seed cascade.
                log::warn!(
                    "[OUTER] {context}: curvature-homotopy entry seed {seed_idx} errored ({err}); \
                     deferring to seed cascade"
                );
                obj.reset();
                false
            }
            None => false,
        };
        if curvature_entry_refused {
            // A refused walk is NEVER a feasibility gate. By contract the walk
            // leaves the term at the full `η = 1` basis (a degenerate anchor or
            // a detected branch bifurcation), so the NORMAL seed cascade below
            // — `accept_seed_without_outer_iterations` and the direct solve at
            // `seed` — takes over from the
            // pristine cold state. Rejecting the seed here instead emptied the
            // candidate set for objectives WITHOUT a continuation path (#1095:
            // a periodic K=1 circle whose walk "buys nothing" and refuses on a
            // small-N pivot bifurcation — periodic K=1 does not advertise
            // reactive domain entry, so every one of its seeds was rejected
            // before any solver started). Reset to the baseline so the cascade
            // opens each seed from its own cold default, exactly as a hard
            // anchor-construction error already does above.
            log::info!(
                "[OUTER] {context}: curvature-homotopy entry refused seed {seed_idx}; deferring \
                 to the seed cascade from the pristine baseline"
            );
            obj.reset();
        }
        install_matching_initial_inner_seed(obj, config, seed, context)?;
        // Zero-iteration acceptance, decided HERE rather than in the objective.
        //
        // Whether a seed is already stationary is a question about the
        // stationarity BAND, and the band lives with `OuterConfig`
        // (`outer_gradient_tolerance`), not with the objective -- which is why
        // `accept_seed_without_outer_iterations` has never once returned `Some`
        // in production: every implementation of it lacks the one input the
        // decision needs, so the branch below it had never executed.
        //
        // Measured (#2363): a fit resumed from a prior fit's terminal
        // certificate is stationary where it starts. |Pg| at the resumed rho is
        // 4.225362e-9 / 6.680405e-8 / 1.369603e-7 on the three estimated-nuisance
        // fixtures, against a band of 1.61e-5 -- inside by three to four orders
        // of magnitude. It then takes one outer iteration to go nowhere, which
        // this skips along with its inner solves.
        //
        // Two things make accepting safe rather than a gamble: the branch
        // RE-CERTIFIES what it accepts through
        // `CertifiedOuterCandidate::from_solver_claim` and falls back into the
        // seed cascade when that fails, so an over-eager acceptance costs
        // nothing; and it fires only for a rho a previous outer run already
        // certified as terminal, never for a heuristic or a mid-run checkpoint.
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
            started_seeds += 1;
            let mut candidate = OuterResult::new(seed.clone(), seed_cost, 0, true, *the_plan);
            candidate.origin = OuterResultOrigin::SeedAcceptedWithoutIteration;
            match CertifiedOuterCandidate::from_solver_claim(obj, config, context, candidate) {
                Ok(candidate) => {
                    if candidate_improves_best(
                        candidate.result(),
                        best.as_ref().map(CertifiedOuterCandidate::result),
                    ) {
                        best = Some(candidate);
                    }
                    break;
                }
                Err((checkpoint, error)) => {
                    log::warn!(
                        "[OUTER] {context}: zero-iteration seed {seed_idx} claimed acceptance but \
                         failed analytic certification: {error}"
                    );
                    if tail_snap_reseed_point.is_none() {
                        tail_snap_reseed_point = checkpoint.tail_snap_reseed.clone();
                    }
                    if saddle_escape_reseed_point.is_none() {
                        saddle_escape_reseed_point = checkpoint.saddle_escape_reseed.clone();
                    }
                    retain_best_outer_checkpoint(&mut best_checkpoint, checkpoint);
                    seed_rejections.push(SeedRejection::from_estimation_error(
                        seed_idx,
                        "certificate",
                        &error,
                    ));
                    continue 'seed_attempts;
                }
            }
        }
        // Typed, reactive domain entry. The literal seed is always evaluated
        // first on the real objective. A finite value keeps the converged probe
        // handoff and pays no continuation work. Only an undefined criterion
        // activates the certified heavy-smoothing path; a hard evaluation error
        // remains a seed refusal and is never converted into a pseudo-value.
        let mut reactive_domain_entry_requested = false;
        // Set when this seed's continuation walk is answered by the recorded
        // cold-entry-leg verdict instead of being walked again (#2080).
        let mut replayed_cold_entry_refusal: Option<String> = None;
        if reactive_domain_entry_available {
            match obj.eval_cost(seed) {
                Ok(cost) if cost.is_finite() => {
                    log::debug!(
                        "[OUTER] {context}: exact seed {seed_idx} is inside the objective domain; \
                         reactive continuation entry not needed"
                    );
                }
                Ok(_) => {
                    reactive_domain_entry_requested = true;
                    if let Some(recorded) = cold_entry_leg_refusal.as_ref() {
                        // The cold entry leg already refused on an earlier seed,
                        // and that leg's `(rho, scalars, warm start)` do not
                        // depend on the seed (see `cold_entry_leg_refusal`). The
                        // walk cannot reach a later, seed-dependent leg without
                        // clearing this one, so re-running it would reproduce the
                        // recorded verdict digit for digit.
                        log::info!(
                            "[OUTER] {context}: exact seed {seed_idx} has undefined criterion, but \
                             the seed-independent cold entry leg has already refused; replaying \
                             that verdict instead of re-walking it"
                        );
                        replayed_cold_entry_refusal = Some(format!(
                            "{recorded} (replayed: the continuation's cold entry leg is evaluated \
                             at the legal upper box with the objective's entry scalars and an \
                             empty warm start, none of which depend on the seed)"
                        ));
                    } else {
                        log::info!(
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
                }
                Err(err) => {
                    log::warn!(
                        "[OUTER] {context}: rejecting seed {seed_idx}: reactive domain-entry \
                         seed probe failed before continuation: {err}"
                    );
                    seed_rejections.push(SeedRejection::from_estimation_error(
                        seed_idx,
                        "domain-entry",
                        &err,
                    ));
                    continue 'seed_attempts;
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
        let mut continuation_arrived =
            continuation_path.is_none() && replayed_cold_entry_refusal.is_none();
        let mut continuation_arrival_refusal: Option<String> = replayed_cold_entry_refusal.take();
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
                            let msg = format!(
                                "reactive domain entry refused before exact-target arrival: {err}"
                            );
                            if legs_descended == 0 {
                                // Nothing has been accepted yet, so this is the
                                // COLD ENTRY leg — the one evaluated at the legal
                                // upper box with the contract's entry scalars and
                                // an empty warm start. Record it so the remaining
                                // seeds replay the verdict rather than recompute
                                // a constant of the problem (#2080).
                                cold_entry_leg_refusal = Some(msg.clone());
                            }
                            continuation_arrival_refusal = Some(msg);
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
                            log::info!(
                                "[OUTER] {context}: continuation seed {seed_idx} coupled \
                                 waypoint struggled below accepted s={s:.4} ({}); refining the \
                                 next attempted distance",
                                failure.message(),
                            );
                        }
                    }
                }
                log::info!(
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
                log::warn!("[OUTER] {context}: rejecting seed {seed_idx}: {msg}");
                seed_rejections.push(SeedRejection::from_message_with_producer_verdict(
                    seed_idx,
                    "domain-entry",
                    msg,
                    false,
                ));
                continue 'seed_attempts;
            }
            // Independently re-evaluate the literal target and require a finite
            // exact criterion before any optimizer can start.
            match obj.eval_cost(seed) {
                Ok(cost) if cost.is_finite() => {
                    log::info!(
                        "[OUTER] {context}: reactive continuation seed {seed_idx} arrived with \
                         finite exact criterion {cost:.6e}"
                    );
                }
                Ok(_) => {
                    let msg = "reactive domain entry refused: exact seed criterion remained \
                               non-finite after certified continuation arrival"
                        .to_string();
                    log::warn!("[OUTER] {context}: rejecting seed {seed_idx}: {msg}");
                    seed_rejections.push(SeedRejection::from_message_with_producer_verdict(
                        seed_idx,
                        "domain-entry",
                        msg,
                        false,
                    ));
                    continue 'seed_attempts;
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
        let seed_slot;
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
                        log::warn!(
                            "[OUTER] {context}: rejecting seed {seed_idx} before solver start: {err}"
                        );
                        seed_rejections.push(SeedRejection::from_objective_error(
                            seed_idx,
                            "validation",
                            &err,
                        ));
                        continue 'seed_attempts;
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
                        log::warn!(
                            "[OUTER] {context}: rejecting seed {seed_idx} before solver start: {err}"
                        );
                        seed_rejections.push(SeedRejection::from_objective_error(
                            seed_idx,
                            "validation",
                            &err,
                        ));
                        continue 'seed_attempts;
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
                        log::warn!(
                            "[OUTER] {context}: rejecting seed {seed_idx} before solver start: {err}"
                        );
                        seed_rejections.push(SeedRejection::from_objective_error(
                            seed_idx,
                            "validation",
                            &err,
                        ));
                        continue 'seed_attempts;
                    }
                    return Err(EstimationError::fatal_objective_evaluation(
                        "outer ARC second-order seed validation",
                        err,
                    ));
                }
                started_seeds += 1;
                seed_slot = started_seeds;

                let cheap_materializable_operator = matches!(
                    seed_eval.hessian,
                    HessianValue::Operator(ref op)
                        if op.materialization().is_available()
                            && op.dim() <= OUTER_HVP_MATERIALIZE_MAX_DIM
                );
                if cheap_materializable_operator {
                    // The operator's own work model says probing every column
                    // is cheap; convert the seed Hessian to dense in-place.
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
                                log::warn!(
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
                                continue 'seed_attempts;
                            }
                        }
                    }
                }
                if matches!(seed_eval.hessian, HessianValue::Operator(_)) {
                    log::debug!(
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

                    let bridge_obj = OuterOperatorBridge {
                        obj,
                        layout,
                        outer_inner_cap: config.outer_inner_cap.clone(),
                        eval_count: 0,
                        g_norm_initial: None,
                        last_g_norm: None,
                        last_value_grad_rho: None,
                    };

                    let mut solver = MatrixFreeTrustRegion::new(seed.clone(), bridge_obj)
                        .with_bounds(bounds_obj)
                        .with_gradient_tolerance(grad_tol)
                        .with_max_iterations(max_iter)
                        .with_initial_sample(seed.clone(), initial_op_sample)
                        // Looser Eisenstat–Walker forcing factor on the
                        // inner Steihaug–Toint CG (default 0.1 → 0.5). The
                        // matrix-free route is reached only after
                        // `prefer_outer_hessian_operator` says Hv is
                        // expensive (large k, n·p crossover, or wide
                        // basis), which is exactly the regime where the
                        // standard inexact-Newton-Krylov 0.5 forcing
                        // factor wins: one extra outer-TR iter is cheap
                        // versus halving the number of inner Hv applies
                        // per outer iter. At large-scale shape (n=300 K,
                        // ~64 outer-TR iters × ~30 trace_logdet calls per
                        // Hv) this halves the dominant per-fit work.
                        .with_cg_tolerance(0.5)
                        // The matrix-free route is exclusively for
                        // exact analytic Hessians; an `Unavailable`
                        // here is a routing/contract violation.
                        .with_hessian_fallback_policy(HessianFallbackPolicy::Error);
                    if let Some(feedback) = config.outer_inner_cap.as_ref() {
                        solver = solver.with_observer(OuterAcceptObserver {
                            feedback: Some(feedback.clone()),
                            accepted_steps: None,
                        });
                    }
                    if let Some(r) = sanitized_operator_trust_restart_radius(
                        config.operator_initial_trust_radius,
                    ) {
                        solver = solver.with_initial_trust_radius(r);
                    }

                    let mf_start = std::time::Instant::now();
                    let report = solver.run_report();
                    let mf_elapsed = mf_start.elapsed().as_secs_f64();
                    let final_radius = report.diagnostics.final_trust_radius;
                    log::info!(
                        "[OUTER summary] matrix-free TR finished status={:?} in {} iters \
                         elapsed={:.3}s final_value={:.6e} final_trust_radius={}",
                        report.status,
                        report.solution.iterations,
                        mf_elapsed,
                        report.solution.final_value,
                        match final_radius {
                            Some(r) => format!("{:.3e}", r),
                            None => "n/a".to_string(),
                        },
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
                            log::warn!(
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
                            log::warn!(
                                "[OUTER warning] {context}: matrix-free TR reached trust-radius reject floor at final_value={:.6e} |g|={:.3e} final_trust_radius={}",
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
                            log::warn!(
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
                        OptimizationStatus::ObjectiveFailed
                            => Err(EstimationError::fatal_outer_evaluation(
                                "matrix-free trust-region evaluation",
                                EstimationError::RemlOptimizationFailed(
                                    "matrix-free trust-region objective evaluation failed"
                                        .to_string(),
                                ),
                            )),
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
                    // `grad_tol.abs` IS the whole band since #2613: the
                    // cost-relative component is no longer anchored on a
                    // trajectory point, so there is nothing left for
                    // `threshold(seed_cost, ‖g₀‖)` to resolve. Using the field
                    // directly keeps this site from reading as if the seed
                    // still decided the guard's stationarity gate.
                    let cost_stall_grad_threshold =
                        grad_tol.abs.max(COST_STALL_PROJECTED_GRAD_FLOOR);

                    // Build the exact seed Hessian before enrolling the seed in
                    // the stall guard. The guard must know whether its incumbent
                    // is a second-order point: repeated infeasible trials cannot
                    // justify halting at a certified strict saddle.
                    let seed_hessian = build_bridge_hessian_for_source(
                        hessian_source,
                        seed_eval.hessian,
                        OUTER_HVP_MATERIALIZE_MAX_DIM,
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
                    let seed_hessian_psd = seed_hessian.as_ref().and_then(|dense| {
                        reduced_hessian_psd_at_point(
                            &seed,
                            &seed_eval.gradient,
                            dense,
                            Some((&seed_rail_bounds.0, &seed_rail_bounds.1)),
                        )
                    });

                    let mut cost_stall_guard = CostStallGuard::new(
                        cost_stall_rel_tol,
                        ARC_COST_STALL_WINDOW,
                        cost_stall_grad_threshold,
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
                        materialize_operator_max_dim: OUTER_HVP_MATERIALIZE_MAX_DIM,
                        eval_count: 0,
                        outer_inner_cap: config.outer_inner_cap.clone(),
                        g_norm_initial: None,
                        last_g_norm: None,
                        last_value_grad_rho: None,
                        cost_stall: Some(cost_stall_guard),
                        cost_stall_bounds: Some((lo.clone(), hi.clone())),
                        },
                        Arc::clone(&last_objective_error),
                    );

                    let initial_sample = SecondOrderSample {
                        value: seed_eval.cost,
                        gradient: seed_eval.gradient,
                        hessian: seed_hessian,
                    };

                    let mut optimizer = ArcOptimizer::new(seed.clone(), objective)
                        .with_bounds(bounds)
                        .with_gradient_tolerance(grad_tol)
                        .with_max_iterations(max_iter)
                        .with_initial_sample(seed.clone(), initial_sample);
                    if let Some(sigma) = config.arc_initial_regularization {
                        optimizer = optimizer.with_initial_regularization(sigma);
                    }
                    if let Some(feedback) = config.outer_inner_cap.as_ref() {
                        optimizer = optimizer.with_observer(OuterAcceptObserver {
                            feedback: Some(feedback.clone()),
                            accepted_steps: None,
                        });
                    }
                    // On the exact-Hessian ARC route, forbid both (a)
                    // finite-difference Hessian estimation if the
                    // objective ever returns
                    // `SecondOrderSample { hessian: None }` and (b)
                    // `opt`'s internal AutoBfgs demotion on step
                    // failure. `HessianFallbackPolicy::Error` plus
                    // `FallbackPolicy::Never` is the precise
                    // expression of "stay inside analytic-Hessian
                    // geometry; surface mismatches loudly". opt 0.3.0
                    // API; previously this was approximated by the
                    // coarse `Profile::Deterministic` knob (which also
                    // tightens unrelated `eta_accept` / history caps).
                    if matches!(hessian_source, HessianSource::Analytic) {
                        optimizer = optimizer
                            .with_hessian_fallback_policy(HessianFallbackPolicy::Error)
                            .with_fallback_policy(OptFallbackPolicy::Never);
                    }
                    match optimizer.run() {
                        Ok(sol) => Ok(solution_into_outer_result(sol, true, *the_plan)),
                        Err(ArcError::MaxIterationsReached { last_solution, .. }) => {
                            log::warn!(
                                "[OUTER warning] {context}: ARC hit max_iter={} at final_value={:.6e} |g|={:.3e}",
                                config.max_iter,
                                last_solution.final_value,
                                last_solution.final_gradient_norm.unwrap_or(f64::NAN),
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
                            // The best-feasible-iterate substitution must produce
                            // THIS seed's `result` (an expression that feeds the
                            // multi-start keep-best below), NOT short-circuit the
                            // whole function with a bare `return`. A bare `return`
                            // here discards any CONVERGED fit an earlier seed already
                            // stored in `best`: on a #1476 concurvity double-penalty
                            // surface the flexible slot-0 seed converges to the
                            // genuine interior optimum (cost ~133), then the promoted
                            // heavy slot-1 seed (#1426) budget-exhausts on the
                            // null-space annihilation shelf and its best-feasible
                            // iterate is a degenerate box corner with a SPURIOUSLY
                            // LOWER cached cost (~65, projected |g| ≫ tol — an invalid
                            // REML the line search could not improve). Returning it
                            // directly shipped that corner (edf_total→1, the supported
                            // smooth annihilated) even though keep-best already held
                            // the converged optimum. Flowing it through keep-best as a
                            // NON-converged candidate lets `candidate_improves_best`
                            // reject it (a converged best always beats a non-converged
                            // candidate). When this seed is the ONLY one (the original
                            // single-start #1371 case) `best` is still None, so
                            // keep-best adopts it unchanged — that behavior is
                            // preserved byte-for-byte.
                            match best_exit {
                                Some(best)
                                    if best.value.is_finite()
                                        && (!last_solution.final_value.is_finite()
                                            || best.value < last_solution.final_value) =>
                                {
                                    log::warn!(
                                        "[OUTER] {context}: ARC budget-exhaustion last iterate \
                                         (value={:.6e}) is worse than the best feasible iterate \
                                         seen (value={:.6e}); substituting the best iterate so a \
                                         degenerate box-corner does not over-shrink a supported \
                                         penalty direction (#1371). The substituted iterate flows \
                                         through the multi-start keep-best as a non-converged \
                                         candidate so an earlier converged seed still wins (#1476).",
                                        last_solution.final_value,
                                        best.value,
                                    );
                                    let mut result = outer_result_with_gradient_norm(
                                        best.rho,
                                        best.value,
                                        best.iterations,
                                        Some(best.grad_norm),
                                        false,
                                        *the_plan,
                                    );
                                    result.origin =
                                        OuterResultOrigin::ArcBestIterateSubstitution;
                                    Ok(result)
                                }
                                _ => {
                                    Ok(solution_into_outer_result(*last_solution, false, *the_plan))
                                }
                            }
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
                                        exit.iterations,
                                        Some(exit.grad_norm),
                                        false,
                                        *the_plan,
                                    );
                                    result.origin =
                                        OuterResultOrigin::ArcInfeasibleStallCheckpoint;
                                    // #2241 — carry the guard's measured probe-
                                    // noise-floor bound so the final analytic
                                    // certificate honors the same flat band the
                                    // guard certified in the loop.
                                    result.flat_noise_grad_bound = exit.noise_grad_bound;
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
                        Err(ArcError::ObjectiveFailed { message }) => {
                            let error = last_objective_error
                                .lock()
                                .expect("ARC objective error publication lock poisoned")
                                .take()
                                .expect(
                                    "ArcError::ObjectiveFailed must follow a failed classified objective evaluation",
                                );
                            assert_eq!(
                                error.message(),
                                message,
                                "ARC returned a different objective error than the bridge published"
                            );
                            Err(EstimationError::fatal_objective_evaluation(
                                "outer ARC evaluation",
                                error,
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
                            log::warn!(
                                "[OUTER] {context}: rejecting seed {seed_idx} before device-BFGS start: {err}"
                            );
                            seed_rejections.push(SeedRejection::from_objective_error(
                                seed_idx,
                                "validation",
                                &err,
                            ));
                            continue 'seed_attempts;
                        }
                        Err(err) => {
                            return Err(EstimationError::fatal_objective_evaluation(
                                "outer device-BFGS seed evaluation",
                                err,
                            ));
                        }
                    };
                    started_seeds += 1;
                    seed_slot = started_seeds;
                    let device_input = crate::gpu::reml_outer::RemlOuterGpuInput {
                        seed_rho: seed.clone(),
                        bounds: bounds_dev,
                        gradient_tolerance: grad_tol_dev,
                        max_iterations: config.max_iter,
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
                    // `seed_slot` is the per-seed index assigned above; it is
                    // consumed only by the host-BFGS logging summary, which
                    // the device-resident branch replaces with its own
                    // device-BFGS summary log below.
                    if seed_slot == 0 {
                        log::debug!(
                            "[OUTER] {context}: device-BFGS seed_slot underflow at seed {seed_idx}"
                        );
                    }
                    match device_outcome {
                        Ok(outcome) => {
                            log::info!(
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
                            log::warn!(
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
                                    continue 'seed_attempts;
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
                                    continue 'seed_attempts;
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
                            log::warn!(
                                "[OUTER] {context}: rejecting seed {seed_idx} before solver start: {err}"
                            );
                            seed_rejections.push(SeedRejection::from_objective_error(
                                seed_idx,
                                "validation",
                                &err,
                            ));
                            continue 'seed_attempts;
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
                            log::warn!(
                                "[OUTER] {context}: rejecting seed {seed_idx} before solver start: {err}"
                            );
                            seed_rejections.push(SeedRejection::from_objective_error(
                                seed_idx,
                                "validation",
                                &err,
                            ));
                            continue 'seed_attempts;
                        }
                        Err(err) => {
                            return Err(EstimationError::fatal_objective_evaluation(
                                "outer BFGS seed validation",
                                err,
                            ));
                        }
                    };
                    started_seeds += 1;
                    seed_slot = started_seeds;
                    // The seed a BFGS run is handed, and the (cost, gradient)
                    // it is handed WITH it. `with_initial_sample` below means
                    // `opt::Bfgs` never re-evaluates here, so a zero gradient
                    // in this sample is a zero-iteration "convergence" at
                    // whatever rho this seed happens to be.
                    log::info!(
                        "[OUTER] {context}: BFGS seed {seed_idx} (slot {seed_slot}) cost={:.6e} \
                         |g|={:.6e} rho={:?}",
                        seed_eval.cost,
                        seed_eval.gradient.iter().map(|g| g * g).sum::<f64>().sqrt(),
                        seed.iter().map(|r| (r * 1e6).round() / 1e6).collect::<Vec<_>>(),
                    );
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
                    // Stationarity gate for the cost-stall exit. Convergence must
                    // mean stationarity, not cost-flatness: a cost stall only
                    // counts as a converged optimum when the projected gradient
                    // norm at the best iterate clears the SAME outer gradient
                    // tolerance the genuine BFGS convergence path uses, with
                    // the same practical floor the ARC guard uses for
                    // bound-pinned separation fits.
                    let seed_grad_norm =
                        seed_eval.gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
                    // `grad_tol.abs` IS the whole band since #2613: the
                    // cost-relative component is no longer anchored on a
                    // trajectory point, so there is nothing left for
                    // `threshold(seed_cost, ‖g₀‖)` to resolve. Using the field
                    // directly keeps this site from reading as if the seed
                    // still decided the guard's stationarity gate.
                    let cost_stall_grad_threshold =
                        grad_tol.abs.max(COST_STALL_PROJECTED_GRAD_FLOOR);
                    let mut cost_stall_guard = CostStallGuard::new(
                        cost_stall_rel_tol,
                        COST_STALL_WINDOW,
                        cost_stall_grad_threshold,
                        cost_stall_exit.clone(),
                    );
                    cost_stall_guard.observe_seed(seed, seed_eval.cost, seed_grad_norm);
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
                            incumbent: Some((seed.clone(), seed_eval.cost)),
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
                        value: seed_eval.cost,
                        gradient: seed_eval.gradient.clone(),
                    };
                    let mut optimizer = Bfgs::new(seed.clone(), objective)
                        .with_initial_sample(seed.clone(), initial_sample)
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
                        .is_some_and(|initial| outer_theta_bitwise_eq(initial, seed));
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
                            .and_then(|h| {
                                match gam_linalg::utils::certified_spd_inverse(
                                    h,
                                    "transferred outer-Hessian BFGS metric",
                                ) {
                                    Ok(inverse) => Some(inverse.into_inverse()),
                                    Err(error) => {
                                        log::info!(
                                            "[OUTER] {context}: rejected transferred BFGS metric: {error}"
                                        );
                                        None
                                    }
                                }
                            });
                        if let Some(h_inv) = dense_metric {
                            log::info!(
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
                        let g0_norm = seed_eval.gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
                        if g0_norm.is_finite() && g0_norm > 0.0 {
                            let scale = (1.0 / g0_norm).clamp(1.0e-3, 1.0e3);
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
                    });
                    let bfgs_start = std::time::Instant::now();
                    let outcome = optimizer.run();
                    let bfgs_elapsed = bfgs_start.elapsed().as_secs_f64();
                    match &outcome {
                        Ok(sol) => log::info!(
                            "[OUTER summary] BFGS converged in {} iters elapsed={:.3}s final_value={:.6e}",
                            sol.iterations,
                            bfgs_elapsed,
                            sol.final_value
                        ),
                        Err(BfgsError::MaxIterationsReached { last_solution }) => log::warn!(
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
                        }) => log::info!(
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
                        Err(e) => log::info!(
                            "[OUTER summary] BFGS failed elapsed={:.3}s err={:?}",
                            bfgs_elapsed,
                            e
                        ),
                    }
                    match outcome {
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
                                    // #2241 — carry the guard's measured probe-
                                    // noise-floor bound so the final analytic
                                    // certificate honors the same flat band the
                                    // guard certified in the loop.
                                    result.flat_noise_grad_bound = exit.noise_grad_bound;
                                    result.cost_stall_probe_scale = exit.probe_scale;
                                    // Preserve HOW BFGS stopped even when the
                                    // guard already certified the stalled score
                                    // surface (mirrors the ARC branch above).
                                    // The mandatory final analytic certificate
                                    // uses this provenance to apply the same
                                    // score-relative flat-valley band as the
                                    // guard; gating the marker on
                                    // `!exit.converged` made the final pass
                                    // silently revert to the much tighter raw
                                    // solver bound and reject the identical
                                    // point the guard certified (#1689 in ARC;
                                    // reproduced live on the BFGS route by the
                                    // GPT-2 E1 structured pass: guard accepted
                                    // |g|=4.97e-1 under the flat band on a
                                    // score of 2.7e3, certificate refused at
                                    // its raw 4.4e-2 bound and the fit died
                                    // with RemlConvergenceError).
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
                            let error = last_objective_error
                                .lock()
                                .expect("BFGS objective error publication lock poisoned")
                                .take()
                                .expect(
                                    "BfgsError::ObjectiveFailed must follow a failed classified objective evaluation",
                                );
                            assert_eq!(
                                error.message(),
                                message,
                                "BFGS returned a different objective error than the bridge published"
                            );
                            Err(EstimationError::fatal_objective_evaluation(
                                "outer BFGS evaluation",
                                error,
                            ))
                        }
                        Err(e) => Err(EstimationError::RemlOptimizationFailed(format!(
                            "BFGS solver failed: {e:?}"
                        ))),
                    }
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
                        seed_slot = started_seeds;
                        Ok(result)
                    }
                    Err(FixedPointOuterRunError::SeedRejected(err)) => {
                        log::warn!(
                            "[OUTER] {context}: rejecting seed {seed_idx} before solver start: {err}"
                        );
                        seed_rejections.push(SeedRejection::from_objective_error(
                            seed_idx,
                            "validation",
                            &err,
                        ));
                        continue 'seed_attempts;
                    }
                    Err(FixedPointOuterRunError::IterationRejected(request)) => {
                        log::warn!(
                            "[OUTER] {context}: EFS trial refused after {} finite iteration(s) \
                             at cost={:.6e}; continuing the exact incumbent with the \
                             analytic-gradient fallback: {}",
                            request.checkpoint.iterations,
                            request.checkpoint.sample.value,
                            request.refusal,
                        );
                        return Ok(PlanRunOutcome::FixedPointContinuationRequested(request));
                    }
                    Err(FixedPointOuterRunError::ImmediateFallback(request)) => {
                        return Ok(PlanRunOutcome::FirstOrderFallbackRequested(request));
                    }
                    Err(FixedPointOuterRunError::Failed(err)) => {
                        started_seeds += 1;
                        seed_slot = started_seeds;
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
                        seed_slot = started_seeds;
                        Ok(result)
                    }
                    Err(FixedPointOuterRunError::SeedRejected(err)) => {
                        log::warn!(
                            "[OUTER] {context}: rejecting seed {seed_idx} before solver start: {err}"
                        );
                        seed_rejections.push(SeedRejection::from_objective_error(
                            seed_idx,
                            "validation",
                            &err,
                        ));
                        continue 'seed_attempts;
                    }
                    Err(FixedPointOuterRunError::IterationRejected(request)) => {
                        log::warn!(
                            "[OUTER] {context}: HybridEFS trial refused after {} finite \
                             iteration(s) at cost={:.6e}; continuing the exact incumbent \
                             with the analytic-gradient fallback: {}",
                            request.checkpoint.iterations,
                            request.checkpoint.sample.value,
                            request.refusal,
                        );
                        return Ok(PlanRunOutcome::FixedPointContinuationRequested(request));
                    }
                    Err(FixedPointOuterRunError::ImmediateFallback(request)) => {
                        return Ok(PlanRunOutcome::FirstOrderFallbackRequested(request));
                    }
                    Err(FixedPointOuterRunError::Failed(err)) => {
                        started_seeds += 1;
                        seed_slot = started_seeds;
                        Err(err)
                    }
                }
            }
        };

        let seed_elapsed = t_seed_start.elapsed().as_secs_f64();
        match result {
            Ok(candidate) => {
                log::debug!(
                    "[outer-timing] seed {}/{} ({:?}): {:.3}s  cost={:.6e}  converged={}",
                    seed_slot,
                    seed_budget,
                    the_plan.solver,
                    seed_elapsed,
                    candidate.final_value,
                    candidate.solver_claimed_convergence(),
                );
                if !candidate.solver_claimed_convergence() {
                    retain_best_outer_checkpoint(&mut best_checkpoint, candidate);
                    // An exhausted iterate is resumable work, not a fit
                    // candidate. Continue the declared multistart budget in
                    // search of a stationary seed; it may never populate or
                    // short-circuit the certified winner slot.
                    continue 'seed_attempts;
                }
                let candidate = match CertifiedOuterCandidate::from_solver_claim(
                    obj, config, context, candidate,
                ) {
                    Ok(candidate) => candidate,
                    Err((checkpoint, error)) => {
                        log::warn!(
                            "[OUTER] {context}: seed {seed_idx} solver convergence claim failed \
                             analytic certification: {error}; retaining only a resume checkpoint"
                        );
                        if tail_snap_reseed_point.is_none() {
                            tail_snap_reseed_point = checkpoint.tail_snap_reseed.clone();
                        }
                        if saddle_escape_reseed_point.is_none() {
                            saddle_escape_reseed_point = checkpoint.saddle_escape_reseed.clone();
                        }
                        retain_best_outer_checkpoint(
                            &mut best_refused_optimum,
                            checkpoint.clone(),
                        );
                        retain_best_outer_checkpoint(&mut best_checkpoint, checkpoint);
                        seed_rejections.push(SeedRejection::from_estimation_error(
                            seed_idx,
                            "certificate",
                            &error,
                        ));
                        continue 'seed_attempts;
                    }
                };
                // #1373: for GLM/survival models the seed screening deliberately
                // places the most-flexible (low-lambda) seed at slot 0 and the
                // heaviest interior (high-lambda) seed at slot 1 so the budget-2
                // multi-start straddles both basins. The flexible basin can
                // converge to a LAML that is epsilon better while overshooting
                // on the response scale. Break that near-tie toward the
                // more-smoothed basin for those families only. Gaussian
                // location-scale needs the same promoted seed order, but keeps
                // Gaussian's plain lowest-cost keep-best policy.
                let parsimonious_keep_best = config
                    .seed_config
                    .risk_profile
                    .uses_parsimonious_keep_best();
                let candidate_improved = if parsimonious_keep_best {
                    candidate_improves_best_parsimonious(
                        candidate.result(),
                        best.as_ref().map(CertifiedOuterCandidate::result),
                        rho_dim,
                    )
                } else {
                    candidate_improves_best(
                        candidate.result(),
                        best.as_ref().map(CertifiedOuterCandidate::result),
                    )
                };
                if candidate_improved {
                    best = Some(candidate);
                }
                let quality_compare_remaining_gaussian_seeds =
                    config.seed_config.risk_profile.uses_lowest_cost_keep_best()
                        && seed_budget > 1
                        && started_seeds < seed_budget;
                // #1373: do not let the first-converged flexible seed (slot 0)
                // short-circuit the multi-start before the deliberately-promoted
                // parsimonious seed (slot 1) has been solved. Without this, the
                // converged break below fires on slot 0 and the heavy basin that
                // the screening order placed at slot 1 — precisely to let
                // keep-best reject an overshoot — is never evaluated. This gate
                // waits for exactly those two comparison roles. Any larger budget
                // exists to recover from failed candidates; it must not launch a
                // third expensive solve after both basins already certified.
                //
                // #1575: but the heavy seed is only ever DECISIVE when slot 0
                // could be beaten (an under-penalized overshoot, a flat-valley
                // near-tie, or a non-converged stall). When slot 0 instead
                // converged to a curvature-pinned, well-penalized optimum (every
                // smoothing λ ≥ 1, residual gradient 100× inside the parsimony tie
                // band), the heavy seed merely re-derives the identical cost/ρ —
                // doubling the binomial/survival outer cost-eval count for
                // nothing. Waive the await in exactly that redundant case; every
                // overshoot/stall/flat-valley path keeps the full guard.
                let promoted_seed_is_redundant = best
                    .as_ref()
                    .is_some_and(|b| parsimony_second_seed_is_redundant(b.result(), rho_dim));
                let non_gaussian_await_parsimony_seed = parsimonious_keep_best
                    && should_await_promoted_parsimony_seed(
                        seed_budget,
                        started_seeds,
                        promoted_seed_is_redundant,
                    );
                if best.is_some()
                    && !quality_compare_remaining_gaussian_seeds
                    && !non_gaussian_await_parsimony_seed
                {
                    break;
                }
            }
            Err(e) => {
                if e.is_fatal_outer_evaluation() {
                    return Err(e);
                }
                log::debug!(
                    "[outer-timing] seed {}/{} ({:?}): {:.3}s  FAILED: {}",
                    seed_slot,
                    seed_budget,
                    the_plan.solver,
                    seed_elapsed,
                    e,
                );
                seed_rejections.push(SeedRejection::from_estimation_error(seed_idx, "solver", &e));
            }
        }
    }

    // #2596 — criterion inversion between the published winner and a refused
    // optimum.
    //
    // A certificate answers "is this ρ stationary?". It does not answer "is
    // this the best ρ we found?". Those come apart at a ρ-box corner: the
    // box-KKT projection zeroes the outward half of the gradient, so a corner
    // certifies with |Pg| identically zero NO MATTER what its criterion says,
    // while an interior optimum has to earn its certificate against a real
    // residual gradient. When the interior optimum's certificate is refused,
    // the corner is the only survivor and is published silently — on #2596 at
    // an outer criterion 26× worse than the interior point the same run had
    // already measured.
    //
    // The screening/mint standard split that produced that particular refusal
    // is fixed at its root in `run.rs` (a screening pass may no longer refuse
    // on a bound the mint would widen). This is the second, independent half:
    // whatever the reason, an inversion of this kind is a fact about the search
    // that must reach the log rather than be absorbed. It changes no verdict —
    // deliberately, because the refused point is by definition uncertified and
    // publishing it would substitute one silent decision for another — but it
    // makes the fallback legible, and it is what turns "the smooth vanished"
    // into "the winner scored 110.94 against a refused 4.19 at ρ=(0.38,−4.98)".
    if let (Some(certified), Some(refused)) = (best.as_ref(), best_refused_optimum.as_ref()) {
        let winner = certified.result();
        let scale = winner
            .final_value
            .abs()
            .max(refused.final_value.abs())
            .max(1.0);
        let inverted_by = winner.final_value - refused.final_value;
        if winner.final_value.is_finite()
            && refused.final_value.is_finite()
            && inverted_by > PARSIMONY_TIE_REL_BAND * scale
        {
            log::warn!(
                "[OUTER] {context}: the published optimum scores WORSE than a refused one — \
                 winner rho={:?} cost={:.6e} (certified), refused rho={:?} cost={:.6e} \
                 (gap {:.3e}, {:.1}× the {:.1e} relative tie band). A ρ-box corner certifies \
                 with |Pg| ≡ 0 whatever its criterion says, so a refused interior optimum \
                 leaves the corner as the only survivor; the published λ is a fallback, not \
                 the best λ this search found (#2596).",
                winner.rho.to_vec(),
                winner.final_value,
                refused.rho.to_vec(),
                refused.final_value,
                inverted_by,
                inverted_by / (PARSIMONY_TIE_REL_BAND * scale),
                PARSIMONY_TIE_REL_BAND,
            );
        }
    }

    if let Some(certified) = best {
        let mut result = certified.into_result();
        result.refused_seed_points =
            certificate_refused_seed_points(&seed_rejections, &seeds);
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
            // Certification may have happened before later multistart trials.
            // Clear every search-state cache before installing the selected
            // point so a rho-only hit cannot leave the objective owning the
            // last rejected trial's inner mode.
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
        log::info!(
            "[OUTER] {context}: retrying once from the confirmed-tail snapped \
             reseed {reseed} (#2348 Inc 2b)"
        );
        let mut retry_config = config.clone();
        retry_config.initial_rho = Some(reseed);
        retry_config.screen_initial_rho = false;
        retry_config.seed_config.max_seeds = 1;
        retry_config.seed_config.seed_budget = 1;
        obj.reset();
        match run_outer_with_plan(obj, &retry_config, context, cap, the_plan, false) {
            Ok(outcome) => return Ok(outcome),
            Err(retry_error) => {
                log::warn!(
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
        log::info!(
            "[OUTER] {context}: retrying once from the negative-curvature saddle-escape \
             reseed {reseed} (#2357)"
        );
        let mut retry_config = config.clone();
        retry_config.initial_rho = Some(reseed);
        retry_config.screen_initial_rho = false;
        retry_config.seed_config.max_seeds = 1;
        retry_config.seed_config.seed_budget = 1;
        obj.reset();
        match run_outer_with_plan(obj, &retry_config, context, cap, the_plan, false) {
            Ok(outcome) => return Ok(outcome),
            Err(retry_error) => {
                log::warn!(
                    "[OUTER] {context}: saddle-escape reseed retry failed ({retry_error}); \
                     falling through to the original exhaustion accounting"
                );
            }
        }
    }

    if let Some(mut checkpoint) = best_checkpoint {
        checkpoint.refused_seed_points =
            certificate_refused_seed_points(&seed_rejections, &seeds);
        return Ok(PlanRunOutcome::Exhausted(checkpoint));
    }

    Err({
        // `screened` reflects how many seeds we actually iterated. With
        // the current cheap-screen pipeline (rank_seeds_with_screening
        // runs upstream), screened equals the size of the consumed
        // candidate list. `exact_validated` counts every seed that
        // attempted a full eval — i.e. either reached the rejection
        // sites in this loop or made it into `started_seeds`.
        let n_generated = seeds.len();
        let n_screened = n_generated;
        let n_exact_validated = seed_rejections.len() + started_seeds;
        let stats = StartupStats::from_rejections(
            n_generated,
            n_screened,
            n_exact_validated,
            started_seeds,
            &seed_rejections,
        );
        let structural = structural_early_exit_key
            .clone()
            .or_else(|| uniform_structural_key(&seed_rejections, 1));
        let early_exit_note = if structural_early_exit_key.is_some() {
            "early-exit triggered: every observed seed reported the same structural rejection"
                .to_string()
        } else if let Some((sig, first_seed, last_seed)) = generic_structural_bail.as_ref() {
            let label = crate::startup_stats::generic_signature_label(sig);
            let skipped = seeds.len().saturating_sub(*last_seed + 1);
            format!(
                "structural: {label} on seeds {first_seed}..{last_seed}; \
                 remaining {skipped} seeds skipped"
            )
        } else {
            String::new()
        };
        if started_seeds == 0 {
            EstimationError::RemlOptimizationFailed(format_no_seeds_passed(
                context,
                &stats,
                &seed_rejections,
                structural.as_ref(),
                &early_exit_note,
            ))
        } else {
            // Mixed outcome: at least one seed started the outer
            // optimiser but none converged. Keep the structured payload
            // so the caller sees both the started_seeds count and the
            // per-rejection breakdown.
            let header = format!(
                "all {started_seeds} seed candidates failed ({context}); \
                 generated={}, screened={}, exact_validated={}, solver_started={}",
                stats.generated, stats.screened, stats.exact_validated, stats.solver_started,
            );
            let body = format_no_seeds_passed(
                context,
                &stats,
                &seed_rejections,
                structural.as_ref(),
                &early_exit_note,
            );
            EstimationError::RemlOptimizationFailed(format!("{header}\n{body}"))
        }
    })
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

/// Is `seed` a prior fit's terminal certificate that is STILL stationary here?
///
/// `Some(cost)` only when all of: the seed is the resumed rho itself; a first
/// order evaluation succeeds and is finite; and the rail-projected gradient sits
/// inside the band the outer certificate demands. Anything else is `None` and
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
    let projected = rail_projected_gradient_norm(seed, &eval.gradient, Some(bounds_template));
    let band = outer_gradient_tolerance(config).threshold(eval.cost, projected);
    if projected > band {
        log::debug!(
            "[OUTER] {context}: resumed terminal certificate seed {seed_idx} is not stationary \
             here (|Pg|={projected:.6e} > band {band:.6e}); running the ordinary cascade"
        );
        return None;
    }
    log::info!(
        "[OUTER] {context}: seed {seed_idx} is a prior fit's terminal certificate and is still \
         stationary (|Pg|={projected:.6e} <= band {band:.6e}); accepting with zero outer iterations"
    );
    Some(eval.cost)
}
