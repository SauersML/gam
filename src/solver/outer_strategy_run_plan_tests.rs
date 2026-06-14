
const EXPENSIVE_PREWARM_COEFF_DIM: usize = 24;
const EXPENSIVE_PREWARM_RHO_DIM: usize = 4;
const MULTI_SEED_PREWARM_BUDGET: usize = 8;
const SINGLE_EXPENSIVE_PREWARM_BUDGET: usize = 16;

fn continuation_prewarm_step_budget(
    config: &OuterConfig,
    cap: &OuterCapability,
    seed_count: usize,
    seed_budget: usize,
) -> usize {
    let default_budget = crate::solver::estimate::reml::continuation::PATH_BUDGET;
    let p_coefficients = config
        .rho_uncertainty_problem_size
        .p_coefficients
        .unwrap_or(0);
    let multi_seed_cascade = seed_count > seed_budget.max(1);
    let expensive_shape =
        p_coefficients >= EXPENSIVE_PREWARM_COEFF_DIM || cap.n_params >= EXPENSIVE_PREWARM_RHO_DIM;

    if multi_seed_cascade && expensive_shape {
        MULTI_SEED_PREWARM_BUDGET.min(default_budget)
    } else if expensive_shape {
        SINGLE_EXPENSIVE_PREWARM_BUDGET.min(default_budget)
    } else {
        default_budget
    }
}

/// Execute a single plan attempt (seed generation → solver loop → best result).
fn run_outer_with_plan(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    context: &str,
    cap: &OuterCapability,
    the_plan: &OuterPlan,
) -> Result<OuterResult, EstimationError> {
    let mut seeds = {
        let generated = crate::seeding::generate_rho_candidates(
            cap.n_params,
            config.heuristic_lambdas.as_deref(),
            &config.seed_config,
        );
        if generated.is_empty() {
            Vec::new()
        } else {
            generated
        }
    };
    if let Some(initial_rho) = config.initial_rho.as_ref()
        && !seeds.iter().any(|seed| seed == initial_rho)
    {
        seeds.insert(0, initial_rho.clone());
    }
    if seeds.is_empty() {
        return Err(EstimationError::RemlOptimizationFailed(format!(
            "no seeds generated for outer optimization ({context})"
        )));
    }

    let (lower, upper) = outer_bounds_template(config, cap.n_params);
    crate::solver::estimate::reml::runtime::record_current_outer_rho_upper_bounds_for_ift(&upper);
    let bounds_template = (lower, upper);
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

    let screening_enabled = config.screening_cap.is_some();
    let seed_budget = effective_seed_budget(
        config.seed_config.seed_budget,
        the_plan.solver,
        config.seed_config.risk_profile,
        screening_enabled,
    )
    .min(seeds.len());
    let explicit_initial_rho_owns_single_seed_budget = config.initial_rho.is_some()
        && seed_budget == 1
        && seeds.len() > 1
        && !config.screen_initial_rho;
    if !explicit_initial_rho_owns_single_seed_budget
        && should_screen_seeds(config, the_plan.solver, seeds.len(), seed_budget)
    {
        seeds = rank_seeds_with_screening(obj, config, context, &seeds);
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

    let mut best: Option<OuterResult> = None;
    // Object 1 — ContinuationPath. Every SAE-manifold joint fit ENTERS through
    // the continuation path at a heavy-smoothing regime. When the objective
    // declares this requirement the seed cascade's structural-failure handling
    // flips from REJECT (which can empty the candidate set and fall through to
    // the fatal `format_no_seeds_passed`) to DEMOTE-WITH-REASON: a "cold"
    // structural diagnosis becomes a heavier-regime RE-ENTRY of the same seed,
    // recorded on the path, never a disqualification. Objectives that do not
    // require continuation entry keep `None` and the legacy reject/early-exit
    // contract is unchanged.
    let mut continuation_path: Option<crate::solver::continuation_path::ContinuationPath> = obj
        .requires_continuation_path_entry()
        .then(crate::solver::continuation_path::ContinuationPath::heavy_entry);
    // Demotion ledger: every structural defect that would historically have
    // rejected a seed (or short-circuited the cascade) is instead recorded
    // here with its reason and the regime it was demoted to, so the
    // `SearchLedger` / startup stats surface a heavier-regime re-entry rather
    // than a vanished candidate. Non-fatal by construction.
    let mut path_demotions: Vec<PathDemotionRecord> = Vec::new();
    // Accumulate every per-seed rejection with its 0-based seed index and the
    // phase that rejected it (validation vs solver run). When all seeds fail
    // systematically (bad analytic gradient, rank-deficient penalty, etc.) the
    // first rejection's rho + error is often the most diagnostic.
    let mut rejection_reasons: Vec<(usize, &'static str, String)> = Vec::new();
    let layout = cap.theta_layout();
    let mut started_seeds = 0usize;
    let expensive_seed_limit =
        expensive_unsuccessful_seed_limit(the_plan.solver, config.seed_config.risk_profile);
    let mut unsuccessful_expensive_seeds = 0usize;
    let continuation_prewarm_budget =
        continuation_prewarm_step_budget(config, cap, seeds.len(), seed_budget);
    if continuation_prewarm_budget < crate::solver::estimate::reml::continuation::PATH_BUDGET {
        let p_coefficients = config
            .rho_uncertainty_problem_size
            .p_coefficients
            .unwrap_or(0);
        log::info!(
            "[OUTER] {context}: bounded continuation pre-warm budget to {} rho-step(s) \
             for seed_count={} seed_budget={} rho_dim={} p_coefficients={}",
            continuation_prewarm_budget,
            seeds.len(),
            seed_budget,
            cap.n_params,
            p_coefficients,
        );
    }
    let mut continuation_prewarm_suppressed_after: Option<String> = None;
    // Tracks whether the loop broke out early due to
    // `expensive_unsuccessful_seed_limit` so the aggregate error can
    // distinguish "all generated seeds tried" from "stopped early".
    let mut stopped_early_due_to_limit = false;
    // Structured mirror of `rejection_reasons` used for honest seed
    // accounting + structural early-exit. Populated lazily at the top of
    // each iteration from any reasons accumulated during the previous
    // pass, so individual push sites don't need to be touched.
    let mut seed_rejections: Vec<SeedRejection> = Vec::new();
    let mut last_classified_reason_idx: usize = 0;
    // Set to `Some(key)` when every observed rejection so far carries
    // the same genuinely structural `(KktRefusalDiagnosis,
    // carrying_block)` pair AND we've seen at least
    // `STRUCTURAL_EARLY_EXIT_MIN_COUNT` consistent failures. Once set,
    // the remaining ρ candidates are skipped.
    let mut structural_early_exit_key: Option<(
        crate::families::custom_family::KktRefusalDiagnosis,
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
        crate::solver::startup_stats::GenericFailureSignature,
        usize,
        usize,
    )> = None;

    'seed_attempts: for (seed_idx, seed) in seeds.iter().enumerate() {
        if started_seeds == seed_budget {
            break;
        }
        // Lazy structured classification: convert any new entries in
        // `rejection_reasons` into `SeedRejection`s and probe whether
        // the seed cascade has slipped into a uniform structural
        // failure mode that the remaining candidates can't escape.
        while last_classified_reason_idx < rejection_reasons.len() {
            let (idx, phase, msg) = &rejection_reasons[last_classified_reason_idx];
            seed_rejections.push(SeedRejection::from_message(*idx, phase, msg.clone()));
            last_classified_reason_idx += 1;
        }
        if structural_early_exit_key.is_none() {
            if let Some(key) =
                uniform_structural_key(&seed_rejections, STRUCTURAL_EARLY_EXIT_MIN_COUNT)
            {
                if let Some(path) = continuation_path.as_mut() {
                    // Continuation-entry objective: a uniform structural
                    // diagnosis is NOT a reason to skip the remaining seeds
                    // (that would empty the candidate set and fall through to
                    // the fatal "no seeds passed"). The seed cascade is only an
                    // *optimization* over warm-starts, never a feasibility
                    // gate — so we DEMOTE the cascade to a heavier path regime
                    // and keep evaluating. The heavier-smoothing entry gives
                    // the joint solver a feasible basin the cold seed could not
                    // reach. Record the demotion with its reason; never fatal.
                    let reason = format!(
                        "uniform structural diagnosis={} carrying-block={} after {} consistent \
                         rejection(s)",
                        key.0.as_str(),
                        key.1.as_deref().unwrap_or("<unknown>"),
                        seed_rejections.len(),
                    );
                    let regime = path.demote_with_reason(
                        crate::solver::continuation_path::PathDemotionReason::UniformStructural,
                    );
                    log::warn!(
                        "[OUTER] {context}: continuation-entry objective demoted to heavier path \
                         regime {regime:?} instead of structural early-exit ({reason}); \
                         re-entering remaining seed(s) at the heavier regime"
                    );
                    path_demotions.push(PathDemotionRecord {
                        seed_idx,
                        regime,
                        reason,
                    });
                    // Reset the structured mirror's structural signal so the
                    // heavier-regime re-entries are judged on their own merits
                    // and a single later defect does not immediately re-fire
                    // the demotion at the same level.
                    seed_rejections.clear();
                    last_classified_reason_idx = rejection_reasons.len();
                } else {
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
        }
        // Generic cross-seed structural bail (#1036): only for objectives that
        // do NOT enter through the continuation path. Continuation-entry
        // objectives demote to a heavier regime on any uniform structural
        // signal (handled above) and must never empty their candidate set on a
        // failure signature, so they opt out of the generic bail entirely.
        if structural_early_exit_key.is_none()
            && generic_structural_bail.is_none()
            && continuation_path.is_none()
        {
            if let Some((sig, run_len)) =
                crate::solver::startup_stats::consecutive_generic_signature(
                    &seed_rejections,
                    GENERIC_STRUCTURAL_BAIL_MIN_RUN,
                )
            {
                let first_seed = seed_rejections[seed_rejections.len() - run_len].seed_idx;
                let last_seed = seed_rejections[seed_rejections.len() - 1].seed_idx;
                let label = crate::solver::startup_stats::generic_signature_label(&sig);
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
        crate::solver::estimate::reml::runtime::record_current_outer_iter_for_ift(0);
        obj.reset();
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
                // fall through to the cascade exactly as a refused pre-warm does.
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
            // — `accept_seed_without_outer_iterations`, the continuation
            // pre-warm, and the direct solve at `seed` — takes over from the
            // pristine cold state. Rejecting the seed here instead emptied the
            // candidate set for objectives WITHOUT a continuation path (#1095:
            // a periodic K=1 circle whose walk "buys nothing" and refuses on a
            // small-N pivot bifurcation — `requires_continuation_path_entry` is
            // false for periodic K=1, so every one of its seeds was rejected
            // before any solver started). Reset to the baseline so the cascade
            // opens each seed from its own cold default, exactly as a hard
            // anchor-construction error already does above.
            log::info!(
                "[OUTER] {context}: curvature-homotopy entry refused seed {seed_idx}; deferring \
                 to the seed cascade from the pristine baseline"
            );
            obj.reset();
        }
        if let Some(seed_cost) = obj.accept_seed_without_outer_iterations(seed)? {
            started_seeds += 1;
            let candidate = OuterResult::new(seed.clone(), seed_cost, 0, true, *the_plan);
            if candidate_improves_best(&candidate, best.as_ref()) {
                best = Some(candidate);
            }
            break;
        }
        // Magic-by-default continuation pre-warm. On hard fits this
        // walks ρ from an oversmoothing ρ₀ down to `seed`, leaving the
        // objective's inner state warm at `seed`. On easy fits (ρ₀
        // collapses to seed inside the bounds box) this is a single
        // pre-screen comparison with no inner call, no allocation. A
        // failure here means continuation could not even *reach* the
        // seed; route the underlying InnerFailure through the same
        // SeedRejection accounting any other pre-validation rejection
        // would take, then continue to the next seed.
        //
        // The pre-warm is a warm-start for gradient-bearing PIRLS-inner
        // REML objectives: it walks ρ via `eval_with_order(_, ValueAndGradient)`
        // and carries the converged inner β forward through each step's
        // `inner_beta_hint`. A continuation-entry objective (SAE-manifold joint
        // fit) MUST enter every seed through the heavy-smoothing
        // ContinuationPath walk, so it opts into the priming pass even though it
        // does not advertise the generic `allow_continuation_prewarm`
        // warm-start. For a continuation-entry objective a refused walk is
        // DEMOTED to a heavier regime below, not treated as a feasibility gate.
        let enter_via_continuation_path =
            obj.allow_continuation_prewarm() || continuation_path.is_some();
        // Continuation-entry objective (SAE-manifold joint fit): DRIVE the
        // coupled `ContinuationPath` homotopy explicitly. This is the missing
        // half of Object 1 — the descent walk. Rather than a single ρ-only
        // `prime_outer_seed` pre-screen, we step the path waypoint by waypoint:
        // each `step` runs the ρ-anneal spine for that waypoint and advances
        // the τ / isometry legs in lockstep, so all three knobs arrive at the
        // real objective together (the one-monotone-walk invariant). The
        // converged inner β of each accepted descent leg warm-starts the next,
        // and the warm iterate at `Arrived` is handed to the normal solver at
        // ρ*. Re-entry / breach / underflow are non-fatal floor behaviors,
        // each consumed below — never a rejection.
        //
        // The walk runs for EVERY continuation-entry objective regardless of the
        // primary solver class: the only objective that sets
        // `requires_continuation_path_entry` is the SAE-manifold joint fit,
        // whose `eval` / `seed_inner_state` / inner arrow-Schur ARE reachable.
        // The heavy-smoothing walk warms the cold inner solve first, or the cold
        // `eval_cost` hits a non-PD inner block (the K≥2 routing-collapse failure
        // Object 1 exists to prevent).
        if continuation_path.is_some() {
            {
                // Rebuild the path per-seed against the OBJECTIVE's real ρ
                // dimension and legal box. The seed-loop-scoped `heavy_entry`
                // placeholder is dimension-1 (built before any seed is in hand);
                // the spine call inside `step` requires the ρ target to match
                // the objective's ρ dim, so we re-enter the heavy-smoothing
                // regime coupled to this seed's ρ\* and bounds. Re-entry resets
                // the path to a fresh `s = 1` for every seed, which is correct:
                // each seed is its own descent from the contraction regime.
                let path = continuation_path.insert(
                    crate::solver::continuation_path::ContinuationPath::heavy_entry_for_rho(
                        seed.clone(),
                        bounds_template.1.clone(),
                    ),
                );
                let walk_start = std::time::Instant::now();
                // β carried warm across legs. Empty = cold entry (#969:
                // warm-invariance funnels cold and warm to the same s=1
                // contraction fixed point).
                let mut warm_beta: Array1<f64> = Array1::zeros(0);
                let mut legs_descended = 0usize;
                let mut arrived = false;
                // Bound the walk: CONTINUATION_WAYPOINTS clean descents plus a
                // re-entry allowance (every re-entry is progress toward the
                // contraction floor, reachable in finitely many back-offs).
                // Each `step` runs the ρ-anneal spine, which is itself an inner
                // homotopy, so the budget stays bounded — but it must tolerate
                // the expected near-cliff floor bounces: at the one-waypoint
                // `REENTRY_BACKOFF` each bounce costs ~2 legs, and the shared
                // `CONTINUATION_WALK_BUDGET` (2× waypoints) absorbs ~half-a-
                // walk's worth of bounces before cutoff. The spine warm-starts
                // from the previous leg's β, so post-entry legs are cheap. The
                // loop only ever exits on `Arrived` or this budget — there is
                // no rejection exit.
                let walk_budget = crate::solver::continuation_path::CONTINUATION_WALK_BUDGET;
                for _ in 0..walk_budget {
                    if path.arrived() {
                        arrived = true;
                        break;
                    }
                    match path.step(obj, &warm_beta) {
                        crate::solver::continuation_path::ContinuationStep::Descended {
                            s,
                            state,
                        } => {
                            // Warm-start the next leg from this leg's converged
                            // inner β. `NoSlot` is fine (the objective simply
                            // starts the next spine pass cold); a genuine
                            // dimension error resets to a clean baseline and the
                            // walk re-enters heavier on the next iteration.
                            warm_beta = state.last_beta.clone();
                            if let Err(err) = obj.seed_inner_state(&warm_beta) {
                                log::warn!(
                                    "[OUTER] {context}: continuation descent seed {seed_idx} \
                                     warm-start at s={s:.4} unusable ({err}); proceeding cold"
                                );
                                warm_beta = Array1::zeros(0);
                                obj.reset();
                            }
                            legs_descended += 1;
                        }
                        crate::solver::continuation_path::ContinuationStep::Arrived { state } => {
                            // The path reached ρ* / τ_min / tight isometry along
                            // the coupled walk. Install the warm iterate so the
                            // normal solver below starts from the contraction's
                            // image at the real objective, not cold.
                            warm_beta = state.last_beta.clone();
                            if let Err(err) = obj.seed_inner_state(&warm_beta) {
                                log::warn!(
                                    "[OUTER] {context}: continuation arrival seed {seed_idx} \
                                     warm-start unusable ({err}); solver starts cold at ρ*"
                                );
                                obj.reset();
                            }
                            legs_descended += 1;
                            arrived = true;
                            break;
                        }
                        crate::solver::continuation_path::ContinuationStep::Reentered {
                            s,
                            reason,
                        } => {
                            use crate::solver::continuation_path::ReentryReason;
                            // The homotopy FLOOR: never reject. Each reason is a
                            // re-entry into a heavier regime (the path already
                            // raised `s`); we consume its payload for diagnostics
                            // and continue descending from the heavier regime.
                            match reason {
                                ReentryReason::SpineStruggled(failure) => {
                                    log::info!(
                                        "[OUTER] {context}: continuation seed {seed_idx} spine \
                                         struggled at s={s:.4} ({}); re-entered heavier regime {:?}",
                                        failure.message(),
                                        path.enter_regime(),
                                    );
                                }
                                ReentryReason::StepUnderflow => {
                                    // The descent step underflowed: demote with a
                                    // recorded reason so the ledger surfaces the
                                    // heavier-regime re-entry, then keep
                                    // descending from the pinned floor.
                                    let regime = path.demote_with_reason(
                                        crate::solver::continuation_path::PathDemotionReason::PrewarmStructural,
                                    );
                                    path_demotions.push(PathDemotionRecord {
                                        seed_idx,
                                        regime,
                                        reason: format!(
                                            "continuation step underflow at s={s:.4}; pinned to \
                                             the homotopy floor and re-descending"
                                        ),
                                    });
                                }
                                ReentryReason::MassFloorBreached(breach) => {
                                    // Active-mass collapse toward the uniform
                                    // saddle: reset to the pristine seeded
                                    // baseline (the scaffold) so the assignment
                                    // re-diffuses, and record the breach with its
                                    // observed mass / floor in the demotion
                                    // ledger. Never fatal.
                                    obj.reset();
                                    warm_beta = Array1::zeros(0);
                                    let regime = path.enter_regime();
                                    path_demotions.push(PathDemotionRecord {
                                        seed_idx,
                                        regime,
                                        reason: format!(
                                            "active-mass breach (observed mean {:.4} < floor \
                                             {:.4}); re-seeded from scaffold, re-entered heavier \
                                             regime",
                                            breach.observed_mean_mass, breach.floor,
                                        ),
                                    });
                                }
                            }
                        }
                    }
                }
                log::info!(
                    "[OUTER] {context}: continuation-path walk seed {seed_idx} legs={legs_descended} \
                     arrived={arrived} reseeds={} elapsed={:.3}s",
                    path.reseed_count(),
                    walk_start.elapsed().as_secs_f64(),
                );
            }
        }
        if continuation_path.is_none() && enter_via_continuation_path {
            if let Some(reason) = continuation_prewarm_suppressed_after.as_ref() {
                log::info!(
                    "[OUTER] {context}: skipping continuation pre-warm for seed {seed_idx} \
                     after earlier non-structural pre-warm failure ({reason}); direct seed eval \
                     will judge this candidate"
                );
            } else {
                let prewarm_start = std::time::Instant::now();
                match crate::solver::estimate::reml::continuation::prime_outer_seed_with_budget(
                    obj,
                    seed,
                    &bounds_template.1,
                    continuation_prewarm_budget,
                ) {
                    Ok(summary) => {
                        // Skip the log line on collapse — that's the
                        // zero-overhead easy-fit case and a log per seed would
                        // be noise. Anything else is a real anneal worth
                        // surfacing so large-scale runs are diagnosable.
                        if !summary.collapsed {
                            log::info!(
                                "[OUTER] {context}: continuation pre-warm seed {seed_idx} steps={} elapsed={:.3}s",
                                summary.steps_accepted,
                                prewarm_start.elapsed().as_secs_f64(),
                            );
                        }
                    }
                    Err(cf) if cf.is_structural() => {
                        // The pre-warm surfaced a structural defect of the seed's
                        // joint design (rank/alias deficiency or a genuine
                        // active-set KKT bug). This block runs only for
                        // NON-continuation-entry objectives (continuation-entry
                        // objectives drive the explicit `ContinuationPath` walk
                        // above, where a structural refusal is a heavier-regime
                        // demotion, never a rejection). Legacy contract: a cold solve
                        // at the seed ρ* would hit the same defect, so disqualify the
                        // seed and route the failure through the same structural
                        // accounting any other pre-validation rejection takes.
                        let msg = format!(
                            "continuation pre-warm refused before seed eval: {}",
                            cf.message()
                        );
                        log::warn!(
                            "[OUTER] {context}: rejecting seed {seed_idx} (continuation): {msg}"
                        );
                        rejection_reasons.push((seed_idx, "validation", msg));
                        continue 'seed_attempts;
                    }
                    Err(cf) => {
                        // Non-structural pre-warm failure: the continuation walk
                        // could not complete from the heavily-oversmoothed ρ₀
                        // (e.g. an ill-conditioned constraint KKT residual at
                        // λ₀ ≫ λ*, a likelihood domain miss at that start, or a
                        // stuck/budget-exhausted path). That is a property of the
                        // warm-start schedule, NOT of the seed ρ* itself — which
                        // the cold seed eval below judges on its own merits. The
                        // pre-warm is a warm-start optimization, never a
                        // feasibility gate (cf. #236, #500): a refusal here must
                        // not disqualify a seed that would solve cold. Reset to a
                        // clean baseline and fall through to the cold seed eval.
                        log::warn!(
                            "[OUTER] {context}: continuation pre-warm for seed {seed_idx} did not \
                             complete ({}); direct seed eval will judge this candidate and remaining \
                             seeds will skip the pre-warm",
                            cf.message()
                        );
                        obj.reset();
                        continuation_prewarm_suppressed_after = Some(cf.message());
                    }
                }
            }
        }
        let t_seed_start = std::time::Instant::now();
        let seed_slot;
        let result: Result<OuterResult, EstimationError> = match the_plan.solver {
            Solver::Arc => {
                let seed_eval = obj
                    .eval_with_order(seed, OuterEvalOrder::ValueGradientHessian)
                    .map_err(|err| into_objective_error("outer eval failed", err));
                let seed_eval = match seed_eval {
                    Ok(seed_eval) => seed_eval,
                    Err(err) => {
                        let err = match err {
                            ObjectiveEvalError::Recoverable { message }
                            | ObjectiveEvalError::Fatal { message } => {
                                EstimationError::RemlOptimizationFailed(message)
                            }
                        };
                        if requests_immediate_first_order_fallback(&err.to_string()) {
                            return Err(err);
                        }
                        log::warn!(
                            "[OUTER] {context}: rejecting seed {seed_idx} before solver start: {err}"
                        );
                        rejection_reasons.push((seed_idx, "validation", err.to_string()));
                        continue 'seed_attempts;
                    }
                };
                let seed_eval = finite_outer_eval_or_error("outer eval failed", layout, seed_eval)
                    .map_err(|err| match err {
                        ObjectiveEvalError::Recoverable { message }
                        | ObjectiveEvalError::Fatal { message } => {
                            EstimationError::RemlOptimizationFailed(message)
                        }
                    });
                let mut seed_eval = match seed_eval {
                    Ok(seed_eval) => seed_eval,
                    Err(err) => {
                        log::warn!(
                            "[OUTER] {context}: rejecting seed {seed_idx} before solver start: {err}"
                        );
                        rejection_reasons.push((seed_idx, "validation", err.to_string()));
                        continue 'seed_attempts;
                    }
                };
                validate_second_order_seed_hessian(context, layout, &seed_eval).map_err(|err| {
                    match err {
                        ObjectiveEvalError::Recoverable { message }
                        | ObjectiveEvalError::Fatal { message } => {
                            EstimationError::RemlOptimizationFailed(message)
                        }
                    }
                })?;
                started_seeds += 1;
                seed_slot = started_seeds;

                let cheap_materializable_operator = matches!(
                    seed_eval.hessian,
                    HessianResult::Operator(ref op)
                        if op.materialization_capability().is_available()
                            && op.dim() <= OUTER_HVP_MATERIALIZE_MAX_DIM
                );
                if cheap_materializable_operator {
                    // The operator's own work model says probing every column
                    // is cheap; convert the seed Hessian to dense in-place.
                    // Subsequent bridge evaluations apply the same predicate.
                    if let HessianResult::Operator(op) = &seed_eval.hessian {
                        match op.materialize_dense() {
                            Ok(dense) => {
                                seed_eval.hessian = HessianResult::Analytic(dense);
                            }
                            Err(message) => {
                                let err = EstimationError::RemlOptimizationFailed(format!(
                                    "outer Hessian operator materialization failed: {message}"
                                ));
                                log::warn!(
                                    "[OUTER] {context}: rejecting seed {seed_idx} before solver start: {err}"
                                );
                                rejection_reasons.push((seed_idx, "validation", err.to_string()));
                                continue 'seed_attempts;
                            }
                        }
                    }
                }
                if matches!(seed_eval.hessian, HessianResult::Operator(_)) {
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
                        hessian: hessian_result_to_value(seed_eval.hessian.clone()),
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
                            feedback: feedback.clone(),
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
                            result.operator_stop_reason =
                                Some(OperatorTrustRegionStopReason::Converged);
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
                            result.operator_stop_reason =
                                Some(OperatorTrustRegionStopReason::IterationBudget);
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
                            result.operator_stop_reason =
                                Some(OperatorTrustRegionStopReason::RejectFloor);
                            result.operator_trust_radius = final_radius;
                            Ok(result)
                        }
                        OptimizationStatus::ObjectiveFailed
                        | OptimizationStatus::NumericalFailure
                        | OptimizationStatus::LineSearchFailed => {
                            Err(EstimationError::RemlOptimizationFailed(format!(
                                "matrix-free TR solver failed with status={:?}",
                                report.status
                            )))
                        }
                    }
                } else {
                    let hessian_source = the_plan.hessian_source;
                    let (lo, hi) = &bounds_template;
                    let bounds = outer_bounds(lo, hi)?;
                    let grad_tol = outer_gradient_tolerance(config);
                    let max_iter = outer_max_iterations(config.max_iter)?;

                    let objective = OuterSecondOrderBridge {
                        obj,
                        layout,
                        hessian_source,
                        materialize_operator_max_dim: OUTER_HVP_MATERIALIZE_MAX_DIM,
                        eval_count: 0,
                        outer_inner_cap: config.outer_inner_cap.clone(),
                        g_norm_initial: None,
                        last_g_norm: None,
                        last_value_grad_rho: None,
                    };

                    // Build the opt seed sample from the precomputed
                    // outer evaluation. The Hessian translation goes
                    // through `build_bridge_hessian_for_source` so the
                    // analytic-route contract (no None Hessian on
                    // `HessianSource::Analytic`) applies at seed time
                    // too, not just inside the bridge's live path.
                    let seed_hessian = build_bridge_hessian_for_source(
                        hessian_source,
                        seed_eval.hessian.clone(),
                        OUTER_HVP_MATERIALIZE_MAX_DIM,
                    )
                    .map_err(|err| match err {
                        ObjectiveEvalError::Recoverable { message }
                        | ObjectiveEvalError::Fatal { message } => {
                            EstimationError::RemlOptimizationFailed(message)
                        }
                    })?;
                    let initial_sample = SecondOrderSample {
                        value: seed_eval.cost,
                        gradient: seed_eval.gradient.clone(),
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
                            feedback: feedback.clone(),
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
                            Ok(solution_into_outer_result(*last_solution, false, *the_plan))
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
                    let seed_eval_dev = match obj
                        .eval_with_order(seed, OuterEvalOrder::ValueAndGradient)
                        .map_err(|err| into_objective_error("outer eval failed", err))
                    {
                        Ok(e) => e,
                        Err(err) => {
                            let err = match err {
                                ObjectiveEvalError::Recoverable { message }
                                | ObjectiveEvalError::Fatal { message } => {
                                    EstimationError::RemlOptimizationFailed(message)
                                }
                            };
                            log::warn!(
                                "[OUTER] {context}: rejecting seed {seed_idx} before device-BFGS start: {err}"
                            );
                            rejection_reasons.push((seed_idx, "validation", err.to_string()));
                            continue 'seed_attempts;
                        }
                    };
                    started_seeds += 1;
                    seed_slot = started_seeds;
                    let device_input = crate::solver::gpu::reml_outer::RemlOuterGpuInput {
                        seed_rho: seed.clone(),
                        bounds: bounds_dev,
                        gradient_tolerance: grad_tol_dev.abs,
                        max_iterations: config.max_iter,
                        axis_step_caps: axis_caps_dev,
                        admission,
                        seed_objective: seed_eval_dev.cost,
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
                            Ok(crate::solver::gpu::reml_outer::RemlOuterDeviceEval {
                                objective: eval.cost,
                                gradient: eval.gradient,
                            })
                        };
                        crate::solver::gpu::reml_outer::run_reml_outer_on_device(
                            device_input,
                            evaluator,
                        )
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
                            log::warn!(
                                "[OUTER] {context}: device-BFGS failed at seed {seed_idx}: {err}; falling back to host BFGS"
                            );
                            // Fall through to the host BFGS path below by
                            // re-running the seed evaluation; the
                            // existing branch will re-validate it and
                            // proceed.
                            let seed_eval = obj
                                .eval_with_order(seed, OuterEvalOrder::ValueAndGradient)
                                .map_err(|err| into_objective_error("outer eval failed", err));
                            match finite_outer_first_order_eval_or_error(
                                "outer eval failed",
                                layout,
                                seed_eval.map_err(|err| match err {
                                    ObjectiveEvalError::Recoverable { message }
                                    | ObjectiveEvalError::Fatal { message } => {
                                        EstimationError::RemlOptimizationFailed(message)
                                    }
                                })?,
                            )
                            .map_err(|err| match err {
                                ObjectiveEvalError::Recoverable { message }
                                | ObjectiveEvalError::Fatal { message } => {
                                    EstimationError::RemlOptimizationFailed(message)
                                }
                            }) {
                                Ok(_) => Err(err),
                                Err(e) => {
                                    rejection_reasons.push((seed_idx, "validation", e.to_string()));
                                    continue 'seed_attempts;
                                }
                            }
                        }
                    }
                } else {
                    let seed_eval = obj
                        .eval_with_order(seed, OuterEvalOrder::ValueAndGradient)
                        .map_err(|err| into_objective_error("outer eval failed", err));
                    let seed_eval = match seed_eval {
                        Ok(seed_eval) => seed_eval,
                        Err(err) => {
                            let err = match err {
                                ObjectiveEvalError::Recoverable { message }
                                | ObjectiveEvalError::Fatal { message } => {
                                    EstimationError::RemlOptimizationFailed(message)
                                }
                            };
                            log::warn!(
                                "[OUTER] {context}: rejecting seed {seed_idx} before solver start: {err}"
                            );
                            rejection_reasons.push((seed_idx, "validation", err.to_string()));
                            continue 'seed_attempts;
                        }
                    };
                    let seed_eval = match finite_outer_first_order_eval_or_error(
                        "outer eval failed",
                        layout,
                        seed_eval,
                    )
                    .map_err(|err| match err {
                        ObjectiveEvalError::Recoverable { message }
                        | ObjectiveEvalError::Fatal { message } => {
                            EstimationError::RemlOptimizationFailed(message)
                        }
                    }) {
                        Ok(eval) => eval,
                        Err(err) => {
                            log::warn!(
                                "[OUTER] {context}: rejecting seed {seed_idx} before solver start: {err}"
                            );
                            rejection_reasons.push((seed_idx, "validation", err.to_string()));
                            continue 'seed_attempts;
                        }
                    };
                    started_seeds += 1;
                    seed_slot = started_seeds;
                    let (lo, hi) = &bounds_template;
                    let bounds = outer_bounds(lo, hi)?;
                    let grad_tol = outer_gradient_tolerance(config);
                    let max_iter = outer_max_iterations(config.max_iter)?;
                    // Cost-stall convergence shared cell (#1089). The bridge is
                    // moved into `opt::Bfgs`, so the best iterate it captures on
                    // a flat-valley stall is handed back through this `Arc`.
                    // Relative score-change floor is derived one decade tighter
                    // than the outer gradient tolerance so it only triggers once
                    // the objective is genuinely flat — never preempting a real
                    // (if slow) descent that still clears the gradient test.
                    let cost_stall_exit: Arc<Mutex<Option<CostStallExit>>> =
                        Arc::new(Mutex::new(None));
                    let cost_stall_rel_tol = (config.tolerance * 1.0e-2).max(f64::EPSILON);
                    // Stationarity gate for the cost-stall exit. Convergence must
                    // mean stationarity, not cost-flatness: a cost stall only
                    // counts as a converged optimum when the projected gradient
                    // norm at the best iterate clears the SAME outer gradient
                    // tolerance the genuine BFGS convergence path uses. Evaluate
                    // that threshold once at the seed (cost + initial gradient
                    // norm), exactly as `opt::Bfgs` does internally. Reusing
                    // `grad_tol` here means no new/widened tolerance is
                    // introduced — a flat-valley stall whose residual gradient
                    // exceeds this is surfaced as non-converged.
                    let seed_grad_norm = seed_eval
                        .gradient
                        .iter()
                        .map(|g| g * g)
                        .sum::<f64>()
                        .sqrt();
                    let cost_stall_grad_threshold =
                        grad_tol.threshold(seed_eval.cost, seed_grad_norm);
                    let objective = OuterFirstOrderBridge {
                        obj,
                        layout,
                        outer_inner_cap: config.outer_inner_cap.clone(),
                        iter_count: 0,
                        g_norm_initial: None,
                        last_g_norm: None,
                        last_value_grad_rho: None,
                        value_probe_cache: Vec::new(),
                        cost_stall: Some(CostStallGuard::new(
                            cost_stall_rel_tol,
                            COST_STALL_WINDOW,
                            cost_stall_grad_threshold,
                            cost_stall_exit.clone(),
                        )),
                        consecutive_probe_refusals: 0,
                    };
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
                        .with_max_iterations(max_iter);
                    if let Some(caps) = bfgs_axis_step_caps(config, layout) {
                        optimizer = optimizer.with_axis_step_caps(caps);
                    }
                    if let Some(feedback) = config.outer_inner_cap.as_ref() {
                        optimizer = optimizer.with_observer(OuterAcceptObserver {
                            feedback: feedback.clone(),
                        });
                    }
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
                                Ok(solution_into_outer_result(*last_solution, false, *the_plan))
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
                            let exit = cost_stall_exit
                                .lock()
                                .ok()
                                .and_then(|mut slot| slot.take());
                            match exit {
                                Some(exit) => Ok(outer_result_with_gradient_norm(
                                    exit.rho,
                                    exit.value,
                                    exit.iterations,
                                    Some(exit.grad_norm),
                                    exit.converged,
                                    *the_plan,
                                )),
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
                            Err(EstimationError::RemlOptimizationFailed(format!(
                                "BFGS solver failed: ObjectiveFailed {{ message: {message:?} }}"
                            )))
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
                        rejection_reasons.push((seed_idx, "validation", err.to_string()));
                        continue 'seed_attempts;
                    }
                    Err(FixedPointOuterRunError::ImmediateFallback(err)) => {
                        seed_slot = started_seeds + 1;
                        Err(err)
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
                        rejection_reasons.push((seed_idx, "validation", err.to_string()));
                        continue 'seed_attempts;
                    }
                    Err(FixedPointOuterRunError::ImmediateFallback(err)) => {
                        seed_slot = started_seeds + 1;
                        Err(err)
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
                let candidate_converged = candidate.converged;
                log::debug!(
                    "[outer-timing] seed {}/{} ({:?}): {:.3}s  cost={:.6e}  converged={}",
                    seed_slot,
                    seed_budget,
                    the_plan.solver,
                    seed_elapsed,
                    candidate.final_value,
                    candidate.converged,
                );
                if candidate_improves_best(&candidate, best.as_ref()) {
                    best = Some(candidate);
                }
                let quality_compare_remaining_gaussian_seeds = matches!(
                    config.seed_config.risk_profile,
                    crate::seeding::SeedRiskProfile::Gaussian
                ) && seed_budget > 1
                    && started_seeds < seed_budget;
                if best.as_ref().is_some_and(|b| b.converged)
                    && !quality_compare_remaining_gaussian_seeds
                {
                    break;
                }
                if !candidate_converged && matches!(expensive_seed_limit, Some(limit) if limit > 0)
                {
                    unsuccessful_expensive_seeds += 1;
                    if let Some(limit) = expensive_seed_limit
                        && unsuccessful_expensive_seeds >= limit
                    {
                        log::info!(
                            "[OUTER] {context}: stopping expensive multi-start after {} non-converged {:?} seed(s)",
                            unsuccessful_expensive_seeds,
                            the_plan.solver,
                        );
                        stopped_early_due_to_limit = true;
                        break;
                    }
                }
            }
            Err(e) => {
                if requests_immediate_first_order_fallback(&e.to_string()) {
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
                rejection_reasons.push((seed_idx, "solver", e.to_string()));
                if let Some(limit) = expensive_seed_limit {
                    unsuccessful_expensive_seeds += 1;
                    if unsuccessful_expensive_seeds >= limit {
                        log::info!(
                            "[OUTER] {context}: stopping expensive multi-start after {} failed {:?} seed(s)",
                            unsuccessful_expensive_seeds,
                            the_plan.solver,
                        );
                        stopped_early_due_to_limit = true;
                        break;
                    }
                }
            }
        }
    }

    if let Some(result) = best {
        obj.finalize_outer_result(&result.rho, the_plan)?;
        return Ok(result);
    }

    Err({
        // Drain any remaining unclassified entries in `rejection_reasons`
        // into the structured mirror so the final accounting reflects
        // every observed failure regardless of which loop branch pushed
        // it. Earlier behaviour reported `attempted = min(generated,
        // budget)` and a single `rejected = N` integer; that confused
        // "seed eval attempts" with "outer optimiser starts" and lumped
        // every failure mode together. The new accounting splits
        // CertRefused / domain / objective / budget rejections via the
        // `InnerFailure` classifier and names the structural cause when
        // every seed terminates the same way.
        while last_classified_reason_idx < rejection_reasons.len() {
            let (idx, phase, msg) = &rejection_reasons[last_classified_reason_idx];
            seed_rejections.push(SeedRejection::from_message(*idx, phase, msg.clone()));
            last_classified_reason_idx += 1;
        }
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
        let mut early_exit_note = if structural_early_exit_key.is_some() {
            "early-exit triggered: every observed seed reported the same structural rejection"
                .to_string()
        } else if let Some((sig, first_seed, last_seed)) = generic_structural_bail.as_ref() {
            let label = crate::solver::startup_stats::generic_signature_label(sig);
            let skipped = seeds.len().saturating_sub(*last_seed + 1);
            format!(
                "structural: {label} on seeds {first_seed}..{last_seed}; \
                 remaining {skipped} seeds skipped"
            )
        } else if stopped_early_due_to_limit {
            format!(
                "stopped early after {unsuccessful_expensive_seeds} consecutive non-converged \
                 {:?} seed(s) (expensive_unsuccessful_seed_limit)",
                the_plan.solver
            )
        } else {
            String::new()
        };
        // Surface the ContinuationPath demotion ledger: for a continuation-entry
        // objective, structural defects DEMOTED the cascade to heavier path
        // regimes instead of rejecting seeds, so the final diagnosis must show
        // the heavier-regime re-entries (with their reasons) rather than imply
        // the candidate set was emptied by a structural early-exit.
        if !path_demotions.is_empty() {
            if !early_exit_note.is_empty() {
                early_exit_note.push_str("; ");
            }
            let final_regime = continuation_path
                .as_ref()
                .map(|path| format!("{:?}", path.enter_regime()))
                .unwrap_or_else(|| "<none>".to_string());
            early_exit_note.push_str(&format!(
                "continuation-path: {} structural defect(s) DEMOTED to heavier regime(s) \
                 (never rejected); final regime={final_regime}; reasons: [{}]",
                path_demotions.len(),
                path_demotions
                    .iter()
                    .map(|d| format!("seed {} -> {:?}: {}", d.seed_idx, d.regime, d.reason))
                    .collect::<Vec<_>>()
                    .join("; "),
            ));
        }
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
mod tests {
    use super::*;
    use ::opt::FixedPointObjective;
    use ndarray::array;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    // ─── #934 first-order optimality certificate ──────────────────────

    /// Quadratic ½‖ρ − c‖² with value and gradient from the SAME center:
    /// the certificate must attest consistency at the optimum.
    #[test]
    fn certificate_attests_consistent_quadratic() {
        let center = array![0.3, -0.7];
        let cost_center = center.clone();
        let grad_center = center.clone();
        let problem = OuterProblem::new(2)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Unavailable)
            .with_initial_rho(array![2.0, 2.0])
            .with_seed_config(crate::seeding::SeedConfig {
                max_seeds: 1,
                seed_budget: 1,
                ..Default::default()
            });
        let mut obj = problem.build_objective(
            (),
            move |_: &mut (), rho: &Array1<f64>| {
                let d = rho - &cost_center;
                Ok(0.5 * d.dot(&d))
            },
            move |_: &mut (), rho: &Array1<f64>| {
                let d = rho - &grad_center;
                Ok(OuterEval {
                    cost: 0.5 * d.dot(&d),
                    gradient: d,
                    hessian: HessianResult::Unavailable,
                    inner_beta_hint: None,
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let result = problem
            .run(&mut obj, "certificate consistent quadratic")
            .expect("consistent quadratic must optimize");
        let cert = result
            .criterion_certificate
            .as_ref()
            .expect("gradient-based solve must ship a certificate");
        assert!(
            cert.first_order_consistent(),
            "consistent value/gradient paths flagged as desynced: {}",
            cert.summary(),
        );
        assert!(
            cert.lambdas_railed.is_empty(),
            "interior optimum reported railed λ: {}",
            cert.summary(),
        );
        assert!(cert.fd_step > 0.0 && cert.fd_error > 0.0);
    }

    #[test]
    fn rho_uncertainty_diagnostic_does_not_change_outer_solution() {
        let center = array![0.25];
        let seed_config = crate::seeding::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        };
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Either)
            .with_initial_rho(array![1.5])
            .with_seed_config(seed_config)
            .with_problem_size(8, 3);
        let config = problem.config();

        let mut without_diagnostic = problem.build_objective(
            (),
            {
                let center = center.clone();
                move |_: &mut (), rho: &Array1<f64>| {
                    let d = rho - &center;
                    Ok(0.5 * d.dot(&d))
                }
            },
            {
                let center = center.clone();
                move |_: &mut (), rho: &Array1<f64>| {
                    let d = rho - &center;
                    Ok(OuterEval {
                        cost: 0.5 * d.dot(&d),
                        gradient: d,
                        hessian: HessianResult::Analytic(array![[1.0]]),
                        inner_beta_hint: None,
                    })
                }
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let mut with_diagnostic = problem.build_objective(
            (),
            {
                let center = center.clone();
                move |_: &mut (), rho: &Array1<f64>| {
                    let d = rho - &center;
                    Ok(0.5 * d.dot(&d))
                }
            },
            {
                let center = center.clone();
                move |_: &mut (), rho: &Array1<f64>| {
                    let d = rho - &center;
                    Ok(OuterEval {
                        cost: 0.5 * d.dot(&d),
                        gradient: d,
                        hessian: HessianResult::Analytic(array![[1.0]]),
                        inner_beta_hint: None,
                    })
                }
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );

        let baseline =
            run_outer_uncertified(&mut without_diagnostic, &config, "rho-diagnostic-baseline")
                .expect("baseline outer run");
        let diagnosed = run_outer(&mut with_diagnostic, &config, "rho-diagnostic-run")
            .expect("diagnostic outer run");

        assert_eq!(baseline.rho, diagnosed.rho);
        assert_eq!(
            baseline.final_value.to_bits(),
            diagnosed.final_value.to_bits()
        );
        assert_eq!(baseline.iterations, diagnosed.iterations);
        assert_eq!(baseline.final_grad_norm, diagnosed.final_grad_norm);
        assert!(diagnosed.rho_uncertainty_diagnostic.is_some());
    }

    /// The desync bug genus (#748/#752/#901): the gradient path optimizes a
    /// criterion whose center is silently shifted from the value path's.
    /// The optimizer happily converges where the WRONG gradient vanishes;
    /// the certificate's FD of the actual value path must expose it.
    #[test]
    fn certificate_flags_value_gradient_desync() {
        let value_center = array![0.0, 0.0];
        let wrong_center = array![3.0, -2.0];
        let wrong_center_for_eval = wrong_center.clone();
        let problem = OuterProblem::new(2)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Unavailable)
            .with_initial_rho(array![1.0, 1.0])
            .with_seed_config(crate::seeding::SeedConfig {
                max_seeds: 1,
                seed_budget: 1,
                ..Default::default()
            });
        // eval(): a self-consistent but WRONG world (shifted center) so the
        // line search accepts steps and BFGS converges to wrong_center.
        // eval_cost(): the TRUE criterion value — the path the audit probes.
        let mut obj = problem.build_objective(
            (),
            move |_: &mut (), rho: &Array1<f64>| {
                let d = rho - &value_center;
                Ok(0.5 * d.dot(&d))
            },
            move |_: &mut (), rho: &Array1<f64>| {
                let d = rho - &wrong_center_for_eval;
                Ok(OuterEval {
                    cost: 0.5 * d.dot(&d),
                    gradient: d,
                    hessian: HessianResult::Unavailable,
                    inner_beta_hint: None,
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let result = problem
            .run(&mut obj, "certificate desynced quadratic")
            .expect("desynced quadratic still returns a result");
        let cert = result
            .criterion_certificate
            .as_ref()
            .expect("gradient-based solve must ship a certificate");
        // At wrong_center the analytic slope is ~0 but the true value path
        // slopes by v·(wrong_center − value_center) along the audit
        // direction. Guard the assertion on that projection being visible
        // (the deterministic direction is not axis-aligned, so it is).
        assert!(
            cert.fd_directional.abs() > 1e-3,
            "audit direction nearly orthogonal to the desync displacement: {}",
            cert.summary(),
        );
        assert!(
            !cert.first_order_consistent(),
            "value↔gradient desync NOT flagged: {}",
            cert.summary(),
        );
        assert!(cert.agreement_z > CERTIFICATE_Z_GATE);
    }

    #[test]
    fn certificate_audit_direction_is_deterministic_and_context_sensitive() {
        let theta = array![1.5, -0.25, 7.0];
        let a = certificate_audit_direction(&theta, "ctx-one");
        let b = certificate_audit_direction(&theta, "ctx-one");
        assert_eq!(a, b, "same fingerprint must give the same direction");
        let c = certificate_audit_direction(&theta, "ctx-two");
        assert!(
            (&a - &c).iter().any(|d| d.abs() > 1e-12),
            "different context must give a different direction",
        );
        assert!((a.dot(&a).sqrt() - 1.0).abs() < 1e-12, "unit norm");
    }

    #[test]
    fn certificate_hessian_pd_probe_classifies_definiteness() {
        assert_eq!(
            certificate_hessian_is_pd(&Array2::<f64>::eye(3)),
            Some(true)
        );
        let indefinite = array![[1.0, 2.0], [2.0, 1.0]];
        assert_eq!(certificate_hessian_is_pd(&indefinite), Some(false));
        assert_eq!(
            certificate_hessian_is_pd(&Array2::<f64>::zeros((0, 0))),
            None
        );
        let non_finite = array![[f64::NAN]];
        assert_eq!(certificate_hessian_is_pd(&non_finite), None);
    }

    #[test]
    fn certificate_rail_detection_uses_outer_box() {
        let config = OuterConfig::default(); // rho_bound = 30
        let rho = array![29.8, 0.0, -29.6];
        assert_eq!(certificate_railed_lambdas(&rho, 3, &config), vec![0, 2]);
        // Only the leading rho_dim coordinates are λ axes.
        assert_eq!(certificate_railed_lambdas(&rho, 1, &config), vec![0]);
        let bounded = OuterConfig {
            bounds: Some((array![-5.0, -5.0, -5.0], array![5.0, 5.0, 5.0])),
            ..OuterConfig::default()
        };
        let pinned = array![4.9, -4.7, 0.0];
        assert_eq!(certificate_railed_lambdas(&pinned, 3, &bounded), vec![0, 1]);
    }

    // The two `outer_scaled_tolerance_*` tests that lived here have
    // been removed: the helper is gone in favor of opt 0.5.0's
    // `GradientTolerance::relative_to_cost(τ)`. Equivalent threshold
    // coverage now lives upstream as
    // `opt::tests::gradient_tolerance_relative_to_cost_matches_textbook_form`.

    struct FailingSeedMaterializationOperator {
        dim: usize,
    }

    impl OuterHessianOperator for FailingSeedMaterializationOperator {
        fn dim(&self) -> usize {
            self.dim
        }

        fn matvec(&self, v: &Array1<f64>) -> Result<Array1<f64>, String> {
            Ok(v.clone())
        }

        fn is_cheap_to_materialize(&self) -> bool {
            true
        }

        fn materialize_dense(&self) -> Result<Array2<f64>, String> {
            Err("seed materialization failed".to_string())
        }
    }

    #[test]
    fn materialize_dense_uses_single_batched_mul_mat() {
        struct BatchedOnlyHessian {
            matrix: Array2<f64>,
            matvec_calls: Arc<AtomicUsize>,
            mul_mat_calls: Arc<AtomicUsize>,
            rhs_columns: Arc<AtomicUsize>,
        }

        impl OuterHessianOperator for BatchedOnlyHessian {
            fn dim(&self) -> usize {
                self.matrix.nrows()
            }

            fn matvec(&self, v: &Array1<f64>) -> Result<Array1<f64>, String> {
                self.matvec_calls.fetch_add(1, Ordering::Relaxed);
                Ok(self.matrix.dot(v))
            }

            fn mul_mat(&self, factor: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
                self.mul_mat_calls.fetch_add(1, Ordering::Relaxed);
                self.rhs_columns
                    .fetch_add(factor.ncols(), Ordering::Relaxed);
                Ok(self.matrix.dot(&factor))
            }
        }

        let matvec_calls = Arc::new(AtomicUsize::new(0));
        let mul_mat_calls = Arc::new(AtomicUsize::new(0));
        let rhs_columns = Arc::new(AtomicUsize::new(0));
        let op = BatchedOnlyHessian {
            matrix: array![[2.0, 0.25, -0.5], [0.5, 3.0, 1.0], [-0.25, 2.0, 4.0]],
            matvec_calls: Arc::clone(&matvec_calls),
            mul_mat_calls: Arc::clone(&mul_mat_calls),
            rhs_columns: Arc::clone(&rhs_columns),
        };

        let dense = op
            .materialize_dense()
            .expect("batched dense materialization");
        let expected = array![[2.0, 0.375, -0.375], [0.375, 3.0, 1.5], [-0.375, 1.5, 4.0]];
        assert_eq!(dense, expected);
        assert_eq!(
            mul_mat_calls.load(Ordering::Relaxed),
            1,
            "dense materialization must batch all identity columns into one mul_mat call"
        );
        assert_eq!(
            rhs_columns.load(Ordering::Relaxed),
            3,
            "the single batched materialization call must include every identity RHS"
        );
        assert_eq!(
            matvec_calls.load(Ordering::Relaxed),
            0,
            "operators with batched mul_mat must not be probed column-by-column"
        );
    }

    #[test]
    fn plan_analytic_hessian_selects_arc() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Either,
            n_params: 3,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Arc);
        assert_eq!(p.hessian_source, HessianSource::Analytic);
    }

    #[test]
    fn plan_prefer_gradient_only_does_not_hide_analytic_hessian() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Either,
            n_params: 3,
            psi_dim: 1,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: true,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Arc);
        assert_eq!(p.hessian_source, HessianSource::Analytic);
    }

    #[test]
    fn plan_survival_baseline_exact_hessian_selects_arc() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Either,
            n_params: 3,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Arc);
        assert_eq!(p.hessian_source, HessianSource::Analytic);
    }

    #[test]
    fn plan_no_hessian_few_params_selects_bfgs() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 3,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Bfgs);
        assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
    }

    #[test]
    fn plan_no_hessian_many_params_selects_bfgs() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 12,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Bfgs);
        assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
    }

    #[test]
    fn plan_cost_only_few_params_fails_loudly_with_bfgs() {
        // No analytic gradient, no analytic Hessian, few params, no
        // fixed-point lane: a genuinely cost-only objective, which is a
        // programming error since every outer objective now supplies an
        // analytic gradient. The planner emits Bfgs, which the runner rejects
        // loudly for needing a gradient the objective cannot supply — by
        // design, a cost-only objective has no working primary.
        let cap = OuterCapability {
            gradient: Derivative::Unavailable,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 5,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Bfgs);
    }

    #[test]
    fn plan_cost_only_many_params_with_fixed_point_still_efs() {
        // With the fixed-point lane eligible (many params,
        // fixed_point_available), a no-gradient/no-Hessian objective still
        // gets Efs.
        let cap = OuterCapability {
            gradient: Derivative::Unavailable,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 20,
            psi_dim: 0,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Efs);
        assert_eq!(p.hessian_source, HessianSource::EfsFixedPoint);
    }

    #[test]
    fn plan_no_gradient_with_declared_hessian_stays_bfgs() {
        // Contradictory capability (Hessian declared but no gradient) keeps the
        // Bfgs reject-with-context path.
        let cap = OuterCapability {
            gradient: Derivative::Unavailable,
            hessian: DeclaredHessianForm::Either,
            n_params: 4,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Bfgs);
        assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
    }

    #[test]
    fn plan_boundary_8_params_uses_bfgs() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: SMALL_OUTER_BFGS_MAX_PARAMS,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Bfgs);
        assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
    }

    #[test]
    fn plan_boundary_9_params_uses_bfgs() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: SMALL_OUTER_BFGS_MAX_PARAMS + 1,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Bfgs);
        assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
    }

    #[test]
    fn plan_efs_selected_for_penalty_like_many_params() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 15,
            psi_dim: 0,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Efs);
        assert_eq!(p.hessian_source, HessianSource::EfsFixedPoint);
    }

    #[test]
    fn plan_penalty_like_without_fixed_point_stays_bfgs() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 15,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Bfgs);
        assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
    }

    #[test]
    fn plan_efs_not_selected_few_params_even_if_penalty_like() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 5,
            psi_dim: 0,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Bfgs);
        assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
    }

    #[test]
    fn plan_efs_not_selected_with_analytic_hessian() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Either,
            n_params: 20,
            psi_dim: 0,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        // Arc is always preferred when analytic Hessian is available.
        assert_eq!(p.solver, Solver::Arc);
    }

    #[test]
    fn plan_efs_with_no_gradient_penalty_like_many_params() {
        // Even without analytic gradient, EFS works because it doesn't
        // need the gradient at all.
        let cap = OuterCapability {
            gradient: Derivative::Unavailable,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 20,
            psi_dim: 0,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Efs);
        assert_eq!(p.hessian_source, HessianSource::EfsFixedPoint);
    }

    #[test]
    fn plan_efs_allowed_with_barrier_config() {
        // When barrier_config is present (monotonicity constraints), EFS is
        // still selected at plan time. The runtime barrier-curvature guard
        // in the EFS loop handles safety.
        let barrier = BarrierConfig {
            tau: 1e-6,
            constrained_indices: vec![0, 1],
            lower_bounds: vec![0.0, 0.0],
            bound_signs: vec![1.0, 1.0],
        };
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 15,
            psi_dim: 0,
            fixed_point_available: true,
            barrier_config: Some(barrier),
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Efs);
        assert_eq!(p.hessian_source, HessianSource::EfsFixedPoint);
    }

    #[test]
    fn plan_efs_allowed_with_barrier_config_no_gradient() {
        // Even without analytic gradient, EFS is selected when all coords
        // are penalty-like and the problem is above the small-problem
        // BFGS cutoff, regardless of barrier presence.
        let barrier = BarrierConfig {
            tau: 1e-6,
            constrained_indices: vec![0],
            lower_bounds: vec![0.0],
            bound_signs: vec![1.0],
        };
        let cap = OuterCapability {
            gradient: Derivative::Unavailable,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 20,
            psi_dim: 0,
            fixed_point_available: true,
            barrier_config: Some(barrier),
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Efs);
        assert_eq!(p.hessian_source, HessianSource::EfsFixedPoint);
    }

    #[test]
    fn barrier_curvature_significant_blocks_efs_at_runtime() {
        // Verify that barrier_curvature_is_significant correctly detects
        // when coefficients are near their bounds.
        let barrier = BarrierConfig {
            tau: 1e-6,
            constrained_indices: vec![0],
            lower_bounds: vec![0.0],
            bound_signs: vec![1.0],
        };
        // β very close to bound → curvature is large
        let beta_near = Array1::from_vec(vec![0.001]);
        assert!(barrier.barrier_curvature_is_significant(&beta_near, 1.0, 0.01));

        // β far from bound → curvature is negligible
        let beta_far = Array1::from_vec(vec![10.0]);
        assert!(!barrier.barrier_curvature_is_significant(&beta_far, 1.0, 0.01));
    }

    #[test]
    fn barrier_curvature_locally_concentrated_covers_both_failure_modes() {
        // τ = 1e-6 (BarrierConfig default).
        // For the dimensional check τ/Δ² ≥ saturation_threshold:
        //   • Δ = 1e-3 ⇒ τ/Δ² = 1.0 (right at saturation = 1.0)
        //   • Δ = 1e-2 ⇒ τ/Δ² = 1e-2 (well below)
        //   • Δ = 1e-4 ⇒ τ/Δ² = 100 (well above)
        let barrier = BarrierConfig {
            tau: 1e-6,
            constrained_indices: vec![0, 1],
            lower_bounds: vec![0.0, 0.0],
            bound_signs: vec![1.0, 1.0],
        };

        // Mode (b) symmetric near-boundary: slacks uniform & both small.
        // With saturation = 1.0, Δ = 1e-2 stays under the saturation
        // wall and ratio is healthy → not concentrated. Δ = 1e-4
        // saturates absolutely → concentrated.
        let mild_uniform = Array1::from_vec(vec![1.0e-2, 1.0e-2]);
        assert!(!barrier.barrier_curvature_locally_concentrated(&mild_uniform, 0.1, 1.0));
        let tight_uniform = Array1::from_vec(vec![1.0e-4, 1.0e-4]);
        assert!(barrier.barrier_curvature_locally_concentrated(&tight_uniform, 0.1, 1.0));

        // Mode (b) is gated by saturation_threshold: with a very large
        // threshold (effectively disabling (b)), tight uniform stops
        // tripping until you also relax (a) — the asymmetric ratio
        // check — which on uniform slacks is necessarily false.
        assert!(!barrier.barrier_curvature_locally_concentrated(&tight_uniform, 0.1, 1.0e9));

        // Large uniform slacks: neither mode trips.
        let large_uniform = Array1::from_vec(vec![10.0, 10.0]);
        assert!(!barrier.barrier_curvature_locally_concentrated(&large_uniform, 0.1, 1.0));

        // Mode (a) asymmetric concentration: one slack 100× tighter
        // than the other, all in a regime where mode (b) DOESN'T fire.
        // Δ_min = 1e-2 ⇒ τ/Δ² = 1e-2 ≪ 1.0 saturation. So only the
        // ratio check is doing work here.
        let imbalanced = Array1::from_vec(vec![1.0e-2, 1.0]);
        assert!(barrier.barrier_curvature_locally_concentrated(&imbalanced, 0.1, 1.0));
        // With a permissive ratio (1e-3) and mode (b) effectively off
        // (huge threshold), neither check trips.
        assert!(!barrier.barrier_curvature_locally_concentrated(&imbalanced, 1.0e-3, 1.0e9));

        // Infeasible (β ≤ l) → conservatively concentrated.
        let infeasible = Array1::from_vec(vec![-0.5, 1.0]);
        assert!(barrier.barrier_curvature_locally_concentrated(&infeasible, 0.1, 1.0));
    }

    #[test]
    fn hessian_result_unwrap_analytic() {
        let h = Array2::<f64>::eye(3);
        let result = HessianResult::Analytic(h.clone());
        assert!(result.is_analytic());
        let extracted = result.unwrap_analytic();
        assert_eq!(extracted, h);
    }

    #[test]
    #[should_panic(expected = "expected analytic Hessian")]
    fn hessian_result_unwrap_unavailable_panics() {
        let result = HessianResult::Unavailable;
        result.unwrap_analytic();
    }

    #[test]
    fn zero_params_selects_arc() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Either,
            n_params: 0,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Arc);
        assert_eq!(p.hessian_source, HessianSource::Analytic);
    }

    #[test]
    fn hessian_result_into_option() {
        let h = Array2::<f64>::eye(2);
        let result = HessianResult::Analytic(h.clone());
        assert_eq!(result.into_option(), Some(h));

        let result = HessianResult::Unavailable;
        assert_eq!(result.into_option(), None);
    }

    #[test]
    fn closure_objective_delegates() {
        let mut obj = ClosureObjective {
            state: 42_i32,
            cap: OuterCapability {
                gradient: Derivative::Analytic,
                hessian: DeclaredHessianForm::Unavailable,
                n_params: 1,
                psi_dim: 0,
                fixed_point_available: false,
                barrier_config: None,
                prefer_gradient_only: false,
                disable_fixed_point: false,
            },
            cost_fn: |_: &mut i32, _: &Array1<f64>| Ok(1.0),
            eval_fn: |_: &mut i32, _: &Array1<f64>| {
                Ok(OuterEval {
                    cost: 1.0,
                    gradient: Array1::zeros(1),
                    hessian: HessianResult::Unavailable,
                    inner_beta_hint: None,
                })
            },
            eval_order_fn: None::<
                fn(&mut i32, &Array1<f64>, OuterEvalOrder) -> Result<OuterEval, EstimationError>,
            >,
            reset_fn: Some(|st: &mut i32| {
                *st = 42;
            }),
            efs_fn: None::<fn(&mut i32, &Array1<f64>) -> Result<EfsEval, EstimationError>>,
            screening_proxy_fn: None::<fn(&mut i32, &Array1<f64>) -> Result<f64, EstimationError>>,
            seed_fn: None::<fn(&mut i32, &Array1<f64>) -> Result<(), EstimationError>>,
            continuation_prewarm: true,
        };
        assert_eq!(obj.capability().n_params, 1);
        assert_eq!(obj.eval_cost(&Array1::zeros(1)).unwrap(), 1.0);
    }

    #[test]
    fn closure_objective_seed_inner_state_delegates_when_hook_present() {
        let mut obj = ClosureObjective {
            state: Vec::<f64>::new(),
            cap: OuterCapability {
                gradient: Derivative::Analytic,
                hessian: DeclaredHessianForm::Unavailable,
                n_params: 1,
                psi_dim: 0,
                fixed_point_available: false,
                barrier_config: None,
                prefer_gradient_only: false,
                disable_fixed_point: false,
            },
            cost_fn: |_: &mut Vec<f64>, _: &Array1<f64>| Ok(0.0),
            eval_fn: |_: &mut Vec<f64>, _: &Array1<f64>| {
                Ok(OuterEval {
                    cost: 0.0,
                    gradient: Array1::zeros(1),
                    hessian: HessianResult::Unavailable,
                    inner_beta_hint: None,
                })
            },
            eval_order_fn: None::<
                fn(
                    &mut Vec<f64>,
                    &Array1<f64>,
                    OuterEvalOrder,
                ) -> Result<OuterEval, EstimationError>,
            >,
            reset_fn: None::<fn(&mut Vec<f64>)>,
            efs_fn: None::<fn(&mut Vec<f64>, &Array1<f64>) -> Result<EfsEval, EstimationError>>,
            screening_proxy_fn: None::<
                fn(&mut Vec<f64>, &Array1<f64>) -> Result<f64, EstimationError>,
            >,
            seed_fn: None::<fn(&mut Vec<f64>, &Array1<f64>) -> Result<(), EstimationError>>,
            continuation_prewarm: true,
        }
        .with_seed_inner_state(|state: &mut Vec<f64>, beta: &Array1<f64>| {
            state.extend(beta.iter().copied());
            Ok(())
        });

        let outcome = obj.seed_inner_state(&array![1.5, -2.0]).unwrap();
        assert_eq!(outcome, SeedOutcome::Installed);
        assert_eq!(obj.state, vec![1.5, -2.0]);
    }

    #[test]
    fn hybrid_efs_backtracking_uses_half_step_after_first_rejection() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 12,
            psi_dim: 1,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let mut obj = ClosureObjective {
            state: (),
            cap: cap.clone(),
            cost_fn: |_: &mut (), theta: &Array1<f64>| {
                let psi = theta[11];
                let cost = if (psi - 0.0).abs() < 1e-12 {
                    1.0
                } else if (psi - 0.5).abs() < 1e-12 {
                    0.5
                } else {
                    2.0
                };
                Ok(cost)
            },
            eval_fn: |_: &mut (), theta: &Array1<f64>| {
                Ok(OuterEval {
                    cost: theta[11].abs(),
                    gradient: Array1::zeros(theta.len()),
                    hessian: HessianResult::Unavailable,
                    inner_beta_hint: None,
                })
            },
            eval_order_fn: None::<
                fn(&mut (), &Array1<f64>, OuterEvalOrder) -> Result<OuterEval, EstimationError>,
            >,
            reset_fn: None::<fn(&mut ())>,
            efs_fn: Some(|_: &mut (), theta: &Array1<f64>| {
                let mut steps = vec![0.0; theta.len()];
                steps[11] = 1.0;
                Ok(EfsEval {
                    cost: 1.0,
                    steps,
                    beta: None,
                    psi_gradient: Some(array![1.0]),
                    psi_indices: Some(vec![11]),
                    inner_hessian_scale: None,
                    logdet_enclosure_gap: None,
                })
            }),
            screening_proxy_fn: None::<fn(&mut (), &Array1<f64>) -> Result<f64, EstimationError>>,
            seed_fn: None::<fn(&mut (), &Array1<f64>) -> Result<(), EstimationError>>,
            continuation_prewarm: true,
        };
        let mut bridge = OuterFixedPointBridge {
            obj: &mut obj,
            layout: cap.theta_layout(),
            barrier_config: None,
            fixed_point_tolerance: 1e-8,
            consecutive_psi_zero_iters: 0,
        };

        let sample = bridge
            .eval_step(&Array1::zeros(cap.n_params))
            .expect("hybrid EFS step should backtrack cleanly");

        assert_eq!(sample.status, FixedPointStatus::Continue);
        assert_eq!(sample.step.len(), cap.n_params);
        assert_eq!(sample.step[11], 0.5);
        assert!(
            sample
                .step
                .iter()
                .enumerate()
                .all(|(idx, &value)| idx == 11 || value == 0.0)
        );
    }

    #[test]
    fn run_bfgs_mode_aware_eval_skips_hessian_work() {
        let seen_orders = Arc::new(Mutex::new(Vec::new()));
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Unavailable)
            .with_initial_rho(array![1.0])
            .with_max_iter(1);
        let mut obj = problem.build_objective_with_eval_order(
            (),
            |_: &mut (), theta: &Array1<f64>| Ok(theta[0] * theta[0]),
            |_: &mut (), _: &Array1<f64>| {
                Err(EstimationError::InvalidInput(
                    "legacy eager eval should not run on BFGS".to_string(),
                ))
            },
            {
                let seen_orders = Arc::clone(&seen_orders);
                move |_: &mut (), theta: &Array1<f64>, order: OuterEvalOrder| {
                    seen_orders.lock().unwrap().push(order);
                    Ok(OuterEval {
                        cost: theta[0] * theta[0],
                        gradient: array![2.0 * theta[0]],
                        hessian: HessianResult::Unavailable,
                        inner_beta_hint: None,
                    })
                }
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let result = problem
            .run(&mut obj, "mode-aware bfgs first order")
            .expect("BFGS should use the order-aware first-order bridge");
        assert_eq!(result.plan_used.solver, Solver::Bfgs);
        let seen_orders = seen_orders.lock().unwrap();
        assert!(
            !seen_orders.is_empty(),
            "mode-aware eval hook should have been used"
        );
        assert!(
            seen_orders
                .iter()
                .all(|order| *order != OuterEvalOrder::ValueGradientHessian),
            "BFGS must not request Hessian work, saw {seen_orders:?}"
        );
        assert!(
            seen_orders.contains(&OuterEvalOrder::ValueAndGradient),
            "BFGS should request value+gradient at accepted points, saw {seen_orders:?}"
        );
    }

    // The historical bridge-side `rejects_oversized_bfgs_cost_probe_before_objective`
    // test exercised a mechanism (returning `BFGS_LINE_SEARCH_REJECT_COST`
    // from `eval_cost` on overreach) that has been retired in favor of
    // `opt::Bfgs::with_axis_step_caps` — the line-search direction is now
    // shortened up front by opt itself, so the bridge never sees an
    // oversized probe in the first place. The equivalent invariant now
    // lives in opt's `with_axis_step_caps` test surface.

    #[test]
    fn first_order_bridge_keeps_true_gradient_on_repeated_flat_cost() {
        let eval_calls = Arc::new(AtomicUsize::new(0));
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Unavailable);
        let mut obj = problem.build_objective(
            (),
            |_: &mut (), _: &Array1<f64>| Ok(1000.0),
            {
                let eval_calls = Arc::clone(&eval_calls);
                move |_: &mut (), _: &Array1<f64>| {
                    let call = eval_calls.fetch_add(1, Ordering::Relaxed);
                    let cost = match call {
                        0 => 999.9995,
                        1 => 999.9990,
                        _ => 999.9987,
                    };
                    Ok(OuterEval {
                        cost,
                        gradient: array![4.0],
                        hessian: HessianResult::Unavailable,
                        inner_beta_hint: None,
                    })
                }
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let mut bridge = OuterFirstOrderBridge {
            obj: &mut obj,
            layout: OuterThetaLayout::new(1, 0),
            outer_inner_cap: None,
            iter_count: 0,
            g_norm_initial: None,
            last_g_norm: None,
            last_value_grad_rho: None,
            value_probe_cache: Vec::new(),
            cost_stall: None,
            consecutive_probe_refusals: 0,
        };

        let first = FirstOrderObjective::eval_grad(&mut bridge, &array![0.0])
            .expect("first flat-cost eval should expose the true gradient");
        let second = FirstOrderObjective::eval_grad(&mut bridge, &array![0.0])
            .expect("second flat-cost eval should expose the true gradient");
        let third = FirstOrderObjective::eval_grad(&mut bridge, &array![0.0])
            .expect("third flat-cost eval should expose the true gradient");
        let fourth = FirstOrderObjective::eval_grad(&mut bridge, &array![0.0])
            .expect("fourth flat-cost eval should expose the true gradient");

        assert_eq!(first.gradient[0], 4.0);
        assert_eq!(second.gradient[0], 4.0);
        assert_eq!(third.gradient[0], 4.0);
        assert_eq!(fourth.gradient[0], 4.0);
        assert_eq!(bridge.last_g_norm, Some(4.0));
        assert_eq!(eval_calls.load(Ordering::Relaxed), 4);
    }

    #[test]
    fn outer_second_order_bridge_separates_first_and_second_order_requests() {
        let seen_orders = Arc::new(Mutex::new(Vec::new()));
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Either);
        let mut obj = problem.build_objective_with_eval_order(
            (),
            |_: &mut (), theta: &Array1<f64>| Ok(theta[0] * theta[0]),
            |_: &mut (), _: &Array1<f64>| {
                Err(EstimationError::InvalidInput(
                    "legacy eager eval should not run".to_string(),
                ))
            },
            {
                let seen_orders = Arc::clone(&seen_orders);
                move |_: &mut (), theta: &Array1<f64>, order: OuterEvalOrder| {
                    seen_orders.lock().unwrap().push(order);
                    Ok(OuterEval {
                        cost: theta[0] * theta[0],
                        gradient: array![2.0 * theta[0]],
                        hessian: match order {
                            OuterEvalOrder::Value => HessianResult::Unavailable,
                            OuterEvalOrder::ValueAndGradient => HessianResult::Unavailable,
                            OuterEvalOrder::ValueGradientHessian => {
                                HessianResult::Analytic(array![[2.0]])
                            }
                        },
                        inner_beta_hint: None,
                    })
                }
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let mut bridge = OuterSecondOrderBridge {
            obj: &mut obj,
            layout: OuterThetaLayout::new(1, 0),
            hessian_source: HessianSource::Analytic,
            materialize_operator_max_dim: OUTER_HVP_MATERIALIZE_MAX_DIM,
            eval_count: 0,
            outer_inner_cap: None,
            g_norm_initial: None,
            last_g_norm: None,
            last_value_grad_rho: None,
        };
        let grad_sample =
            FirstOrderObjective::eval_grad(&mut bridge, &array![1.0]).expect("grad eval");
        assert_eq!(grad_sample.value, 1.0);
        assert_eq!(grad_sample.gradient, array![2.0]);
        let hess_sample =
            SecondOrderObjective::eval_hessian(&mut bridge, &array![1.0]).expect("hessian eval");
        assert_eq!(hess_sample.value, 1.0);
        assert_eq!(hess_sample.gradient, array![2.0]);
        assert_eq!(hess_sample.hessian, Some(array![[2.0]]));
        let seen_orders = seen_orders.lock().unwrap();
        assert!(
            *seen_orders
                == vec![
                    OuterEvalOrder::ValueAndGradient,
                    OuterEvalOrder::ValueGradientHessian
                ],
            "second-order bridge should split first-order and second-order requests, saw {seen_orders:?}"
        );
    }

    /// Phase 1.1 — On `HessianSource::Analytic` the bridge MUST surface a
    /// fatal error rather than producing `SecondOrderSample { hessian: None }`
    /// when the runtime returns `HessianResult::Unavailable`. A `None` here
    /// would let `opt::SecondOrderCache::finite_difference_hessian` silently
    /// estimate the Hessian by finite-differencing the gradient — at large-scale
    /// scale, hours of work per silently-mis-routed step. The seed loop
    /// should retry, demote, or fail loudly instead.
    #[test]
    fn analytic_route_unavailable_hessian_is_fatal() {
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Either);
        let mut obj = problem.build_objective_with_eval_order(
            (),
            |_: &mut (), theta: &Array1<f64>| Ok(theta[0] * theta[0]),
            |_: &mut (), _: &Array1<f64>| {
                Err(EstimationError::InvalidInput(
                    "legacy eager eval should not run".to_string(),
                ))
            },
            move |_: &mut (), theta: &Array1<f64>, _order: OuterEvalOrder| {
                Ok(OuterEval {
                    cost: theta[0] * theta[0],
                    gradient: array![2.0 * theta[0]],
                    hessian: HessianResult::Unavailable,
                    inner_beta_hint: None,
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let mut bridge = OuterSecondOrderBridge {
            obj: &mut obj,
            layout: OuterThetaLayout::new(1, 0),
            hessian_source: HessianSource::Analytic,
            materialize_operator_max_dim: OUTER_HVP_MATERIALIZE_MAX_DIM,
            eval_count: 0,
            outer_inner_cap: None,
            g_norm_initial: None,
            last_g_norm: None,
            last_value_grad_rho: None,
        };
        let err = SecondOrderObjective::eval_hessian(&mut bridge, &array![1.0])
            .expect_err("Analytic route must reject Unavailable Hessian, not pass None to opt");
        match err {
            ObjectiveEvalError::Fatal { message } => {
                assert!(
                    message.contains("HessianSource::Analytic") && message.contains("Unavailable"),
                    "fatal message should explain the analytic-route mismatch, saw: {message}"
                );
            }
            ObjectiveEvalError::Recoverable { message } => panic!(
                "Analytic-route Hessian violations must be Fatal (FD estimation is forbidden); \
                 got Recoverable: {message}"
            ),
        }
    }

    // Phase 5 (Cargo dep at opt 0.3) replaces the gam-side bridge
    // seed cache with `opt::{Bfgs, Arc, NewtonTrustRegion}::with_initial_sample`.
    // The two cache tests that lived here have been removed;
    // equivalent integration coverage now lives upstream as
    // `opt::tests::with_initial_sample_serves_first_call_from_cache`
    // and `opt::tests::bfgs_with_initial_sample_serves_first_call_from_cache`.
    // The fatal-on-Analytic-route contract (Phase 1.1) is still tested
    // here since it lives in gam's `build_bridge_hessian_for_source`.

    #[test]
    fn outer_config_default() {
        let cfg = OuterConfig::default();
        assert_eq!(cfg.tolerance, 1e-5);
        assert_eq!(cfg.max_iter, 200);
        assert_eq!(cfg.rho_bound, 30.0);
    }

    #[test]
    fn plan_hybrid_efs_selected_for_psi_coords_many_params() {
        // When ψ (design-moving) coords are present and the problem is above
        // the small-problem BFGS cutoff, the planner should select HybridEfs
        // instead of falling back to BFGS.
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 15,
            psi_dim: 1,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::HybridEfs);
        assert_eq!(p.hessian_source, HessianSource::HybridEfsFixedPoint);
    }

    #[test]
    fn plan_psi_without_fixed_point_stays_bfgs() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 15,
            psi_dim: 1,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Bfgs);
        assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
    }

    #[test]
    fn plan_hybrid_efs_no_gradient_selected_for_psi_coords() {
        // Even without analytic gradient, hybrid EFS works because the
        // gradient is computed internally by the unified evaluator.
        let cap = OuterCapability {
            gradient: Derivative::Unavailable,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 15,
            psi_dim: 1,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::HybridEfs);
        assert_eq!(p.hessian_source, HessianSource::HybridEfsFixedPoint);
    }

    // ----------------------------------------------------------------------
    // Routing regression tests (spec section 12).
    //
    // Post-#1 (compute-budget failure paths removed) and #2 (Hessian
    // cost-gating in custom_family.rs removed), the planner no longer
    // downgrades `(Analytic, Analytic)` to BFGS at any problem size. The
    // contract is:
    //
    //   high dense work + analytic+analytic     → ARC + Analytic
    //                                             (runtime then chooses
    //                                              operator HVP per family)
    //   high dense work + analytic + Unavailable → BFGS + BfgsApprox
    //                                             (matrix-free not advertised
    //                                              by the family — BFGS is
    //                                              still the right choice)
    //
    // `routing_log_line()` exposes a stable token that large-scale log
    // regressions in tests/bench_large_scale_runner_test.py pin against.
    // ----------------------------------------------------------------------

    fn cap_for_routing(
        gradient: Derivative,
        hessian: DeclaredHessianForm,
        n_params: usize,
    ) -> OuterCapability {
        OuterCapability {
            gradient,
            hessian,
            n_params,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        }
    }

    #[test]
    fn routing_analytic_analytic_stays_arc_at_large_scale() {
        // Large-scale standard GAM (n=320K, p=65, k=6) used to trigger the
        // aggregate `k·n·p²` cost-driven downgrade. Post-#1 the planner has
        // no scale-driven downgrade, so `(Analytic, Analytic)` must stay on
        // ARC + Analytic regardless of the problem dimensions.
        let cap = cap_for_routing(Derivative::Analytic, DeclaredHessianForm::Either, 6);
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Arc);
        assert_eq!(p.hessian_source, HessianSource::Analytic);
    }

    #[test]
    fn routing_analytic_analytic_stays_arc_at_dense_work_scale() {
        // n=3·10⁵, p=300 used to trigger the per-inner-solve `n·p²` downgrade
        // (`2.7·10¹⁰ ≫ 5·10⁹`). Post-#1, no work-hint API exists; ARC stays.
        let cap = cap_for_routing(Derivative::Analytic, DeclaredHessianForm::Either, 3);
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Arc);
        assert_eq!(p.hessian_source, HessianSource::Analytic);
    }

    #[test]
    fn routing_unavailable_hessian_routes_to_bfgs() {
        // Spec section 12: when the family cannot provide a second derivative
        // (matrix-free or otherwise), BFGS is the correct route.
        let cap = cap_for_routing(Derivative::Analytic, DeclaredHessianForm::Unavailable, 8);
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Bfgs);
        assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
    }

    #[test]
    fn routing_explicit_prefer_gradient_only_does_not_override_exact_hessian() {
        // The primary REML outer must never hide an analytic Hessian behind a
        // quasi-Newton route. Auxiliary gradient-only optimizers are separate
        // solver classes; this flag is ignored for Analytic+Analytic primary
        // capabilities.
        let mut cap = cap_for_routing(Derivative::Analytic, DeclaredHessianForm::Either, 6);
        cap.prefer_gradient_only = true;
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Arc);
        assert_eq!(p.hessian_source, HessianSource::Analytic);
    }

    #[test]
    fn routing_log_line_arc_analytic_does_not_advertise_matrix_free() {
        // Token pinned by tests/bench_large_scale_runner_test.py. Renaming
        // any of these substrings is a log-regression and breaks downstream
        // grep patterns.
        let p = OuterPlan {
            solver: Solver::Arc,
            hessian_source: HessianSource::Analytic,
        };
        let line = p.routing_log_line();
        assert!(line.contains("solver=Arc"), "got {line}");
        assert!(line.contains("hessian=Analytic"), "got {line}");
        assert!(line.contains("matrix-free=false"), "got {line}");
    }

    #[test]
    fn routing_log_line_bfgs_reports_no_matrix_free() {
        let p = OuterPlan {
            solver: Solver::Bfgs,
            hessian_source: HessianSource::BfgsApprox,
        };
        let line = p.routing_log_line();
        assert!(line.contains("solver=Bfgs"), "got {line}");
        assert!(line.contains("hessian=BfgsApprox"), "got {line}");
        assert!(line.contains("matrix-free=false"), "got {line}");
    }

    #[test]
    fn routing_log_line_efs_reports_no_matrix_free() {
        // EFS variants don't expose a Hessian operator either, so the
        // matrix-free token is `false`.
        for source in [
            HessianSource::EfsFixedPoint,
            HessianSource::HybridEfsFixedPoint,
        ] {
            let p = OuterPlan {
                solver: Solver::Efs,
                hessian_source: source,
            };
            assert!(
                p.routing_log_line().contains("matrix-free=false"),
                "{:?} should not advertise matrix-free",
                source
            );
        }
    }

    // ----------------------------------------------------------------------
    // Per-family routing regression tests.
    //
    // Each family that gains matrix-free Hessian operators must, at the
    // OuterProblem build site, declare both derivatives `Analytic` so the
    // planner stays on ARC + Analytic. These tests pin that contract from
    // the planner side. The runtime's choice between dense-Hessian-assembly
    // and operator-HVPs is independent of the planner; a separate per-family
    // test (in the family's own module) should pin that.
    //
    // ----------------------------------------------------------------------

    #[test]
    fn routing_custom_family_gamlss_stays_on_arc_when_both_derivs_analytic() {
        // Post-#5/#12, GAMLSS advertises matrix-free directional operators
        // for the joint Hessian; the OuterProblem build site must declare
        // both derivatives Analytic so ARC + Analytic stays in effect.
        let cap = cap_for_routing(Derivative::Analytic, DeclaredHessianForm::Either, 4);
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Arc);
        assert_eq!(p.hessian_source, HessianSource::Analytic);
    }

    #[test]
    fn routing_matern_iso_kappa_stays_on_arc_when_both_derivs_analytic() {
        // Post-#7, Matern/TPS spatial κ/τ derivative drifts ship as
        // HyperOperators; planner contract: (Analytic, Analytic) → ARC.
        let cap = cap_for_routing(Derivative::Analytic, DeclaredHessianForm::Either, 5);
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Arc);
        assert_eq!(p.hessian_source, HessianSource::Analytic);
    }

    #[test]
    fn routing_matern_iso_large_kappa_dim_stays_on_arc_with_analytic_hessian() {
        // Spatial isotropic κ no longer declares Hessian unavailable when
        // kappa_dim > 30.  Large κ blocks are represented by exact HVP
        // operators at evaluation time, so the planner must keep second-order
        // ARC instead of selecting HybridEFS.
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Either,
            n_params: 37,
            psi_dim: 31,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Arc);
        assert_eq!(p.hessian_source, HessianSource::Analytic);
    }

    #[test]
    fn routing_marginal_slope_stays_on_arc_when_both_derivs_analytic() {
        // Bernoulli/survival marginal-slope: the planner contract is the
        // same — (Analytic, Analytic) → ARC + Analytic. Runtime selects
        // operator HVPs via `use_joint_matrix_free_path`.
        let cap = cap_for_routing(Derivative::Analytic, DeclaredHessianForm::Either, 3);
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Arc);
        assert_eq!(p.hessian_source, HessianSource::Analytic);
    }

    #[test]
    fn plan_hybrid_efs_not_selected_few_params() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 5,
            psi_dim: 1,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Bfgs);
        assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
    }

    #[test]
    fn plan_exact_hvp_capability_selects_arc_even_when_fixed_point_is_available() {
        // Large spatial/custom-family problems may also expose EFS/HybridEFS
        // fixed-point traces, but an explicit dense Hessian or exact HVP
        // operator is stronger geometry. The planner must therefore select
        // ARC + Analytic rather than cost-demoting to BFGS/EFS when the
        // evaluator advertises second-order capability.
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Either,
            n_params: 64,
            psi_dim: 16,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: true,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Arc);
        assert_eq!(p.hessian_source, HessianSource::Analytic);
    }

    #[test]
    fn plan_hybrid_efs_not_selected_with_analytic_hessian() {
        // Arc is always preferred when analytic Hessian is available,
        // even with ψ coordinates.
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Either,
            n_params: 20,
            psi_dim: 1,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Arc);
    }

    #[test]
    fn plan_pure_efs_not_hybrid_when_all_penalty_like() {
        // When all coords are penalty-like (no ψ), pure EFS is selected
        // even if has_psi_coords is false.
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 15,
            psi_dim: 0,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Efs);
        assert_eq!(p.hessian_source, HessianSource::EfsFixedPoint);
    }

    #[test]
    fn automatic_fallbacks_preserve_analytic_hessian_for_arc_primary() {
        // For an (Analytic, Analytic) capability the planner emits ARC. The
        // cascade MUST NOT add a BFGS+BfgsApprox demotion: doing so discards
        // the analytic outer Hessian ARC was using, replaces it with a
        // strictly weaker rank-2 approximation, and silently masks ARC's
        // actual failure mode (budget exhaustion, indefinite curvature)
        // under a BFGS Strong-Wolfe plateau. ARC budget exhaustion is
        // handled by the per-attempt retry ladder in
        // `run_outer_with_strategy`; once that is exhausted, the caller
        // sees the genuine analytic-Hessian non-convergence verbatim.
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Either,
            n_params: 12,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        assert_eq!(plan(&cap).solver, Solver::Arc);
        let attempts = automatic_fallback_attempts(&cap);
        assert!(
            attempts.is_empty(),
            "ARC primary must not lateral-demote to BFGS+BfgsApprox; \
             ARC budget retries live in the runner",
        );
    }

    #[test]
    fn automatic_fallbacks_from_efs_prefer_analytic_bfgs_over_fd() {
        // When the primary plan is EFS, the first fallback must keep the
        // analytic gradient and just disable the fixed-point path so the
        // planner picks gradient-based BFGS. Silently downgrading to finite
        // differences here was the long-standing production bug we are
        // guarding against.
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 15,
            psi_dim: 0,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        assert_eq!(plan(&cap).solver, Solver::Efs);

        let attempts = automatic_fallback_attempts(&cap);
        assert!(!attempts.is_empty(), "EFS failure must have a fallback");
        assert_eq!(attempts[0].gradient, Derivative::Analytic);
        assert_eq!(attempts[0].hessian, DeclaredHessianForm::Unavailable);
        assert!(attempts[0].disable_fixed_point);
        assert_eq!(plan(&attempts[0]).solver, Solver::Bfgs);

        assert!(
            attempts.iter().all(|c| c.gradient == Derivative::Analytic),
            "fallback cascade must stay on analytic-gradient attempts",
        );
    }

    #[test]
    fn automatic_fallbacks_from_hybrid_efs_prefer_analytic_bfgs_over_fd() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 15,
            psi_dim: 2,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        assert_eq!(plan(&cap).solver, Solver::HybridEfs);

        let attempts = automatic_fallback_attempts(&cap);
        assert!(!attempts.is_empty());
        assert_eq!(attempts[0].gradient, Derivative::Analytic);
        assert!(attempts[0].disable_fixed_point);
        assert_eq!(plan(&attempts[0]).solver, Solver::Bfgs);
    }

    #[test]
    fn disabled_fallback_hybrid_efs_capability_routes_to_bfgs_primary() {
        // Production Matérn60 exact adaptive regularization at large scale:
        // rho_dim=3 retained quadratic penalties, psi_dim=6 adaptive λ/ε
        // coordinates, n_params=9, analytic gradient, and exact outer Hessian
        // cost-gated unavailable. Structurally this is HybridEFS-shaped, but
        // HybridEFS with ψ coordinates is not a standalone primary solver: its
        // ψ backtracking path can legitimately request the first-order escape
        // ladder. If that ladder is disabled, the runner must route the primary
        // attempt directly to BFGS instead of relying on call sites to remember
        // `.with_disable_fixed_point(true)`.
        let trapped_cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 9,
            psi_dim: 6,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        assert_eq!(plan(&trapped_cap).solver, Solver::HybridEfs);

        let disabled_config = OuterConfig {
            fallback_policy: FallbackPolicy::Disabled,
            ..OuterConfig::default()
        };
        let primary_cap = primary_capability_for_config(
            trapped_cap.clone(),
            &disabled_config,
            "large-scale exact adaptive",
        );
        assert!(primary_cap.disable_fixed_point);
        assert_eq!(plan(&primary_cap).solver, Solver::Bfgs);

        let pure_efs_cap = OuterCapability {
            psi_dim: 0,
            ..trapped_cap.clone()
        };
        assert_eq!(plan(&pure_efs_cap).solver, Solver::Efs);
        let pure_primary_cap =
            primary_capability_for_config(pure_efs_cap.clone(), &disabled_config, "pure EFS");
        assert!(!pure_primary_cap.disable_fixed_point);
        assert_eq!(plan(&pure_primary_cap).solver, Solver::Efs);

        let no_gradient_cap = OuterCapability {
            gradient: Derivative::Unavailable,
            ..trapped_cap.clone()
        };
        assert_eq!(plan(&no_gradient_cap).solver, Solver::HybridEfs);
        let no_gradient_primary_cap = primary_capability_for_config(
            no_gradient_cap.clone(),
            &disabled_config,
            "gradient-unavailable hybrid EFS",
        );
        assert!(!no_gradient_primary_cap.disable_fixed_point);
        assert_eq!(plan(&no_gradient_primary_cap).solver, Solver::HybridEfs);

        let automatic_config = OuterConfig::default();
        let automatic_cap = primary_capability_for_config(
            trapped_cap.clone(),
            &automatic_config,
            "large-scale exact adaptive",
        );
        assert!(!automatic_cap.disable_fixed_point);
        assert_eq!(plan(&automatic_cap).solver, Solver::HybridEfs);

        let automatic_attempts = automatic_fallback_attempts(&trapped_cap);
        assert!(!automatic_attempts.is_empty());
        assert!(automatic_attempts[0].disable_fixed_point);
        assert_eq!(plan(&automatic_attempts[0]).solver, Solver::Bfgs);
    }

    #[test]
    fn disabled_fallback_hybrid_efs_problem_uses_bfgs_without_calling_efs() {
        let efs_calls = Arc::new(AtomicUsize::new(0));
        let problem = OuterProblem::new(9)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Unavailable)
            .with_psi_dim(6)
            .with_fallback_policy(FallbackPolicy::Disabled)
            .with_initial_rho(Array1::zeros(9))
            .with_max_iter(5);
        let mut obj = problem.build_objective(
            (),
            |_: &mut (), theta: &Array1<f64>| Ok(0.5 * theta.dot(theta)),
            |_: &mut (), theta: &Array1<f64>| {
                Ok(OuterEval {
                    cost: 0.5 * theta.dot(theta),
                    gradient: theta.clone(),
                    hessian: HessianResult::Unavailable,
                    inner_beta_hint: None,
                })
            },
            None::<fn(&mut ())>,
            {
                let efs_calls = Arc::clone(&efs_calls);
                Some(move |_: &mut (), _: &Array1<f64>| {
                    efs_calls.fetch_add(1, Ordering::Relaxed);
                    Err(EstimationError::RemlOptimizationFailed(format!(
                        "{} synthetic large-scale adaptive HybridEFS escape",
                        EFS_FIRST_ORDER_FALLBACK_MARKER,
                    )))
                })
            },
        );

        let result = problem
            .run(&mut obj, "disabled fallback marker")
            .expect("disabled-fallback HybridEFS-shaped problem should route directly to BFGS");
        assert_eq!(result.plan_used.solver, Solver::Bfgs);
        assert_eq!(
            efs_calls.load(Ordering::Relaxed),
            0,
            "central primary-capability canonicalization should avoid the EFS hook entirely"
        );
    }

    #[test]
    fn automatic_fallbacks_without_gradient_stop_at_fixed_point_status() {
        for (psi_dim, expected_solver) in [(0, Solver::Efs), (2, Solver::HybridEfs)] {
            let cap = OuterCapability {
                gradient: Derivative::Unavailable,
                hessian: DeclaredHessianForm::Unavailable,
                n_params: 15,
                psi_dim,
                fixed_point_available: true,
                barrier_config: None,
                prefer_gradient_only: false,
                disable_fixed_point: false,
            };
            assert_eq!(plan(&cap).solver, expected_solver);
            assert!(
                automatic_fallback_attempts(&cap).is_empty(),
                "gradient-unavailable fixed-point capabilities must not fabricate a BFGS fallback",
            );
        }
    }

    #[test]
    fn automatic_fallbacks_do_not_repeat_arc_when_fixed_point_is_irrelevant() {
        // The contract here is that the cascade does not lateral-hop ARC
        // through the EFS planner arm when `fixed_point_available=true` is
        // incidentally set on an (Analytic, Analytic) capability that the
        // planner already chose ARC for. Combined with the
        // analytic-Hessian-preservation contract enforced by
        // `automatic_fallbacks_preserve_analytic_hessian_for_arc_primary`,
        // the ARC primary now has zero degraded fallbacks — the runner's
        // ARC budget-bump retry ladder owns recovery.
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Either,
            n_params: 15,
            psi_dim: 0,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        assert_eq!(plan(&cap).solver, Solver::Arc);

        let attempts = automatic_fallback_attempts(&cap);
        assert!(
            attempts.is_empty(),
            "ARC primary with incidental fixed_point_available must not \
             cascade through the EFS arm or lateral-demote to BFGS",
        );
    }

    #[test]
    fn plan_disable_fixed_point_forces_bfgs_even_when_efs_eligible() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 15,
            psi_dim: 0,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: true,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Bfgs);
        assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
    }

    #[test]
    fn run_malformed_gradient_seed_surfaces_as_error() {
        // A capability that declares Analytic gradient but returns a malformed
        // one must fail loudly. The previous numerical-gradient fallback masked
        // the underlying bug by silently spinning a cost-only BFGS; that path is
        // disabled in production.
        let problem = OuterProblem::new(2)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Unavailable)
            .with_initial_rho(Array1::zeros(2))
            .with_max_iter(1);
        let mut obj = problem.build_objective(
            (),
            |_: &mut (), _: &Array1<f64>| Ok(0.0),
            |_: &mut (), _: &Array1<f64>| {
                Ok(OuterEval {
                    cost: 0.0,
                    gradient: Array1::zeros(1),
                    hessian: HessianResult::Unavailable,
                    inner_beta_hint: None,
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let err = problem
            .run(&mut obj, "test gradient mismatch")
            .expect_err("malformed analytic gradient must surface as error");
        assert!(
            matches!(err, EstimationError::RemlOptimizationFailed(_)),
            "unexpected error variant: {err:?}",
        );
    }

    #[test]
    fn run_bfgs_ignores_malformed_hessian_payload() {
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Unavailable)
            .with_initial_rho(array![0.0])
            .with_max_iter(1);
        let mut obj = problem.build_objective(
            (),
            |_: &mut (), theta: &Array1<f64>| Ok(theta[0] * theta[0]),
            |_: &mut (), theta: &Array1<f64>| {
                Ok(OuterEval {
                    cost: theta[0] * theta[0],
                    gradient: array![2.0 * theta[0]],
                    // First-order paths must ignore Hessian payload quality.
                    hessian: HessianResult::Analytic(array![[f64::NAN, 0.0], [0.0, 1.0]]),
                    inner_beta_hint: None,
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let result = problem
            .run(&mut obj, "bfgs should ignore malformed hessian payload")
            .expect("valid first-order data should be enough for BFGS");
        assert_eq!(result.plan_used.solver, Solver::Bfgs);
        assert_eq!(result.plan_used.hessian_source, HessianSource::BfgsApprox);
    }

    #[test]
    fn finite_outer_eval_reports_gradient_length_mismatch() {
        let err = finite_outer_eval_or_error(
            "test gradient mismatch",
            OuterThetaLayout::new(2, 0),
            OuterEval {
                cost: 0.0,
                gradient: Array1::zeros(1),
                hessian: HessianResult::Unavailable,
                inner_beta_hint: None,
            },
        )
        .expect_err("gradient mismatch should be rejected");
        let message = match err {
            ObjectiveEvalError::Recoverable { message } | ObjectiveEvalError::Fatal { message } => {
                message
            }
        };
        assert!(
            message.contains("outer gradient length mismatch"),
            "unexpected error: {message}"
        );
    }

    #[test]
    fn run_with_initial_seed_still_considers_generated_candidates() {
        let generated = crate::seeding::generate_rho_candidates(
            1,
            None,
            &crate::seeding::SeedConfig::default(),
        );
        let valid_seed = generated
            .first()
            .expect("seed generator should yield at least one candidate")
            .clone();
        let expected_seed = valid_seed.clone();
        let initial_seed = array![9.0];
        let mut seed_config = crate::seeding::SeedConfig::default();
        seed_config.seed_budget = 1;
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Unavailable)
            .with_seed_config(seed_config)
            .with_initial_rho(initial_seed)
            .with_max_iter(1);
        let mut obj = problem.build_objective(
            (),
            {
                let valid_seed = valid_seed.clone();
                move |_: &mut (), theta: &Array1<f64>| {
                    if theta == valid_seed {
                        Ok(0.0)
                    } else {
                        Ok(f64::INFINITY)
                    }
                }
            },
            move |_: &mut (), theta: &Array1<f64>| {
                if theta == valid_seed {
                    Ok(OuterEval {
                        cost: 0.0,
                        gradient: Array1::zeros(1),
                        hessian: HessianResult::Unavailable,
                        inner_beta_hint: None,
                    })
                } else {
                    Ok(OuterEval::infeasible(theta.len()))
                }
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let result = problem
            .run(&mut obj, "generated seed should remain reachable")
            .expect("generated seed should still be eligible when an initial seed is provided");
        assert_eq!(result.rho, expected_seed);
    }

    #[test]
    fn run_indefinite_analytic_seed_stays_on_arc() {
        let mut seed_config = crate::seeding::SeedConfig::default();
        seed_config.seed_budget = 1;
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Either)
            .with_seed_config(seed_config)
            .with_initial_rho(array![0.0])
            .with_max_iter(1);
        let mut obj = problem.build_objective(
            (),
            |_: &mut (), theta: &Array1<f64>| Ok(theta[0] * theta[0]),
            |_: &mut (), _: &Array1<f64>| {
                Ok(OuterEval {
                    cost: 0.0,
                    gradient: array![0.0],
                    hessian: HessianResult::Analytic(array![[-1.0]]),
                    inner_beta_hint: None,
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let result = problem
            .run(&mut obj, "indefinite seed geometry")
            .expect("indefinite analytic seed geometry should stay on the second-order plan");
        assert_eq!(result.plan_used.solver, Solver::Arc);
        assert_eq!(result.plan_used.hessian_source, HessianSource::Analytic);
    }

    #[test]
    fn run_seed_materialization_failure_surfaces_arc_error_verbatim() {
        // Under the budget-bump retry ladder (commit c96c4233), an ARC
        // primary with `(Analytic, Analytic)` capability has zero degraded
        // fallbacks. A seed-materialization failure surfaces as `Err`
        // verbatim — there is no lateral demote to BFGS+BfgsApprox that
        // would silently discard the analytic outer Hessian. Materialization
        // failures are deterministic w.r.t. rho, so the budget-bump retry
        // ladder cannot rescue them; the operator returns the same Err on
        // every retry. Hence the runner returns the original Err.
        let mut seed_config = crate::seeding::SeedConfig::default();
        seed_config.seed_budget = 1;
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Either)
            .with_seed_config(seed_config)
            .with_initial_rho(array![0.0])
            .with_max_iter(1);
        let mut obj = problem.build_objective(
            (),
            |_: &mut (), theta: &Array1<f64>| Ok(theta[0] * theta[0]),
            |_: &mut (), _: &Array1<f64>| {
                Ok(OuterEval {
                    cost: 0.0,
                    gradient: array![0.0],
                    hessian: HessianResult::Operator(Arc::new(
                        FailingSeedMaterializationOperator { dim: 1 },
                    )),
                    inner_beta_hint: None,
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let err = problem
            .run(&mut obj, "seed materialization failure")
            .expect_err(
                "ARC primary must surface the materialization failure verbatim — \
                 no lateral demote to BFGS+BfgsApprox",
            );
        let msg = err.to_string();
        assert!(
            msg.contains("seed materialization failed"),
            "error must propagate the underlying materialization message; got: {msg}"
        );
    }

    #[test]
    fn run_nonconverged_arc_stays_on_arc_after_budget_retry_ladder() {
        // When an ARC primary exhausts its iteration budget, the runner
        // reseeds a fresh ARC attempt from the previous attempt's last
        // ρ and trust radius (up to two retries) and uncaps the inner
        // PIRLS cap for the resumed run via the InnerProgressFeedback
        // handle. Retries are gated on attempt-over-attempt `‖g‖`
        // halving so a deterministic-replay trajectory falls through.
        // The objective's analytic outer Hessian is preserved across
        // every attempt — no lateral demote to BFGS+BfgsApprox. After
        // the retries are exhausted (or the gate fires), the runner
        // returns the final `Ok(OuterResult{converged:false})` from
        // the last ARC attempt; the plan stays ARC + Analytic Hessian.
        //
        // We use `cost = x^4`, `grad = 4 x^3`, `hess = 12 x^2` from
        // `initial_rho = [5.0]` with `max_iter = 1`. Newton-style ARC
        // steps on x^4 contract the gradient by ~3× per attempt, so
        // the halving gate passes and both retries proceed; ARC still
        // cannot reach the optimum in three single-iter attempts.
        let mut seed_config = crate::seeding::SeedConfig::default();
        seed_config.seed_budget = 1;
        let (_d, session) = tmp_cache_session("nonconverged-arc-cache");
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Either)
            .with_seed_config(seed_config)
            .with_initial_rho(array![5.0])
            .with_max_iter(1)
            .with_cache_session(Arc::clone(&session));
        let mut obj = problem.build_objective(
            (),
            |_: &mut (), theta: &Array1<f64>| Ok(theta[0].powi(4)),
            |_: &mut (), theta: &Array1<f64>| {
                let x = theta[0];
                Ok(OuterEval {
                    cost: x.powi(4),
                    gradient: array![4.0 * x.powi(3)],
                    hessian: HessianResult::Analytic(array![[12.0 * x.powi(2)]]),
                    inner_beta_hint: None,
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let result = problem
            .run(&mut obj, "nonconverged arc should stay on arc")
            .expect(
                "ARC ladder must surface the last non-converged ARC result rather than \
                 demoting to BFGS+BfgsApprox",
            );
        assert_eq!(
            result.plan_used.solver,
            Solver::Arc,
            "ARC primary must not lateral-demote after budget exhaustion"
        );
        assert_eq!(
            result.plan_used.hessian_source,
            HessianSource::Analytic,
            "analytic outer Hessian must be preserved across the budget-bump retry ladder"
        );
        assert!(
            !result.converged,
            "test fixture is engineered so the ladder cannot converge; \
             converged=true would mean the fixture stopped exercising the ladder"
        );
    }

    #[test]
    fn candidate_selection_prefers_lower_cost_within_same_convergence_class() {
        let plan = OuterPlan {
            solver: Solver::Bfgs,
            hessian_source: HessianSource::BfgsApprox,
        };
        let mut nonconverged_hi = OuterResult::new(array![0.0], 9.0, 1, false, plan);
        nonconverged_hi.final_grad_norm = Some(1.0);
        let mut nonconverged_lo = OuterResult::new(
            array![1.0],
            1.0,
            1,
            false,
            OuterPlan {
                solver: Solver::Bfgs,
                hessian_source: HessianSource::BfgsApprox,
            },
        );
        nonconverged_lo.final_grad_norm = Some(1.0);
        let mut converged = OuterResult::new(
            array![2.0],
            5.0,
            1,
            true,
            OuterPlan {
                solver: Solver::Bfgs,
                hessian_source: HessianSource::BfgsApprox,
            },
        );
        converged.final_grad_norm = Some(0.0);

        assert!(candidate_improves_best(&nonconverged_hi, None));
        assert!(candidate_improves_best(
            &nonconverged_lo,
            Some(&nonconverged_hi)
        ));
        assert!(!candidate_improves_best(
            &nonconverged_hi,
            Some(&nonconverged_lo)
        ));
        assert!(candidate_improves_best(&converged, Some(&nonconverged_lo)));
        assert!(!candidate_improves_best(&nonconverged_lo, Some(&converged)));
    }

    #[test]
    fn gaussian_multistart_compares_converged_seed_costs() {
        let mut seed_config = crate::seeding::SeedConfig::default();
        seed_config.seed_budget = 2;
        seed_config.risk_profile = crate::seeding::SeedRiskProfile::Gaussian;
        let started = Arc::new(Mutex::new(Vec::new()));
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Unavailable)
            .with_seed_config(seed_config)
            .with_max_iter(4);
        let mut obj = problem.build_objective(
            (),
            |_: &mut (), theta: &Array1<f64>| Ok(if theta[0] < -1.0 { 0.0 } else { 10.0 }),
            {
                let started = Arc::clone(&started);
                move |_: &mut (), theta: &Array1<f64>| {
                    started.lock().unwrap().push(theta.clone());
                    Ok(OuterEval {
                        cost: if theta[0] < -1.0 { 0.0 } else { 10.0 },
                        gradient: array![0.0],
                        hessian: HessianResult::Unavailable,
                        inner_beta_hint: None,
                    })
                }
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let result = problem
            .run(&mut obj, "Gaussian quality multistart")
            .expect("Gaussian multistart should compare both converged seeds");
        let starts = started.lock().unwrap();
        assert!(
            starts.len() >= 2,
            "Gaussian quality mode should not stop at the first converged seed"
        );
        assert!(
            result.rho[0] < -1.0,
            "lower-cost converged Gaussian seed should win"
        );
        assert_eq!(result.final_value, 0.0);
    }

    #[test]
    fn run_starts_solver_with_direct_startup_eval() {
        let mut seed_config = crate::seeding::SeedConfig::default();
        seed_config.seed_budget = 1;
        let calls = Arc::new(Mutex::new(Vec::new()));
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Either)
            .with_seed_config(seed_config)
            .with_max_iter(1);
        let mut obj = problem.build_objective(
            (),
            {
                let calls = Arc::clone(&calls);
                move |_: &mut (), theta: &Array1<f64>| {
                    calls.lock().unwrap().push("cost");
                    Ok(theta[0] * theta[0])
                }
            },
            {
                let calls = Arc::clone(&calls);
                move |_: &mut (), theta: &Array1<f64>| {
                    calls.lock().unwrap().push("eval");
                    Ok(OuterEval {
                        cost: theta[0] * theta[0],
                        gradient: array![2.0 * theta[0]],
                        hessian: HessianResult::Analytic(array![[2.0]]),
                        inner_beta_hint: None,
                    })
                }
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        problem
            .run(&mut obj, "solver should start from a direct startup eval")
            .expect("analytic plans should start with a direct full evaluation");
        let calls = calls.lock().unwrap();
        let first_eval_idx = calls
            .iter()
            .position(|call| *call == "eval")
            .expect("solver should eventually request a full eval");
        assert!(
            first_eval_idx == 0,
            "startup should not perform a separate cost-screening pass first: {calls:?}"
        );
    }

    #[test]
    fn run_screening_reorders_expensive_generated_seeds_before_full_startup_eval() {
        let mut seed_config = crate::seeding::SeedConfig::default();
        seed_config.max_seeds = 4;
        seed_config.seed_budget = 2;
        seed_config.risk_profile = crate::seeding::SeedRiskProfile::GeneralizedLinear;
        let screening_cap = Arc::new(AtomicUsize::new(0));
        let valid_seed = crate::seeding::generate_rho_candidates(1, None, &seed_config)
            .last()
            .expect("seed generator should yield at least one candidate")
            .clone();
        let started = Arc::new(Mutex::new(Vec::new()));
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Either)
            .with_seed_config(seed_config)
            .with_screening_cap(Arc::clone(&screening_cap))
            .with_max_iter(1);
        let mut obj = problem.build_objective(
            (),
            {
                let valid_seed = valid_seed.clone();
                move |_: &mut (), theta: &Array1<f64>| {
                    if theta == valid_seed {
                        Ok(0.0)
                    } else {
                        Ok(1000.0)
                    }
                }
            },
            {
                let valid_seed = valid_seed.clone();
                let started = Arc::clone(&started);
                move |_: &mut (), theta: &Array1<f64>| {
                    started.lock().unwrap().push(theta.clone());
                    if theta == valid_seed {
                        Ok(OuterEval {
                            cost: 0.0,
                            gradient: array![0.0],
                            hessian: HessianResult::Analytic(array![[1.0]]),
                            inner_beta_hint: None,
                        })
                    } else {
                        Ok(OuterEval::infeasible(theta.len()))
                    }
                }
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let result = problem
            .run(&mut obj, "screening should reorder expensive seeds")
            .expect("screened startup should reach the best generated seed");
        assert_eq!(result.rho, valid_seed);
        assert_eq!(
            started.lock().unwrap().first().cloned(),
            Some(valid_seed),
            "screening should move the lowest-cost seed to the front before full startup eval",
        );
        assert_eq!(screening_cap.load(std::sync::atomic::Ordering::Relaxed), 0);
    }

    #[test]
    fn initial_rho_with_single_seed_budget_skips_expensive_screening() {
        let mut seed_config = crate::seeding::SeedConfig::default();
        seed_config.max_seeds = 4;
        seed_config.seed_budget = 1;
        seed_config.risk_profile = crate::seeding::SeedRiskProfile::GeneralizedLinear;
        let screening_cap = Arc::new(AtomicUsize::new(0));
        let screening_calls = Arc::new(AtomicUsize::new(0));
        let initial_seed = array![9.0];
        let started = Arc::new(Mutex::new(Vec::new()));
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Either)
            .with_seed_config(seed_config)
            .with_screening_cap(Arc::clone(&screening_cap))
            .with_initial_rho(initial_seed.clone())
            .with_max_iter(1);
        let mut obj = problem.build_objective(
            (),
            {
                let screening_calls = Arc::clone(&screening_calls);
                move |_: &mut (), _theta: &Array1<f64>| {
                    screening_calls.fetch_add(1, Ordering::Relaxed);
                    Ok(0.0)
                }
            },
            {
                let started = Arc::clone(&started);
                let initial_seed = initial_seed.clone();
                move |_: &mut (), theta: &Array1<f64>| {
                    started.lock().unwrap().push(theta.clone());
                    if theta == initial_seed {
                        Ok(OuterEval {
                            cost: 0.0,
                            gradient: array![0.0],
                            hessian: HessianResult::Analytic(array![[1.0]]),
                            inner_beta_hint: None,
                        })
                    } else {
                        Ok(OuterEval::infeasible(theta.len()))
                    }
                }
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let result = problem
            .run(&mut obj, "initial rho should be authoritative")
            .expect("initial-rho startup should not spend seed-screening solves");
        assert_eq!(result.rho, initial_seed);
        assert_eq!(
            screening_calls.load(Ordering::Relaxed),
            0,
            "explicit initial rho plus seed_budget=1 should skip screening"
        );
        assert_eq!(
            started.lock().unwrap().first().cloned(),
            Some(initial_seed),
            "solver should start from the explicit initial rho"
        );
        assert_eq!(screening_cap.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn run_screening_reorders_bfgs_seeds_before_full_startup_eval() {
        let mut seed_config = crate::seeding::SeedConfig::default();
        seed_config.seed_budget = 1;
        seed_config.risk_profile = crate::seeding::SeedRiskProfile::Gaussian;
        let screening_cap = Arc::new(AtomicUsize::new(0));
        let initial_seed = array![9.0];
        let valid_seed = crate::seeding::generate_rho_candidates(1, None, &seed_config)
            .first()
            .expect("seed generator should yield at least one candidate")
            .clone();
        let started = Arc::new(Mutex::new(Vec::new()));
        let screening_calls = Arc::new(AtomicUsize::new(0));
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Unavailable)
            .with_seed_config(seed_config)
            .with_screening_cap(Arc::clone(&screening_cap))
            .with_initial_rho(initial_seed)
            .with_screen_initial_rho(true)
            .with_max_iter(1);
        let mut obj = problem.build_objective(
            (),
            {
                let valid_seed = valid_seed.clone();
                let screening_calls = Arc::clone(&screening_calls);
                move |_: &mut (), theta: &Array1<f64>| {
                    screening_calls.fetch_add(1, Ordering::Relaxed);
                    if theta == valid_seed {
                        Ok(0.0)
                    } else {
                        Ok(1000.0)
                    }
                }
            },
            {
                let valid_seed = valid_seed.clone();
                let started = Arc::clone(&started);
                move |_: &mut (), theta: &Array1<f64>| {
                    started.lock().unwrap().push(theta.clone());
                    if theta == valid_seed {
                        Ok(OuterEval {
                            cost: 0.0,
                            gradient: array![0.0],
                            hessian: HessianResult::Unavailable,
                            inner_beta_hint: None,
                        })
                    } else {
                        Ok(OuterEval::infeasible(theta.len()))
                    }
                }
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let result = problem
            .run(&mut obj, "BFGS screening should reorder expensive seeds")
            .expect("screened BFGS startup should reach the best generated seed");
        assert_eq!(result.plan_used.solver, Solver::Bfgs);
        assert_eq!(result.rho, valid_seed);
        assert_eq!(
            started.lock().unwrap().first().cloned(),
            Some(valid_seed),
            "BFGS screening should move the lowest-cost seed to the front before full startup eval",
        );
        assert!(
            screening_calls.load(Ordering::Relaxed) > 1,
            "BFGS seed screening should rank candidates with cost-only probes first",
        );
        assert_eq!(screening_cap.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn screening_cap_survives_per_seed_reset_before_proxy_eval() {
        let mut seed_config = crate::seeding::SeedConfig::default();
        seed_config.max_seeds = 3;
        seed_config.seed_budget = 1;
        seed_config.risk_profile = crate::seeding::SeedRiskProfile::Gaussian;
        let screening_cap = Arc::new(AtomicUsize::new(0));
        let proxy_saw_cap = Arc::new(AtomicBool::new(false));
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Unavailable)
            .with_seed_config(seed_config)
            .with_screening_cap(Arc::clone(&screening_cap))
            .with_max_iter(1);
        let mut obj = problem.build_objective_with_screening_proxy(
            (),
            |_: &mut (), _: &Array1<f64>| Ok(0.0),
            |_: &mut (), theta: &Array1<f64>| {
                Ok(OuterEval {
                    cost: theta[0].abs(),
                    gradient: array![0.0],
                    hessian: HessianResult::Unavailable,
                    inner_beta_hint: None,
                })
            },
            |_: &mut (), theta: &Array1<f64>, _: OuterEvalOrder| {
                Ok(OuterEval {
                    cost: theta[0].abs(),
                    gradient: array![0.0],
                    hessian: HessianResult::Unavailable,
                    inner_beta_hint: None,
                })
            },
            {
                let screening_cap = Arc::clone(&screening_cap);
                Some(move |_: &mut ()| {
                    screening_cap.store(0, Ordering::Relaxed);
                })
            },
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
            {
                let screening_cap = Arc::clone(&screening_cap);
                let proxy_saw_cap = Arc::clone(&proxy_saw_cap);
                move |_: &mut (), theta: &Array1<f64>| {
                    let cap = screening_cap.load(Ordering::Relaxed);
                    if cap > 0 {
                        proxy_saw_cap.store(true, Ordering::Relaxed);
                        Ok(theta[0].abs())
                    } else {
                        Err(EstimationError::RemlOptimizationFailed(
                            "screening proxy ran without an active cap".to_string(),
                        ))
                    }
                }
            },
        );
        problem
            .run(&mut obj, "screening cap reset regression")
            .expect("screening cap should be restored after each per-seed reset");
        assert!(
            proxy_saw_cap.load(Ordering::Relaxed),
            "screening proxy should observe a nonzero cap"
        );
        assert_eq!(screening_cap.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn rank_seeds_cascade_escalates_when_initial_cap_collapses_all() {
        // When every seed's cost is non-finite at the initial screening cap
        // we must NOT jump straight to a fully uncapped re-evaluation on
        // every seed (the original two-stage protocol). Instead the cap
        // should escalate geometrically (initial → 4× → 16× → uncapped),
        // exiting the moment any cap stage produces a finite cost. This
        // test forces a cost function that returns non-finite for cap < 12
        // and finite for cap ≥ 12, then asserts the cascade exits at the
        // 4× stage with a meaningful ranking — never reaching the uncapped
        // pass.
        let mut seed_config = crate::seeding::SeedConfig::default();
        seed_config.seed_budget = 1;
        seed_config.screen_max_inner_iterations = 3;
        let screening_cap = Arc::new(AtomicUsize::new(0));
        let initial_seed = array![5.0];
        let valid_seed = crate::seeding::generate_rho_candidates(1, None, &seed_config)
            .first()
            .expect("seed generator should yield at least one candidate")
            .clone();
        let max_cap_seen = Arc::new(AtomicUsize::new(0));
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Either)
            .with_seed_config(seed_config)
            .with_screening_cap(Arc::clone(&screening_cap))
            .with_initial_rho(initial_seed.clone())
            .with_screen_initial_rho(true)
            .with_max_iter(1);
        let mut obj = problem.build_objective(
            (),
            {
                let screening_cap = Arc::clone(&screening_cap);
                let max_cap_seen = Arc::clone(&max_cap_seen);
                let valid_seed = valid_seed.clone();
                move |_: &mut (), theta: &Array1<f64>| {
                    let cap = screening_cap.load(Ordering::Relaxed);
                    max_cap_seen.fetch_max(cap, Ordering::Relaxed);
                    // Mimic an inner solver that needs ≥ 12 iterations of
                    // budget to certify a finite cost; below that it returns
                    // a non-finite "could not converge" signal.
                    if cap > 0 && cap < 12 {
                        return Ok(f64::NAN);
                    }
                    if theta == valid_seed {
                        Ok(0.0)
                    } else {
                        Ok(1000.0)
                    }
                }
            },
            {
                let valid_seed = valid_seed.clone();
                move |_: &mut (), theta: &Array1<f64>| {
                    if theta == valid_seed {
                        Ok(OuterEval {
                            cost: 0.0,
                            gradient: array![0.0],
                            hessian: HessianResult::Analytic(array![[1.0]]),
                            inner_beta_hint: None,
                        })
                    } else {
                        Ok(OuterEval::infeasible(theta.len()))
                    }
                }
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        problem
            .run(&mut obj, "cascade should escalate")
            .expect("cascade should reach a finite cost at the 4× cap stage");
        // The cascade is [3, 12, 48, 0]; the 4× stage (cap=12) is the first
        // stage that produces a finite cost, so the cascade must exit there
        // and never escalate to 48 or to the uncapped (0) stage.
        let max_cap = max_cap_seen.load(Ordering::Relaxed);
        assert_eq!(
            max_cap, 12,
            "cascade should stop at the 4× cap stage; observed max cap = {max_cap}"
        );
        assert_eq!(
            screening_cap.load(Ordering::Relaxed),
            0,
            "screening cap must be restored to its previous value after cascade"
        );
    }

    #[test]
    fn run_efs_skips_global_cost_screening() {
        let mut seed_config = crate::seeding::SeedConfig::default();
        seed_config.max_seeds = 6;
        seed_config.seed_budget = 1;
        let screening_calls = Arc::new(AtomicUsize::new(0));
        let problem = OuterProblem::new(15)
            .with_gradient(Derivative::Unavailable)
            .with_hessian(DeclaredHessianForm::Unavailable)
            .with_seed_config(seed_config)
            .with_max_iter(1);
        let mut obj = problem.build_objective(
            (),
            {
                let screening_calls = Arc::clone(&screening_calls);
                move |_: &mut (), _: &Array1<f64>| {
                    screening_calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    Ok(0.0)
                }
            },
            |_: &mut (), theta: &Array1<f64>| Ok(OuterEval::infeasible(theta.len())),
            None::<fn(&mut ())>,
            Some(|_: &mut (), theta: &Array1<f64>| {
                Ok(EfsEval {
                    cost: 0.0,
                    steps: vec![0.0; theta.len()],
                    beta: None,
                    psi_gradient: None,
                    psi_indices: None,
                    inner_hessian_scale: None,
                    logdet_enclosure_gap: None,
                })
            }),
        );
        problem
            .run(
                &mut obj,
                "EFS should not use a separate global cost-screening pass",
            )
            .expect("first generated EFS seed should be sufficient");
        assert_eq!(
            screening_calls.load(std::sync::atomic::Ordering::Relaxed),
            0,
            "EFS startup should not call eval_cost just to screen seeds"
        );
    }

    #[test]
    fn run_efs_skips_invalid_leading_seed_without_spending_budget() {
        let generated = crate::seeding::generate_rho_candidates(
            15,
            None,
            &crate::seeding::SeedConfig::default(),
        );
        let valid_seed = generated
            .first()
            .expect("seed generator should yield at least one candidate")
            .clone();
        let invalid_seed = Array1::from_elem(15, 9.0);
        let mut seed_config = crate::seeding::SeedConfig::default();
        seed_config.seed_budget = 1;
        let problem = OuterProblem::new(15)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Unavailable)
            .with_seed_config(seed_config)
            .with_initial_rho(invalid_seed)
            .with_max_iter(1);
        let mut obj = problem.build_objective(
            (),
            |_: &mut (), _: &Array1<f64>| Ok(0.0),
            |_: &mut (), theta: &Array1<f64>| Ok(OuterEval::infeasible(theta.len())),
            None::<fn(&mut ())>,
            {
                let valid_seed = valid_seed.clone();
                Some(move |_: &mut (), theta: &Array1<f64>| {
                    if theta == valid_seed {
                        Ok(EfsEval {
                            cost: 0.0,
                            steps: vec![0.0; theta.len()],
                            beta: None,
                            psi_gradient: None,
                            psi_indices: None,
                            inner_hessian_scale: None,
                            logdet_enclosure_gap: None,
                        })
                    } else {
                        Err(EstimationError::RemlOptimizationFailed(
                            "invalid EFS seed".to_string(),
                        ))
                    }
                })
            },
        );
        let result = problem
            .run(&mut obj, "efs generated seed should remain reachable")
            .expect("invalid startup seeds should not consume the only EFS seed slot");
        assert_eq!(result.rho, valid_seed);
        assert_eq!(result.plan_used.solver, Solver::Efs);
    }

    #[test]
    fn run_efs_runtime_fallback_marker_degrades_to_bfgs_immediately() {
        let mut seed_config = crate::seeding::SeedConfig::default();
        seed_config.seed_budget = 2;
        let efs_calls = Arc::new(AtomicUsize::new(0));
        let problem = OuterProblem::new(12)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Unavailable)
            .with_seed_config(seed_config)
            .with_initial_rho(Array1::zeros(12))
            .with_max_iter(5);
        let mut obj = problem.build_objective(
            (),
            |_: &mut (), theta: &Array1<f64>| Ok(0.5 * theta.dot(theta)),
            |_: &mut (), theta: &Array1<f64>| {
                Ok(OuterEval {
                    cost: 0.5 * theta.dot(theta),
                    gradient: theta.clone(),
                    hessian: HessianResult::Unavailable,
                    inner_beta_hint: None,
                })
            },
            None::<fn(&mut ())>,
            {
                let efs_calls = Arc::clone(&efs_calls);
                Some(move |_: &mut (), _: &Array1<f64>| {
                    efs_calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    Err(EstimationError::RemlOptimizationFailed(format!(
                        "{} synthetic runtime escape hatch",
                        EFS_FIRST_ORDER_FALLBACK_MARKER,
                    )))
                })
            },
        );
        let result = problem
            .run(&mut obj, "efs runtime fallback marker")
            .expect("runtime EFS escape hatch should degrade to BFGS");
        assert_eq!(result.plan_used.solver, Solver::Bfgs);
        assert_eq!(
            efs_calls.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "runtime fallback marker should abort the EFS attempt immediately"
        );
    }

    #[test]
    fn run_rejects_invalid_theta_layout() {
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Unavailable)
            .with_psi_dim(2)
            .with_initial_rho(Array1::zeros(1))
            .with_max_iter(1);
        let mut obj = problem.build_objective(
            (),
            |_: &mut (), _: &Array1<f64>| Ok(0.0),
            |_: &mut (), _: &Array1<f64>| {
                Ok(OuterEval {
                    cost: 0.0,
                    gradient: Array1::zeros(1),
                    hessian: HessianResult::Unavailable,
                    inner_beta_hint: None,
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let err = problem
            .run(&mut obj, "test invalid layout")
            .expect_err("invalid theta layout should fail cleanly");
        assert!(
            err.to_string().contains("invalid outer theta layout"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn effective_seed_budget_caps_expensive_solver_retries() {
        assert_eq!(
            effective_seed_budget(
                4,
                Solver::Efs,
                crate::seeding::SeedRiskProfile::GeneralizedLinear,
                false,
            ),
            1
        );
        assert_eq!(
            effective_seed_budget(
                4,
                Solver::HybridEfs,
                crate::seeding::SeedRiskProfile::Survival,
                false,
            ),
            1
        );
        assert_eq!(
            effective_seed_budget(
                3,
                Solver::Arc,
                crate::seeding::SeedRiskProfile::GeneralizedLinear,
                true,
            ),
            1
        );
        assert_eq!(
            effective_seed_budget(
                3,
                Solver::Arc,
                crate::seeding::SeedRiskProfile::Survival,
                false,
            ),
            1
        );
        assert_eq!(
            effective_seed_budget(
                3,
                Solver::Bfgs,
                crate::seeding::SeedRiskProfile::Survival,
                false,
            ),
            3
        );
    }

    #[test]
    fn run_arc_projects_seed_before_seed_validation_eval() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut seed_config = crate::seeding::SeedConfig::default();
        seed_config.max_seeds = 1;
        seed_config.seed_budget = 1;
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Either)
            .with_bounds(array![0.0], array![1.0])
            .with_initial_rho(array![2.0])
            .with_seed_config(seed_config)
            .with_max_iter(1);
        let mut obj = problem.build_objective(
            (),
            |_: &mut (), theta: &Array1<f64>| Ok((theta[0] - 0.25).powi(2)),
            {
                let seen = Arc::clone(&seen);
                move |_: &mut (), theta: &Array1<f64>| {
                    seen.lock().unwrap().push(theta.clone());
                    Ok(OuterEval {
                        cost: (theta[0] - 0.25).powi(2),
                        gradient: array![2.0 * (theta[0] - 0.25)],
                        hessian: HessianResult::Analytic(array![[2.0]]),
                        inner_beta_hint: None,
                    })
                }
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        problem
            .run(&mut obj, "arc seed projection")
            .expect("arc should evaluate the projected seed");
        assert_eq!(
            seen.lock().unwrap().first().cloned(),
            Some(array![1.0]),
            "Arc must project the seed before validating the initial sample",
        );
    }

    #[test]
    fn run_bfgs_projects_seed_before_seed_validation_eval() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut seed_config = crate::seeding::SeedConfig::default();
        seed_config.max_seeds = 1;
        seed_config.seed_budget = 1;
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Unavailable)
            .with_bounds(array![0.0], array![1.0])
            .with_initial_rho(array![2.0])
            .with_seed_config(seed_config)
            .with_max_iter(1);
        let mut obj = problem.build_objective(
            (),
            |_: &mut (), theta: &Array1<f64>| Ok((theta[0] - 0.25).powi(2)),
            {
                let seen = Arc::clone(&seen);
                move |_: &mut (), theta: &Array1<f64>| {
                    seen.lock().unwrap().push(theta.clone());
                    Ok(OuterEval {
                        cost: (theta[0] - 0.25).powi(2),
                        gradient: array![2.0 * (theta[0] - 0.25)],
                        hessian: HessianResult::Unavailable,
                        inner_beta_hint: None,
                    })
                }
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        problem
            .run(&mut obj, "bfgs seed projection")
            .expect("BFGS should evaluate the projected seed");
        assert_eq!(
            seen.lock().unwrap().first().cloned(),
            Some(array![1.0]),
            "BFGS must project the seed before validating the initial sample",
        );
    }

    fn tmp_cache_session(label: &str) -> (tempfile::TempDir, Arc<CacheSession>) {
        let dir = tempfile::tempdir().unwrap();
        let store = crate::cache::WarmStartStore::open(
            dir.path().to_path_buf(),
            crate::cache::StoreOptions {
                size_budget_bytes: 1024 * 1024,
                ttl: std::time::Duration::from_secs(60),
            },
        )
        .unwrap();
        let mut fp = crate::cache::Fingerprinter::new();
        fp.absorb_str(b"outer-test", label);
        let key = fp.finalize();
        (dir, Arc::new(CacheSession::open(store, key)))
    }

    #[test]
    fn checkpointing_objective_persists_finite_evals() {
        let (_d, session) = tmp_cache_session("ckpt-persist");
        let problem = OuterProblem::new(1).with_gradient(Derivative::Unavailable);
        let mut inner: ClosureObjective<_, _, _> = problem.build_objective(
            (),
            |_: &mut (), theta: &Array1<f64>| Ok(theta[0] * theta[0]),
            |_: &mut (), _: &Array1<f64>| {
                Err(EstimationError::InvalidInput("eval not used".into()))
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let mut wrapped = CheckpointingObjective::new(&mut inner, Arc::clone(&session), Vec::new());
        // Initial: nothing on disk.
        assert!(session.try_load().is_none());
        // First eval persists.
        let v0 = wrapped.eval_cost(&array![3.0]).unwrap();
        assert!((v0 - 9.0).abs() < 1e-12);
        let on_disk = session.try_load().expect("first eval should checkpoint");
        let payload = decode_iterate(&on_disk.payload, 1).expect("payload decodes");
        assert!((payload.cost - 9.0).abs() < 1e-12);
        assert_eq!(payload.rho, vec![3.0]);
        // Strictly improving eval must bypass the 2-second rate limit.
        let v1 = wrapped.eval_cost(&array![0.5]).unwrap();
        assert!((v1 - 0.25).abs() < 1e-12);
        let on_disk = session
            .try_load()
            .expect("improving eval should checkpoint");
        let payload = decode_iterate(&on_disk.payload, 1).expect("payload decodes");
        assert!((payload.cost - 0.25).abs() < 1e-12);
        assert_eq!(payload.rho, vec![0.5]);
        // Non-finite values must not corrupt the on-disk best-known iterate.
        let v_inf = wrapped.eval_cost(&array![f64::NAN]);
        match v_inf {
            Ok(value) => assert!(!value.is_finite()),
            Err(err) => assert!(!err.to_string().is_empty()),
        }
        let on_disk = session.try_load().expect("prior best preserved");
        let payload = decode_iterate(&on_disk.payload, 1).expect("payload decodes");
        assert!((payload.cost - 0.25).abs() < 1e-12);
    }

    #[test]
    fn checkpointing_objective_rejects_wrong_dim_on_decode() {
        // A payload from a 3-dim fit is invalid input for a 5-dim resume.
        let bytes = encode_iterate(&array![1.0, 2.0, 3.0], None, 0.5, 0).expect("encode");
        assert!(decode_iterate(&bytes, 3).is_some());
        assert!(decode_iterate(&bytes, 5).is_none());
    }

    #[test]
    fn iterate_payload_round_trips_beta() {
        // Every persisted entry that comes with an inner-β hint round-trips
        // (ρ, β) together — that pair lets a resume open inner PIRLS in the
        // basin of quadratic attraction regardless of where ρ sits.
        let rho = array![10.0, -10.0, 5.0];
        let beta = array![0.12, -0.34, 0.56, 7.89];
        let bytes = encode_iterate(&rho, Some(&beta), 1.0, 7).expect("encode");
        let decoded = decode_iterate(&bytes, rho.len()).expect("decode");
        assert_eq!(decoded.rho, rho.to_vec());
        assert_eq!(decoded.beta, beta.to_vec());
        // ρ-only writes (β = None) still encode but with an empty beta slot.
        let ro_bytes = encode_iterate(&rho, None, 1.0, 7).expect("encode-rho-only");
        let ro = decode_iterate(&ro_bytes, rho.len()).expect("decode-rho-only");
        assert!(ro.beta.is_empty());
    }

    #[test]
    fn note_persists_inner_beta_hint_from_eval() {
        // Write-side proof of the principled fix: when the inner solver
        // surfaces β via OuterEval::inner_beta_hint, CheckpointingObjective
        // captures it on every accepted eval AND exposes it for finalize.
        let (_d, session) = tmp_cache_session("note-persists-beta");
        let problem = OuterProblem::new(1).with_gradient(Derivative::Unavailable);
        let mut inner: ClosureObjective<_, _, _> = problem.build_objective(
            (),
            |_: &mut (), _: &Array1<f64>| Ok(1.0),
            |_: &mut (), theta: &Array1<f64>| {
                Ok(OuterEval {
                    cost: theta[0] * theta[0],
                    gradient: array![2.0 * theta[0]],
                    hessian: HessianResult::Unavailable,
                    inner_beta_hint: Some(array![1.5, 2.5, 3.5]),
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let mut wrapped = CheckpointingObjective::new(&mut inner, Arc::clone(&session), Vec::new());
        let eval = wrapped.eval(&array![0.5]).expect("eval ok");
        assert!((eval.cost - 0.25).abs() < 1e-12);
        let on_disk = session
            .try_load()
            .expect("eval with finite β must persist a (ρ,β) checkpoint");
        let payload = decode_iterate(&on_disk.payload, 1).expect("payload decodes");
        assert_eq!(payload.beta, vec![1.5, 2.5, 3.5]);
        let captured = wrapped.last_inner_beta().expect("β was captured");
        assert_eq!(captured.to_vec(), vec![1.5, 2.5, 3.5]);
    }

    #[test]
    fn note_rejects_nonfinite_inner_beta() {
        // A divergent inner state must NOT poison the cache: persisting a
        // non-finite β would re-create the inner-PIRLS budget-exhaustion
        // failure mode at boundary ρ where the cached β is supposed to
        // place the resume inside Newton's quadratic basin.
        let (_d, session) = tmp_cache_session("note-rejects-bad-beta");
        let problem = OuterProblem::new(1).with_gradient(Derivative::Unavailable);
        let mut inner: ClosureObjective<_, _, _> = problem.build_objective(
            (),
            |_: &mut (), _: &Array1<f64>| Ok(1.0),
            |_: &mut (), theta: &Array1<f64>| {
                Ok(OuterEval {
                    cost: theta[0] * theta[0],
                    gradient: array![2.0 * theta[0]],
                    hessian: HessianResult::Unavailable,
                    inner_beta_hint: Some(array![f64::NAN, 0.5]),
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let mut wrapped = CheckpointingObjective::new(&mut inner, Arc::clone(&session), Vec::new());
        let eval = wrapped.eval(&array![0.5]).expect("eval ok");
        assert!((eval.cost - 0.25).abs() < 1e-12);
        assert!(
            session.try_load().is_none(),
            "non-finite β must abort the checkpoint write, not poison the cache",
        );
        assert!(
            wrapped.last_inner_beta().is_none(),
            "non-finite β must not be exposed via last_inner_beta()",
        );
    }

    #[test]
    fn classify_extracts_beta_from_v2_payload() {
        // The classifier propagates `beta` from the v2 payload onto its
        // Seed/ExactFinal decisions so the dispatcher can hand it to
        // OuterObjective::seed_inner_state. Without this, the (ρ, β) payload
        // would write β but never resurface it on resume.
        let rho = array![1.0, 2.0];
        let beta = array![10.0, 20.0, 30.0];
        let payload = encode_iterate(&rho, Some(&beta), 1.0, 0).expect("encode");
        let loaded = crate::cache::LoadedEntry {
            entry: crate::cache::CachedEntry {
                payload,
                objective: Some(1.0),
                iteration: Some(0),
                kind: crate::cache::EntryKind::Checkpoint,
                written_unix_secs: 0,
            },
            source: crate::cache::LoadSource::Preloaded,
        };
        let CacheSeedDecision::Seed {
            beta: decoded_beta, ..
        } = classify_cache_entry_for_outer(&loaded, 2)
        else {
            panic!("expected Seed decision");
        };
        assert_eq!(decoded_beta, beta.to_vec());

        // ρ-only payload (legacy or family-without-β) decodes to empty beta.
        let payload = encode_iterate(&rho, None, 1.0, 0).expect("encode");
        let loaded = crate::cache::LoadedEntry {
            entry: crate::cache::CachedEntry {
                payload,
                objective: Some(1.0),
                iteration: Some(0),
                kind: crate::cache::EntryKind::Checkpoint,
                written_unix_secs: 0,
            },
            source: crate::cache::LoadSource::Preloaded,
        };
        let CacheSeedDecision::Seed {
            beta: decoded_beta, ..
        } = classify_cache_entry_for_outer(&loaded, 2)
        else {
            panic!("expected Seed decision");
        };
        assert!(
            decoded_beta.is_empty(),
            "ρ-only payload must produce an empty beta so the dispatcher skips seed_inner_state"
        );
    }

    #[test]
    fn run_calls_seed_inner_state_with_cached_beta() {
        // End-to-end read-side wiring: a cache hit carrying β must call
        // OuterObjective::seed_inner_state(&beta) *before* the first BFGS
        // eval. We verify this by routing through a custom OuterObjective
        // that records the β it was seeded with.
        struct RecordingObj {
            seeded: Arc<Mutex<Option<Array1<f64>>>>,
            eval_count: Arc<Mutex<usize>>,
        }
        impl OuterObjective for RecordingObj {
            fn capability(&self) -> OuterCapability {
                // Analytic gradient AND analytic Hessian so the planner picks
                // the same Hessian-bearing path a real fit takes; using
                // Unavailable here would test a degenerate plan.
                OuterCapability {
                    gradient: Derivative::Analytic,
                    hessian: DeclaredHessianForm::Dense,
                    n_params: 2,
                    psi_dim: 0,
                    fixed_point_available: false,
                    barrier_config: None,
                    prefer_gradient_only: false,
                    disable_fixed_point: false,
                }
            }
            fn eval_cost(&mut self, theta: &Array1<f64>) -> Result<f64, EstimationError> {
                Ok(theta.dot(theta))
            }
            fn eval(&mut self, theta: &Array1<f64>) -> Result<OuterEval, EstimationError> {
                *self.eval_count.lock().unwrap() += 1;
                // f(θ) = ‖θ‖² → ∇f = 2θ, ∇²f = 2I.
                Ok(OuterEval {
                    cost: theta.dot(theta),
                    gradient: 2.0 * theta,
                    hessian: HessianResult::Analytic(2.0 * Array2::<f64>::eye(theta.len())),
                    inner_beta_hint: None,
                })
            }
            fn reset(&mut self) {}
            fn seed_inner_state(
                &mut self,
                beta: &Array1<f64>,
            ) -> Result<SeedOutcome, EstimationError> {
                *self.seeded.lock().unwrap() = Some(beta.clone());
                Ok(SeedOutcome::Installed)
            }
        }

        let (_d, session) = tmp_cache_session("seed-inner-state-call");
        let bytes = encode_iterate(&array![1.0, 2.0], Some(&array![7.5, 8.5, 9.5]), 5.0, 3)
            .expect("encode");
        session.checkpoint(&bytes, Some(5.0), Some(3));

        let seeded: Arc<Mutex<Option<Array1<f64>>>> = Arc::new(Mutex::new(None));
        let eval_count: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
        let mut obj = RecordingObj {
            seeded: Arc::clone(&seeded),
            eval_count: Arc::clone(&eval_count),
        };

        let problem = OuterProblem::new(2)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Unavailable)
            .with_max_iter(1)
            .with_cache_session(Arc::clone(&session));
        match problem.run(&mut obj, "seed-inner-state-call") {
            Ok(result) => assert!(result.final_value.is_finite()),
            Err(err) => assert!(!err.to_string().is_empty()),
        }

        let observed = seeded.lock().unwrap().clone();
        assert_eq!(
            observed,
            Some(array![7.5, 8.5, 9.5]),
            "dispatcher must call seed_inner_state with the cached β before run_outer",
        );
    }

    #[test]
    fn run_skips_seed_inner_state_when_payload_has_no_beta() {
        // Symmetric guard: a ρ-only cache entry must NOT invoke
        // seed_inner_state — calling it with an empty / zero / garbage β
        // would silently degrade a family that has a non-trivial inner
        // default into one started at zeros.
        struct CountingObj {
            seed_calls: Arc<Mutex<usize>>,
        }
        impl OuterObjective for CountingObj {
            fn capability(&self) -> OuterCapability {
                // Analytic gradient AND analytic Hessian so the planner picks
                // the same Hessian-bearing path a real fit takes; using
                // Unavailable here would test a degenerate plan.
                OuterCapability {
                    gradient: Derivative::Analytic,
                    hessian: DeclaredHessianForm::Dense,
                    n_params: 2,
                    psi_dim: 0,
                    fixed_point_available: false,
                    barrier_config: None,
                    prefer_gradient_only: false,
                    disable_fixed_point: false,
                }
            }
            fn eval_cost(&mut self, theta: &Array1<f64>) -> Result<f64, EstimationError> {
                Ok(theta.dot(theta))
            }
            fn eval(&mut self, theta: &Array1<f64>) -> Result<OuterEval, EstimationError> {
                // f(θ) = ‖θ‖² → ∇f = 2θ, ∇²f = 2I.
                Ok(OuterEval {
                    cost: theta.dot(theta),
                    gradient: 2.0 * theta,
                    hessian: HessianResult::Analytic(2.0 * Array2::<f64>::eye(theta.len())),
                    inner_beta_hint: None,
                })
            }
            fn reset(&mut self) {}
            fn seed_inner_state(
                &mut self,
                beta: &Array1<f64>,
            ) -> Result<SeedOutcome, EstimationError> {
                *self.seed_calls.lock().unwrap() += beta.len().max(1);
                Ok(SeedOutcome::Installed)
            }
        }

        let (_d, session) = tmp_cache_session("seed-inner-state-skip");
        // ρ-only payload — no β.
        let bytes = encode_iterate(&array![1.0, 2.0], None, 5.0, 3).expect("encode");
        session.checkpoint(&bytes, Some(5.0), Some(3));

        let seed_calls: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
        let mut obj = CountingObj {
            seed_calls: Arc::clone(&seed_calls),
        };

        let problem = OuterProblem::new(2)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Unavailable)
            .with_max_iter(1)
            .with_cache_session(Arc::clone(&session));
        match problem.run(&mut obj, "seed-inner-state-skip") {
            Ok(result) => assert!(result.final_value.is_finite()),
            Err(err) => assert!(!err.to_string().is_empty()),
        }

        assert_eq!(
            *seed_calls.lock().unwrap(),
            0,
            "seed_inner_state must not fire when the cached payload carries no β",
        );
    }

    #[test]
    fn cache_entry_classifier_honors_finite_seeds_regardless_of_saturation() {
        // The classifier no longer reshapes ρ based on shape. Any finite,
        // correctly-dimensioned payload is honored as the next run's seed.
        // Boundary-saturated entries written under the v2 (ρ, β) invariant
        // are a *legitimate* finding — the smoothness wants to be near-null
        // — and the persisted β puts the next inner solve at zero-gradient,
        // making the cold-β failure mode impossible to re-create from cache.
        for rho_seed in [array![9.0, 0.0], array![10.0, -10.0], array![-10.0, 10.0]] {
            let payload = encode_iterate(&rho_seed, None, 1.0, 0).expect("encode");
            let loaded = crate::cache::LoadedEntry {
                entry: crate::cache::CachedEntry {
                    payload,
                    objective: Some(1.0),
                    iteration: Some(0),
                    kind: crate::cache::EntryKind::Checkpoint,
                    written_unix_secs: 0,
                },
                source: crate::cache::LoadSource::Preloaded,
            };

            assert!(cache_entry_would_help_outer(&loaded, 2));
            let CacheSeedDecision::Seed { rho, .. } = classify_cache_entry_for_outer(&loaded, 2)
            else {
                panic!(
                    "finite seed {:?} must be honored unchanged; the read-side clamp / \
                     all-saturated-discard branches were band-aids over the missing β cache",
                    rho_seed
                );
            };
            assert_eq!(rho, rho_seed, "ρ must round-trip without reshaping");
        }
    }

    #[test]
    fn cache_entry_classifier_rejects_only_structural_failures() {
        // Only structural failures discard: payload shape (wrong rho_dim,
        // non-finite payload internals → decode None → "payload-shape-mismatch")
        // and non-finite cache metadata → "non-finite-payload". Saturation
        // and β presence are NOT discards here: saturation is honored, and
        // ρ-only payloads decode cleanly with an empty β slot.

        // Non-finite metadata objective: decode succeeds (finite payload
        // cost), but the entry-level objective is NaN — discard as
        // non-finite-payload.
        let payload = encode_iterate(&array![0.5, 0.5], None, 1.0, 0).expect("encode");
        let loaded = crate::cache::LoadedEntry {
            entry: crate::cache::CachedEntry {
                payload,
                objective: Some(f64::NAN),
                iteration: Some(0),
                kind: crate::cache::EntryKind::Checkpoint,
                written_unix_secs: 0,
            },
            source: crate::cache::LoadSource::Preloaded,
        };
        assert!(matches!(
            classify_cache_entry_for_outer(&loaded, 2),
            CacheSeedDecision::Discard {
                reason: "non-finite-payload",
                ..
            }
        ));

        // Dimension mismatch: 2-D payload viewed as a 3-D problem → decode
        // rejects shape → "payload-shape-mismatch".
        let payload = encode_iterate(&array![0.5, 0.5], None, 1.0, 0).expect("encode");
        let loaded = crate::cache::LoadedEntry {
            entry: crate::cache::CachedEntry {
                payload,
                objective: Some(1.0),
                iteration: Some(0),
                kind: crate::cache::EntryKind::Checkpoint,
                written_unix_secs: 0,
            },
            source: crate::cache::LoadSource::Preloaded,
        };
        assert!(matches!(
            classify_cache_entry_for_outer(&loaded, 3),
            CacheSeedDecision::Discard {
                reason: "payload-shape-mismatch",
                ..
            }
        ));
    }

    #[test]
    fn exact_final_cache_hit_is_helpful_even_at_boundary() {
        let payload = encode_iterate(&array![10.0, -10.0], None, 1.0, 3).expect("encode");
        let loaded = crate::cache::LoadedEntry {
            entry: crate::cache::CachedEntry {
                payload,
                objective: Some(1.0),
                iteration: Some(3),
                kind: crate::cache::EntryKind::Final,
                written_unix_secs: 0,
            },
            source: crate::cache::LoadSource::Exact,
        };

        assert!(cache_entry_would_help_outer(&loaded, 2));
        assert!(matches!(
            classify_cache_entry_for_outer(&loaded, 2),
            CacheSeedDecision::ExactFinal { iterations: 3, .. }
        ));
    }

    #[test]
    fn checkpointing_objective_mirrors_checkpoints() {
        let (_primary_dir, primary) = tmp_cache_session("ckpt-primary");
        let (_mirror_dir, mirror) = tmp_cache_session("ckpt-mirror");
        let problem = OuterProblem::new(1).with_gradient(Derivative::Unavailable);
        let mut inner: ClosureObjective<_, _, _> = problem.build_objective(
            (),
            |_: &mut (), theta: &Array1<f64>| Ok(theta[0] * theta[0]),
            |_: &mut (), _: &Array1<f64>| {
                Err(EstimationError::InvalidInput("eval not used".into()))
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let mut wrapped = CheckpointingObjective::new(
            &mut inner,
            Arc::clone(&primary),
            vec![Arc::clone(&mirror)],
        );

        let value = wrapped.eval_cost(&array![4.0]).unwrap();
        assert_eq!(value, 16.0);

        let primary_payload =
            decode_iterate(&primary.try_load().expect("primary checkpoint").payload, 1)
                .expect("primary decode");
        let mirror_payload =
            decode_iterate(&mirror.try_load().expect("mirror checkpoint").payload, 1)
                .expect("mirror decode");
        assert_eq!(primary_payload.rho, vec![4.0]);
        assert_eq!(mirror_payload.rho, vec![4.0]);
        assert_eq!(primary_payload.cost, mirror_payload.cost);
    }

    #[test]
    fn cached_rho_is_prepended_as_first_seed() {
        // Whitebox: pre-seed the session with a known iterate, then run
        // an OuterProblem with a deliberately-different `initial_rho`.
        // The runner must visit the cached rho before the configured
        // `initial_rho` because `try_load` overrode it.
        let (_d, session) = tmp_cache_session("seed-prepend");
        // Hand-write the cached checkpoint: rho = [2.5], cost = 0.25.
        // Final exact hits return immediately; checkpoints still exercise the
        // regular seed-prepend path.
        let payload = encode_iterate(&array![2.5], None, 0.25, 0).expect("encode");
        session.checkpoint(&payload, Some(0.25), Some(0));
        assert!(
            session.try_load().is_some(),
            "precondition: cache populated"
        );

        let seen: Arc<Mutex<Vec<Array1<f64>>>> = Arc::new(Mutex::new(Vec::new()));
        // A gradient-bearing BFGS problem. Bounds must contain the cached rho
        // so the projector doesn't snap it away.
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Unavailable)
            .with_bounds(array![-5.0], array![5.0])
            .with_initial_rho(array![-3.0]) // deliberately not 2.5
            .with_max_iter(8)
            .with_cache_session(Arc::clone(&session));
        let mut obj = problem.build_objective(
            seen.clone(),
            |seen: &mut Arc<Mutex<Vec<Array1<f64>>>>, theta: &Array1<f64>| {
                seen.lock().unwrap().push(theta.clone());
                Ok((theta[0] - 2.5).powi(2))
            },
            |_: &mut Arc<Mutex<Vec<Array1<f64>>>>, theta: &Array1<f64>| {
                Ok(OuterEval {
                    cost: (theta[0] - 2.5).powi(2),
                    gradient: array![2.0 * (theta[0] - 2.5)],
                    hessian: HessianResult::Unavailable,
                    inner_beta_hint: None,
                })
            },
            None::<fn(&mut Arc<Mutex<Vec<Array1<f64>>>>)>,
            None::<
                fn(
                    &mut Arc<Mutex<Vec<Array1<f64>>>>,
                    &Array1<f64>,
                ) -> Result<EfsEval, EstimationError>,
            >,
        );
        match problem.run(&mut obj, "seed-prepend") {
            Ok(result) => assert!(result.final_value.is_finite()),
            Err(err) => assert!(!err.to_string().is_empty()),
        }
        // The cached rho (2.5) must appear in the eval trace, and it must
        // appear no later than the configured initial_rho (−3.0). Both
        // are inside the bounds so the projector cannot rewrite them.
        let evals = seen.lock().unwrap();
        let pos_cached = evals.iter().position(|r| (r[0] - 2.5).abs() < 1e-9);
        let pos_initial = evals.iter().position(|r| (r[0] + 3.0).abs() < 1e-9);
        assert!(
            pos_cached.is_some(),
            "cached rho must be evaluated; saw {:?}",
            *evals
        );
        if let (Some(c), Some(i)) = (pos_cached, pos_initial) {
            assert!(
                c <= i,
                "cached rho (idx {c}) must precede initial_rho (idx {i})",
            );
        }
    }

    #[test]
    fn all_saturated_cached_rho_is_honored_as_seed() {
        // Inverse of the prior `all_saturated_cached_rho_is_discarded_before_seed_validation`
        // test. Under v1 the cache stored ρ-only, so resuming at boundary ρ
        // forced PIRLS to cold-start β against a Hessian with condition
        // number `≈ e^{2·rho_bound}` — Newton degraded to O(1/k) descent
        // that exhausted the cycle budget. The "discard if all-saturated"
        // branch was a read-side band-aid; it suppressed a legitimate
        // resume signal in exchange for tolerating the broken contract.
        //
        // Under v2 the iterate payload carries (ρ, β). When β is persisted
        // alongside boundary ρ the next inner solve opens at zero gradient,
        // and the conditioning is no longer a barrier. Therefore the
        // classifier no longer reshapes ρ based on saturation: every
        // finite, correctly-dimensioned entry is used as the seed. This
        // test pins that contract.
        let (_d, session) = tmp_cache_session("all-saturated-honored");
        let payload = encode_iterate(&array![10.0, -10.0], None, 1.0, 0).expect("encode");
        session.checkpoint(&payload, Some(1.0), Some(0));
        assert!(
            session.try_load().is_some(),
            "precondition: cache populated"
        );

        let seen: Arc<Mutex<Vec<Array1<f64>>>> = Arc::new(Mutex::new(Vec::new()));
        let mut seed_config = crate::seeding::SeedConfig::default();
        seed_config.max_seeds = 4;
        seed_config.seed_budget = 1;
        let problem = OuterProblem::new(2)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Unavailable)
            .with_seed_config(seed_config)
            .with_initial_rho(array![0.0, 0.0])
            .with_rho_bound(10.0)
            .with_max_iter(1)
            .with_cache_session(Arc::clone(&session));

        let mut obj = problem.build_objective(
            seen.clone(),
            |_: &mut Arc<Mutex<Vec<Array1<f64>>>>, theta: &Array1<f64>| Ok(theta.dot(theta)),
            |seen: &mut Arc<Mutex<Vec<Array1<f64>>>>, theta: &Array1<f64>| {
                seen.lock().unwrap().push(theta.clone());
                Ok(OuterEval {
                    cost: theta.dot(theta),
                    gradient: theta.clone(),
                    hessian: HessianResult::Unavailable,
                    inner_beta_hint: None,
                })
            },
            None::<fn(&mut Arc<Mutex<Vec<Array1<f64>>>>)>,
            None::<
                fn(
                    &mut Arc<Mutex<Vec<Array1<f64>>>>,
                    &Array1<f64>,
                ) -> Result<EfsEval, EstimationError>,
            >,
        );

        match problem.run(&mut obj, "all-saturated-honored") {
            Ok(result) => assert!(result.final_value.is_finite()),
            Err(err) => assert!(!err.to_string().is_empty()),
        }
        let evals = seen.lock().unwrap();
        assert!(
            evals.iter().any(|rho| rho == array![10.0, -10.0]),
            "cached saturated ρ must be evaluated unchanged under v2 (ρ, β) invariant; saw {:?}",
            *evals
        );
    }

    #[test]
    fn exact_final_cache_hit_skips_outer_validation() {
        let (_d, session) = tmp_cache_session("final-skip");
        let payload = encode_iterate(&array![2.5], None, 0.25, 7).expect("encode");
        session.finalize(&payload, Some(0.25), Some(7));

        let seen: Arc<Mutex<Vec<Array1<f64>>>> = Arc::new(Mutex::new(Vec::new()));
        // The exact final cache hit short-circuits before any solver runs, so
        // the declared derivatives only need to make a well-formed plan.
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Unavailable)
            .with_bounds(array![-5.0], array![5.0])
            .with_initial_rho(array![-3.0])
            .with_max_iter(8)
            .with_cache_session(Arc::clone(&session));
        let mut obj = problem.build_objective(
            seen.clone(),
            |seen: &mut Arc<Mutex<Vec<Array1<f64>>>>, theta: &Array1<f64>| {
                seen.lock().unwrap().push(theta.clone());
                Ok((theta[0] - 2.5).powi(2))
            },
            |_: &mut Arc<Mutex<Vec<Array1<f64>>>>, theta: &Array1<f64>| {
                Ok(OuterEval {
                    cost: (theta[0] - 2.5).powi(2),
                    gradient: array![2.0 * (theta[0] - 2.5)],
                    hessian: HessianResult::Unavailable,
                    inner_beta_hint: None,
                })
            },
            None::<fn(&mut Arc<Mutex<Vec<Array1<f64>>>>)>,
            None::<
                fn(
                    &mut Arc<Mutex<Vec<Array1<f64>>>>,
                    &Array1<f64>,
                ) -> Result<EfsEval, EstimationError>,
            >,
        );

        let result = problem
            .run(&mut obj, "final-skip")
            .expect("final exact hit should return cached outer result");
        assert_eq!(result.rho, array![2.5]);
        assert_eq!(result.final_value, 0.25);
        assert_eq!(result.iterations, 7);
        assert!(result.converged);
        assert!(
            seen.lock().unwrap().is_empty(),
            "exact final hit should not evaluate the outer objective"
        );
    }
}
