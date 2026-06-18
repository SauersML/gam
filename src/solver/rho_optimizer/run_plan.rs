use super::*;

pub(crate) const EXPENSIVE_PREWARM_COEFF_DIM: usize = 24;
pub(crate) const EXPENSIVE_PREWARM_RHO_DIM: usize = 4;
pub(crate) const MULTI_SEED_PREWARM_BUDGET: usize = 8;
pub(crate) const SINGLE_EXPENSIVE_PREWARM_BUDGET: usize = 16;

pub(crate) fn continuation_prewarm_step_budget(
    config: &OuterConfig,
    cap: &OuterCapability,
    seed_count: usize,
    seed_budget: usize,
) -> usize {
    // Warm-start cache hit: the seed (ρ, and since 0.1.204 the inner β) was
    // populated from a prior fit's persisted near-optimal iterate, so the
    // continuation pre-warm — which only exists to anneal a COLD seed toward the
    // optimum — has nothing to anneal. Skip it entirely; the outer BFGS/Newton
    // still runs to its REML/KKT certificate from the cached iterate, so the
    // converged optimum is identical. Cold-start fits (no hit) fall through to
    // the existing shape-based budget byte-for-byte.
    if config.warm_start_cache_hit {
        return 0;
    }
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
pub(crate) fn run_outer_with_plan(
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
    crate::solver::estimate::reml::outer_eval::record_current_outer_rho_upper_bounds_for_ift(
        &upper,
    );
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
    if config.warm_start_cache_hit {
        log::info!(
            "[OUTER] {context}: continuation pre-warm skipped: warm-start cache hit \
             (seed already near-optimal); proceeding straight to BFGS/Newton certificate"
        );
    } else if continuation_prewarm_budget < crate::solver::estimate::reml::continuation::PATH_BUDGET
    {
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
        crate::solver::estimate::reml::outer_eval::record_current_outer_iter_for_ift(0);
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
        if continuation_path.is_none()
            && enter_via_continuation_path
            && continuation_prewarm_budget > 0
        {
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
                    let cost_stall_rel_tol =
                        (config.tolerance * 1.0e-2).max(COST_STALL_REL_TOL_FLOOR);
                    let arc_seed_grad_norm =
                        seed_eval.gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
                    let cost_stall_grad_threshold = grad_tol
                        .threshold(seed_eval.cost, arc_seed_grad_norm)
                        .max(COST_STALL_PROJECTED_GRAD_FLOOR);

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
                        cost_stall: Some(CostStallGuard::new(
                            cost_stall_rel_tol,
                            ARC_COST_STALL_WINDOW,
                            cost_stall_grad_threshold,
                            cost_stall_exit.clone(),
                        )),
                        cost_stall_bounds: Some((lo.clone(), hi.clone())),
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
                        Err(ArcError::ObjectiveFailed { message })
                            if message == COST_STALL_CONVERGED_SENTINEL =>
                        {
                            // The bridge's cost-stall guard halted ARC because
                            // the REML score stopped decreasing (#1089/#1237).
                            // Rebuild the outer result from the published best
                            // iterate; the converged flag rides on the guard's
                            // bound-projected stationarity test (`exit.converged`)
                            // exactly as the BFGS branch does. A non-converged
                            // cost-stall flows into the same best-so-far
                            // non-convergence reporting as MaxIterations.
                            let exit = cost_stall_exit.lock().ok().and_then(|mut slot| slot.take());
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
                                    "ARC cost-stall sentinel fired without a published best \
                                     iterate ({context})"
                                ))),
                            }
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
                    // Relative score-change floor is derived from the outer
                    // tolerance but has a numerical floor so very tight user
                    // tolerances do not disable the mgcv-style flat-valley stop.
                    let cost_stall_exit: Arc<Mutex<Option<CostStallExit>>> =
                        Arc::new(Mutex::new(None));
                    let cost_stall_rel_tol =
                        (config.tolerance * 1.0e-2).max(COST_STALL_REL_TOL_FLOOR);
                    // Stationarity gate for the cost-stall exit. Convergence must
                    // mean stationarity, not cost-flatness: a cost stall only
                    // counts as a converged optimum when the projected gradient
                    // norm at the best iterate clears the SAME outer gradient
                    // tolerance the genuine BFGS convergence path uses, with
                    // the same practical floor the ARC guard uses for
                    // bound-pinned separation fits.
                    let seed_grad_norm =
                        seed_eval.gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
                    let cost_stall_grad_threshold = grad_tol
                        .threshold(seed_eval.cost, seed_grad_norm)
                        .max(COST_STALL_PROJECTED_GRAD_FLOOR);
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
                        cost_stall_bounds: Some((lo.clone(), hi.clone())),
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
                    // Warm-start first-step scaling. `opt::Bfgs` begins with an
                    // UNSCALED identity inverse-Hessian (`B_inv = I`) on iter 0:
                    // the search direction is the raw `d = -g`, so the unit
                    // line-search step (`α = 1`) is `-g` in ρ-space. The
                    // optimizer's Barzilai-Borwein self-scaling (`γ = sᵀy/yᵀy`)
                    // only fires AFTER the first line search completes, so when a
                    // warm start lands a near-optimal seed whose residual gradient
                    // still has a large component along a weakly-curved (heavily
                    // penalized) log-λ direction, the raw `-g` step overshoots and
                    // the StrongWolfe search has to bracket/zoom — each bracketing
                    // probe is a full inner joint-Newton re-solve. On the biobank
                    // LOSO fold that is the observed three ~65 s `outer eval Value`
                    // probes before the single accepted step.
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
                    // Gated on "this seed is the pinned warm start" so cold
                    // multistart seeds keep the optimizer's historical internal
                    // scaling. Two warm-start mechanisms both pin `initial_rho`:
                    // the in-process / disk persistent cache (which also flips
                    // `warm_start_cache_hit`) AND the biobank cross-fit β
                    // projection (`consume_fit_artifact`, logged `[CACHE]
                    // beta-warm action=projected source=cross-fit`), which sets
                    // `initial_rho` to the transferred ρ but leaves
                    // `warm_start_cache_hit` false. Cover both by testing seed
                    // identity against `initial_rho`. The scale is clamped to the
                    // same `[1e-3, 1e3]` band the optimizer applies to its own BB
                    // estimate so a pathological seed gradient cannot produce a
                    // degenerate metric.
                    let is_warm_seed = config.warm_start_cache_hit
                        || config
                            .initial_rho
                            .as_ref()
                            .is_some_and(|initial| initial == seed);
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
                        // joint-Newton re-solve. `invert_spd_with_ridge` only
                        // returns when the curvature is SPD (after a tiny ridge),
                        // which is exactly when it is a valid (descent-preserving)
                        // metric; a non-PD transferred Hessian falls through to
                        // the scalar magnitude metric. Either way the converged
                        // optimum is unchanged: BFGS reaches ∇V=0 under any SPD
                        // initial metric, and the gradient/KKT tests are identical.
                        let dense_metric = config
                            .warm_start_outer_hessian
                            .as_ref()
                            .filter(|h| {
                                h.nrows() == layout.n_params
                                    && h.ncols() == layout.n_params
                                    && h.iter().all(|v| v.is_finite())
                            })
                            .and_then(|h| {
                                crate::linalg::utils::invert_spd_with_ridge(h, 1.0e-8).ok()
                            })
                            .filter(|h_inv| h_inv.iter().all(|v| v.is_finite()));
                        if let Some(h_inv) = dense_metric {
                            log::info!(
                                "[OUTER] {context}: warm-start BFGS metric = transferred \
                                 H(θ̂)⁻¹ (dim={}); quasi-Newton first step",
                                layout.n_params,
                            );
                            optimizer = optimizer
                                .with_initial_metric(InitialMetric::DenseInverseHessian(h_inv));
                        } else {
                            let g0_norm =
                                seed_eval.gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
                            if g0_norm.is_finite() && g0_norm > 0.0 {
                                let scale = (1.0 / g0_norm).clamp(1.0e-3, 1.0e3);
                                optimizer =
                                    optimizer.with_initial_metric(InitialMetric::Scalar(scale));
                            }
                        }
                    }
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
                            let exit = cost_stall_exit.lock().ok().and_then(|mut slot| slot.take());
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
#[path = "run_plan_tests.rs"]
mod run_plan_tests;
