use super::*;

pub(crate) const EXPENSIVE_PREWARM_COEFF_DIM: usize = 24;
pub(crate) const EXPENSIVE_PREWARM_RHO_DIM: usize = 4;
pub(crate) const MULTI_SEED_PREWARM_BUDGET: usize = 8;
pub(crate) const SINGLE_EXPENSIVE_PREWARM_BUDGET: usize = 16;

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

/// Coefficient dimension at which the per-step inner solve cost begins to grow
/// steeply (the empirical #979 "centers≈8→10" cliff). For a custom-family
/// marginal-slope fit `p_coefficients` ≈ Σ over both formulas of the basis
/// dim, so two `matern(centers=K)` formulas land at `p ≈ 2K`; the per-step
/// inner joint-Newton solve becomes multi-second by `K ≈ 8` (`p ≈ 16`), well
/// BELOW the `EXPENSIVE_PREWARM_COEFF_DIM = 24` "expensive shape" gate. Below
/// this floor the pre-warm keeps the full `PATH_BUDGET` (cheap fits anneal
/// fully and the seed-continuation accuracy is untouched); at or above it the
/// per-seed step budget is scaled DOWN inversely with `p_coefficients` so the
/// TOTAL pre-warm inner-solve work stays bounded as the problem grows past the
/// cliff, instead of paying `PATH_BUDGET` (= 64) full inner solves per seed.
pub(crate) const PREWARM_COST_CLIFF_COEFF_DIM: usize = 12;

/// Target ceiling on `budget × p_coefficients` once past the cost cliff: the
/// per-step inner solve cost scales roughly with `p_coefficients`, so holding
/// `budget · p` constant keeps the per-seed pre-warm wall-clock flat across
/// center counts (the #979 acceptance workloads centers ∈ {4, 12, 20} all land
/// at a comparable, bounded pre-warm cost instead of the centers=20 non-finish).
pub(crate) const PREWARM_COST_BUDGET_COEFF_PRODUCT: usize =
    PREWARM_COST_CLIFF_COEFF_DIM * SINGLE_EXPENSIVE_PREWARM_BUDGET;

/// Floor on the scaled budget: even on the largest problems the pre-warm must
/// still anneal a few continuation legs from the oversmoothing ρ₀ toward the
/// seed so the warm β it forwards is genuinely near-optimal (capping must not
/// regress the seed-continuation accuracy the pre-warm exists to provide).
pub(crate) const PREWARM_MIN_SCALED_BUDGET: usize = 4;

/// Scale the per-seed continuation pre-warm step budget by `p_coefficients`
/// once the problem is past the cost cliff, so the TOTAL pre-warm inner-solve
/// work stays bounded as center count grows. Returns a budget in
/// `[PREWARM_MIN_SCALED_BUDGET, base_budget]` that is non-increasing in
/// `p_coefficients`. Below the cliff this is the identity (`base_budget`).
pub(crate) fn cost_scaled_prewarm_budget(base_budget: usize, p_coefficients: usize) -> usize {
    if p_coefficients <= PREWARM_COST_CLIFF_COEFF_DIM {
        return base_budget;
    }
    let scaled =
        (PREWARM_COST_BUDGET_COEFF_PRODUCT / p_coefficients).max(PREWARM_MIN_SCALED_BUDGET);
    scaled.min(base_budget)
}

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
    let default_budget = crate::estimate::reml::continuation::PATH_BUDGET;
    let p_coefficients = config
        .rho_uncertainty_problem_size
        .p_coefficients
        .unwrap_or(0);
    let multi_seed_cascade = seed_count > seed_budget.max(1);
    // An "expensive shape" for pre-warm bounding is ANY problem at or past the
    // #979 per-step cost cliff. The legacy gate (p ≥ 24 or rho dim ≥ 4) MISSED
    // the marginal-slope cliff: two `matern/duchon(centers=K)` formulas give
    // p ≈ 2K, so the centers≈8 cliff lands at p ≈ 16 — below the legacy p ≥ 24
    // tier and below the rho-dim ≥ 4 tier (two formulas ⇒ rho_dim ≈ 2). Without
    // the cliff term `base_budget` stayed at the full PATH_BUDGET (64) and the
    // ONLY thing that bounded the cold walk was the inverse-p `cost_scaled_*`
    // taper (64 → ~12 at p = 16). Twelve multi-second inner solves is still the
    // ~108s the owner measured under parallel (all-cold) load, while a sequential
    // rerun that happened to hit the persisted warm-start cache skipped it
    // entirely — so the SAME inputs were "seconds vs intractable" purely as a
    // function of disk-cache/scheduling state (#979 Experiment-2). Folding the
    // cost cliff into `expensive_shape` collapses the cold base to the small
    // bounded tier at the cliff, making the fired magnitude a deterministic
    // function of the PROBLEM (p_coefficients, rho dim) — the warm-start cache
    // hit then only ever turns the bounded tier into a redundant skip-to-0,
    // never the difference between bounded and unbounded.
    let expensive_shape = p_coefficients >= EXPENSIVE_PREWARM_COEFF_DIM
        || p_coefficients >= PREWARM_COST_CLIFF_COEFF_DIM
        || cap.n_params >= EXPENSIVE_PREWARM_RHO_DIM;

    // True once the coefficient dim is at or past the #979 per-step cost cliff,
    // where each inner joint-Newton solve is already multi-second. This is the
    // regime where the cold pre-warm became intractable (~108s for ~12 steps at
    // the centers≈8 cliff), so the cold base is bounded by the SMALL documented
    // tier (`MULTI_SEED_PREWARM_BUDGET`) here — NOT the larger
    // `SINGLE_EXPENSIVE_PREWARM_BUDGET` — before the inverse-p taper scales it
    // down further. The taper alone (from a base of 64) left 12 steps at the
    // cliff; capping the base to the small tier first brings the cold magnitude
    // to the owner's 4–6 / `MULTI_SEED_PREWARM_BUDGET` "small number" intent.
    let past_cost_cliff = p_coefficients >= PREWARM_COST_CLIFF_COEFF_DIM;

    // Shape-derived base budget: the legacy "expensive shape" tiers, with the
    // #979 cost cliff bounding the cold base to the small tier. This caps the
    // pre-warm once the problem is large enough to declare an expensive shape
    // (p ≥ 24, p ≥ the cost cliff, or rho dim ≥ 4).
    let base_budget = if past_cost_cliff || (multi_seed_cascade && expensive_shape) {
        MULTI_SEED_PREWARM_BUDGET.min(default_budget)
    } else if expensive_shape {
        SINGLE_EXPENSIVE_PREWARM_BUDGET.min(default_budget)
    } else {
        default_budget
    };

    // #979 cost-cliff cap: the per-step inner solve cost grows steeply with
    // `p_coefficients` (the centers≈8→10 cliff for two-formula marginal-slope
    // fits, where p ≈ 2·centers). The legacy "expensive shape" gate only fires
    // at p ≥ 24, so a centers ∈ {8..12} fit still paid the FULL PATH_BUDGET (64)
    // multi-second inner solves per seed — the binary marginal-slope slowdown.
    // Scale the base budget DOWN inversely with `p_coefficients` past the cliff
    // so total pre-warm work stays bounded, while preserving the full budget on
    // cheap (small-p) fits and never collapsing below
    // `PREWARM_MIN_SCALED_BUDGET` legs (so the warm β stays near-optimal).
    cost_scaled_prewarm_budget(base_budget, p_coefficients)
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
        match certify_outer_optimality(obj, config, context, &mut candidate) {
            Ok(certificate) => {
                candidate.criterion_certificate = Some(certificate);
                Ok(Self(candidate))
            }
            Err(error) => {
                candidate.converged = false;
                Err((candidate, error))
            }
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

/// Execute a single plan attempt (seed generation → solver loop → best result).
pub(crate) fn run_outer_with_plan(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    context: &str,
    cap: &OuterCapability,
    the_plan: &OuterPlan,
) -> Result<PlanRunOutcome, EstimationError> {
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
    crate::estimate::reml::outer_eval::record_current_outer_rho_upper_bounds_for_ift(&upper);
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

    let seed_budget = effective_seed_budget(
        config.seed_config.seed_budget,
        the_plan.solver,
        config.seed_config.risk_profile,
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

    let mut best: Option<CertifiedOuterCandidate> = None;
    let mut best_checkpoint: Option<OuterResult> = None;
    // A reactive domain-entry path is created inside a seed attempt only after
    // that objective's exact seed cost is non-finite. Already-feasible seeds
    // therefore stay on the zero-heavy-entry path.
    let reactive_domain_scalar_contract = obj.reactive_domain_scalar_contract()?;
    let reactive_domain_entry_available = reactive_domain_scalar_contract.is_some();
    // Accumulate every per-seed rejection with its 0-based seed index and the
    // phase that rejected it (validation vs solver run). When all seeds fail
    // systematically (bad analytic gradient, rank-deficient penalty, etc.) the
    // first rejection's rho + error is often the most diagnostic.
    let mut rejection_reasons: Vec<(usize, &'static str, String)> = Vec::new();
    let layout = cap.theta_layout();
    // Number of smoothing (ρ) coordinates, used to break a near-LAML-tie toward
    // the more-penalized basin in the non-Gaussian multi-start keep-best.
    let rho_dim = layout.rho_dim();
    let mut started_seeds = 0usize;
    let continuation_prewarm_budget =
        continuation_prewarm_step_budget(config, cap, seeds.len(), seed_budget);
    if config.warm_start_cache_hit {
        log::info!(
            "[OUTER] {context}: continuation pre-warm skipped: warm-start cache hit \
             (seed already near-optimal); proceeding straight to BFGS/Newton certificate"
        );
    } else if continuation_prewarm_budget < crate::estimate::reml::continuation::PATH_BUDGET {
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

    'seed_attempts: for (seed_idx, seed) in seeds.iter().enumerate() {
        if started_seeds == seed_budget {
            break;
        }
        // Domain entry is a property of this literal seed. A loop-local path
        // cannot leak its state or regime into another candidate.
        let mut continuation_path: Option<crate::continuation_path::ContinuationPath> = None;
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
        crate::estimate::reml::outer_eval::record_current_outer_iter_for_ift(0);
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
        if let Some(seed_cost) = obj.accept_seed_without_outer_iterations(seed)? {
            started_seeds += 1;
            let candidate = OuterResult::new(seed.clone(), seed_cost, 0, true, *the_plan);
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
                    retain_best_outer_checkpoint(&mut best_checkpoint, checkpoint);
                    rejection_reasons.push((seed_idx, "certificate", error.to_string()));
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
        if reactive_domain_entry_available {
            match obj.eval_cost(seed) {
                Ok(cost) if cost.is_finite() => {
                    log::debug!(
                        "[OUTER] {context}: exact seed {seed_idx} is inside the objective domain; \
                         reactive continuation entry not needed"
                    );
                }
                Ok(_) => {
                    log::info!(
                        "[OUTER] {context}: exact seed {seed_idx} has undefined criterion; \
                         entering through certified heavy-smoothing continuation"
                    );
                    // The failed cold probe may have left objective-owned trial
                    // state. Re-enter from the pristine baseline; successful
                    // path evaluations establish a fresh exact-seed handoff.
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
                    reactive_domain_entry_requested = true;
                }
                Err(err) => {
                    let msg = format!(
                        "reactive domain-entry seed probe failed before continuation: {err}"
                    );
                    log::warn!("[OUTER] {context}: rejecting seed {seed_idx}: {msg}");
                    rejection_reasons.push((seed_idx, "domain-entry", msg));
                    continue 'seed_attempts;
                }
            }
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
        // `inner_beta_hint`. A reactive-domain objective enters the explicit
        // path only after the exact real-objective probe above returned a
        // non-finite criterion; already-finite seeds never allocate or drive it.
        let enter_via_continuation_path =
            obj.allow_continuation_prewarm() || continuation_path.is_some();
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
                rejection_reasons.push((seed_idx, "domain-entry", msg));
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
                    rejection_reasons.push((seed_idx, "domain-entry", msg));
                    continue 'seed_attempts;
                }
                Err(err) => {
                    let msg = format!(
                        "reactive domain entry refused: exact seed verification failed after \
                         certified continuation arrival: {err}"
                    );
                    log::warn!("[OUTER] {context}: rejecting seed {seed_idx}: {msg}");
                    rejection_reasons.push((seed_idx, "domain-entry", msg));
                    continue 'seed_attempts;
                }
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
                match crate::estimate::reml::continuation::prime_outer_seed_with_budget(
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
                        // Objectives without an active reactive-domain path.
                        // An active path was already driven and exact-verified
                        // above, so it cannot enter this generic pre-warm block.
                        // Legacy contract: a cold solve
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
                                rejection_reasons.push((seed_idx, "validation", err.to_string()));
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
                            result.operator_stop_reason =
                                Some(OperatorTrustRegionStopReason::Converged);
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
                            result.operator_stop_reason =
                                Some(OperatorTrustRegionStopReason::CostStallFlatValley);
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
                    let cost_stall_rel_tol = config
                        .rel_cost_tolerance
                        .unwrap_or(config.tolerance * 1.0e-2)
                        .max(COST_STALL_REL_TOL_FLOOR);
                    let arc_seed_grad_norm =
                        seed_eval.gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
                    let cost_stall_grad_threshold = grad_tol
                        .threshold(seed_eval.cost, arc_seed_grad_norm)
                        .max(COST_STALL_PROJECTED_GRAD_FLOOR);

                    let mut cost_stall_guard = CostStallGuard::new(
                        cost_stall_rel_tol,
                        ARC_COST_STALL_WINDOW,
                        cost_stall_grad_threshold,
                        cost_stall_exit.clone(),
                    );
                    cost_stall_guard.observe_seed(
                        &seed,
                        seed_eval.cost,
                        projected_gradient_norm(
                            &seed,
                            &seed_eval.gradient,
                            Some(&(lo.clone(), hi.clone())),
                        ),
                    );

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
                        cost_stall: Some(cost_stall_guard),
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
                                    Ok(outer_result_with_gradient_norm(
                                        best.rho,
                                        best.value,
                                        best.iterations,
                                        Some(best.grad_norm),
                                        false,
                                        *the_plan,
                                    ))
                                }
                                _ => {
                                    Ok(solution_into_outer_result(*last_solution, false, *the_plan))
                                }
                            }
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
                                Some(exit) => {
                                    let mut result = outer_result_with_gradient_norm(
                                        exit.rho,
                                        exit.value,
                                        exit.iterations,
                                        Some(exit.grad_norm),
                                        exit.converged,
                                        *the_plan,
                                    );
                                    // #2241 — carry the guard's measured probe-
                                    // noise-floor bound so the final analytic
                                    // certificate honors the same flat band the
                                    // guard certified in the loop.
                                    result.flat_noise_grad_bound = exit.noise_grad_bound;
                                    // Preserve HOW ARC stopped even when the
                                    // guard already certified the stalled score
                                    // surface. The mandatory final analytic
                                    // certificate uses this provenance to apply
                                    // the same derived score-relative flat-valley
                                    // bound as the guard. Dropping the marker on
                                    // `exit.converged=true` made the final pass
                                    // silently revert to the much tighter raw
                                    // solver bound and reject the identical
                                    // point (#1689: |g|=.042 on |V|≈982).
                                    result.operator_stop_reason =
                                        Some(OperatorTrustRegionStopReason::CostStallFlatValley);
                                    Ok(result)
                                }
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
                    let cost_stall_grad_threshold = grad_tol
                        .threshold(seed_eval.cost, seed_grad_norm)
                        .max(COST_STALL_PROJECTED_GRAD_FLOOR);
                    let mut cost_stall_guard = CostStallGuard::new(
                        cost_stall_rel_tol,
                        COST_STALL_WINDOW,
                        cost_stall_grad_threshold,
                        cost_stall_exit.clone(),
                    );
                    cost_stall_guard.observe_seed(seed, seed_eval.cost, seed_grad_norm);
                    let objective = OuterFirstOrderBridge {
                        obj,
                        layout,
                        outer_inner_cap: config.outer_inner_cap.clone(),
                        iter_count: 0,
                        g_norm_initial: None,
                        last_g_norm: None,
                        last_value_grad_rho: None,
                        value_probe_cache: Vec::new(),
                        cost_stall: Some(cost_stall_guard),
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
                    // starts, because it is local to the parent fit. Two warm-start
                    // mechanisms both pin `initial_rho`: the in-process / disk
                    // persistent cache (which also flips `warm_start_cache_hit`)
                    // and the biobank cross-fit beta projection, which sets
                    // `initial_rho` to the transferred rho but leaves
                    // `warm_start_cache_hit` false. Cover both by testing seed
                    // identity against `initial_rho`. The scalar scale is clamped
                    // to the same `[1e-3, 1e3]` band the optimizer applies to its
                    // own BB estimate so a pathological seed gradient cannot
                    // produce a degenerate metric.
                    let is_warm_seed = config.warm_start_cache_hit
                        || config
                            .initial_rho
                            .as_ref()
                            .is_some_and(|initial| initial == seed);
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
                                Some(exit) => {
                                    let mut result = outer_result_with_gradient_norm(
                                        exit.rho,
                                        exit.value,
                                        exit.iterations,
                                        Some(exit.grad_norm),
                                        exit.converged,
                                        *the_plan,
                                    );
                                    // #2241 — carry the guard's measured probe-
                                    // noise-floor bound so the final analytic
                                    // certificate honors the same flat band the
                                    // guard certified in the loop.
                                    result.flat_noise_grad_bound = exit.noise_grad_bound;
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
                                    result.operator_stop_reason =
                                        Some(OperatorTrustRegionStopReason::CostStallFlatValley);
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
                log::debug!(
                    "[outer-timing] seed {}/{} ({:?}): {:.3}s  cost={:.6e}  converged={}",
                    seed_slot,
                    seed_budget,
                    the_plan.solver,
                    seed_elapsed,
                    candidate.final_value,
                    candidate.converged,
                );
                if !candidate.converged {
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
                        retain_best_outer_checkpoint(&mut best_checkpoint, checkpoint);
                        rejection_reasons.push((seed_idx, "certificate", error.to_string()));
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
                // keep-best reject an overshoot — is never evaluated. Bounded to
                // the existing seed_budget (typically 2 for non-Gaussian ARC), so
                // this solves at most one additional seed before the break.
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
                let non_gaussian_await_parsimony_seed = parsimonious_keep_best
                    && seed_budget > 1
                    && started_seeds < seed_budget
                    && !best
                        .as_ref()
                        .is_some_and(|b| parsimony_second_seed_is_redundant(b.result(), rho_dim));
                if best.is_some()
                    && !quality_compare_remaining_gaussian_seeds
                    && !non_gaussian_await_parsimony_seed
                {
                    break;
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
            }
        }
    }

    if let Some(certified) = best {
        let result = certified.into_result();
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
            .map(TerminalInnerCapGuard::lift);
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

    if let Some(checkpoint) = best_checkpoint {
        return Ok(PlanRunOutcome::Exhausted(checkpoint));
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
#[path = "run_plan_tests.rs"]
mod run_plan_tests;
