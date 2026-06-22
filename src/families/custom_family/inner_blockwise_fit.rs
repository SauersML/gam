//! The blockwise inner-fit driver (`inner_blockwise_fit`), the joint
//! Newton polish step, and inner-result assembly, split out of
//! `outer_objective.rs` by concern (#1145). Re-exported via
//! `custom_family` so existing paths stay stable.

use super::*;

pub(crate) fn inner_blockwise_fit<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    block_log_lambdas: &[Array1<f64>],
    options: &BlockwiseFitOptions,
    warm_start: Option<&ConstrainedWarmStart>,
) -> Result<BlockwiseInnerResult, String> {
    // Inner-blockwise prelude waypoints. At large-scale n the cold-start
    // path between function entry and the first PIRLS/JN cycle-summary
    // log can run for many minutes (sometimes hours) silently while
    // row-kernel workspace builds run. Emit a `[STAGE] PIRLS/inner`
    // line at each transition so the next failed run pinpoints which
    // named step holds time. Gated on large-scale n so small-fit
    // tests stay quiet.
    let inner_started = std::time::Instant::now();
    let mut states = buildblock_states(family, specs)?;
    refresh_all_block_etas(family, specs, &mut states)?;
    let total_joint_p = specs.iter().map(|spec| spec.design.ncols()).sum::<usize>();
    let total_joint_n = joint_observation_count(&states);
    const INNER_PRELUDE_LOG_MIN_N: usize = 100_000;
    let prelude_log = total_joint_n >= INNER_PRELUDE_LOG_MIN_N;
    if prelude_log {
        log::info!(
            "[STAGE] PIRLS/inner step=buildblock_states+refresh_etas elapsed={:.3}s n={} p={} blocks={}",
            inner_started.elapsed().as_secs_f64(),
            total_joint_n,
            total_joint_p,
            specs.len(),
        );
    }
    let matrix_free_joint_requested = use_joint_matrix_free_path(total_joint_p, total_joint_n)
        || family.prefers_matrix_free_inner_joint(specs, &states);
    let has_workspace_source = family.inner_coefficient_hessian_hvp_available(specs);
    // Probe the *spec-aware* joint Hessian: it is the canonical source of the
    // coupled joint curvature. A family may override only
    // `exact_newton_joint_hessian_with_specs` (the variant that has access to
    // the realized block designs needed to assemble the cross-block
    // `X_aᵀ diag(w_ab) X_b` blocks — e.g. the Dirichlet common-parameterization
    // family, whose `evaluate` emits diagonal working sets so the spec-less
    // default block assembler returns `None`). Routing the inner joint-Newton
    // availability gate through the spec-less `exact_newton_joint_hessian`
    // would then mis-classify such a family as "no joint Hessian" and drop it
    // onto pure block-diagonal backfitting, which fails to reach KKT on small,
    // concentrated coupled likelihoods. The `_with_specs` path subsumes the
    // spec-less one for every family (single-block / uncoupled delegate
    // identically), so it is the correct probe here.
    let has_joint_exacthessian = if has_workspace_source {
        true
    } else {
        family
            .exact_newton_joint_hessian_with_specs(&states, specs)?
            .is_some()
    };
    let structurally_coupled_joint_hessian = specs.len() >= 2
        && !family.likelihood_blocks_uncoupled()
        && !family.has_explicit_joint_hessian()
        && !has_workspace_source
        && family.joint_hessian_is_structurally_coupled(&states)?;
    let coupled_exact_joint_required = specs.len() >= 2
        && !family.likelihood_blocks_uncoupled()
        && (family.has_explicit_joint_hessian()
            || has_workspace_source
            || structurally_coupled_joint_hessian);
    // When the family declares its likelihood blocks UNCOUPLED
    // (`∂²L/∂β_a∂β_b = 0` for every a ≠ b) the joint penalized objective is
    // fully separable across blocks: the joint Hessian is exactly
    // block-diagonal and each block carries only its own penalty. On a
    // separable objective block-coordinate descent solves each block's
    // (possibly inequality-constrained) subproblem to its own exact optimum —
    // it IS the joint solve, and each block gets its OWN trust radius, its OWN
    // active-set QP, and its OWN KKT certificate.
    //
    // Forcing the coupled joint-Newton onto such a problem instead couples two
    // independent blocks under ONE shared trust radius and ONE concatenated
    // KKT residual. That is actively harmful when the blocks differ sharply in
    // conditioning — the competing-risks twin time-basis fit (#1025) is the
    // canonical case: two cause-specific baselines share the same I-spline
    // evaluated at the same event times, but one cause sits near its
    // monotonicity-constraint boundary with an O(1e5) hazard-derivative
    // gradient while the other is interior. The shared globalization cannot
    // satisfy both blocks' KKT conditions at once; the joint residual stalls
    // far above tolerance, the inner solve burns its whole cycle budget on
    // every outer ρ-eval, and the fit only survives by falling through to the
    // block-coordinate path anyway (which then converges in a handful of
    // cycles). Route uncoupled multi-block specs straight to that exact
    // separable path. `coupled_exact_joint_required` is already gated the same
    // way (uncoupled families are designed to fall through to blockwise), so
    // this only stops the engine from attempting — and grinding on — a joint
    // solve it was never required to run.
    //
    // Single-block families and genuinely coupled multi-block families are
    // unaffected: the former never had cross-block coupling to begin with, the
    // latter still take the joint path (their objective is NOT separable, so
    // block-coordinate descent would drop the cross-block ∂²L/∂β_a∂β_b
    // curvature).
    let blocks_separable = specs.len() >= 2 && family.likelihood_blocks_uncoupled();
    let use_joint_newton =
        has_joint_exacthessian && (specs.len() >= 2 || has_workspace_source) && !blocks_separable;
    let joint_workspace_requested = use_joint_newton && has_workspace_source;
    let inner_tol = options.inner_tol;
    let inner_max_cycles_base = options.inner_max_cycles;
    // Per-outer-call inner-cycle cap. The earlier "adaptive inner cycle
    // cap" doubled this mid-loop on plateaus, but that turned out to be
    // the wrong response to stalled descent (descent ratios pinned at
    // ~0.999 paired with a sub-tolerance objective change is the
    // no-descent signal, not a "give Newton more cycles" signal). The
    // plateau-flat-objective convergence certificate in the inner-cycle
    // body now handles that case directly, so the cap stays fixed at the
    // baseline for the lifetime of this outer call.
    let inner_max_cycles = capped_inner_max_cycles(options, inner_max_cycles_base);
    // Each block's assembled penalty matrix depends only on that block's
    // penalties and smoothing parameters. Build these setup matrices in
    // parallel, but keep the coordinate-descent and line-search loops below
    // strictly serial because each accepted block update changes the state seen
    // by later blocks.
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let s_lambdas_launch_started = std::time::Instant::now();
    let s_lambdas_par_iter = (0..specs.len()).into_par_iter().map(|b| {
        let spec = &specs[b];
        let Some(block_log_lambda) = block_log_lambdas.get(b) else {
            return Err(CustomFamilyError::UnsupportedConfiguration {
                reason: format!("missing log-smoothing parameter vector for block {b}"),
            }
            .into());
        };
        if block_log_lambda.len() != spec.penalties.len() {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "block {b} log-smoothing parameter length {} does not match penalties {}",
                    block_log_lambda.len(),
                    spec.penalties.len()
                ),
            }
            .into());
        }

        let p = spec.design.ncols();
        let lambdas = block_log_lambda.mapv(f64::exp);
        let mut s_lambda = Array2::<f64>::zeros((p, p));
        for (k, s) in spec.penalties.iter().enumerate() {
            s.add_scaled_to(lambdas[k], &mut s_lambda);
        }
        Ok(s_lambda)
    });
    let s_lambdas_collect_started = std::time::Instant::now();
    let s_lambdas_launch_elapsed = s_lambdas_launch_started.elapsed();
    let s_lambdas = s_lambdas_par_iter.collect::<Result<Vec<_>, String>>()?;
    if prelude_log {
        log::info!(
            "[STAGE] PIRLS/inner step=s_lambdas par_iter launch={:.3}s collect={:.3}s blocks={} (since inner-start={:.3}s)",
            s_lambdas_launch_elapsed.as_secs_f64(),
            s_lambdas_collect_started.elapsed().as_secs_f64(),
            specs.len(),
            inner_started.elapsed().as_secs_f64(),
        );
    }
    let ridge = effective_solverridge(options.ridge_floor);
    let joint_bundle: Option<&crate::families::joint_penalty::JointPenaltyBundle> =
        options.joint_penalties.as_deref();
    if let Some(bundle) = joint_bundle {
        for (i, spec) in bundle.specs.iter().enumerate() {
            if spec.dim() != total_joint_p {
                return Err(format!(
                    "joint penalty {i}: dim {} != total compiled p {}",
                    spec.dim(),
                    total_joint_p,
                ));
            }
        }
        if bundle.specs.len() != bundle.log_lambdas.len() {
            return Err(format!(
                "joint penalty bundle: {} specs vs {} log_lambdas",
                bundle.specs.len(),
                bundle.log_lambdas.len(),
            ));
        }
    }
    let mut cached_active_sets: Vec<Option<Vec<usize>>> = vec![None; specs.len()];
    if let Some(seed) = warm_start
        && seed.block_beta.len() == states.len()
        && seed.active_sets.len() == states.len()
    {
        if warm_start_matches_block_log_lambdas(seed, block_log_lambdas)
            && let Some(cached) = seed.cached_inner.as_ref()
            && cached.converged
            && seed
                .block_beta
                .iter()
                .zip(&states)
                .all(|(beta_seed, state)| beta_seed.len() == state.beta.len())
        {
            for (state, beta_seed) in states.iter_mut().zip(&seed.block_beta) {
                state.beta.assign(beta_seed);
            }
            cached_active_sets = seed.active_sets.clone();
            refresh_all_block_etas(family, specs, &mut states)?;
            log::info!(
                "[PIRLS/joint-Newton warm-start] reused cached same-rho inner mode | cycles={} logdet_h={:.6e} logdet_s={:.6e}",
                cached.cycles,
                cached.block_logdet_h,
                cached.block_logdet_s,
            );
            return Ok(BlockwiseInnerResult {
                block_states: states,
                active_sets: normalize_active_sets(cached_active_sets),
                log_likelihood: cached.log_likelihood,
                penalty_value: cached.penalty_value,
                cycles: cached.cycles,
                converged: cached.converged,
                block_logdet_h: cached.block_logdet_h,
                block_logdet_s: cached.block_logdet_s,
                s_lambdas,
                joint_workspace: cached.joint_workspace.clone(),
                kkt_residual: cached.kkt_residual.clone(),
                active_constraints: cached.active_constraints.clone(),
            });
        }
        // Cold-start path: copy prior β where dimensions match
        // (best-effort; mismatched blocks keep the freshly-built
        // initial state).
        for (b, beta_seed) in seed.block_beta.iter().enumerate() {
            if beta_seed.len() == states[b].beta.len() {
                let beta_projected =
                    family.post_update_block_beta(&states, b, &specs[b], beta_seed.clone())?;
                states[b].beta.assign(&beta_projected);
            }
        }
        cached_active_sets = seed.active_sets.clone();
        refresh_all_block_etas(family, specs, &mut states)?;
    }
    let load_joint_started = std::time::Instant::now();
    if prelude_log {
        log::info!(
            "[STAGE] PIRLS/inner step=load_joint_gradient_evaluation begin use_joint_newton={} joint_workspace_requested={} (since inner-start={:.3}s)",
            use_joint_newton,
            joint_workspace_requested,
            inner_started.elapsed().as_secs_f64(),
        );
    }
    let (
        mut current_log_likelihood,
        mut cached_eval,
        mut cached_joint_gradient,
        mut cached_joint_workspace,
    ) = if use_joint_newton {
        let (log_likelihood, gradient, eval, workspace) = load_joint_gradient_evaluation(
            family,
            specs,
            options,
            &states,
            joint_workspace_requested,
            None,
        )?;
        (log_likelihood, eval, gradient, workspace)
    } else {
        let eval = family.evaluate(&states)?;
        let log_likelihood = eval.log_likelihood;
        (log_likelihood, Some(eval), None, None)
    };
    if prelude_log {
        log::info!(
            "[STAGE] PIRLS/inner step=load_joint_gradient_evaluation end elapsed={:.3}s log_likelihood={:.6e} has_gradient={} has_workspace={}",
            load_joint_started.elapsed().as_secs_f64(),
            current_log_likelihood,
            cached_joint_gradient.is_some(),
            cached_joint_workspace.is_some(),
        );
    }
    // Validate exact-Newton block Hessians at the family-evaluation
    // boundary. A non-finite entry is a contract violation against the
    // family's analytic second derivative; refuse to iterate before
    // any factorization rather than letting it slip through to a
    // downstream logdet check that may be gated off by the outer
    // optimizer's flags.
    let validate_started = std::time::Instant::now();
    if let Some(eval) = cached_eval.as_ref() {
        validate_block_hessians_finite(eval)?;
    }
    if prelude_log {
        log::info!(
            "[STAGE] PIRLS/inner step=validate_block_hessians_finite elapsed={:.3}s checked={}",
            validate_started.elapsed().as_secs_f64(),
            cached_eval.is_some(),
        );
    }
    let penalty_started = std::time::Instant::now();
    let mut current_penalty = total_quadratic_penalty(
        &states,
        &s_lambdas,
        ridge,
        options.ridge_policy,
        joint_bundle,
        Some(specs),
    );
    if prelude_log {
        log::info!(
            "[STAGE] PIRLS/inner step=total_quadratic_penalty elapsed={:.3}s penalty={:.6e} (prelude_total={:.3}s)",
            penalty_started.elapsed().as_secs_f64(),
            current_penalty,
            inner_started.elapsed().as_secs_f64(),
        );
    }
    let mut lastobjective = -current_log_likelihood + current_penalty;
    let mut converged = false;
    let mut cycles_done = 0usize;
    // Pre-allocate per-block eta backup buffers to avoid O(n) allocation
    // per block per cycle in the backtracking line search.
    let mut eta_backups: Vec<Array1<f64>> =
        states.iter().map(|s| Array1::zeros(s.eta.len())).collect();

    // ── Joint Newton fast path ──
    //
    // When the family provides an exact joint Hessian (GAMLSS location-scale),
    // solve the full (p_mu + p_ls) × (p_mu + p_ls) system in one Newton step
    // per cycle instead of iterating between blocks. This converges quadratically
    // (5-10 steps) instead of linearly (20-100+ blockwise cycles).
    //
    // Generic block-diagonal surrogate families may still fall back to
    // blockwise iteration if the joint surrogate is unavailable. Families that
    // advertise a real coupled joint Hessian must not: the blockwise loop only
    // sees principal blocks, so it drops the cross-block curvature that makes
    // the joint problem well conditioned near saturated optima.

    // `last_residual_tol` mirrors the per-cycle KKT tolerance computed inside
    // the joint-Newton loop (`inner_tol · (1 + max(‖∇L‖∞, ‖Sβ‖∞))`). It must
    // live at function scope so both the post-converged exit block inside
    // `if use_joint_newton` AND the post-block-fit IFT residual builder
    // outside that branch can thread the same tolerance into the
    // `ProjectedKktResidual::with_metadata(...)` builder. Seed at `inner_tol`
    // so a path that skips the loop entirely (no joint-Newton, or zero
    // cycles) still records a finite, non-NaN tolerance on the residual
    // carrier rather than NaN.
    let mut last_residual_tol: f64 = inner_tol;

    if use_joint_newton {
        // Build block ranges for the joint system.
        let ranges: Vec<(usize, usize)> = {
            let mut offset = 0;
            specs
                .iter()
                .map(|s| {
                    let start = offset;
                    offset += s.design.ncols();
                    (start, offset)
                })
                .collect()
        };
        let total_p: usize = ranges.last().map_or(0, |r| r.1);

        // Universal full-span Jeffreys/Firth robustness. Build `Z_J` once and
        // use the same term in the coupled Newton step, objective value, and
        // stationarity checks so a near-separating coefficient is bounded by
        // the likelihood's own Fisher geometry instead of an ad-hoc ridge.
        // `None` (empty coefficient system) leaves every step and objective at
        // the un-augmented inner Newton.
        //
        // Continuous-response families (the canonical example: transformation-
        // normal h(Y|x) ~ N(0,1)) opt out via
        // `joint_jeffreys_term_required() = false`. They have no separation
        // regime, the Fisher information is `O(n)` on every identified
        // direction by construction, and each Jeffreys evaluation costs
        // `p` directional-derivative calls into the family's exact joint
        // Hessian — at large scale (CTN duchon16d, p=144, n=20000) that
        // is the dominant per-cycle cost (~200 s/cycle on three calls per
        // cycle), exhausting the inner budget before the algorithm converges
        // while contributing essentially zero to the gradient/curvature.
        let joint_jeffreys_subspace = if family.joint_jeffreys_term_required() {
            build_joint_jeffreys_subspace(specs, &ranges)?
        } else {
            None
        };
        // FIRTH MERIT BOOKKEEPING (gam#826/#872 — per-cycle Φ fold, not a carried
        // value). `current_penalty` / `lastobjective` hold ONLY the quadratic
        // penalty `½βᵀSβ` (NO Φ). The Firth value `−Φ` is folded into the
        // accept/reject comparison FRESH at each β under the same
        // `jeffreys_skippable_this_cycle` gate the step and KKT residual use, so
        // `old_objective` (old β) and `trialobjective` (trial β) are always on the
        // same objective `−ℓ + ½βᵀSβ − Φ` regardless of whether a cycle skips the
        // term. Carrying Φ in `current_penalty` (the previous design) desynced
        // old-vs-trial by ±Φ whenever the per-cycle skippable decision flipped —
        // and the cycle-0 baseline folded Φ UNCONDITIONALLY while the trial folded
        // it gated, so a skippable cycle 0 saw a spurious `Δobj = ±Φ`, rejected
        // every backtrack, and refused as a `phantom_multiplier` at a zero step
        // (the binomial location-scale coupled non-convergence). SIGN: Firth ADDS
        // ½log|I| to the log-likelihood ⇒ the NLL objective SUBTRACTS Φ, matching
        // the Newton step rhs / KKT residual which ADD `∇Φ` to `∇L − Sβ`.

        let joint_mode_diagonal_ridge =
            if ridge > 0.0 && options.ridge_policy.include_quadratic_penalty {
                ridge
            } else {
                0.0
            };

        // Exact joint Newton steps are guarded by two independent mechanisms:
        // family-owned feasibility (`max_feasible_step_size`) and the adaptive
        // trust region below. There is intentionally no family hook for a
        // hard per-attempt coefficient-space clamp; keeping the policy local
        // avoids stale no-op configuration and makes the trust-region behavior
        // explicit at the only place it is used.

        // Cross-cycle convergence carry-over: set at the end of every
        // accepted cycle so the next cycle can distinguish a true KKT
        // optimum on a rank-deficient null mode (objective stuck
        // because every direction is along the null space) from
        // genuine non-convergence. The residual signal does not need
        // a carry-over — `residual <= residual_tol` is the canonical
        // KKT certificate and the end-of-cycle test consumes it
        // directly when it fires.

        // Predicted-reduction tracker for the principled trust-region
        // stopping criterion (Conn-Gould-Toint, *Trust-Region Methods*,
        // Theorem 6.4.6). The Newton model at the accepted step has a
        // predicted decrease `m(0) − m(δ) = −g·δ − 0.5·δ·H·δ`. For an
        // unclipped Newton step (H·δ = −g) this is `0.5·g·H⁻¹·g`, the
        // Newton decrement squared / 2. When the model itself predicts
        // a decrease smaller than the objective tolerance, no descent
        // direction the Hessian can resolve will lower the objective
        // by more than `objective_tol`, and continuing is wall-clock
        // waste regardless of whether the raw gradient residual or
        // step-norm gates have closed.
        //
        // Cross-cycle convergence carry-over: set at the end of every
        // accepted cycle so the next cycle's line-search-failure path
        // can distinguish a true KKT optimum on a rank-deficient
        // Hessian (no meaningful trial step, even though step_inf is
        // O(1) along the null mode) from genuine non-convergence.
        let mut last_cycle_residual_below_tol = false;
        let mut last_cycle_obj_change_below_tol = false;

        let mut joint_trust_radius = 1.0_f64;
        let mut joint_block_trust_radii = vec![1.0_f64; ranges.len()];
        let mut last_accepted_hit_joint_trust_boundary = false;
        // Hard upper bound for the for-loop's range. The cap is fixed at
        // `inner_max_cycles` for the lifetime of this outer call (the
        // earlier mid-loop cap extension was removed in favor of the
        // plateau-flat-objective convergence certificate), but the
        // sentinel pattern is retained — the `.max(200)` floor is a
        // harmless safety pad and the explicit `cycle >= inner_max_cycles`
        // break keeps the existing `continue` statements in the body
        // working
        // (they advance `cycle` via the iterator), unlike a `while` +
        // manual-counter rewrite.
        let inner_loop_hard_ceiling = inner_max_cycles.max(200);
        // Verbose cadence for the inner joint-Newton log block. Boring cycles
        // (first-attempt accepts with no convergence event) emit ONE compact
        // one-liner instead of the 4-line pre-cycle/TR/cycle-summary/convergence
        // block. Verbose cycles (first, last, every 20th, all rejections,
        // convergence events) keep the full detail. JOINT_LOG_VERBOSE_PERIOD is
        // tuned so a 200-cycle inner solve emits ~10 detailed waypoints plus
        // 1 compact line per remaining cycle (~210 lines), down from ~800.
        const JOINT_LOG_VERBOSE_PERIOD: usize = 50;
        // Residual-stall detector for joint Newton. Distinct from the
        // blockwise loglik-frozen divergence detector lower in the file:
        // that one requires the log-likelihood to be unchanged for K
        // cycles AND the per-block Newton step pinned at the cap.
        //
        // Large-scale survival marginal-slope hits a different pattern —
        // the joint objective decreases monotonically by O(1) per cycle
        // (so loglik is NOT frozen), the TR repeatedly clamps proposals
        // with |prop|∞ >> trust_radius, and the post-step KKT residual
        // oscillates in a band orders of magnitude above residual_tol
        // without trending down. Burning the rest of the cycle budget on
        // this pattern reaches inner_max_cycles "non-converged", which
        // then drops the outer optimizer into the first-order bridge
        // fallback with a stale-mode gradient that ‖g‖ ≈ 10⁷ kills BFGS
        // line search at iter 0.
        //
        // Track the best residual seen and the number of cycles since
        // any meaningful improvement (≥10% drop). Once we've burned at
        // least RESIDUAL_STALL_MIN_CYCLES with no improvement AND the
        // TR has been clamping aggressively, exit `converged=false` so
        // the outer optimizer sees a non-converged signal while we still
        // have a finite, in-range β to return (instead of running to the
        // hard ceiling and then handing BFGS a junk gradient).
        const RESIDUAL_STALL_NO_IMPROVE_CYCLES: usize = 30;
        const RESIDUAL_STALL_MIN_CYCLES: usize = 40;
        const RESIDUAL_STALL_IMPROVEMENT_FACTOR: f64 = 0.9;
        const RESIDUAL_STALL_BLOCK_GRADIENT_FACTOR: f64 = 50.0;
        let mut best_residual_seen: f64 = f64::INFINITY;
        // Smallest *certified* stationarity residual the solve actually computed,
        // tracked independently of `best_residual_seen` (whose updates are bound
        // to the residual-stall counters at the post-step site below and so are
        // skipped by every head-of-cycle / pre-line-search certificate exit). The
        // terminal verdict reports THIS so a legitimate early-certificate exit
        // (e.g. the cycle-0 pre-line-search KKT exit on intercept-only / already-
        // stationary data) reports the finite residual it certified on instead of
        // the sentinel `inf` — converged=true must never be paired with a non-
        // finite residual in the log (#1040 inner-report truthfulness).
        let mut min_certified_residual: f64 = f64::INFINITY;
        let mut cycles_since_residual_improved: usize = 0;
        // Number of consecutive non-improving cycles after which the
        // conditioning-based self-vanishing Levenberg–Marquardt damping is
        // ARMED inside the spectral-range Newton solve, for EVERY family
        // (#826/#808). The undamped range-restricted Newton step oscillates on a
        // full-rank-but-ill-conditioned penalized Hessian at the oversmoothed-ρ
        // operating point: the tiny-but-above-cutoff curvature of the lightly
        // identified mean/threshold/wiggle block takes an enormous `component/λ`
        // proposal that the trust region clips every cycle, so the residual on
        // that block freezes while its β stays ≈0 (the exact #826 signature).
        // The conditioning-gated `μ = c·‖∇L − Sβ‖∞` caps that component into a
        // bounded descent step. It is SELF-VANISHING (μ → 0 as the residual → 0)
        // so the converged β and the KKT certificate are byte-identical to the
        // undamped solve — zero REML/LAML bias. Arming it on OBSERVED non-
        // progress rather than a static per-family flag keeps the AFT /
        // constant-scale endgame (which converges quadratically and never
        // stalls) byte-identical: a quadratically-converging solve reaches
        // tolerance in a handful of cycles and never trips this threshold, so μ
        // is never engaged there. Only a genuinely oscillating ill-conditioned
        // solve crosses it, which is exactly when the damping is sound. Set a
        // few cycles below the stall-exit window so the damping gets a chance to
        // rescue the solve well before the early-exit / budget tripwire fires.
        // (The conditioning-gated self-vanishing μ this armed now lives ONLY in the
        // test-retained `solve_joint_newton_step_on_spectral_range`; the production
        // joint step takes the exact trust-region multiplier λ instead — gam#979.)
        // Recent KKT-residual values (oldest→newest) used to detect STEADY
        // geometric descent at the certificate-refusal gate. A still-converging
        // Newton direction (residual dropping by a steady factor < 1 each cycle)
        // must not be misclassified as a multiplier/null plateau and exited
        // early (gam#787 duchon centers≥20: the logslope block converges
        // geometrically — residual ~0.33×/cycle — but `linearized_rel ≥ 0.5`
        // routed it into the plateau-refusal break a few cycles short of tol).
        const RESIDUAL_DESCENT_WINDOW: usize = 3;
        let mut residual_descent_history: std::collections::VecDeque<f64> =
            std::collections::VecDeque::with_capacity(RESIDUAL_DESCENT_WINDOW);
        let mut tr_clamped_during_stall: bool = false;
        // Fully-rejected stall guard. The residual-stall guard below
        // (post-grad-reload) only fires on cycles that produced an accepted
        // step, because every termination check it gates lives after the
        // `if !accepted { continue; }` exit at the bottom of the trust-region
        // attempt loop. When every cycle in a row is fully rejected — all
        // JOINT_TRUST_MAX_ATTEMPTS trial steps fail the line-search check —
        // none of those guards ever see the iterate, the cycle loop spins
        // up to `inner_loop_hard_ceiling` cycles, and the inner solver burns
        // ~120 s of wall-clock per outer ρ-evaluation that the outer
        // optimizer will reject anyway. The signature is exact and local:
        // (i) every trust attempt this cycle was rejected by SOME path —
        // model, likelihood, objective, OR feasibility (the four counters
        // partition the JOINT_TRUST_MAX_ATTEMPTS attempts), so `model_rejects +
        // likelihood_rejects + objective_rejects + feasibility_rejects ==
        // JOINT_TRUST_MAX_ATTEMPTS`,
        // AND (ii) the joint trust radius has NOT shrunk relative to the
        // previous fully-rejected cycle. Condition (i) was originally
        // objective-only (`objective_rejects == MAX`, others 0), which never
        // fired on the biobank gauge-flat marginal/logslope fit: there the
        // objective is flat to f64 precision along the residual direction and
        // the BMS line search rejects every trial on the LIKELIHOOD early-exit
        // path, so the guard's increment was unreachable and the loop spun to
        // the cap. A full likelihood-path rejection at a collapsed radius is
        // the same no-descent stall, so any-path full rejection counts.
        // Condition (ii) is what proves no progress is possible: β is
        // reverted to its pre-cycle value on every fully-rejected cycle, so
        // with an identical Newton system AND an identical trust radius the
        // next cycle's trust-region search is byte-deterministically the
        // same as this one's. The radius can stall above the 1e-12 floor
        // when `shrink_active_joint_block_trust_radii` only shrinks blocks
        // that hit their per-block boundary — an interior block keeps its
        // radius forever, so `max(block_radii)` is held by that block while
        // the boundary block's radius collapses to 1e-12 without changing
        // the max. After `FULLY_REJECTED_STALL_MAX_CYCLES` consecutive cycles
        // with both conditions, judge convergence on the identified (range)
        // subspace: a stall at a collapsed radius proves the descent direction
        // is gauge-flat, so if the range-projected KKT residual is at tolerance
        // the fit is at a numerically-stationary penalized optimum and is
        // returned converged; only when the identified-subspace residual is
        // ALSO above tol is this a genuine non-convergence the outer optimizer
        // should reject — exit non-converged so it rejects this ρ cleanly
        // instead of waiting for the cycle cap.
        const FULLY_REJECTED_STALL_MAX_CYCLES: usize = 8;
        let mut prev_rejected_trust_radius: Option<f64> = None;
        let mut consecutive_held_rejected_cycles: usize = 0;
        // Byte-identical fixed-point detector for the fully-rejected stall.
        // Tracks the first-attempt trial objective of the previous fully-
        // rejected cycle; when the current fully-rejected cycle reproduces it
        // bit-for-bit the iterate is provably stationary at the f64 floor (the
        // n≈3e5 marginal/logslope coupling case where the line search rejects
        // every step on a 1-ULP cross-path round-off gap and β reverts
        // identically each cycle). One repeat is conclusive, so the guard fires
        // after two such cycles regardless of the `radius_held` heuristic.
        let mut prev_rejected_first_attempt_objective: Option<f64> = None;
        let mut consecutive_identical_rejected_cycles: usize = 0;
        const IDENTICAL_REJECTED_STALL_MAX_CYCLES: usize = 2;
        // Collapsed-trust-region all-reject-at-floor guard (gam#979 survival
        // hang / binary high-`centers` `IntegrationError`). DISTINCT from the
        // two detectors above:
        //   * `consecutive_held_rejected_cycles` requires the radius to be HELD
        //     relative to the *previous reject* — which it is at any pinned
        //     value, floor or not — and only fires after 8 cycles.
        //   * `consecutive_identical_rejected_cycles` requires the trial
        //     objective to repeat BIT-FOR-BIT, which a near-singular coupled
        //     marginal↔logslope system need not do: tiny non-deterministic
        //     round-off in the per-row tower contraction perturbs the trial
        //     objective in its last ULPs even while the step is otherwise
        //     stuck, so the byte-identical detector never latches.
        // The unambiguous deterministic signal of "stuck and cannot recover" is
        // the trust radius sitting at its absolute `1e-12` floor WHILE every
        // line-search attempt is rejected: no smaller step is representable, so
        // the radius cannot shrink further, and the all-reject means the step
        // makes no progress. After `JOINT_COLLAPSED_FLOOR_ALL_REJECT_MAX_CYCLES`
        // consecutive such cycles the loop is provably grinding to its budget on
        // a near-singular system (`phantom_multiplier_with_well_conditioned_H`),
        // so exit cleanly through the SAME identified-subspace / fixed-point
        // certificate path the other two detectors use — converged if the
        // range-space residual is stationary, give-best non-converged otherwise
        // — instead of spinning out the full `inner_max_cycles`. The absolute-
        // floor requirement is why this CANNOT fire on a genuinely progressing
        // fit: a fit that is descending keeps the radius well above `1e-12`
        // (it grows on `rho>0.75`/boundary and only collapses to the floor after
        // a sustained reject streak), so the counter resets on every accepted
        // cycle and never reaches the threshold.
        // Threshold + floor ceiling live in `joint_newton.rs` so the loop and
        // the `joint_newton_collapsed_trust_region_all_reject_exits_before_grinding_budget`
        // unit test assert against one source of truth.
        let mut consecutive_all_reject_at_floor_cycles: usize = 0;
        let mut last_joint_math: Option<JointNewtonMathDiagnostic> = None;
        // Cross-cycle cache of the joint Jeffreys/Firth triple `(β_key, ∇Φ, H_Φ)`
        // (gam#729/#826/#808). Computing `(∇Φ, H_Φ)` costs `p` family
        // directional-derivative calls plus the `½ S Sᵀ` GEMM; for a K-block
        // coupled family that is the dominant per-inner-cycle cost. The post-step
        // KKT residual recomputes the triple at the just-accepted β; the NEXT
        // cycle's head needs the SAME triple at that SAME β. Carry it forward
        // keyed on the flattened β so the head reuses the post-step result instead
        // of recomputing — collapsing two O(p)-directional-derivative evaluations
        // per accepted cycle to one. The key is an exact-equality check on the
        // flattened β (β is byte-identical between an accepted post-step residual
        // and the next head), so the reused term is the exact term at the current
        // iterate — no staleness, no tolerance fudge.
        let mut jeffreys_triple_cache: Option<(Array1<f64>, Array1<f64>, Array2<f64>)> = None;
        // Stash for the structured cert-REFUSED report computed inside the
        // cycle loop, so the post-loop bubbled error (`coupled exact-joint
        // inner solve exited the joint Newton path …`) can emit the same
        // per-block + spectrum breakdown without re-materializing H_pen.
        let mut last_kkt_refusal_report: Option<KktRefusalReport> = None;
        let mut prev_kkt_norm: Option<f64> = None;
        // Convergence-endgame flag for the Jeffreys second-order completion
        // (gam#979): set once the post-step KKT residual enters
        // `JEFFREYS_COMPLETION_RESIDUAL_BAND × residual_tol`, consumed by the
        // next cycle's dense-spectral step assembly.
        let mut jeffreys_completion_endgame = false;
        // Plateau streak on |Δobj| ≤ objective_tol. The scale-aware
        // flatness predicate stays local to this loop; the streak/window
        // discipline (grow on flat, reset on recovery) is the shared
        // loop_guard::FlatStreak so it cannot drift from the other
        // stagnation detectors in the tree (#968).
        let mut obj_flat_streak = crate::solver::loop_guard::FlatStreak::new(
            crate::solver::loop_guard::PLATEAU_DEFAULT_WINDOW,
        );
        // Total descent budget across the joint-Newton loop, used by
        // the end-of-loop summary to report `descent_total`.
        let initial_joint_objective: f64 = lastobjective;
        // Per-cycle |Δobjective| history for the geometric-tail trigger of
        // the constrained-stationary certificate below. When the cycles
        // settle into a linear-rate plateau (|Δobj_next| / |Δobj_prev|
        // approaching 1 monotonically over the window), the total
        // *remaining* objective descent is rigorously bounded above by the
        // geometric series sum |Δobj_now| / (1 − max_ratio). When that
        // bound is below `objective_tol` the cert can fire many cycles
        // earlier than waiting for any single |Δobj| to individually
        // cross obj_tol — the bound is mathematically the same precision
        // contract, applied to the asymptotic tail rather than one step.
        const GEOMETRIC_TAIL_WINDOW: usize = 5;
        let mut geometric_tail_history: std::collections::VecDeque<f64> =
            std::collections::VecDeque::with_capacity(GEOMETRIC_TAIL_WINDOW);

        // The exact joint-Hessian route solves the penalized Newton system
        // directly. Extra damping must be wired through an accepted/rejected
        // step policy before it belongs here; keep the matvec faithful to the
        // objective until then.
        for cycle in 0..inner_loop_hard_ceiling {
            if cycle >= inner_max_cycles {
                break;
            }
            let verbose_cycle = cycle == 0
                || cycle + 1 == inner_max_cycles
                || (cycle + 1) % JOINT_LOG_VERBOSE_PERIOD == 0;
            // Pre-cycle header line removed: the post-cycle one-liner below
            // carries cycle/objective/Δobj/step/residual/time and on verbose
            // cadence the expanded convergence line additionally carries
            // -loglik and penalty. Suppressing this avoids emitting a second
            // info-level line per cycle just to repeat numbers we already
            // log at end of cycle.
            // Per-cycle phase-timing accumulators. Surface where the inner
            // joint-Newton spends time so a 18-min silent cycle 0 (the
            // bernoulli marginal-slope FLEX large-scale failure mode) becomes a
            // logged timeline at the end of the cycle. Phases:
            //   * hessian: joint Hessian source build (matrix-free workspace
            //     OR dense fallback assembly)
            //   * pcg:     matrix-free QP solve via solve_spd_pcg_with_info_into
            //              (already logs its own diagnostics; we accumulate
            //              here for the end-of-cycle summary)
            //   * line_search: backtracking step-size search (up to 8 attempts)
            //   * grad_reload: post-accept joint gradient + workspace refresh
            let cycle_started = std::time::Instant::now();
            // Top-of-cycle row-measure capture. The trust-region ratio
            // ρ = [F(β) − F(β + δ)] / [−g·δ − ½·δᵀHδ] is only meaningful when
            // every input (Hessian, gradient, objective at β, trial objective
            // at β + δ) is evaluated against the same row measure. We freeze
            // the measure here and re-read it at each of the four sites later
            // in the cycle, then hard-fail (Err) just before ρ if any of them
            // diverged. Cf. `src/solver/row_measure.rs`.
            let tr_row_measure_top =
                crate::solver::row_measure::RowSubsampleMask::from_options(options, total_joint_n);
            let hessian_started = std::time::Instant::now();
            let hessian_scope_guard = crate::process_monitor::track_scope(format!(
                "joint Newton hessian_qp cycle={cycle} n={total_joint_n} p={total_p}"
            ));
            log::info!(
                "[joint-newton-tr] phase=hessian_qp cycle={} r={:.3e}",
                cycle,
                joint_trust_radius,
            );
            let cycle_log = prelude_log;
            let constraints_started = std::time::Instant::now();
            let block_constraints = collect_block_linear_constraints(family, &states, specs)?;
            let joint_constraints =
                assemble_joint_linear_constraints(&block_constraints, &ranges, total_p)?;
            if cycle_log && cycle == 0 {
                log::info!(
                    "[STAGE] PIRLS/inner step=cycle0 block+joint constraints elapsed={:.3}s n={} p={}",
                    constraints_started.elapsed().as_secs_f64(),
                    total_joint_n,
                    total_p,
                );
            }
            let workspace_build_started = std::time::Instant::now();
            // Get joint Hessian and block gradients from the current evaluation.
            // Hold the cycle's exact-Newton workspace (cache of per-row kernel
            // evaluations at the current β) so a REJECTED cycle can hand it back
            // to `cached_joint_workspace` for the next cycle. After a reject the
            // line search restores β to `old_beta` — exactly the β this workspace
            // was built at — so reusing the cache is bit-identical and skips the
            // O(n) row-kernel re-evaluation (`build_row_kernel_cache`) that
            // otherwise reruns the full data through the per-row CDF/derivative
            // math on every rejected cycle. The converged-exit paths below null
            // this (no carry-forward needed once the inner solve returns).
            let mut hessian_workspace_for_cycle: Option<Arc<dyn ExactNewtonJointHessianWorkspace>> =
                None;
            let joint_hessian_source = if joint_workspace_requested {
                let cached_hit = cached_joint_workspace.is_some();
                let workspace = match cached_joint_workspace.take() {
                    Some(workspace) => Some(workspace),
                    None => family.exact_newton_joint_hessian_workspace_with_options(
                        &states, specs, options,
                    )?,
                };
                if cycle_log && cycle == 0 {
                    log::info!(
                        "[STAGE] PIRLS/inner step=cycle0 hessian-workspace cached_hit={} elapsed={:.3}s n={} p={}",
                        cached_hit,
                        workspace_build_started.elapsed().as_secs_f64(),
                        total_joint_n,
                        total_p,
                    );
                }
                hessian_workspace_for_cycle = workspace.clone();
                workspace
                    .as_ref()
                    .map(|workspace| {
                        exact_newton_joint_hessian_source_from_workspace(
                            workspace,
                            total_p,
                            MaterializationIntent::InnerSolve,
                            "joint Newton inner exact-newton operator mismatch",
                        )
                    })
                    .transpose()?
                    .flatten()
            } else {
                None
            };
            // Row measure observed by the Hessian build above.
            let tr_row_measure_hessian =
                crate::solver::row_measure::RowSubsampleMask::from_options(options, total_joint_n);
            let joint_hessian_source = match joint_hessian_source {
                Some(source) => source,
                None => {
                    // Spec-aware joint Hessian: canonical coupled-curvature
                    // source (see the availability gate above). Families that
                    // only override `_with_specs` (Dirichlet common-parameter)
                    // would otherwise hand back `None` from the spec-less
                    // default and silently drop off the joint-Newton path.
                    let h_joint_opt =
                        family.exact_newton_joint_hessian_with_specs(&states, specs)?;
                    let Some(h_joint) = h_joint_opt else {
                        break; // Fall back to blockwise if joint Hessian unavailable
                    };
                    match symmetrized_square_matrix(
                        h_joint,
                        total_p,
                        "joint Newton inner exact-newton Hessian shape mismatch",
                    ) {
                        Ok(matrix) => JointHessianSource::Dense(matrix),
                        Err(_) => break,
                    }
                }
            };
            let hessian_source_elapsed = workspace_build_started.elapsed();
            if hessian_source_elapsed.as_secs_f64() >= 1.0 || (cycle_log && cycle == 0) {
                let source_kind = if matches!(&joint_hessian_source, JointHessianSource::Dense(_)) {
                    "dense"
                } else {
                    "operator"
                };
                log::info!(
                    "[STAGE] PIRLS/inner step=cycle{} hessian-source joint_workspace_requested={} source={} elapsed={:.3}s n={} p={}",
                    cycle,
                    joint_workspace_requested,
                    source_kind,
                    hessian_source_elapsed.as_secs_f64(),
                    total_joint_n,
                    total_p,
                );
            }

            // Concatenate block gradients and betas.
            let Some(grad_joint) = cached_joint_gradient.clone() else {
                break;
            };
            // Row measure observed by the gradient at β. `cached_joint_gradient`
            // was loaded earlier under `options`; if the auto-subsample
            // installer or any sibling path swapped the mask between then and
            // now, the id captured here will diverge from the rest and the
            // pre-ρ check below will Err. Cf. `src/solver/row_measure.rs`.
            let tr_row_measure_gradient =
                crate::solver::row_measure::RowSubsampleMask::from_options(options, total_joint_n);
            if grad_joint.len() != total_p {
                break;
            }
            let mut beta_joint = Array1::<f64>::zeros(total_p);
            for b in 0..specs.len() {
                let (start, end) = ranges[b];
                beta_joint
                    .slice_mut(ndarray::s![start..end])
                    .assign(&states[b].beta);
            }

            // Non-finite-curvature guard (gam#1088). A `NaN`/`Inf` in the
            // family curvature `H` makes the penalized Hessian `H_pen = H +
            // S(λ)` — and therefore its spectrum — degenerate, so the KKT
            // certificate is structurally unreachable: the spectral step
            // solve produces garbage, the projected residual neither converges
            // nor trends down, and the residual-based divergence/stall guards
            // below (gated on a *finite* residual that a corrupted-but-not-yet-
            // propagated curvature can still leave finite) do not catch it.
            // Left unguarded the loop then burns the full `inner_loop_hard_
            // ceiling` (1200 cycles) on every outer ρ-eval / seed — the
            // multi-hour link-wiggle & location-scale benchmark timeouts. The
            // penalty is finite by construction, so this is a curvature defect:
            // the trial is degenerate. Exit immediately as non-converged with
            // the current finite β so the outer optimizer rejects this ρ-eval
            // cleanly (mirrors the residual divergence guard below), rather
            // than grinding to the ceiling and reporting a `NaN` H_pen
            // spectrum at the refusal point.
            if !joint_hessian_source_curvature_is_finite(&joint_hessian_source) {
                cycles_done = cycle + 1;
                log::warn!(
                    "[PIRLS/joint-Newton convergence] cycle {:>3} | non-finite-curvature guard (gam#1088): the joint Hessian source carries a non-finite entry, so the penalized Hessian H_pen = H + S(λ) and its spectrum (λ_max/λ_min/cond) are degenerate and the KKT certificate can never be issued; returning unconverged with finite β so the outer optimizer rejects this ρ evaluation instead of grinding to inner_max_cycles={}.",
                    cycle,
                    inner_max_cycles,
                );
                converged = false;
                break;
            }

            let trace_diagonal_ridge = joint_mode_diagonal_ridge + JOINT_TRACE_STABILITY_RIDGE;
            let joint_hessian_is_dense =
                matches!(&joint_hessian_source, JointHessianSource::Dense(_));
            let joint_solver_diagonal_ridge = stabilized_joint_solver_diagonal_ridge(
                family,
                &joint_hessian_source,
                &ranges,
                &s_lambdas,
                trace_diagonal_ridge,
                options.ridge_floor,
                joint_bundle,
            );
            // CHEAP CONDITIONING PRE-CHECK (always-on robustness, zero-cost on
            // easy/large fits). Before paying for the dense joint-Hessian
            // materialization + `O(p³)` reduced eigendecomposition inside the
            // Jeffreys term, ask whether the term is PROVABLY skippable from a few
            // matrix-free Hessian-vector products against the source we just built.
            // When `true`, the exact conditioning gate is certain to return the
            // zero term, so every Jeffreys call this cycle short-circuits to the
            // exact-zero contribution WITHOUT forming anything dense — byte-
            // identical to the gated-off path, and preserving the matrix-free path
            // on wide well-conditioned fits. Only runs the estimate when a Jeffreys
            // subspace exists and `total_p` is wide enough that the dense eigh is
            // the cost we want to avoid (the helper itself gates on the size
            // threshold and conservatively returns `false` if unsure). Computed
            // once per inner cycle and reused across the cycle's head-KKT, step,
            // and trial-value calls; the conditioning changes slowly across cycles
            // so re-estimating per cycle (one `O(p·k)` burst) is already cheap
            // against the work it guards.
            let jeffreys_skippable_this_cycle: bool = if options.seed_screening {
                // Seed screening only ranks seeds: skip the O(p · per-axis-Hdot)
                // full Jeffreys gradient/curvature loop. The value-only Jeffreys
                // term (folded into the objective baseline / trial penalties via
                // `custom_family_joint_jeffreys_value`, gated independently on
                // `joint_jeffreys_subspace.is_some()`) still bounds the screening
                // score on separating directions; only the per-axis step curvature
                // — the wrong cost class for ranking on a K-block coupled family —
                // is dropped here (gam#729/#808).
                true
            } else if joint_jeffreys_subspace.is_some() {
                // EXPECTED-INFORMATION GUARD (gam#1020): the skippable
                // certificate probes the OBSERVED Hessian source; it only
                // transfers to the Jeffreys gate when the family's Jeffreys
                // information IS the observed Hessian. Expected-information
                // families (probit-class) bypass the pre-check — observed
                // information grows on saturated rows exactly where the
                // expected information collapses and the gate must arm.
                family.joint_jeffreys_information_matches_observed_hessian()
                    && jeffreys_term_skippable_for_source(&joint_hessian_source, total_p)
                        .unwrap_or(false)
            } else {
                false
            };
            let joint_trust_metric_diag = match &joint_hessian_source {
                JointHessianSource::Dense(h_joint) => joint_penalty_preconditioner_diag(
                    &h_joint.diag().to_owned(),
                    &ranges,
                    &s_lambdas,
                    joint_solver_diagonal_ridge,
                    joint_bundle,
                ),
                JointHessianSource::Operator { diagonal, .. } => joint_penalty_preconditioner_diag(
                    diagonal,
                    &ranges,
                    &s_lambdas,
                    joint_solver_diagonal_ridge,
                    joint_bundle,
                ),
            };
            // HEAD-β JEFFREYS CACHE (gam#729/#808). The full Jeffreys/Firth triple
            // `(Φ, ∇Φ, H_Φ)` costs `p` family directional-derivative calls (the
            // `for k in 0..p` loop in `joint_jeffreys_term`); for a K-block coupled
            // family (Dirichlet/multinomial) that is the dominant per-cycle cost.
            // The head-of-cycle KKT residual, the constrained-QP step, and the
            // spectral/dense Newton step are ALL built at the SAME cycle-start β
            // (`&states`, before any step is accepted), so they need the SAME
            // triple. Compute it ONCE here and reuse, instead of three independent
            // O(p)-directional-derivative evaluations per cycle. The post-step
            // residual below is at the accepted β, so it correctly recomputes.
            // `None` when the term is condition-gated/skippable (∇Φ=0, H_Φ=0).
            let head_beta_key: Array1<f64> = flatten_state_betas(&states, specs);
            let head_jeffreys_term: Option<(Array1<f64>, Array2<f64>)> =
                if jeffreys_skippable_this_cycle {
                    None
                } else if let Some((_, grad_phi, hphi)) = jeffreys_triple_cache
                    .as_ref()
                    .filter(|(key, _, _)| *key == head_beta_key)
                {
                    // Cross-cycle cache hit: the previous cycle's post-step KKT
                    // residual already computed the exact triple at this β. Reuse.
                    Some((grad_phi.clone(), hphi.clone()))
                } else if let Some(z_joint) = joint_jeffreys_subspace.as_ref() {
                    let term = match custom_family_joint_jeffreys_term(
                        family, &states, specs, &ranges, z_joint,
                    )? {
                        Some((_phi, grad_phi, hphi))
                            if grad_phi.len() == grad_joint.len()
                                && hphi.nrows() == total_p
                                && hphi.ncols() == total_p =>
                        {
                            Some((grad_phi, hphi))
                        }
                        _ => None,
                    };
                    if let Some((grad_phi, hphi)) = term.as_ref() {
                        jeffreys_triple_cache =
                            Some((head_beta_key.clone(), grad_phi.clone(), hphi.clone()));
                    }
                    term
                } else {
                    None
                };
            // Fold the Firth/Jeffreys score `∇Φ` into the head-of-cycle KKT
            // residual when the term is armed, for the same reason as the
            // post-step residual below: the inner objective is `−ℓ + ½βᵀSβ − Φ`,
            // so the certifiable stationarity is `∇L − Sβ + ∇Φ = 0`. Without
            // this the head-of-cycle KKT exit (`current_stationarity_residual ≤
            // residual_tol`) can never fire on the near-separating span, even
            // when the iterate is the Firth optimum. No-op when the Jeffreys
            // term is unavailable or condition-gated to zero.
            let head_kkt_gradient: Option<Array1<f64>> = head_jeffreys_term
                .as_ref()
                .map(|(grad_phi, _hphi)| &grad_joint + grad_phi);
            let current_kkt_norm = exact_newton_joint_stationarity_inf_norm_from_gradient(
                head_kkt_gradient.as_ref().unwrap_or(&grad_joint),
                &states,
                specs,
                &s_lambdas,
                ridge,
                options.ridge_policy,
                &block_constraints,
                Some(cached_active_sets.as_slice()),
            )?;
            if current_kkt_norm.is_finite() {
                min_certified_residual = min_certified_residual.min(current_kkt_norm);
            }
            let pcg_rel_tol = joint_pcg_eisenstat_walker_forcing(prev_kkt_norm, current_kkt_norm);

            let solve_joint_constraints_dense = joint_constraints.is_some()
                || !matrix_free_joint_requested
                || joint_hessian_is_dense;
            if cycle == 0 {
                log::info!(
                    "[JN-BRANCH-DIAG #1040] cycle=0 joint_constraints_is_some={} matrix_free_joint_requested={} joint_hessian_is_dense={} solve_joint_constraints_dense={} -> branch={} total_p={} levenberg_on_ill_cond={}",
                    joint_constraints.is_some(),
                    matrix_free_joint_requested,
                    joint_hessian_is_dense,
                    solve_joint_constraints_dense,
                    if solve_joint_constraints_dense && joint_constraints.is_some() {
                        "CONSTRAINED_QP"
                    } else if matrix_free_joint_requested && !joint_hessian_is_dense {
                        "MATRIX_FREE_PCG"
                    } else {
                        "DENSE_SPECTRAL"
                    },
                    total_p,
                    family.levenberg_on_ill_conditioning(),
                );
            }
            // Exact trust-region subproblem factorization (gam#979). Populated on
            // the unconstrained dense-spectral path with the metric-whitened
            // eigendecomposition of the penalized Hessian, so the trust loop below
            // re-solves the *exact* Moré–Sorensen subproblem at each trust radius
            // from one factorization — replacing the dogleg/Cauchy/box-truncation
            // globalization with the single object they all approximate. `None` on
            // the constrained-QP and matrix-free PCG paths, which keep their
            // existing globalization untouched.
            let mut joint_spectrum: Option<whitened_spectrum::WhitenedHessianSpectrum> = None;
            // DENSE-FALLBACK OPERATOR MATERIALIZATION REUSE (gam#1040). On the
            // DENSE_SPECTRAL path the inner Hessian `source` can be a matrix-free
            // `Operator` (BMS flex, large n, p below the matrix-free joint-dim
            // threshold so PCG is not requested): the dense-fallback below then
            // calls `materialize_joint_hessian_source` to form the unpenalized
            // dense `H` ONCE for the spectral `decompose`. Without capturing it,
            // the per-cycle Cauchy leg and the up-to-`JOINT_TRUST_MAX_ATTEMPTS`
            // predicted-reduction matvecs each re-apply the operator's `apply_into`
            // — an `O(n·p)` row sweep over n≈196k rows, ~25× per cycle — when the
            // identical action is already available as an `O(p²)` dense matvec.
            // Capturing the unpenalized dense here and routing those matvecs
            // through a `Dense` source is byte-identical (the dense build IS the
            // operator's action by construction of `materialize_joint_hessian_source`)
            // and removes the dominant residual per-cycle row work on this path.
            let mut materialized_dense_unpenalized: Option<Array2<f64>> = None;
            let (candidate_beta, joint_active_set, joint_step_spectral_nullity) =
                if solve_joint_constraints_dense
                    && let Some(constraints) = joint_constraints.as_ref()
                {
                    let mut lhs = match materialize_joint_hessian_source(
                        &joint_hessian_source,
                        total_p,
                        "joint Newton inner constrained Hessian materialization",
                    ) {
                        Ok(matrix) => matrix,
                        Err(_) => break,
                    };
                    add_joint_penalty_to_matrix(
                        &mut lhs,
                        &ranges,
                        &s_lambdas,
                        trace_diagonal_ridge,
                        joint_bundle,
                    );
                    if joint_solver_diagonal_ridge != trace_diagonal_ridge {
                        for d in 0..lhs.nrows() {
                            lhs[[d, d]] += joint_solver_diagonal_ridge - trace_diagonal_ridge;
                        }
                    }
                    check_linear_feasibility(&beta_joint, constraints, 1e-8).map_err(|e| {
                        format!("joint Newton constrained solve [cycle={cycle}]: {e}")
                    })?;
                    let warm_joint_active =
                        flatten_joint_active_set(&cached_active_sets, &block_constraints);
                    let lower_bounds = match extract_simple_lower_bounds(constraints, total_p) {
                        Ok(bounds) => bounds,
                        Err(_) => break,
                    };
                    // Newton IRLS step in absolute-β space:
                    //
                    //   β_new = H_pen⁻¹ (H_L β + ∇ℓ)
                    //
                    // where H_pen = H_L + S, derived from Newton's update
                    //   β_new = β + H_pen⁻¹(∇ℓ − Sβ)
                    //         = H_pen⁻¹(H_pen β + ∇ℓ − Sβ)
                    //         = H_pen⁻¹(H_L β + ∇ℓ).
                    //
                    // The QP `min 0.5 β' H_pen β − rhs_beta' β` has unconstrained
                    // optimum β = H_pen⁻¹ rhs_beta, so rhs_beta = H_pen β + (∇ℓ − Sβ)
                    // gives the correct Newton update. Passing raw grad_joint (=∇ℓ)
                    // would collapse to β = H_pen⁻¹ ∇ℓ, which at the true optimum
                    // (∇ℓ = Sβ̂) gives H_pen⁻¹ Sβ̂ ≠ β̂ — wrong fixed point.
                    let penalty_beta_joint = apply_joint_block_penalty(
                        &ranges,
                        &s_lambdas,
                        &beta_joint,
                        joint_mode_diagonal_ridge,
                        joint_bundle,
                    );
                    let mut rhs_step = &grad_joint - &penalty_beta_joint;
                    // Reuse the head-β Jeffreys triple (consistently attenuated in
                    // `head_jeffreys_term` — both ∇Φ and H_Φ scaled by one scalar,
                    // gam#826/#872/#715). Skipped when the cheap pre-check certifies
                    // well-conditioning: ∇Φ = 0 and H_Φ = 0 there, so neither
                    // rhs_step nor lhs change.
                    // PSD PROJECTION (gam#979). The exact divided-difference H_Φ is
                    // indefinite exactly where Φ is (mixed-sign reduced spectrum at
                    // off-mode trial points). The unconstrained dense-spectral path
                    // consumes it exactly — the Moré–Sorensen subproblem handles
                    // indefiniteness rigorously — but THIS active-set QP requires a
                    // convex model (an indefinite QP cycles its active set and the
                    // inner grinds the budget). Use the PSD part of H_Φ here: honest
                    // magnitudes (unlike the old `K²` vec-Gram phantom), guaranteed
                    // solvable QP, and the exact ∇Φ in the rhs keeps the fixed point
                    // unchanged — only the convergence rate on indefinite stretches
                    // degrades to the damped-Newton rate the constrained path always
                    // had.
                    if let Some((grad_phi, hphi)) = head_jeffreys_term.as_ref()
                        && grad_phi.len() == rhs_step.len()
                    {
                        rhs_step += grad_phi;
                        lhs += &symmetric_psd_projection(hphi);
                    }
                    // Self-vanishing Levenberg–Marquardt damping for the
                    // CONSTRAINED active-set QP, mirroring the spectral-range
                    // branch below (μ = JOINT_SPECTRAL_LEVENBERG_FACTOR·‖rhs‖∞).
                    //
                    // When the joint design carries inequality constraints
                    // (the monotone I-spline time-warp of a survival
                    // location-scale / AFT fit) the spectral range step that
                    // drops ker(H_pen) is NOT taken — this dense active-set QP
                    // runs instead. On a constant-scale AFT the 12-col monotone
                    // time-warp's non-affine deviation is statistically
                    // UNIDENTIFIED, so H_pen is rank-deficient along that gauge
                    // direction. An undamped QP then has a continuum of optima
                    // differing only by the free gauge component, and the
                    // active set slides along the monotone constraint face
                    // taking an O(1) proposal step in that direction every
                    // cycle. The proposal `step_inf` never exhausts, so the
                    // identified-subspace KKT certificate (gated on
                    // `step_inf ≤ step_tol`) never fires and the inner
                    // joint-Newton grinds the full `inner_max_cycles` on EVERY
                    // outer ρ-eval — the survival-LS AFT "hang" (#736/#735/#721).
                    //
                    // Adding μ·I to the QP Hessian gives ker(H_pen) a tiny
                    // positive curvature, so the constrained minimizer is unique
                    // and its gauge component is driven toward zero; the proposal
                    // step then exhausts at the identified-subspace optimum and
                    // the certificate fires in a handful of cycles. Because
                    // μ ∝ ‖∇L − Sβ‖∞ → 0 at the KKT fixed point, the converged β
                    // and the well-identified flexible-scale fast path (where the
                    // time-warp IS identified and H_pen is non-singular) are
                    // unchanged — a genuinely flexible survival-LS fit still
                    // performs its full search.
                    //
                    // CRITICAL: the floor is only correct on a genuinely
                    // rank-deficient `H_pen`. Gate it strictly on
                    // `nullity > 0`. On a FULLY IDENTIFIED constrained fit
                    // (e.g. the post-reduction constant-scale loglogistic AFT,
                    // #736/#735/#721/#733/#734 — a 3-parameter model with
                    // block_widths = [1,1,1] and an empty `ker(H_pen)`) the QP
                    // minimizer is already unique, so the floor adds nothing it
                    // is needed for but everything it costs: with residual r and
                    // factor 1e-3 the floor is μ≈1e-3·r, and on an unpenalized
                    // location intercept whose likelihood curvature H is small
                    // at n=23 the damped Newton component shrinks the residual
                    // only by the GEOMETRIC ratio H/(H+μ) per cycle instead of
                    // quadratically. With μ≈1e-6 and a small H that ratio is far
                    // from 1, so the threshold-block stationarity residual
                    // plateaus at ~1e-3–1e-4 and the inner solve burns its whole
                    // cycle budget without ever reaching `residual_tol`. The
                    // self-vanishing μ→0 is too slow because it vanishes only as
                    // fast as the residual it is throttling. Disabling the floor
                    // when `nullity == 0` makes the constrained QP solve the
                    // EXACT undamped Newton/KKT system, recovering quadratic
                    // convergence to `residual_tol` in a handful of cycles. The
                    // rank-deficient case (`nullity > 0`, the pre-reduction
                    // unidentified time-warp gauge) keeps the floor and its hang
                    // fix unchanged. `None` (eigensolve failed / zero Hessian)
                    // falls back to the damped path conservatively.
                    // gam#1040: the survival marginal-slope joint shares one
                    // matern PC basis between the marginal and the log-slope
                    // surface, so `H_pen` is FULL RANK (`nullity == 0`) yet
                    // severely ill-conditioned (cond ≈ 5.8e6). With the floor
                    // gated on `nullity > 0` alone the undamped active-set QP has
                    // a constrained minimiser that is unique only up to round-off
                    // along the near-null mode: the active set slides an O(1)
                    // proposal step every cycle, `step_inf` never exhausts, the
                    // constrained-fixed-point / KKT certificate never fires, and
                    // the inner joint-Newton grinds the full cycle budget on EVERY
                    // outer ρ-eval (the hours-long survival-MS hang). The family
                    // opts into damping this case via
                    // `levenberg_on_ill_conditioning()`; the self-vanishing μ
                    // (∝ projected residual → 0 at the KKT fixed point) gives the
                    // near-null mode a tiny positive curvature so the minimiser is
                    // unique and `step_inf` exhausts, WITHOUT moving the converged
                    // β. Apply it only when the matrix is genuinely ill-conditioned
                    // (`cond > LEVENBERG_ILL_CONDITIONING_THRESHOLD`); a
                    // well-conditioned full-rank constrained fit (the tiny
                    // unpenalised loglogistic AFT, #736/#735/#721, where the floor
                    // would cap the convergence rate at the geometric H/(H+μ) ratio)
                    // keeps the EXACT undamped Newton/KKT solve and its quadratic
                    // convergence. `None` (eigensolve failed / zero Hessian) falls
                    // back to the damped path conservatively.
                    let (hpen_nullity, hpen_condition) =
                        match symmetric_penalized_hessian_nullity_and_condition(&lhs) {
                            Some((n, c)) => (Some(n), c),
                            None => (None, f64::INFINITY),
                        };
                    let nullity_floor = hpen_nullity.map(|n| n > 0).unwrap_or(true);
                    let ill_conditioned_floor = family.levenberg_on_ill_conditioning()
                        && hpen_nullity == Some(0)
                        && hpen_condition > LEVENBERG_ILL_CONDITIONING_THRESHOLD;
                    let apply_constrained_floor = nullity_floor || ill_conditioned_floor;
                    // Self-vanishing scale = the PROJECTED stationarity residual
                    // (`current_kkt_norm`), NOT the raw ‖∇ℓ − Sβ + ∇Φ‖∞. At a
                    // CONSTRAINED optimum the raw RHS converges to the active-set
                    // multiplier mass ‖Aᵀλ‖∞ — an O(1) quantity that never
                    // vanishes — so a floor scaled by it never lifts, throttling
                    // every weakly-curved identified direction to a geometric
                    // H/(H+μ) contraction and exhausting the inner budget with the
                    // projected residual stalled just above tolerance (#1025: the
                    // competing-risks twin time-basis fit, per_block_resid stuck at
                    // 1.457 for the full budget). The projected residual is the
                    // honest distance-from-KKT measure: it equals the raw RHS on
                    // unconstrained fits (no behavior change there) and → 0 at a
                    // constrained optimum, so the floor vanishes exactly where the
                    // comment above promises it does.
                    let rhs_inf = rhs_step.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
                    let floor_scale = if current_kkt_norm.is_finite() {
                        current_kkt_norm.min(rhs_inf)
                    } else {
                        rhs_inf
                    };
                    let constrained_levenberg_mu = JOINT_SPECTRAL_LEVENBERG_FACTOR * floor_scale;
                    if apply_constrained_floor
                        && constrained_levenberg_mu > 0.0
                        && constrained_levenberg_mu.is_finite()
                    {
                        for d in 0..lhs.nrows() {
                            lhs[[d, d]] += constrained_levenberg_mu;
                        }
                    }
                    // MODIFIED-NEWTON CONVEXIFICATION (gam#1040 / gam#979). The
                    // exact survival marginal-slope joint NLL Hessian is INDEFINITE
                    // on the flat baseline-hazard λ valley (the linear baseline +
                    // the z·exp(logslope) cross-coupling carry genuine negative
                    // curvature away from the optimum). The active-set QP below
                    // minimizes `½βᵀHβ − rhs_betaᵀβ`; with an indefinite `H` that
                    // model has a direction that LOWERS the local quadratic
                    // objective while moving AWAY from the KKT point. The
                    // trust-region wrapper gates acceptance on the objective-
                    // reduction ratio ρ — NOT on the stationarity residual — so it
                    // accepts every such step at ρ≈1 and GROWS its radius while the
                    // stationarity residual DIVERGES (the measured 3.5e4 → 9.5e6
                    // blow-up on the time block). The unconstrained dense-spectral
                    // path never exhibits this because `WhitenedHessianSpectrum`
                    // already reflects negative-curvature modes to `|γ|`; the
                    // constrained branch must do the same to its dense `lhs`.
                    // Reflecting (not clamping-to-zero) keeps the curvature
                    // magnitude so the QP stays bounded and the step length matches
                    // the dense path; at a genuine constrained optimum the reduced
                    // Hessian is PSD so this is a no-op and the converged β is
                    // unchanged.
                    //
                    // NEWTON-DECREMENT CERTIFICATE ON THE CONSTRAINED PATH
                    // (gam#1040 / gam#1088). The dense-spectral branch populates
                    // `joint_spectrum` (line ~1493) so the convergence loop's
                    // Newton-decrement exit can terminate the geometric/linear tail
                    // when the achievable model descent `½ Σ c_k²/|γ_k|` drops below
                    // `objective_tol`. The constrained branch never set it, so a
                    // weakly-identified survival-MS fit (the n≈2e5 logslope block,
                    // step clamped by the trust region, residual creeping ~7%/cycle)
                    // had no early-exit and ground the whole budget. Build the same
                    // D-whitened spectrum from the penalized `lhs` (decrement reflects
                    // negative modes via `.abs()` internally, so the pre-reflection
                    // `lhs` is the right input) and the augmented stationarity RHS, so
                    // the decrement read is consistent with the dense path. Diagnostic
                    // only for the convergence test — it does NOT change the QP step.
                    if let Ok(spectrum) = whitened_spectrum::WhitenedHessianSpectrum::decompose(
                        &lhs,
                        &rhs_step,
                        &joint_trust_metric_diag,
                        KKT_REFUSAL_RANK_TOL,
                    ) {
                        joint_spectrum = Some(spectrum);
                    }
                    let lhs_reflected = symmetric_negative_curvature_reflected(&lhs);
                    if cycle <= 2 {
                        let min_eval_raw = symmetric_min_eigenvalue_signed(&lhs);
                        let min_eval_refl = symmetric_min_eigenvalue_signed(&lhs_reflected);
                        log::info!(
                            "[JN-REFLECT-DIAG #1040] cycle={cycle} CONSTRAINED_QP lambda_min_signed_raw={min_eval_raw:.3e} lambda_min_signed_reflected={min_eval_refl:.3e} (reflection {})",
                            if min_eval_refl > min_eval_raw + min_eval_raw.abs() * 1e-9 {
                                "CHANGED the spectrum"
                            } else {
                                "NO-OP (already PSD)"
                            },
                        );
                    }
                    let lhs = lhs_reflected;
                    let rhs_beta = &lhs.dot(&beta_joint) + &rhs_step;
                    let solve_result = if let Some(bounds) = lower_bounds.as_ref() {
                        solve_quadratic_with_simple_lower_bounds(
                            &lhs,
                            &rhs_beta,
                            &beta_joint,
                            bounds,
                            warm_joint_active.as_deref(),
                        )
                    } else {
                        solve_quadratic_with_linear_constraints(
                            &lhs,
                            &rhs_beta,
                            &beta_joint,
                            constraints,
                            warm_joint_active.as_deref(),
                        )
                        .map_err(|e| e.to_string())
                    };
                    match solve_result {
                        Ok((beta_new, active_set)) => (beta_new, Some(active_set), 0usize),
                        Err(_) => break,
                    }
                } else {
                    // Stationarity residual: r = S*beta - gradient (for penalized NLL)
                    let penalty_beta = apply_joint_block_penalty(
                        &ranges,
                        &s_lambdas,
                        &beta_joint,
                        joint_mode_diagonal_ridge,
                        joint_bundle,
                    );
                    let mut rhs = &grad_joint - &penalty_beta;
                    // Universal robustness: fold the family-general
                    // Jeffreys/Firth curvature `H_Φ` and score `∇Φ` into BOTH the
                    // matrix-free PCG step AND the dense spectral fallback below,
                    // scoped to the full-span basis `Z_J`. Computed ONCE here
                    // so the matvec closure and the RHS share the SAME term and the
                    // fallback does not recompute it. The inner objective is
                    // `−ℓ + ½βᵀSβ − Φ`, so the Newton system the step must solve is
                    //   (H + S_λ + H_Φ) δ = (∇ℓ − S_λβ) + ∇Φ.
                    // Previously the PCG matvec applied only `H + S_λ` and its RHS
                    // omitted `∇Φ`, so on the matrix-free path (large p / large n)
                    // Firth was a SILENT NO-OP: the proper-prior never reached the
                    // step that actually moves β, leaving separation/under-
                    // identification uncured exactly where the dense route is not
                    // taken. The dense route (small p, e.g. BMS p≈51) was already
                    // correct. `H_Φ` is the full-span Gauss-Newton surrogate
                    // `½ J H_id⁻¹ Jᵀ` (Z_J = identity ⇒ p×p, not low-rank), but the
                    // conditioning gate in `joint_jeffreys_term` returns the zero
                    // term on every well-conditioned fit, so this only arms on the
                    // near-separating span
                    // — and `hphi` is materialized once per cycle regardless, so the
                    // matvec adds only one O(p²) HVP, preserving the matrix-free
                    // path's asymptotics where Firth is negligible (term = `None`).
                    // Cheap pre-check certified well-conditioned ⇒ the exact term
                    // is the zero contribution (∇Φ = 0, H_Φ = 0). Short-circuit to
                    // `None` WITHOUT materializing the dense joint Hessian or running
                    // the O(p³) reduced eigendecomposition — this is the matrix-free
                    // PCG hot path, where forming a dense p×p H_Φ every cycle was the
                    // regression. Byte-identical to the gated-off dense path: `rhs`
                    // is left as `∇ℓ − S_λβ` and no H_Φ is folded into the matvec.
                    // Reuse the head-β Jeffreys triple (computed once this cycle);
                    // this Newton step is built at the same cycle-start β.
                    let inner_jeffreys_term: Option<(Array1<f64>, Array2<f64>)> =
                        match head_jeffreys_term.as_ref() {
                            Some((grad_phi, hphi)) if grad_phi.len() == rhs.len() => {
                                rhs += grad_phi;
                                Some((grad_phi.clone(), hphi.clone()))
                            }
                            _ => None,
                        };
                    // PSD PROJECTION for the SPD-PCG matvec (gam#979): the exact
                    // divided-difference H_Φ can be indefinite at off-mode trial
                    // points, which breaks the SPD-CG contract. The matvec uses its
                    // PSD part; the dense spectral fallback below keeps the EXACT
                    // (possibly indefinite) H_Φ — the Moré–Sorensen subproblem
                    // handles it rigorously.
                    let inner_jeffreys_hphi: Option<Arc<Array2<f64>>> = inner_jeffreys_term
                        .as_ref()
                        .map(|(_grad_phi, hphi)| Arc::new(symmetric_psd_projection(hphi)));
                    let pcg_started = std::time::Instant::now();
                    let pcg_requested = matrix_free_joint_requested && !joint_hessian_is_dense;
                    let mut spectral_nullity_for_step = 0usize;
                    let mut delta = if pcg_requested {
                        let preconditioner_diag = match &joint_hessian_source {
                            JointHessianSource::Dense(h_joint) => {
                                joint_penalty_preconditioner_diag(
                                    &h_joint.diag().to_owned(),
                                    &ranges,
                                    &s_lambdas,
                                    joint_solver_diagonal_ridge,
                                    joint_bundle,
                                )
                            }
                            JointHessianSource::Operator { diagonal, .. } => {
                                joint_penalty_preconditioner_diag(
                                    diagonal,
                                    &ranges,
                                    &s_lambdas,
                                    joint_solver_diagonal_ridge,
                                    joint_bundle,
                                )
                            }
                        };
                        // Pre-allocate the penalty workspace ONCE outside the
                        // PCG closure so each CG iter (called hundreds-to-
                        // thousands of times per outer iter at large scale)
                        // reuses the buffer instead of allocating per call.
                        // RefCell because solve_spd_pcg* expects `Fn` (immutable
                        // borrow of captures) and we need interior mutability
                        // to write into the workspace.
                        let penalty_workspace = RefCell::new(Array1::<f64>::zeros(total_p));
                        // Capture the Jeffreys/Firth curvature for the matvec. When
                        // armed (and nonzero past the conditioning gate) the PCG
                        // operator becomes `H + S_λ + H_Φ`, matching the augmented
                        // RHS `(∇ℓ − S_λβ) + ∇Φ` set above and the dense spectral
                        // fallback. `None` keeps the unaugmented matvec.
                        let pcg_hphi_dense = inner_jeffreys_hphi.clone();
                        let pcg_hphi_op = inner_jeffreys_hphi.clone();
                        match &joint_hessian_source {
                            JointHessianSource::Dense(h_joint) => {
                                crate::linalg::utils::solve_spd_pcg_with_info_into(
                                    |v, out| {
                                        // h_joint * v -> out (faer-backed, no alloc)
                                        crate::faer_ndarray::fast_av_view_into(
                                            h_joint,
                                            v,
                                            out.view_mut(),
                                        );
                                        let mut pen = penalty_workspace.borrow_mut();
                                        apply_joint_block_penalty_into(
                                            &ranges,
                                            &s_lambdas,
                                            v,
                                            joint_solver_diagonal_ridge,
                                            &mut pen,
                                            joint_bundle,
                                        );
                                        *out += &*pen;
                                        if let Some(hphi) = pcg_hphi_dense.as_ref() {
                                            *out += &hphi.dot(v);
                                        }
                                    },
                                    &rhs,
                                    &preconditioner_diag,
                                    pcg_rel_tol,
                                    JOINT_PCG_MAX_ITER_MULTIPLIER * total_p.max(1),
                                )
                                .map(|(solution, info)| {
                                    log_joint_pcg_diagnostics(
                                        cycle,
                                        total_p,
                                        total_joint_n,
                                        &preconditioner_diag,
                                        &info,
                                    );
                                    solution
                                })
                            }
                            JointHessianSource::Operator { apply_into, .. } => {
                                let apply_h_into = Arc::clone(apply_into);
                                crate::linalg::utils::solve_spd_pcg_with_info_into(
                                    |v, out| {
                                        if let Err(error) = apply_h_into(v, out) {
                                            log::warn!(
                                                "joint Newton inner operator matvec failed: {error}"
                                            );
                                            out.fill(0.0);
                                        }
                                        let mut pen = penalty_workspace.borrow_mut();
                                        apply_joint_block_penalty_into(
                                            &ranges,
                                            &s_lambdas,
                                            v,
                                            joint_solver_diagonal_ridge,
                                            &mut pen,
                                            joint_bundle,
                                        );
                                        *out += &*pen;
                                        if let Some(hphi) = pcg_hphi_op.as_ref() {
                                            *out += &hphi.dot(v);
                                        }
                                    },
                                    &rhs,
                                    &preconditioner_diag,
                                    pcg_rel_tol,
                                    JOINT_PCG_MAX_ITER_MULTIPLIER * total_p.max(1),
                                )
                                .map(|(solution, info)| {
                                    log_joint_pcg_diagnostics(
                                        cycle,
                                        total_p,
                                        total_joint_n,
                                        &preconditioner_diag,
                                        &info,
                                    );
                                    solution
                                })
                            }
                        }
                    } else {
                        None
                    };
                    if pcg_requested {
                        log::info!(
                            "[PIRLS/joint-PCG] cycle {:>3} | n={} p={} solved={} elapsed={:.3}s",
                            cycle,
                            total_joint_n,
                            total_p,
                            delta.is_some(),
                            pcg_started.elapsed().as_secs_f64()
                        );
                    }
                    if delta.is_none() {
                        if pcg_requested {
                            break;
                        }
                        let mut lhs_true = match materialize_joint_hessian_source(
                            &joint_hessian_source,
                            total_p,
                            "joint Newton inner dense fallback Hessian materialization",
                        ) {
                            Ok(matrix) => matrix,
                            Err(_) => break,
                        };
                        // Capture the unpenalized dense `H` for the rest of this
                        // cycle (gam#1040): the Cauchy leg and trust-region
                        // predicted-reduction matvecs below can then reuse it as a
                        // cheap `O(p²)` dense matvec instead of re-applying a
                        // matrix-free operator `O(n·p)` per attempt. Only when the
                        // source is an `Operator` — a `Dense` source already gives
                        // those matvecs the fast path, so cloning would be waste.
                        if matches!(&joint_hessian_source, JointHessianSource::Operator { .. }) {
                            materialized_dense_unpenalized = Some(lhs_true.clone());
                        }
                        // Snapshot the Jeffreys information matrix only when a
                        // family supplies the contracted completion. The generic
                        // pairwise fallback costs p(p+1)/2 full second-directional
                        // Hessian passes; at biobank scale (BMS p=35, n≈196k) it
                        // turns a near-converged polishing cycle into ~50s of row
                        // work. Without a contracted hook the divided-difference
                        // H_phi model remains first-order correct and the KKT
                        // certificate owns convergence.
                        let jeffreys_completion_requested =
                            family.joint_jeffreys_information_contracted_trace_hessian_available();
                        let h_info_for_completion = (jeffreys_completion_endgame
                            && inner_jeffreys_term.is_some()
                            && jeffreys_completion_requested)
                            .then(|| family.joint_jeffreys_information_with_specs(&states, specs))
                            .transpose()?
                            .flatten();
                        add_joint_penalty_to_matrix(
                            &mut lhs_true,
                            &ranges,
                            &s_lambdas,
                            joint_mode_diagonal_ridge,
                            joint_bundle,
                        );
                        // Universal robustness: add the
                        // family-general Jeffreys curvature `H_Phi` to the
                        // penalized Hessian. This is the Tier-B coupled-Newton form
                        // of Firth: the reduced Fisher information `Z_J^T H Z_J`
                        // supplies the missing O(n) curvature that bounds a
                        // near-separating coefficient to O(1). When the Jeffreys
                        // term is unavailable, the step stays unaugmented.
                        //
                        // `∇Φ` is NOT re-added here: `rhs` (and thus `spectral_rhs`)
                        // already carries `+∇Φ` from the single shared computation
                        // above, and we REUSE that same `H_Φ` here rather than
                        // recomputing the (O(p) directional-derivative) term — the
                        // dense fallback and the matrix-free PCG step now solve the
                        // SAME Jeffreys-augmented Newton system.
                        let spectral_rhs = rhs.clone();
                        if let Some((_grad_phi, hphi)) = inner_jeffreys_term.as_ref() {
                            lhs_true += hphi;
                            // ENDGAME EXACTNESS (gam#979). The divided-difference
                            // H_Φ omits the second-directional-Hessian remainder
                            // `½ tr(K · D_ab)`; near a Firth-active mode that
                            // remainder is comparable to the kept curvature, so
                            // Newton converges only linearly (a residual sawtooth
                            // plateauing just above the certificate tolerance —
                            // enough mode noise to swamp outer finite differences
                            // and feed the IFT near-flat-kernel amplification).
                            // Once the residual enters the convergence band, add
                            // the exact completion so the model is the true
                            // Hessian of the Φ-augmented objective and the endgame
                            // is quadratic. A family contracted trace hook can
                            // supply it at any width; the pairwise `p(p+1)/2`
                            // fallback remains limited to moderate p. `None`
                            // degrades safely to the divided-difference model.
                            if let (Some(h_info), Some(z_joint)) = (
                                h_info_for_completion.as_ref(),
                                joint_jeffreys_subspace.as_ref(),
                            ) && let Some(completion) =
                                custom_family_joint_jeffreys_second_order_completion(
                                    family, &states, specs, h_info, z_joint, false,
                                )?
                            {
                                lhs_true += &completion;
                            }
                        }
                        // Single metric-whitened eigendecomposition drives BOTH the
                        // seed step and every trust-region re-solve this cycle
                        // (gam#979). The prior code ran a SECOND O(p³)
                        // eigendecomposition of the raw Hessian here purely to form
                        // the seed step — doubling the dominant per-cycle cost on the
                        // ~5 s/cycle ill-conditioned survival marginal-slope inner.
                        // The exact trust-region multiplier λ (chosen so ‖δ‖_D = r)
                        // subsumes the old self-vanishing Levenberg-μ seed: `decompose`
                        // whitens by the trust metric so the penalty (λ~e²⁴) and the
                        // likelihood scales are throttled uniformly — the scale
                        // invariance the multiplicative μ approximated. `lhs_true`
                        // already carries the penalty and the Firth/Jeffreys curvature
                        // H_Φ and `spectral_rhs` the augmented stationarity RHS, so the
                        // subproblem model matches the predicted-reduction model and the
                        // accept/reject gain ratio exactly.
                        let spectrum = whitened_spectrum::WhitenedHessianSpectrum::decompose(
                            &lhs_true,
                            &spectral_rhs,
                            &joint_trust_metric_diag,
                            KKT_REFUSAL_RANK_TOL,
                        )?;
                        // Seed = the unconstrained (Moore–Penrose, range-restricted)
                        // exact step, so cycle 0 can take the full Newton step on a
                        // well-conditioned model (the cycle-0 radius bump below relies
                        // on this); the trust loop re-solves at finite radius for every
                        // subsequent attempt. An indefinite model reflects negative
                        // curvature to |λ|, exactly as the prior spectral solve did.
                        let spectral_step = spectrum.trust_region_step(f64::INFINITY);
                        spectral_nullity_for_step = spectral_step.nullity;
                        if spectral_step.reflected_negative_modes > 0 {
                            log::info!(
                                "[PIRLS/joint-Newton] cycle {cycle:>3} | indefinite inner \
                                 Hessian: reflected {}/{} negative-curvature modes to |λ| \
                                 (λ_min={:.3e}); proceeding with modified-Newton descent step \
                                 under trust-region globalization",
                                spectral_step.reflected_negative_modes,
                                total_p,
                                spectral_step.most_negative_eigenvalue,
                            );
                        }
                        if spectral_step.nullity > 0 {
                            log::debug!(
                                "[PIRLS/joint-Newton] spectral reduced solve: nullity@{:.0e}={}/{} \
                             |P0 rhs|∞={:.3e} |P+ rhs|∞={:.3e} λ_min+={:.3e} λ_max={:.3e}",
                                spectral_step.rank_tol,
                                spectral_step.nullity,
                                total_p,
                                spectral_step.null_rhs_inf,
                                spectral_step.range_rhs_inf,
                                spectral_step.lambda_min_positive,
                                spectral_step.lambda_max_abs,
                            );
                        }
                        delta = Some(spectral_step.delta);
                        // The same factorization powers every trust-radius re-solve
                        // in the loop below (gam#979) — no second eigendecomposition.
                        joint_spectrum = Some(spectrum);
                    }

                    let Some(delta) = delta else {
                        break; // Fall back to blockwise
                    };
                    if !delta.iter().all(|v| v.is_finite()) {
                        break; // Fall back to blockwise
                    }
                    (beta_joint.clone() + &delta, None, spectral_nullity_for_step)
                };
            // Hessian-source build (and any QP solve immediately above) are
            // done by the time we reach `delta`. Capture the wall-clock
            // before the line-search phase so the end-of-cycle summary can
            // attribute time correctly between the Hessian/QP and the
            // backtracking step search.
            let hessian_and_qp_elapsed = hessian_started.elapsed();
            drop(hessian_scope_guard);
            let line_search_started = std::time::Instant::now();
            log::info!(
                "[joint-newton-tr] phase=line_search cycle={} r={:.3e} hessian_qp_elapsed={:.3}s",
                cycle,
                joint_trust_radius,
                hessian_and_qp_elapsed.as_secs_f64(),
            );
            let delta = &candidate_beta - &beta_joint;
            // Effective Hessian source for the remaining per-cycle matvecs
            // (Cauchy leg + trust-region predicted reduction). When the dense
            // fallback above materialized a matrix-free `Operator` to dense, route
            // those matvecs through that `Dense` snapshot so each is an `O(p²)`
            // GEMV rather than an `O(n·p)` operator row-sweep repeated up to
            // `JOINT_TRUST_MAX_ATTEMPTS` times (gam#1040). Byte-identical action
            // (the dense build IS the operator's action by construction); falls
            // back to the original source when no dense snapshot was taken (the
            // already-`Dense` and PCG paths).
            let dense_snapshot_source =
                materialized_dense_unpenalized.map(JointHessianSource::Dense);
            let effective_hessian_source: &JointHessianSource = dense_snapshot_source
                .as_ref()
                .unwrap_or(&joint_hessian_source);

            // Trust-region globalization for the joint Newton proposal.  The
            // previous implementation used up to eight backtracking likelihood
            // evaluations (each can build the exact joint workspace at large-scale
            // scale).  Here the step is truncated before evaluation and the
            // single trial objective is accepted only when the actual decrease
            // is positive relative to the local quadratic model.
            let step_inf = delta.iter().copied().map(f64::abs).fold(0.0_f64, f64::max);

            let old_beta: Vec<Array1<f64>> = states.iter().map(|s| s.beta.clone()).collect();
            // Firth value Φ at the OLD (start-of-cycle) β, folded under the SAME
            // skippable gate the trial uses below — so `actual_reduction =
            // old_objective − trialobjective` compares two points on one objective
            // `−ℓ + ½βᵀSβ − Φ` (gam#826/#872). `lastobjective` is the pure
            // quadratic-penalized objective; subtract the gated old-β Φ here.
            let old_phi = if !jeffreys_skippable_this_cycle {
                joint_jeffreys_subspace
                    .as_ref()
                    .map(|z_joint| {
                        custom_family_joint_jeffreys_value(family, &states, specs, &ranges, z_joint)
                    })
                    .unwrap_or(0.0)
            } else {
                0.0
            };
            let old_objective = lastobjective - old_phi;
            // Row measure observed by the objective at β. `lastobjective` was
            // set on the previous cycle (or at function entry) under `options`;
            // see top-of-cycle capture for rationale.
            let tr_row_measure_old_objective =
                crate::solver::row_measure::RowSubsampleMask::from_options(options, total_joint_n);
            let mut accepted = false;
            let mut accepted_joint_workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>> =
                None;
            let mut line_search_attempts = 0usize;

            // Pure Newton must take a full step on the first cycle of an
            // exact quadratic problem (i.e. converge in one cycle when the
            // model is exact). The trust-region globalization above must not
            // truncate the very first proposal merely because the hard-coded
            // initial radius (1.0) is smaller than the natural Newton-step norm.
            //
            // There are two norms in play:
            //   * the constrained-QP / dogleg paths truncate per block against
            //     `joint_block_trust_radii`;
            //   * the exact spectral trust-region path solves one global
            //     Moré–Sorensen problem against `joint_trust_radius`.
            //
            // The old cycle-0 bump only raised the per-block radii and then set the
            // global radius to `max(block_norms)`. For a multiblock exact quadratic
            // with a diagonal metric that leaves a full Newton step like
            // `[0.8, 0.8]` inside every per-block ball (`max = 0.8`) but outside
            // the global spectral ball (`sqrt(0.8² + 0.8²) = 1.13`). Once the
            // constrained branch started populating `joint_spectrum` for the
            // Newton-decrement certificate, the line search correctly used the
            // spectral path and incorrectly clipped that exact feasible Newton
            // step to radius 1.0, preventing one-cycle KKT convergence. Bump the
            // global radius to the full metric norm while still bumping each block
            // radius to its own block norm; this keeps the first exact Newton step
            // untruncated in both globalization modes and leaves the standard
            // adaptive shrink/expand for subsequent cycles.
            if cycle == 0 && joint_step_spectral_nullity == 0 {
                let initial_global_norm =
                    joint_trust_region_metric_step_norm(&delta, &joint_trust_metric_diag);
                let initial_block_norms = joint_trust_region_block_metric_norms(
                    &delta,
                    &ranges,
                    &joint_trust_metric_diag,
                );
                for (radius, norm) in joint_block_trust_radii.iter_mut().zip(initial_block_norms) {
                    if norm.is_finite() && norm > *radius {
                        *radius = norm;
                    }
                }
                let block_radius = joint_block_trust_radii
                    .iter()
                    .copied()
                    .fold(0.0_f64, f64::max);
                joint_trust_radius = if initial_global_norm.is_finite() {
                    block_radius.max(initial_global_norm)
                } else {
                    block_radius
                };
                if !joint_trust_radius.is_finite() || joint_trust_radius <= 0.0 {
                    joint_trust_radius = 1.0;
                }
            }

            let penalty_beta = apply_joint_block_penalty(
                &ranges,
                &s_lambdas,
                &beta_joint,
                joint_mode_diagonal_ridge,
                joint_bundle,
            );
            // Stationarity RHS for the trust-region quadratic model. When the
            // Jeffreys/Firth term is armed the inner objective is `−ℓ+½βᵀSβ+Φ`, so
            // the model RHS is `∇L − Sβ + ∇Φ` — the SAME augmented RHS the Newton
            // step solves and the H_Φ-augmented `hpen_delta` below pairs with. Using
            // the bare `∇L − Sβ` here desyncs `predicted_reduction` from the
            // augmented step + the Φ-augmented `actual_reduction`, which is what
            // froze the coupled K-block line search (gam#729/#715). No-op when the
            // term is condition-gated/unavailable (∇Φ=0).
            let mut rhs = &grad_joint - &penalty_beta;
            if let Some((grad_phi, _hphi)) = head_jeffreys_term.as_ref()
                && grad_phi.len() == rhs.len()
            {
                rhs += grad_phi;
            }
            let beta_inf = states
                .iter()
                .flat_map(|s| s.beta.iter().copied())
                .map(f64::abs)
                .fold(0.0_f64, f64::max);
            let step_tol = inner_tol * (1.0 + beta_inf);
            let objective_tol = inner_tol * (1.0 + old_objective.abs());
            // Scale the KKT residual tolerance against the natural magnitude
            // of ‖Sβ − ∇L‖∞ (i.e. max(‖∇L‖∞, ‖Sβ‖∞)), not the objective. The
            // gradient and Sβ scale independently of the likelihood — at
            // large scale with |β|∞ ~ 10²–10³ and non-trivial smoothing,
            // ‖Sβ‖∞ can sit orders of magnitude above |obj| and FP noise
            // alone keeps the residual above any obj-scaled tol, so KKT is
            // never certified even when the iterate is the true optimum.
            let grad_inf = grad_joint
                .iter()
                .map(|x: &f64| x.abs())
                .fold(0.0_f64, f64::max);
            let penalty_inf = penalty_beta
                .iter()
                .map(|x: &f64| x.abs())
                .fold(0.0_f64, f64::max);
            let residual_tol = inner_tol * (1.0 + grad_inf.max(penalty_inf));
            last_residual_tol = residual_tol;
            let current_stationarity_residual = current_kkt_norm;
            // KKT certificate: ‖∇L − Sβ‖_∞ ≤ residual_tol together with
            // ‖δ‖_∞ ≤ step_tol is sufficient first-order optimality of the
            // penalized objective; no descent direction exists from the
            // current point. Conditioning that exit on additional evidence
            // of objective progress in the previous cycle would refuse to
            // recognize convergence at a starting point that already sits
            // at the optimum (e.g. balanced data with an intercept-only
            // fit, where ∇ℓ vanishes by symmetry from cycle 0 and the
            // Newton step is identically zero so the trust-region search
            // can never produce a strictly negative actual reduction).
            if current_stationarity_residual <= residual_tol && step_inf <= step_tol {
                log::info!(
                    "[PIRLS/joint-Newton convergence] cycle {:>3} | pre-line-search converged: proposal_inf={:.3e} (tol={:.3e}) | residual={:.3e} (tol={:.3e})",
                    cycle,
                    step_inf,
                    step_tol,
                    current_stationarity_residual,
                    residual_tol,
                );
                // Pre-line-search convergence: β did not move this cycle (the
                // proposal was at the step-tolerance floor), so the cycle
                // workspace is still at the converged β and the post-loop
                // covariance/IFT assembly can reuse it instead of rebuilding the
                // full per-row kernel cache at the same β.
                cached_joint_workspace = hessian_workspace_for_cycle.take();
                cycles_done = cycle;
                converged = true;
                break;
            }

            // Trust-region retries preserve the objective-decrease guarantee
            // when the initial radius is too optimistic. If the Newton proposal
            // is not a descent direction for the penalized quadratic model,
            // switch once to a diagonally preconditioned gradient step and keep
            // the same exact full-objective accept/reject test.
            const JOINT_TRUST_MAX_ATTEMPTS: usize = 24;
            let mut search_delta = delta.clone();
            let search_joint_active_set: Option<Vec<usize>> = joint_active_set.clone();
            let mut tried_preconditioned_descent = false;
            // Dogleg Cauchy leg (gam#826/#808). Compute the unconstrained Cauchy
            // point of the penalized (Firth-augmented) quadratic model ONCE per
            // cycle: the M-metric steepest-descent direction `p_sd = M⁻¹·rhs`
            // and its curvature `p_sd·H·p_sd` (a coupled Hessian-vector product,
            // so it must be hoisted out of the radius-shrink loop). When the
            // Newton step exceeds a block's trust radius the dogleg blends
            // toward this Cauchy leg, guaranteeing at least the Cauchy decrease
            // even when the spectral Newton step is numerically frozen at the
            // oversmoothed seed (the high-curvature log_sigma block's Newton
            // component is `O(g/λ) ≈ 5e-21`). `joint_active_set` is the
            // unconstrained joint Newton path; the constrained-QP path keeps its
            // own globalization, so the dogleg is only built (and used) when no
            // active set is in force.
            // Only the dogleg/box-truncation globalization (no spectrum) ever
            // consumes the Cauchy leg; when the exact Moré–Sorensen spectrum is
            // present the trust loop re-solves from it and `dogleg_cauchy` is dead.
            // Skipping its construction there removes one coupled Hessian-vector
            // product per cycle — an `O(n·p)` operator row-sweep on the matrix-free
            // DENSE_SPECTRAL path that produced no value (gam#1040).
            let dogleg_cauchy: Option<Array1<f64>> =
                if search_joint_active_set.is_none() && joint_spectrum.is_none() {
                    let mut p_sd = Array1::<f64>::zeros(total_p);
                    for (i, (r, w)) in rhs.iter().zip(joint_trust_metric_diag.iter()).enumerate() {
                        p_sd[i] = r / positive_joint_diagonal_entry(*w);
                    }
                    let mut h_psd = Array1::<f64>::zeros(total_p);
                    let mut cauchy_penalty_scratch = Array1::<f64>::zeros(total_p);
                    match apply_joint_penalized_hessian_into_with_workspace(
                        effective_hessian_source,
                        &ranges,
                        &s_lambdas,
                        joint_mode_diagonal_ridge,
                        &p_sd,
                        &mut h_psd,
                        &mut cauchy_penalty_scratch,
                        joint_bundle,
                    ) {
                        Ok(()) => {
                            if let Some((_grad_phi, hphi)) = head_jeffreys_term.as_ref() {
                                h_psd += &hphi.dot(&p_sd);
                            }
                            let cauchy = joint_cauchy_step(&rhs, &p_sd, &h_psd);
                            if cauchy.iter().all(|v| v.is_finite()) {
                                Some(cauchy)
                            } else {
                                None
                            }
                        }
                        Err(_) => None,
                    }
                } else {
                    None
                };
            let mut model_rejects = 0usize;
            let mut likelihood_rejects = 0usize;
            let mut objective_rejects = 0usize;
            // Feasibility-path rejections (gam#979 survival monotone cone). The two
            // constrained-path `continue`s — the `apply_joint_feasibility_limit`
            // α-crush `Err` (current iterate infeasible / no positive step) and the
            // `project_point_strictly_into_feasible_cone` `None` (degenerate /
            // empty-interior cone at this trial) — consume a trust attempt but were
            // NOT counted by any of model/likelihood/objective. On the survival
            // marginal-slope monotone-cone pathology this is the DOMINANT reject
            // path (the trial step keeps crossing the binding time-derivative cone
            // at slack≈0), so `model + likelihood + objective < MAX_ATTEMPTS`
            // ALWAYS, `all_attempts_rejected` was permanently false, and the
            // fully-rejected stall guard below NEVER armed — the inner joint-Newton
            // spun to `inner_loop_hard_ceiling` every outer ρ-evaluation (the 1322 s
            // hang; #1040). A feasibility rejection IS a "no descent the local model
            // can reconcile at this β" signal exactly like an objective rejection,
            // so counting it restores the partition invariant
            // `model + likelihood + objective + feasibility == MAX_ATTEMPTS` the
            // stall guard relies on. Off the constrained pathology this counter
            // stays 0 (those `continue`s are never taken on a feasible/unconstrained
            // arm), so every converging fit is byte-identical.
            let mut feasibility_rejects = 0usize;
            let mut first_likelihood_reject: Option<String> = None;
            // Fixed-point signature for the fully-rejected stall guard. The
            // FIRST trust attempt of a cycle evaluates the proposal at the
            // pre-cycle β (the attempt that has not yet shrunk the radius), so
            // its trial objective is a deterministic function of (β, S, λ)
            // alone. On a fully-rejected cycle β is reverted to that same
            // pre-cycle value, so if two consecutive fully-rejected cycles
            // report a *byte-identical* first-attempt trial objective the next
            // cycle's entire Newton system, trust-region search, and reject
            // outcome are provably byte-identical — the iterate is at an exact
            // fixed point and every further cycle is a pure no-op. This is
            // strictly stronger evidence than the `radius_held` heuristic (which
            // can keep resetting while the boundary block's radius oscillates),
            // so it lets the guard fire in two cycles instead of grinding to the
            // 8-cycle held-radius count or the hard ceiling. Captured only on
            // the first attempt; a successful first-attempt likelihood-reject
            // leaves it `None` (and the early-exit reject path captures it
            // explicitly below) so a non-finite trial cannot masquerade as a
            // fixed point.
            let mut first_attempt_trial_objective: Option<f64> = None;
            // Frozen-step line-search short-circuit (n≈3e5 marginal-slope floor
            // stall). Once the joint trust radius is pinned (the shrink rule
            // clamps every block radius at the `1e-12` floor, so a reject can no
            // longer reduce it), the Moré–Sorensen / dogleg step the next attempt
            // builds is a deterministic function of the unchanged radii and the
            // reverted β — byte-identical to this attempt, hence the same trial
            // objective. The floor-stalled cycle therefore logged
            // `JOINT_TRUST_MAX_ATTEMPTS` identical `reject_floor` lines, each a
            // redundant full-data (320k-row) line-search sweep (~0.5 s apiece),
            // every cycle until the cross-cycle stall guard fired — pure waste on
            // the dominant cost of the inner solve. Track the previous rejected
            // attempt's trial objective; when the radius is held AND the current
            // rejected trial reproduces it bit-for-bit the step is provably frozen
            // and the remaining attempts are no-ops, so stop. `frozen_floor_full_reject`
            // records that the cycle was nonetheless fully rejected, preserving the
            // `all_attempts_rejected` partition the cross-cycle stall guard relies
            // on; `first_attempt_trial_objective` is still captured on attempt 0,
            // so the byte-identical cross-cycle detector is unaffected and the
            // converged/non-converged decision is byte-identical to exhausting the
            // loop.
            let mut prev_rejected_attempt_objective: Option<f64> = None;
            let mut frozen_floor_full_reject = false;
            // Coalesce consecutive trust-region attempts whose accept/reject
            // outcome and numeric signature round to the same values, so a long
            // run of identical retries collapses into a single "attempts a..b
            // (×N)" line at flush time instead of spamming one line per try.
            let mut tr_log_sig: Option<String> = None;
            let mut tr_log_first: usize = 0;
            let mut tr_log_last: usize = 0;
            // Hoist the two full-size scratch buffers used in the predicted-
            // reduction computation outside the trust-region attempt loop.
            // The loop runs up to JOINT_TRUST_MAX_ATTEMPTS times per outer
            // Newton step, so allocating these per-attempt would add O(total_p)
            // heap traffic on every radius shrink/expand iteration.
            let mut hpen_delta = Array1::<f64>::zeros(total_p);
            let mut tr_penalty_scratch = Array1::<f64>::zeros(total_p);
            for trust_attempt in 0..JOINT_TRUST_MAX_ATTEMPTS {
                line_search_attempts = trust_attempt + 1;
                accepted_joint_workspace = None;
                // Dogleg globalization (gam#826/#808): when the unconstrained
                // Newton path is in force and a finite Cauchy leg was built,
                // construct the dogleg blend of the Cauchy and Newton points at
                // the current per-block radii. Otherwise (constrained-QP path,
                // or after the preconditioned-descent fallback replaced
                // `search_delta`) fall back to box-truncating the search step.
                let mut trial_delta;
                // gam#979: set when the constrained-QP candidate is taken
                // untruncated because the global α-crush would otherwise collapse
                // a feasible, within-trust QP step (see the gated bypass below).
                let mut qp_feasible_bypass = false;
                let mut block_step_norms = if let Some(spectrum) = joint_spectrum.as_ref() {
                    // Exact Moré–Sorensen trust-region step at the current radius
                    // (gam#979). The step already lies in the `D`-metric ball, so
                    // no dogleg blend or box-truncation is applied: on a shrink the
                    // direction is RE-SOLVED (bending toward the gradient), the
                    // property the dogleg/truncation lacked. Re-solving reuses the
                    // cached factorization at O(p) cost. On the constrained path the
                    // resulting (unconstrained) step is projected back onto the cone
                    // just below (gam#1108), preserving this step's fast convergence
                    // while keeping every accepted iterate feasible.
                    //
                    // If the already-computed Newton/QP step lies inside the
                    // current global trust ball, take it directly instead of asking
                    // the trust-region solver to recover the boundary solution at
                    // `r == ‖δ_N‖`. The boundary multiplier is mathematically zero
                    // in that case, but finite precision can produce a tiny positive
                    // multiplier and perturb an exact quadratic one-step solve by
                    // O(1e-6), which is large relative to the inner KKT floor. The
                    // direct step is the exact unconstrained minimizer of the local
                    // model and is still trust-region feasible.
                    let search_norm = joint_trust_region_metric_step_norm(
                        &search_delta,
                        &joint_trust_metric_diag,
                    );
                    if search_norm.is_finite()
                        && joint_trust_radius.is_finite()
                        && search_norm <= joint_trust_radius * (1.0 + 1e-12)
                    {
                        trial_delta = search_delta.clone();
                    } else {
                        trial_delta = spectrum.trust_region_step(joint_trust_radius).delta;
                    }
                    joint_trust_region_block_metric_norms(
                        &trial_delta,
                        &ranges,
                        &joint_trust_metric_diag,
                    )
                } else if let Some(cauchy) = dogleg_cauchy.as_ref()
                    && !tried_preconditioned_descent
                {
                    trial_delta = Array1::<f64>::zeros(total_p);
                    joint_dogleg_step_to_block_metric_radii(
                        &search_delta,
                        cauchy,
                        &ranges,
                        &joint_trust_metric_diag,
                        &joint_block_trust_radii,
                        &mut trial_delta,
                    )
                } else {
                    // Box-truncation branch — taken on the CONSTRAINED-QP path
                    // (search_joint_active_set is Some, so no spectrum / dogleg).
                    // `search_delta` is the active-set QP's Newton step, FEASIBLE
                    // by construction.
                    //
                    // gam#979 GATED QP-FEASIBILITY BYPASS. The default behaviour
                    // box-truncates this feasible QP step per-block (which can push
                    // it off the monotone cone face) and then the global
                    // fraction-to-boundary α-crush scales the whole joint step by a
                    // single α; on a binding monotonicity row at slack≈0 that α
                    // collapses to ~1e-4, freezing β and hanging the inner solve.
                    //
                    // Bypass that ONLY when the observable pathology is present —
                    // never on a healthy step — so every currently-converging arm
                    // (binary BMS score-warp monotonicity especially) is byte-
                    // identical. The three gate conditions:
                    //   (i)   the candidate came from the constrained QP
                    //         (search_joint_active_set.is_some()),
                    //   (ii)  the QP step is already within the joint trust region
                    //         (step_norm ≤ joint_trust_radius), so truncation is
                    //         not needed for globalization, AND
                    //   (iii) the α the legacy path WOULD apply is below the crush
                    //         threshold (it would gut the step).
                    // When all hold, take the untruncated QP step and let the
                    // magnitude-preserving cone projection below enforce
                    // feasibility. Otherwise truncate exactly as before.
                    let qp_norms = joint_trust_region_block_metric_norms(
                        &search_delta,
                        &ranges,
                        &joint_trust_metric_diag,
                    );
                    let qp_step_norm = qp_norms.iter().copied().fold(0.0_f64, f64::max);
                    let within_trust = qp_step_norm.is_finite()
                        && joint_trust_radius.is_finite()
                        && qp_step_norm <= joint_trust_radius;
                    let alpha_would_crush = if search_joint_active_set.is_some() && within_trust {
                        match compute_joint_feasibility_alpha(
                            family,
                            &states,
                            &ranges,
                            &search_delta,
                        ) {
                            Ok((alpha, _)) => alpha < JOINT_FEASIBILITY_ALPHA_CRUSH_THRESHOLD,
                            // Err = current iterate infeasible / no positive step;
                            // fall back to the legacy truncate + α path, which will
                            // surface the same Err and shrink the radius.
                            Err(_) => false,
                        }
                    } else {
                        false
                    };
                    if search_joint_active_set.is_some() && within_trust && alpha_would_crush {
                        qp_feasible_bypass = true;
                        trial_delta = search_delta.clone();
                        qp_norms
                    } else {
                        trial_delta = search_delta.clone();
                        truncate_joint_step_to_block_metric_radii(
                            &mut trial_delta,
                            &ranges,
                            &joint_trust_metric_diag,
                            &joint_block_trust_radii,
                        )
                    }
                };
                // FEASIBILITY ENFORCEMENT (gam#979 survival flex non-convergence).
                //
                // The global `apply_joint_feasibility_limit` enforces feasibility
                // by a single fraction-to-boundary scalar `α` applied to the WHOLE
                // joint step. On a binding monotonicity row at slack≈0 with
                // negative drift, `α` collapses to ~1e-4, globally crushing the
                // step so β moves ~1e-4/cycle while a huge time-block gradient
                // (|g|≈720) persists: the objective drifts down ~50/cycle but the
                // KKT residual never clears, the inner joint-Newton grinds the full
                // cycle budget, and the seed is rejected — the survival
                // marginal-slope hang.
                //
                // When the gated bypass above fired (`qp_feasible_bypass`), the
                // pathology is present (constrained-QP candidate, within trust, α
                // below the crush threshold): SKIP the α-crush and let the
                // magnitude-preserving cone projection below enforce feasibility,
                // which keeps the unconstrained step components and only corrects
                // the binding directions. Off the pathology the α-crush runs
                // exactly as before — a no-op when `α = 1` (healthy), the legacy
                // scaling when `α ∈ [threshold, 1)` — so every converging arm is
                // byte-identical. The α-crush `Err` (current iterate infeasible /
                // no positive step) still triggers a radius shrink + retry.
                if !qp_feasible_bypass
                    && apply_joint_feasibility_limit(family, &states, &ranges, &mut trial_delta)
                        .is_err()
                {
                    feasibility_rejects += 1;
                    joint_trust_radius = shrink_active_joint_block_trust_radii(
                        &mut joint_block_trust_radii,
                        &block_step_norms,
                        0.25,
                    );
                    continue;
                }
                // CONSTRAINED-PATH FEASIBILITY PROJECTION (gam#1108 / gam#979). The
                // trust-region trial step (Moré–Sorensen / dogleg / box-trunc) is
                // taken in the UNCONSTRAINED D-metric ball, so the step can cross
                // the monotone time-derivative cone `Aβ ≥ b`. The next cycle's
                // `check_linear_feasibility` gate would then reject the accepted
                // iterate — the interval-censored survival warm-start abort.
                // Project the trial iterate back onto the cone with the exact
                // identity-Hessian active-set projection, preserving the trust
                // step's fast convergence while guaranteeing every accepted iterate
                // is feasible. This is the feasibility mechanism for the gated
                // QP-bypass case (gam#979, where the α-crush is skipped) and the
                // safety net for any truncation-induced infeasibility on the
                // constrained path. No-op when the joint design is unconstrained or
                // the trial is already feasible (the common case — including a
                // bypassed QP step, which is feasible by construction);
                // `block_step_norms` is recomputed from the projected step just
                // below so the trust-radius bookkeeping stays consistent.
                if let Some(constraints) = joint_constraints.as_ref() {
                    let trial_beta = &beta_joint + &trial_delta;
                    if check_linear_feasibility(&trial_beta, constraints, 1e-8).is_err() {
                        match crate::solver::active_set::project_point_strictly_into_feasible_cone(
                            &trial_beta,
                            constraints,
                        ) {
                            Some(projected) => {
                                trial_delta = &projected - &beta_joint;
                            }
                            None => {
                                // Projection failed to find a strictly-interior
                                // point (degenerate / empty-interior cone at this
                                // trial). Since the global α-crush is gated off on
                                // the constrained path, preserve the old safety
                                // net here: shrink the active block trust radii and
                                // retry, exactly as the α-crush `Err` branch did
                                // (gam#979). Without this an infeasible trial would
                                // reach the next cycle's `check_linear_feasibility`
                                // QP gate and hard-error.
                                feasibility_rejects += 1;
                                joint_trust_radius = shrink_active_joint_block_trust_radii(
                                    &mut joint_block_trust_radii,
                                    &block_step_norms,
                                    0.25,
                                );
                                continue;
                            }
                        }
                    }
                }
                block_step_norms = joint_trust_region_block_metric_norms(
                    &trial_delta,
                    &ranges,
                    &joint_trust_metric_diag,
                );
                let step_norm = block_step_norms.iter().copied().fold(0.0_f64, f64::max);
                let trial_step_inf = trial_delta
                    .iter()
                    .copied()
                    .map(f64::abs)
                    .fold(0.0_f64, f64::max);
                let step_hit_trust_boundary = block_step_norms
                    .iter()
                    .zip(&joint_block_trust_radii)
                    .any(|(step_norm, radius)| {
                        joint_block_step_hit_trust_boundary(*step_norm, *radius)
                    });
                // Predicted reduction must use the TRUE penalized Hessian
                // (the one that appears in `f(β) = -ℓ + ½βᵀSβ + ½·joint_mode_diagonal_ridge·‖β‖²`),
                // NOT the SPD-stabilized version. The stabilizing shift
                // in `joint_solver_diagonal_ridge` is purely a solver-side
                // tool to make the Newton system invertible when H_NLL
                // has negative eigenvalues; it is not part of the true
                // objective the trial-likelihood evaluator computes.
                //
                // If we use `joint_solver_diagonal_ridge` here, then for
                // any Newton step lying in null(H_true) (e.g. the
                // marginal-block cancellation direction in the saturated
                // probit regime — see
                // `marginal_block_hessian_cancels_in_saturated_regime`),
                // predicted = ½·rhs·δ while actual = rhs·δ, giving ρ = 2
                // exactly. The trust-region loop then accepts the step
                // (ρ > 0.75 expands the radius), and the same regime
                // repeats every cycle — exactly the large-scale-saturated
                // failure trace. Pinned by
                // `ridge_stabilization_gap_produces_exact_rho_two_in_null_direction`.
                //
                // `hpen_delta` and `tr_penalty_scratch` are hoisted outside
                // this loop; the workspace variant reuses them without
                // allocating per attempt.
                hpen_delta.fill(0.0);
                if apply_joint_penalized_hessian_into_with_workspace(
                    effective_hessian_source,
                    &ranges,
                    &s_lambdas,
                    joint_mode_diagonal_ridge,
                    &trial_delta,
                    &mut hpen_delta,
                    &mut tr_penalty_scratch,
                    joint_bundle,
                )
                .is_err()
                {
                    break;
                }
                // JEFFREYS/FIRTH CURVATURE IN THE TRUST-REGION MODEL (gam#729/#715).
                // When the Jeffreys term is armed, the inner objective the merit
                // (`trialobjective = −ℓ + ½βᵀSβ + Φ`) measures and the Newton step
                // (`(H+Sλ+H_Φ)δ = ∇L−Sβ+∇Φ`) target both include the Firth term, so
                // the trust-region quadratic model's curvature MUST include `H_Φδ`
                // too. Omitting it (bare `(H+Sλ)δ`) makes `predicted_reduction`
                // inconsistent with the H_Φ-augmented `rhs` and the Φ-augmented
                // `actual_reduction`: for a coupled K-block family near the Firth
                // optimum (residual floored at ‖∇Φ‖) the resulting trust_ratio is
                // wrong, the line search rejects the genuine descent step (accepts
                // ~0), and β freezes with the residual stalled at a constant ≫ tol
                // — the unbounded-cycle non-convergence the inner solve exhibits on
                // the Dirichlet/multinomial fits. Adding `H_Φδ` makes the model
                // curvature match the augmented system the step solves and the
                // merit the accept test uses, so the step is accepted and the
                // residual descends. No-op when the term is condition-gated (∇Φ=0,
                // H_Φ=0) or unavailable.
                if let Some((_grad_phi, hphi)) = head_jeffreys_term.as_ref() {
                    let hphi_delta = hphi.dot(&trial_delta);
                    hpen_delta += &hphi_delta;
                }
                let predicted_reduction =
                    joint_quadratic_predicted_reduction(&rhs, &hpen_delta, &trial_delta);
                let linearized_next_kkt_inf = hpen_delta
                    .iter()
                    .zip(rhs.iter())
                    .map(|(hpen, rhs)| (hpen - rhs).abs())
                    .fold(0.0_f64, f64::max);
                // Reject only non-descent directions on the quadratic model.
                // A small-but-positive predicted reduction is what Newton
                // *should* produce near the optimum of a large-magnitude
                // objective: ½δᵀHδ scales with curvature×step², so it can be
                // far below the (relative) objective_tol = inner_tol·(1+|obj|)
                // while still being a correct Newton step. Trust-region ρ
                // shrink/expand handles small-but-valid Newton steps; the
                // preconditioned branch below is only for model-invalid
                // directions, and preserves linear constraints when present.
                //
                // NEAR-FLOOR CARVE-OUT (gam#787 binary matern centers=12). When
                // the Newton proposal is already at the step-tolerance floor —
                // `step_inf ≤ 4·step_tol`, the same round-off band the cert path
                // uses — the iterate is doing KKT polishing on a flat objective,
                // not global descent: there `predicted_reduction = rhs·δ − ½δᵀHδ`
                // is two near-equal O(step²) quantities and its SIGN is round-off
                // noise (a true Newton step gives +½δᵀHδ but the damped/range-
                // restricted spectral solve leaves rhs·δ a hair below ½δᵀHδ). The
                // `predicted_reduction ≤ 0` branch then mistook this for a model-
                // invalid direction and substituted `joint_preconditioned_descent_delta`,
                // a step sized for OBJECTIVE descent (diagonal-preconditioned
                // gradient, O(900×) larger than the polishing proposal). That step
                // bought a round-off-level objective gain but catapulted the KKT
                // residual off a near-converged iterate (‖∇L−Sβ‖ 1.7e-4 → 4.7e-1),
                // which then never recovered — every later cycle re-triggered the
                // same substitution (proposal stays pred≤0), pinning the residual
                // far above tol until the cycle budget exhausted → seed rejected →
                // hard raise. At the step floor we instead take the tiny proposal
                // as-is and let the trust-region noise-floor guard accept it at
                // ρ=1 (it neither helps nor hurts the objective beyond round-off),
                // so the inner keeps polishing the KKT residual to tol.
                let proposal_at_step_floor = joint_proposal_at_step_floor(step_inf, step_tol);
                if (!predicted_reduction.is_finite() || predicted_reduction <= 0.0)
                    && !proposal_at_step_floor
                {
                    model_rejects += 1;
                    // CONSTRAINED-PATH GUARD (#1108). The preconditioned-descent
                    // substitution replaces `search_delta` with an UNCONSTRAINED
                    // diagonally-preconditioned gradient step (`δ = M⁻¹·rhs`). That
                    // direction respects neither the active set nor the linear
                    // inequality cone `Aβ ≥ b`, and nothing downstream re-projects
                    // it: a constrained family that maintains feasibility purely
                    // through the QP (e.g. `LatentSurvivalFamily`, whose
                    // `max_feasible_step_size` is `None` and whose
                    // `post_update_block_beta` is the identity) has no barrier clip
                    // in `apply_joint_feasibility_limit` to pull the gradient step
                    // back onto the monotone time-derivative cone. The trial β then
                    // leaves the cone, the objective-descent test ACCEPTS it (the
                    // gradient step does lower the unconstrained merit), and the
                    // NEXT cycle's `check_linear_feasibility` rejects the accepted
                    // iterate as an "infeasible iterate" (raw `Aβ−b` violation
                    // ~5.5e-3) — aborting the whole interval-censored warm start.
                    // The QP's `search_delta` is a feasible-to-feasible chord
                    // (`candidate_beta − beta_joint`, both endpoints in the convex
                    // cone), so box-truncating it to a SMALLER trust radius keeps
                    // every sub-step feasible. On the constrained path we therefore
                    // never swap in the unconstrained descent direction; we only
                    // shrink the radius and re-truncate the constrained chord. The
                    // comment on the preconditioned branch already promised it
                    // "preserves linear constraints when present" — this makes the
                    // implementation honor that contract.
                    let constrained_path_active = search_joint_active_set.is_some();
                    if !tried_preconditioned_descent && !constrained_path_active {
                        match joint_preconditioned_descent_delta(
                            effective_hessian_source,
                            &ranges,
                            &s_lambdas,
                            joint_solver_diagonal_ridge,
                            &rhs,
                            joint_bundle,
                        ) {
                            Ok(descent_delta) => {
                                search_delta = descent_delta;
                            }
                            Err(_) => {
                                joint_trust_radius = shrink_active_joint_block_trust_radii(
                                    &mut joint_block_trust_radii,
                                    &block_step_norms,
                                    0.25,
                                );
                            }
                        }
                        tried_preconditioned_descent = true;
                    } else {
                        joint_trust_radius = shrink_active_joint_block_trust_radii(
                            &mut joint_block_trust_radii,
                            &block_step_norms,
                            0.25,
                        );
                    }
                    continue;
                }

                for b in 0..specs.len() {
                    let (start, end) = ranges[b];
                    let mut trial_beta = old_beta[b].clone();
                    trial_beta += &trial_delta.slice(ndarray::s![start..end]);
                    let projected =
                        family.post_update_block_beta(&states, b, &specs[b], trial_beta.clone())?;
                    reject_constrained_post_update_repair(
                        b,
                        &specs[b],
                        &trial_beta,
                        &projected,
                        block_constraints[b].as_ref(),
                    )?;
                    states[b].beta.assign(&projected);
                }
                refresh_all_block_etas(family, specs, &mut states)?;
                let mut trial_penalty = total_quadratic_penalty(
                    &states,
                    &s_lambdas,
                    ridge,
                    options.ridge_policy,
                    joint_bundle,
                    Some(specs),
                );
                // Jeffreys objective contribution at the trial point keeps the
                // accept/reject objective consistent with the Jeffreys-modified
                // Newton step. `states` already holds the trial coefficients
                // (assigned + eta-refreshed above). No-op when the Jeffreys term
                // is unavailable or condition-gated to zero. When the cheap pre-
                // check certified this cycle well-conditioned, the step used H_Φ=0
                // / ∇Φ=0, so the consistent accept/reject objective also uses Φ=0:
                // skipping here keeps value and step on the SAME objective (the
                // value/step consistency the term exists to enforce) and avoids the
                // dense H/eigh at the trial point. The 8× conditioning margin makes
                // a single damped Newton step incapable of crossing the gate.
                // SUBTRACT Φ: the inner NLL objective is `−ℓ + ½βᵀSβ − Φ` (Firth
                // adds ½log|I| to the log-likelihood). Must match the cycle-0
                // baseline, the Newton step, and the KKT residual — INCLUDING the
                // `jeffreys_skippable_this_cycle` gate, so that on a well-conditioned
                // cycle the trial, the step (H_Φ=0/∇Φ=0), and the residual all sit
                // on the SAME Φ=0 objective (gam#729/#715 sign fix; the baseline and
                // post-accept folds carry the matching skippable gate).
                if !jeffreys_skippable_this_cycle
                    && let Some(z_joint) = joint_jeffreys_subspace.as_ref()
                {
                    trial_penalty -= custom_family_joint_jeffreys_value(
                        family, &states, specs, &ranges, z_joint,
                    );
                }
                // Cheap-LL line-search path: rejected backtracking attempts
                // discard the exact-Newton workspace they build, so we evaluate
                // just the scalar full-data log-likelihood for the accept/reject
                // decision and only build the full state once the step is
                // accepted (via the gradient reload below).
                //
                // EARLY-EXIT THRESHOLD MUST BOUND THE NLL, NOT THE FULL OBJECTIVE
                // (was a stall — gam#787/#785, duchon centers≥20). The family's
                // `bernoulli_margslope_line_search_ll_with_early_exit` short-
                // circuits the row sweep when the accumulated `-Σ wᵢ log CDF` (the
                // NLL ALONE — no penalty, no Jeffreys Φ) exceeds the threshold; its
                // monotone-lower-bound proof is valid only for the NLL term. But the
                // accept test is on the FULL augmented objective
                // `F = -ℓ + ½βᵀSβ + Φ_trial`, accepted iff `F ≤ old_objective + slack`,
                // i.e. iff `-ℓ_trial ≤ old_objective + slack − penalty_trial`. Passing
                // the full `old_objective` as the NLL threshold therefore over-rejects
                // by exactly `penalty_trial`: where the trial penalty is NEGATIVE
                // (the Jeffreys term subtracts Φ, and `½βᵀSβ` can be net-negative
                // under the reparam) the NLL threshold sits BELOW the true accept
                // bound, so the early exit kills net-descent steps the trust region
                // would accept — every backtracking attempt false-rejects, the radius
                // collapses, and the inner exits non-converged at cycle ~2 (seed
                // rejected pre-solver → hard raise, β pinned). Subtract the trial
                // penalty so the threshold is the NLL the trial must beat.
                let line_search_options =
                    coefficient_line_search_options(options, old_objective + 1e-10 - trial_penalty);
                // Accept-on-first-attempt fast path (gam#979 `gradient_reload`
                // cost). On the FIRST trust-region attempt of a cycle the step
                // is the undamped (radius-bumped) Newton proposal, which on the
                // common ρ≈1 `hold_inside` large-scale pattern accepts outright.
                // The cheap scalar sweep below would then run a full row stream
                // and immediately discard it, leaving `gradient_reload` to
                // re-stream every row at the SAME β to build the gradient
                // workspace — the ~5s redundant second pass per accepted cycle.
                //
                // Instead, when a workspace gradient source is available, build
                // the joint-Newton workspace ONCE at the trial β and read its
                // `joint_log_likelihood_evaluation()` (the same `Σ wᵢ log Φ` the
                // cheap sweep computes, on the same row measure — both derive
                // from `options`). The materialised per-row cache is threaded
                // forward as `accepted_joint_workspace`, so on accept the reload
                // short-circuits through `joint_gradient_evaluation()` with NO
                // second stream — collapsing the accepted cycle to one row pass.
                //
                // Only the first attempt takes this path: it is the only one
                // expected to accept, so a rejected first attempt pays a single
                // full (non-early-exited) sweep — paid back many-fold on the
                // dominant accept-on-first-attempt cycle. Later backtracking
                // attempts keep the cheap early-exiting sweep (they are expected
                // to reject and the workspace they would build is discarded).
                // A workspace build that *errors* (e.g. an infeasible trial η on
                // the radius-bumped first proposal) must NOT abort the whole
                // solve: the cheap-sweep path below classifies that same failure
                // as a recoverable likelihood-reject and shrinks the radius. So
                // treat an `Err`/`None` here as "fast path unavailable" and fall
                // through to the cheap sweep, which owns the reject bookkeeping.
                let fused_first_attempt = if trust_attempt == 0 && joint_workspace_requested {
                    joint_line_search_log_likelihood_with_workspace(family, options, specs, &states)
                        .ok()
                        .flatten()
                } else {
                    None
                };
                let trial_ll = if let Some((value, workspace)) = fused_first_attempt {
                    accepted_joint_workspace = Some(workspace);
                    value
                } else {
                    match joint_line_search_log_likelihood(family, &line_search_options, &states) {
                        Ok((value, workspace)) => {
                            accepted_joint_workspace = workspace;
                            value
                        }
                        Err(e) => {
                            likelihood_rejects += 1;
                            if first_likelihood_reject.is_none() {
                                first_likelihood_reject = Some(e);
                            }
                            for (b, old) in old_beta.iter().enumerate() {
                                states[b].beta.assign(old);
                            }
                            refresh_all_block_etas(family, specs, &mut states)?;
                            joint_trust_radius = shrink_active_joint_block_trust_radii(
                                &mut joint_block_trust_radii,
                                &block_step_norms,
                                0.25,
                            );
                            continue;
                        }
                    }
                };
                let trialobjective = -trial_ll + trial_penalty;
                if trust_attempt == 0 && trialobjective.is_finite() {
                    // Deterministic fixed-point signature (see declaration). The
                    // first attempt evaluates at the unshrunk pre-cycle β, so this
                    // value identifies the iterate exactly.
                    first_attempt_trial_objective = Some(trialobjective);
                }
                // Row measure observed by the trial objective at β + δ. The
                // line-search helper above runs under `coefficient_line_search_options`,
                // which now preserves `outer_score_subsample` and disables
                // any further auto-install; if either contract is broken the
                // id will diverge from `tr_row_measure_top` and we Err below.
                let tr_row_measure_trial =
                    crate::solver::row_measure::RowSubsampleMask::from_options(
                        options,
                        total_joint_n,
                    );
                // Hard invariant: the trust-region ratio numerator (objective
                // at β minus trial at β+δ) and denominator (rhs·δ − ½δᵀH δ)
                // MUST share a row measure with the Hessian/gradient build.
                // Bubble out via `Err` rather than panic; this function
                // already returns `Result<_, String>`.
                let top_id = tr_row_measure_top.id;
                if tr_row_measure_hessian.id != top_id {
                    return Err(format!(
                        "trust-region row-measure invariant violated: \
                         Hessian id 0x{:016x} differs from top-of-cycle id 0x{:016x} \
                         (cycle {}); the joint Hessian was built against a different \
                         row mask than the trust-region globalization captured at the \
                         top of the cycle. ρ would compare ½δᵀHδ on one measure to \
                         F(β)−F(β+δ) on another.",
                        tr_row_measure_hessian.id, top_id, cycle
                    ));
                }
                if tr_row_measure_gradient.id != top_id {
                    return Err(format!(
                        "trust-region row-measure invariant violated: \
                         gradient id 0x{:016x} differs from top-of-cycle id 0x{:016x} \
                         (cycle {}); `cached_joint_gradient` was loaded against a \
                         different row mask than the trust-region globalization \
                         captured at the top of the cycle. rhs·δ in the predicted \
                         reduction would not match the rest of the ρ inputs.",
                        tr_row_measure_gradient.id, top_id, cycle
                    ));
                }
                if tr_row_measure_old_objective.id != top_id {
                    return Err(format!(
                        "trust-region row-measure invariant violated: \
                         objective-at-β id 0x{:016x} differs from top-of-cycle id \
                         0x{:016x} (cycle {}); `lastobjective` was computed against \
                         a different row mask than the trust-region globalization \
                         captured at the top of the cycle.",
                        tr_row_measure_old_objective.id, top_id, cycle
                    ));
                }
                if tr_row_measure_trial.id != top_id {
                    return Err(format!(
                        "trust-region row-measure invariant violated: \
                         trial-objective id 0x{:016x} differs from top-of-cycle id \
                         0x{:016x} (cycle {}, attempt {}); the line-search trial \
                         likelihood evaluated against a different row mask than the \
                         Hessian/gradient/old-objective build. Cf. \
                         `coefficient_line_search_options` and \
                         `install_auto_outer_subsample_options`.",
                        tr_row_measure_trial.id, top_id, cycle, trust_attempt
                    ));
                }
                let actual_reduction = old_objective - trialobjective;
                let trust_update = update_joint_trust_region_radius(
                    joint_trust_radius,
                    step_norm,
                    actual_reduction,
                    predicted_reduction,
                    old_objective,
                );
                let old_radius = joint_trust_radius;
                // Classify the outcome of this attempt so the diagnostic line
                // says *why* the step was taken or rejected rather than just
                // dumping numbers. The four phases partition the post-log
                // branches below; computing them up front lets the log line
                // and the dispatch agree.
                let floor_reached = trust_update.accepted
                    && current_stationarity_residual <= residual_tol
                    && joint_objective_floor_reached(
                        old_objective,
                        trialobjective,
                        actual_reduction,
                        predicted_reduction,
                        objective_tol,
                    );
                let roundoff_slack = joint_objective_roundoff_slack(old_objective, trialobjective);
                let secondary_ok = !floor_reached
                    && trialobjective.is_finite()
                    && trust_update.accepted
                    && trialobjective <= old_objective + roundoff_slack;
                let phase: &'static str = if floor_reached {
                    "converged"
                } else if secondary_ok {
                    "accepted"
                } else if trust_update.accepted {
                    "stall"
                } else {
                    "reject"
                };
                if floor_reached || secondary_ok {
                    for (block_radius, block_step_norm) in joint_block_trust_radii
                        .iter_mut()
                        .zip(block_step_norms.iter())
                    {
                        let block_update = update_joint_trust_region_radius(
                            *block_radius,
                            *block_step_norm,
                            actual_reduction,
                            predicted_reduction,
                            old_objective,
                        );
                        if block_update.radius >= *block_radius
                            || joint_block_step_hit_trust_boundary(*block_step_norm, *block_radius)
                        {
                            *block_radius = block_update.radius;
                        }
                    }
                    joint_trust_radius = joint_block_trust_radii
                        .iter()
                        .copied()
                        .fold(0.0_f64, f64::max);
                } else {
                    joint_trust_radius = shrink_active_joint_block_trust_radii(
                        &mut joint_block_trust_radii,
                        &block_step_norms,
                        0.25,
                    );
                }
                let radius_held =
                    (joint_trust_radius - old_radius).abs() <= 1e-12 * old_radius.abs().max(1.0);
                let joint_math = JointNewtonMathDiagnostic {
                    old_kkt_inf: current_kkt_norm,
                    linearized_next_kkt_inf,
                    predicted_reduction,
                    actual_reduction,
                    trust_ratio: trust_update.rho,
                    step_inf: trial_step_inf,
                    proposal_inf: step_inf,
                };
                let radius_field = if radius_held {
                    format!("r={:.3e} (held)", old_radius)
                } else {
                    format!("r={:.3e}->{:.3e}", old_radius, joint_trust_radius)
                };
                // Surface the TR-policy decision so future failures
                // distinguish "TR is throttling Newton" from "TR is not
                // the bottleneck — Newton itself finds short steps".
                // For the large-scale linear-convergence pattern the policy
                // is consistently `hold_inside` (ρ≈1, |δ| ≪ radius),
                // which proves the TR is not what is keeping the step
                // small — that came up before via "(held)" alone but
                // the explicit decision label makes the inference
                // immediate instead of requiring step/radius arithmetic
                // in the reader's head.
                let tr_attempt_sig = format!(
                    "{:<9}  ρ={:+.3e}  Δobj={:+.3e}  pred={:+.3e}  {}  decision={:<22}  |δ|={:.3e}  |δ|∞={:.3e}  |prop|∞={:.3e}",
                    phase,
                    trust_update.rho,
                    actual_reduction,
                    predicted_reduction,
                    radius_field,
                    trust_update.decision.label(),
                    step_norm,
                    trial_step_inf,
                    step_inf,
                );
                match tr_log_sig.as_deref() {
                    Some(prev) if prev == tr_attempt_sig.as_str() => {
                        tr_log_last = line_search_attempts;
                    }
                    Some(prev) => {
                        if tr_log_first == tr_log_last {
                            log::info!(
                                "[PIRLS/joint-Newton/TR cycle={} attempt={}] {}",
                                cycle,
                                tr_log_first,
                                prev,
                            );
                        } else {
                            log::info!(
                                "[PIRLS/joint-Newton/TR cycle={} attempts={}..{} ×{}] {}",
                                cycle,
                                tr_log_first,
                                tr_log_last,
                                tr_log_last - tr_log_first + 1,
                                prev,
                            );
                        }
                        tr_log_sig = Some(tr_attempt_sig);
                        tr_log_first = line_search_attempts;
                        tr_log_last = line_search_attempts;
                    }
                    None => {
                        tr_log_sig = Some(tr_attempt_sig);
                        tr_log_first = line_search_attempts;
                        tr_log_last = line_search_attempts;
                    }
                }
                if floor_reached {
                    if let Some(sig) = tr_log_sig.take() {
                        if tr_log_first == tr_log_last {
                            log::info!(
                                "[PIRLS/joint-Newton/TR cycle={} attempt={}] {}",
                                cycle,
                                tr_log_first,
                                sig,
                            );
                        } else {
                            log::info!(
                                "[PIRLS/joint-Newton/TR cycle={} attempts={}..{} ×{}] {}",
                                cycle,
                                tr_log_first,
                                tr_log_last,
                                tr_log_last - tr_log_first + 1,
                                sig,
                            );
                        }
                    }
                    for (b, old) in old_beta.iter().enumerate() {
                        states[b].beta.assign(old);
                    }
                    refresh_all_block_etas(family, specs, &mut states)?;
                    last_joint_math = Some(joint_math);
                    accepted = true;
                    converged = true;
                    break;
                }
                if secondary_ok {
                    if let Some(sig) = tr_log_sig.take() {
                        if tr_log_first == tr_log_last {
                            log::info!(
                                "[PIRLS/joint-Newton/TR cycle={} attempt={}] {}",
                                cycle,
                                tr_log_first,
                                sig,
                            );
                        } else {
                            log::info!(
                                "[PIRLS/joint-Newton/TR cycle={} attempts={}..{} ×{}] {}",
                                cycle,
                                tr_log_first,
                                tr_log_last,
                                tr_log_last - tr_log_first + 1,
                                sig,
                            );
                        }
                    }
                    current_penalty = trial_penalty;
                    if let Some(joint_active_set) = search_joint_active_set.as_ref() {
                        cached_active_sets =
                            scatter_joint_active_set(joint_active_set, &block_constraints);
                    }
                    let tight_rows_added = augment_active_sets_with_tight_constraint_rows(
                        &mut cached_active_sets,
                        &block_constraints,
                        &states,
                        crate::solver::active_set::ACTIVE_SET_PRIMAL_FEASIBILITY_TOL,
                    );
                    if tight_rows_added > 0 {
                        log::info!(
                            "[PIRLS/joint-Newton] added {tight_rows_added} tight constraint row(s) \
                             to the cached active set after accepted constrained step"
                        );
                    }
                    last_joint_math = Some(joint_math);
                    last_accepted_hit_joint_trust_boundary = step_hit_trust_boundary;
                    accepted = true;
                    break;
                }
                for (b, old) in old_beta.iter().enumerate() {
                    states[b].beta.assign(old);
                }
                refresh_all_block_etas(family, specs, &mut states)?;
                objective_rejects += 1;
                // Frozen-step short-circuit (see declaration). `radius_held` here
                // means the post-reject shrink did not change the joint trust
                // radius — i.e. the radii are pinned at the `1e-12` floor. If the
                // trial objective also reproduces the previous rejected attempt's
                // bit-for-bit, the dogleg/Moré–Sorensen step is frozen and every
                // remaining attempt would re-reject the identical step: skip the
                // redundant full-data sweeps and let the cross-cycle stall guard
                // certify the fixed point.
                if radius_held && trialobjective.is_finite() {
                    if let Some(prev) = prev_rejected_attempt_objective {
                        if prev.to_bits() == trialobjective.to_bits() {
                            frozen_floor_full_reject = true;
                            break;
                        }
                    }
                    prev_rejected_attempt_objective = Some(trialobjective);
                } else {
                    prev_rejected_attempt_objective = None;
                }
            }
            if let Some(sig) = tr_log_sig.take() {
                if tr_log_first == tr_log_last {
                    log::info!(
                        "[PIRLS/joint-Newton/TR cycle={} attempt={}] {}",
                        cycle,
                        tr_log_first,
                        sig,
                    );
                } else {
                    log::info!(
                        "[PIRLS/joint-Newton/TR cycle={} attempts={}..{} ×{}] {}",
                        cycle,
                        tr_log_first,
                        tr_log_last,
                        tr_log_last - tr_log_first + 1,
                        sig,
                    );
                }
            }
            let line_search_elapsed = line_search_started.elapsed();
            if accepted && converged {
                log::info!(
                    "[PIRLS/joint-Newton/cycle-summary] cycle={} accepted=true hessian_qp={:.3}s line_search={:.3}s line_search_attempts={} reject_model={} reject_likelihood={} reject_objective={} reject_feasibility={} first_likelihood_reject={} grad_reload=0.000s total={:.3}s",
                    cycle,
                    hessian_and_qp_elapsed.as_secs_f64(),
                    line_search_elapsed.as_secs_f64(),
                    line_search_attempts,
                    model_rejects,
                    likelihood_rejects,
                    objective_rejects,
                    feasibility_rejects,
                    first_likelihood_reject.as_deref().unwrap_or("none"),
                    cycle_started.elapsed().as_secs_f64(),
                );
                // Accepted step moved β; the cycle workspace is at the OLD
                // (pre-step) β, so it must NOT be carried into the post-loop
                // covariance/IFT assembly (which needs the converged β). Drop it.
                cached_joint_workspace = None;
                cycles_done = cycle + 1;
                break;
            }
            if !accepted {
                // Retry the joint Newton loop from the same state after a
                // failed trust-region search. Falling through into blockwise
                // would switch a coupled exact-Hessian problem onto a
                // principal-block surrogate, which is the ridge-drift failure
                // mode this path is meant to avoid. The trust-region radius
                // already collapsed via the attempt loop's shrink rules, so
                // the next cycle's Newton proposal will be evaluated under
                // a tighter L2 bound without any parallel adaptation here.
                log::info!(
                    "[PIRLS/joint-Newton/cycle-summary] cycle={} accepted=false hessian_qp={:.3}s line_search={:.3}s line_search_attempts={} reject_model={} reject_likelihood={} reject_objective={} reject_feasibility={} first_likelihood_reject={} grad_reload=0.000s total={:.3}s",
                    cycle,
                    hessian_and_qp_elapsed.as_secs_f64(),
                    line_search_elapsed.as_secs_f64(),
                    line_search_attempts,
                    model_rejects,
                    likelihood_rejects,
                    objective_rejects,
                    feasibility_rejects,
                    first_likelihood_reject.as_deref().unwrap_or("none"),
                    cycle_started.elapsed().as_secs_f64(),
                );
                // Restore original betas
                for (b, old) in old_beta.iter().enumerate() {
                    states[b].beta.assign(old);
                }
                refresh_all_block_etas(family, specs, &mut states)?;
                // β is now back at `old_beta`, the exact β this cycle's
                // exact-Newton workspace was built at. A rejected cycle does NOT
                // run the post-accept gradient reload (which is what otherwise
                // re-stashes a workspace), so without this the next cycle's
                // `cached_joint_workspace.take()` is `None` and re-streams all n
                // rows through the per-row kernel cache build — pure redundancy
                // at the identical β. Hand the still-valid workspace back so the
                // next cycle hits the cache. Bit-identical: same family, same β,
                // so the rebuilt cache would be byte-for-byte this one; the inner
                // solve still re-derives the Newton step and runs to its KKT
                // certificate unchanged. On the loop-exit `break`s below this is
                // a harmless assignment to a value that is then dropped.
                cached_joint_workspace = hessian_workspace_for_cycle.take();
                // If the previous cycle's bookkeeping certified KKT
                // stationarity (residual ≤ tol and objective change ≤
                // tol), the line-search failure here is round-off on a
                // rank-deficient null mode rather than non-convergence:
                // the proposed `H⁻¹ g` step stays O(1) along the null
                // direction at the optimum, every trial moves β along
                // it without changing the objective, and round-off
                // flips the sign of `actual − predicted` so the
                // sufficient-decrease check rejects every trial. The
                // iterate ALREADY satisfies the first-order optimality
                // conditions; we accept that as convergence rather
                // than fail the outer "inner solve did not converge"
                // panic on a fully resolved fit.
                if last_cycle_residual_below_tol && last_cycle_obj_change_below_tol {
                    converged = true;
                    break;
                }
                // Fully-rejected stall guard. See the constant declaration
                // at the top of this function for the full rationale. The
                // condition is: every trust attempt this cycle was rejected by
                // SOME path (model OR likelihood OR objective OR feasibility; the
                // four reject counters partition the JOINT_TRUST_MAX_ATTEMPTS
                // attempts) AND
                // the joint trust radius did not shrink relative to the previous
                // fully-rejected cycle. Both together prove the next cycle's
                // Newton system, trust radius, and trust-region search are
                // bytewise identical to this cycle's — there is no descent
                // direction the local quadratic model can reconcile at this β.
                //
                // The earlier form required objective_rejects ==
                // JOINT_TRUST_MAX_ATTEMPTS && likelihood_rejects == 0, so it
                // NEVER fired on the biobank gauge-flat marginal/logslope fit:
                // there the objective is flat to f64 precision along the
                // residual direction and the BMS line search rejects every
                // trial on the *likelihood* early-exit path
                // (likelihood_rejects == 24), so the stall guard's increment
                // condition was unreachable and the loop spun to its cap. A
                // full rejection by the likelihood path at a collapsed trust
                // radius is the same numerically-flat-no-descent stall as a
                // full objective rejection; counting either lets the guard fire.
                let all_attempts_rejected = frozen_floor_full_reject
                    || model_rejects + likelihood_rejects + objective_rejects + feasibility_rejects
                        == JOINT_TRUST_MAX_ATTEMPTS;
                let radius_held_since_last_reject = match prev_rejected_trust_radius {
                    Some(prev) => {
                        joint_trust_radius.is_finite()
                            && prev.is_finite()
                            && joint_trust_radius >= prev * (1.0 - 1e-12)
                    }
                    None => false,
                };
                if all_attempts_rejected && radius_held_since_last_reject {
                    consecutive_held_rejected_cycles =
                        consecutive_held_rejected_cycles.saturating_add(1);
                } else {
                    consecutive_held_rejected_cycles = 0;
                }
                prev_rejected_trust_radius = Some(joint_trust_radius);
                // Byte-identical fixed-point detector. A fully-rejected cycle
                // whose first-attempt trial objective reproduces the previous
                // fully-rejected cycle's value bit-for-bit proves β reverted to
                // the same iterate and the Newton system is identical, so every
                // further cycle is a provable no-op. This is stronger than the
                // held-radius count and fires in two cycles, fixing the n≈3e5
                // marginal/logslope grind where the held-radius path's off-by-one
                // let the inner solve spin past the wall-clock budget (gam#979).
                match (
                    all_attempts_rejected,
                    first_attempt_trial_objective,
                    prev_rejected_first_attempt_objective,
                ) {
                    (true, Some(current), Some(prev)) if current.to_bits() == prev.to_bits() => {
                        consecutive_identical_rejected_cycles =
                            consecutive_identical_rejected_cycles.saturating_add(1);
                    }
                    _ => {
                        consecutive_identical_rejected_cycles = 0;
                    }
                }
                if all_attempts_rejected {
                    prev_rejected_first_attempt_objective = first_attempt_trial_objective;
                } else {
                    prev_rejected_first_attempt_objective = None;
                }
                // Collapsed-trust-region all-reject-at-floor detector (gam#979).
                // Increment only when EVERY attempt this cycle was rejected AND
                // the joint trust radius has reached its absolute `1e-12` floor:
                // the radius cannot shrink further and the step makes no
                // progress, so the next cycle is forced to repeat this one. Any
                // accepted cycle (handled below via the post-grad-reload reset)
                // or any cycle whose radius is still above the floor breaks the
                // streak, so a progressing fit never accumulates it.
                let all_attempts_rejected_at_floor_this_cycle = all_attempts_rejected
                    && joint_trust_radius_at_absolute_floor(joint_trust_radius);
                if all_attempts_rejected_at_floor_this_cycle {
                    consecutive_all_reject_at_floor_cycles =
                        consecutive_all_reject_at_floor_cycles.saturating_add(1);
                } else {
                    consecutive_all_reject_at_floor_cycles = 0;
                }
                let collapsed_floor_exit = joint_collapsed_floor_all_reject_exit(
                    consecutive_all_reject_at_floor_cycles,
                    all_attempts_rejected_at_floor_this_cycle,
                );
                if consecutive_held_rejected_cycles >= FULLY_REJECTED_STALL_MAX_CYCLES
                    || consecutive_identical_rejected_cycles >= IDENTICAL_REJECTED_STALL_MAX_CYCLES
                    || collapsed_floor_exit
                {
                    let last_math_summary = last_joint_math
                        .as_ref()
                        .map(|math| {
                            format!(
                                "last_newton_math={{old_kkt={:.3e}, linearized_next={:.3e}, actual={:+.3e}, pred={:+.3e}, rho={:+.3e}, scalar_relerr={:.3e}, step_inf={:.3e}, proposal_inf={:.3e}}}",
                                math.old_kkt_inf,
                                math.linearized_next_kkt_inf,
                                math.actual_reduction,
                                math.predicted_reduction,
                                math.trust_ratio,
                                math.scalar_model_relative_error(),
                                math.step_inf,
                                math.proposal_inf,
                            )
                        })
                        .unwrap_or_else(|| "last_newton_math=<none>".to_string());
                    let stall_trigger = if consecutive_identical_rejected_cycles
                        >= IDENTICAL_REJECTED_STALL_MAX_CYCLES
                    {
                        format!(
                            "byte-identical first-attempt trial objective for {} consecutive \
                             fully-rejected cycles (exact fixed point)",
                            consecutive_identical_rejected_cycles
                        )
                    } else if consecutive_all_reject_at_floor_cycles
                        >= JOINT_COLLAPSED_FLOOR_ALL_REJECT_MAX_CYCLES
                    {
                        format!(
                            "{} consecutive fully-rejected cycles with the joint trust radius \
                             collapsed to its absolute 1e-12 floor (no smaller step \
                             representable, step makes no progress)",
                            consecutive_all_reject_at_floor_cycles
                        )
                    } else {
                        format!(
                            "{} consecutive fully-rejected cycles with joint trust radius held",
                            consecutive_held_rejected_cycles
                        )
                    };
                    log::warn!(
                        "[PIRLS/joint-Newton convergence] cycle {:>3} | fully-rejected stall \
                         early-exit: every trust-region attempt rejected (by any of the model / \
                         likelihood / objective paths) — {} at joint trust radius {:.3e}. Reverted β \
                         + identical Newton system mean the next cycle's step is byte-identical to \
                         this one's; no descent direction is reachable from this iterate under the \
                         current local model. {}. Checking identified-subspace stationarity before \
                         declaring non-convergence.",
                        cycle,
                        stall_trigger,
                        joint_trust_radius,
                        last_math_summary,
                    );
                    // Judge convergence on the IDENTIFIED (range) subspace
                    // before declaring non-convergence. A fully-rejected stall
                    // at a collapsed trust radius (every trial rejected at
                    // ~noise, ΔNLL ≈ 1 ULP) is the PROOF the descent direction
                    // is gauge-flat: the raw KKT residual (the biobank fit's
                    // 0.5) lives in the unidentified ker(H_pen) direction (the
                    // gauge-flat marginal/logslope coupling, same family as the
                    // c5d327ba4 separation false-positive), which the outer IFT
                    // pseudo-inverse projects out. Reuse the EXACT machinery the
                    // normal converged path uses (gam#979 commit 09b584024):
                    // the active-set-projected stationarity vector
                    // (`exact_newton_joint_projected_stationarity_vector_from_gradient`)
                    // restricted to range(H+Sλ) via
                    // `projected_residual_range_space_inf`. If the identified-
                    // subspace residual is at tolerance the fit IS at a
                    // numerically-stationary penalized optimum and must be
                    // RETURNED converged; only if it is ALSO above tol is this a
                    // genuine non-convergence. `cached_joint_gradient` was loaded
                    // at the cycle-entry β, which is exactly the reverted
                    // `old_beta` here, so the residual is evaluated at the
                    // returned iterate.
                    let stall_converged_on_identified_subspace = match cached_joint_gradient
                        .as_ref()
                    {
                        Some(stall_gradient) => {
                            match exact_newton_joint_projected_stationarity_vector_from_gradient(
                                stall_gradient,
                                &states,
                                specs,
                                &s_lambdas,
                                ridge,
                                options.ridge_policy,
                                &block_constraints,
                                Some(cached_active_sets.as_slice()),
                            ) {
                                Ok(stall_projected_residual_vec) => {
                                    projected_residual_range_space_inf(
                                        &stall_projected_residual_vec,
                                        &joint_hessian_source,
                                        &ranges,
                                        &s_lambdas,
                                        ridge,
                                        options.ridge_policy,
                                        total_p,
                                    )
                                    .filter(|range_residual| range_residual.is_finite())
                                    .filter(|range_residual| *range_residual <= last_residual_tol)
                                }
                                Err(_) => None,
                            }
                        }
                        None => None,
                    };
                    if let Some(stall_range_residual) = stall_converged_on_identified_subspace {
                        log::info!(
                            "[PIRLS/joint-Newton convergence] cycle {:>3} | fully-rejected stall \
                             resolved as identified-subspace KKT convergence (gam#979): every \
                             trust-region attempt rejected for {} cycles at trust radius {:.3e} \
                             (objective flat to f64 precision along the proposal — the proof the \
                             descent direction is gauge-flat), but the range-space \
                             (identified-subspace) residual {:.3e} ≤ tol {:.3e}; the leftover raw \
                             residual lives entirely in the unidentified ker(H_pen) gauge mode the \
                             outer IFT projects out (gam#553). The iterate is at a \
                             numerically-stationary penalized optimum — returning converged.",
                            cycle,
                            consecutive_held_rejected_cycles,
                            joint_trust_radius,
                            stall_range_residual,
                            last_residual_tol,
                        );
                        if stall_range_residual.is_finite() {
                            min_certified_residual =
                                min_certified_residual.min(stall_range_residual);
                        }
                        converged = true;
                        break;
                    }
                    // FULL-RANK FIXED-POINT FALLBACK (gam#979 survival marginal-slope).
                    //
                    // The range-space certificate above returns `None` when
                    // `projected_residual_range_space_inf` finds `nullity == 0`:
                    // at the REML optimum ρ the penalty `S(λ)` lifts the
                    // gauge-flat marginal≈logslope alias direction (the n≈133
                    // heart-failure fit's 0.956-overlap confound) to an
                    // eigenvalue just ABOVE the `1e-10·λ_max` rank cutoff, so
                    // H_pen reads numerically full-rank and there is no null
                    // space to project the residual onto. The previous code then
                    // declared non-convergence and ABORTED the whole fit — even
                    // though the iterate is a provably-stationary penalized
                    // optimum whose residual is floored at a small multiple of
                    // tol by the near-singular conditioning, NOT by a reachable
                    // descent direction.
                    //
                    // The honest discriminator is the SAME conjunction the
                    // fully-rejected stall is built on, asserted explicitly so a
                    // genuine non-convergence (a fit stalled FAR from tol with
                    // real reducible range-space mass) still exits unconverged:
                    //   (a) the joint trust radius has collapsed to its `1e-12`
                    //       floor — no smaller step is representable;
                    //   (b) `FULLY_REJECTED_STALL_MAX_CYCLES` consecutive cycles
                    //       (or a byte-identical first-attempt objective) were
                    //       fully rejected — β reverts identically, so the next
                    //       step is bit-for-bit the same: a fixed point;
                    //   (c) the last accepted Newton model was essentially exact
                    //       (`trust_ratio ≈ 1` and scalar relative model error
                    //       below `inner_tol`) — the quadratic model has nothing
                    //       left to reduce.
                    // Only when all three hold AND the best residual the solve
                    // actually achieved is within the same `4×residual_tol` band
                    // the sibling identified-subspace certificates use (lines
                    // ~4497 / ~4659) AND the objective is flat do we certify. A
                    // fit stalled at, say, 10×tol with reducible mass fails the
                    // `best_residual_seen ≤ 4×tol` gate and still aborts.
                    const FIXED_POINT_TRUST_RADIUS_CEIL: f64 = 1e-9;
                    let model_exact = last_joint_math
                        .as_ref()
                        .map(|math| {
                            (math.trust_ratio - 1.0).abs() <= 0.5
                                && math.scalar_model_relative_error() <= inner_tol.max(1e-6)
                        })
                        .unwrap_or(false);
                    let provable_fixed_point = joint_trust_radius <= FIXED_POINT_TRUST_RADIUS_CEIL
                        && (consecutive_held_rejected_cycles >= FULLY_REJECTED_STALL_MAX_CYCLES
                            || consecutive_identical_rejected_cycles
                                >= IDENTICAL_REJECTED_STALL_MAX_CYCLES
                            || consecutive_all_reject_at_floor_cycles
                                >= JOINT_COLLAPSED_FLOOR_ALL_REJECT_MAX_CYCLES)
                        && model_exact;
                    if provable_fixed_point
                        && last_cycle_obj_change_below_tol
                        && best_residual_seen.is_finite()
                        && best_residual_seen <= 4.0 * last_residual_tol
                    {
                        log::info!(
                            "[PIRLS/joint-Newton convergence] cycle {:>3} | fully-rejected stall \
                             resolved as full-rank fixed-point KKT convergence (gam#979): trust \
                             radius collapsed to {:.3e} (≤ floor), {} consecutive fully-rejected \
                             cycles, last Newton model exact (ρ≈1, scalar_relerr small), and the \
                             objective is flat — a provably-stationary penalized optimum. H_pen reads \
                             full-rank at the optimal ρ (the gauge-flat alias eigenvalue sits above \
                             the rank cutoff), so the range-space certificate has no null space to \
                             project; the best achieved KKT residual {:.3e} ≤ 4×tol {:.3e} is the \
                             near-singular conditioning floor, not reducible descent — returning \
                             converged.",
                            cycle,
                            joint_trust_radius,
                            consecutive_held_rejected_cycles,
                            best_residual_seen,
                            4.0 * last_residual_tol,
                        );
                        if best_residual_seen.is_finite() {
                            min_certified_residual = min_certified_residual.min(best_residual_seen);
                        }
                        converged = true;
                        break;
                    }
                    converged = false;
                    break;
                }
                // CONTINUE rather than break (gam#826/#872/#715). The comment
                // above documents the intent — "retry the joint Newton loop from
                // the same state after a failed trust-region search" — but the old
                // code BROKE instead, giving up after a SINGLE cycle of failed line
                // search. On a severely near-separating coupled fit (matern
                // binomial location-scale, quasi-separating multinomial, flexible
                // linkwiggle) the cycle-0 Newton proposal is huge (the separation
                // gradient ÷ the Firth-bounded curvature), the trust region clamps
                // it, and the clamped step does not yet reduce the merit — so the
                // FIRST cycle's backtracking exhausts without acceptance. The
                // attempt loop already shrank `joint_trust_radius` /
                // `joint_block_trust_radii` (carried across cycles), so the NEXT
                // cycle re-proposes under the tighter radius and eventually accepts
                // a productive step — standard trust-region globalization. Breaking
                // at cycle 0 aborted the coupled solve ("exited the joint Newton
                // path before convergence — no math snapshot") before the trust
                // region could adapt. The inner cycle cap and the residual-stall /
                // trust-region-floor guards above still bound the loop, so a
                // genuinely stuck fit exits with a diagnosed non-convergence rather
                // than spinning. Falling through to blockwise (the old `break`)
                // would switch the coupled exact-Hessian problem onto a
                // principal-block surrogate (the ridge-drift mode this path avoids).
                continue;
            }

            let grad_reload_started = std::time::Instant::now();
            log::info!(
                "[joint-newton-tr] phase=gradient_reload cycle={} attempts={} r={:.3e}",
                cycle,
                line_search_attempts,
                joint_trust_radius,
            );
            let (log_likelihood, gradient, eval, workspace) = load_joint_gradient_evaluation(
                family,
                specs,
                options,
                &states,
                joint_workspace_requested,
                accepted_joint_workspace.take(),
            )?;
            let grad_reload_elapsed = grad_reload_started.elapsed();
            // Reset the fully-rejected stall guard's bookkeeping: an accepted
            // cycle moved β and may have grown the trust radius, so the next
            // rejected-cycle comparison must start fresh rather than carry
            // forward a stale radius snapshot from the previous reject streak.
            prev_rejected_trust_radius = None;
            consecutive_held_rejected_cycles = 0;
            // An accepted step moved β, so the fixed-point signature is stale;
            // reset it so a later reject streak compares only consecutive
            // fully-rejected cycles at the SAME iterate.
            prev_rejected_first_attempt_objective = None;
            consecutive_identical_rejected_cycles = 0;
            // An accepted step moved β and (via the trust-region grow rules)
            // lifts the radius off its floor, so the collapsed-floor all-reject
            // streak no longer holds; reset it (gam#979).
            consecutive_all_reject_at_floor_cycles = 0;
            // Accepted-cycle timing breakdown is debug-only. The per-cycle
            // info line below already includes total cycle time; emitting a
            // four-phase split on every verbose cycle adds a redundant info
            // line. Rejected cycles still keep the detailed phase log since
            // the reject reason and per-phase split is the diagnostic.
            log::debug!(
                "[PIRLS/joint-Newton/cycle-summary] cycle={} accepted=true hessian_qp={:.3}s line_search={:.3}s line_search_attempts={} grad_reload={:.3}s total={:.3}s",
                cycle,
                hessian_and_qp_elapsed.as_secs_f64(),
                line_search_elapsed.as_secs_f64(),
                line_search_attempts,
                grad_reload_elapsed.as_secs_f64(),
                cycle_started.elapsed().as_secs_f64(),
            );
            current_log_likelihood = log_likelihood;
            cached_joint_gradient = gradient;
            cached_eval = eval;
            cached_joint_workspace = workspace;
            current_penalty = total_quadratic_penalty(
                &states,
                &s_lambdas,
                ridge,
                options.ridge_policy,
                joint_bundle,
                Some(specs),
            );
            // `current_penalty` / `lastobjective` stay the pure quadratic-penalized
            // objective (NO Φ folded in) — the Firth value is applied per cycle at
            // each β (see `old_objective` above and `trialobjective` below). The
            // gated Φ at the accepted β is captured separately so the convergence
            // `objective_change` compares the augmented objective at the new vs old
            // β consistently (gam#826/#872).
            lastobjective = -current_log_likelihood + current_penalty;
            let new_phi = if !jeffreys_skippable_this_cycle {
                joint_jeffreys_subspace
                    .as_ref()
                    .map(|z_joint| {
                        custom_family_joint_jeffreys_value(family, &states, specs, &ranges, z_joint)
                    })
                    .unwrap_or(0.0)
            } else {
                0.0
            };
            let accepted_step_inf = states
                .iter()
                .zip(old_beta.iter())
                .flat_map(|(state, old)| {
                    state
                        .beta
                        .iter()
                        .zip(old.iter())
                        .map(|(new, old)| (new - old).abs())
                })
                .fold(0.0_f64, f64::max);
            cycles_done = cycle + 1;

            // Check convergence via joint stationarity. When the family-general
            // Firth/Jeffreys term is armed, the penalized objective the inner
            // Newton actually optimizes is `−ℓ + ½βᵀSβ − Φ`, so its KKT
            // stationarity is `∇L − Sβ + ∇Φ = 0`. The Newton STEP already folds
            // `∇Φ` into its RHS (`spectral_rhs += grad_phi`), but the bare
            // `exact_newton_joint_stationarity_*` residual omits it — at the
            // Firth fixed point `∇L − Sβ = −∇Φ`, so the certificate floors at
            // `‖∇Φ‖∞` and never certifies, stalling the inner solve on exactly
            // the near-separating span Firth is meant to bound (the residual the
            // outer REML then rejects). Fold `∇Φ` into the gradient used for the
            // KKT residual so the convergence criterion matches the augmented
            // objective the step descends. No-op when the Jeffreys term is
            // unavailable or condition-gated to zero.
            let Some(gradient) = cached_joint_gradient.as_ref() else {
                break;
            };
            let jeffreys_augmented_gradient: Option<Array1<f64>> = if jeffreys_skippable_this_cycle
            {
                // Well-conditioned ⇒ ∇Φ = 0, so the KKT residual is the bare
                // stationarity (and floors at 0, not ‖∇Φ‖) — matching the step,
                // which folded H_Φ=0/∇Φ=0 this cycle. Avoids the dense H/eigh.
                None
            } else if let Some(z_joint) = joint_jeffreys_subspace.as_ref() {
                match custom_family_joint_jeffreys_term(family, &states, specs, &ranges, z_joint)? {
                    Some((_phi, grad_phi, hphi))
                        if grad_phi.len() == gradient.len()
                            && hphi.nrows() == total_p
                            && hphi.ncols() == total_p =>
                    {
                        let augmented = gradient + &grad_phi;
                        // Cache the exact triple at the just-accepted β so the next
                        // cycle's head reuses it instead of recomputing the
                        // O(p)-directional-derivative + GEMM term (gam#729).
                        let post_beta_key = flatten_state_betas(&states, specs);
                        jeffreys_triple_cache = Some((post_beta_key, grad_phi, hphi));
                        Some(augmented)
                    }
                    _ => None,
                }
            } else {
                None
            };
            let residual_gradient = jeffreys_augmented_gradient.as_ref().unwrap_or(gradient);
            let residual = exact_newton_joint_stationarity_inf_norm_from_gradient(
                residual_gradient,
                &states,
                specs,
                &s_lambdas,
                ridge,
                options.ridge_policy,
                &block_constraints,
                Some(cached_active_sets.as_slice()),
            )?;
            prev_kkt_norm = Some(residual);
            // Record this cycle's KKT residual for the steady-geometric-descent
            // test at the certificate-refusal gate below (gam#787 centers≥20).
            if residual.is_finite() {
                min_certified_residual = min_certified_residual.min(residual);
                residual_descent_history.push_back(residual);
                while residual_descent_history.len() > RESIDUAL_DESCENT_WINDOW {
                    residual_descent_history.pop_front();
                }
            }

            // Scale-aware tolerances. The objective check was already
            // relative (`inner_tol * (1 + |obj|)`), but the step and
            // residual checks were absolute against the bare `inner_tol`
            // — at large scale (n ≈ 320k), β iterates can keep moving
            // by ~1e-5 per cycle along the monotonicity-feasible
            // manifold even after the likelihood has gone flat, and the
            // joint gradient ‖·‖_∞ is O(|obj|), not O(1). Running
            // 50-100 cycles past objective convergence is the
            // dominant inner-PIRLS cost at large scale. Switching to
            // relative scaling (`inner_tol * (1 + ‖β‖_∞)` for steps,
            // `inner_tol * (1 + |obj|)` for the gradient residual)
            // exits PIRLS as soon as the optimum is statistically
            // resolved, without loosening behavior at small n where
            // ‖β‖_∞ ≈ 1 and |obj| ≈ 1 give tolerances within 2× of
            // the historical absolute 1e-6.
            let beta_inf = states
                .iter()
                .flat_map(|s| s.beta.iter().copied())
                .map(f64::abs)
                .fold(0.0_f64, f64::max);
            let step_tol = inner_tol * (1.0 + beta_inf);
            let objective_tol = inner_tol * (1.0 + lastobjective.abs());
            // KKT residual tolerance must scale with the natural magnitude of
            // ‖Sβ − ∇L‖∞ (i.e. max(‖∇L‖∞, ‖Sβ‖∞)), not the objective. At
            // large scale with |β|∞ in the 10²–10³ range the gradient and
            // penalty norms can sit orders of magnitude above |obj| and FP
            // noise alone keeps the residual above any obj-scaled tol. The
            // pre-line-search check at the head of the cycle already uses
            // `inner_tol * (1 + max(grad_inf, pen_inf))`; using only grad_inf
            // here created an asymmetry where the same convergence criterion
            // would accept at one site and reject at the other, and on
            // marginal-slope models where Sβ is the larger term it shrank
            // the post-accept tolerance below the achievable FP floor.
            let mut block_gradient_norms = Vec::with_capacity(states.len());
            let mut block_penalty_norms = Vec::with_capacity(states.len());
            for (block_idx, (start, end)) in ranges.iter().copied().enumerate() {
                block_gradient_norms.push(
                    gradient
                        .slice(s![start..end])
                        .iter()
                        .map(|x: &f64| x.abs())
                        .fold(0.0_f64, f64::max),
                );
                let mut penalty_block = s_lambdas[block_idx].dot(&states[block_idx].beta);
                if options.ridge_policy.include_quadratic_penalty && ridge > 0.0 {
                    penalty_block += &states[block_idx].beta.mapv(|v| ridge * v);
                }
                block_penalty_norms.push(
                    penalty_block
                        .iter()
                        .map(|x: &f64| x.abs())
                        .fold(0.0_f64, f64::max),
                );
            }
            let grad_inf = block_gradient_norms.iter().copied().fold(0.0_f64, f64::max);
            let pen_inf = block_penalty_norms.iter().copied().fold(0.0_f64, f64::max);
            // Firth/Jeffreys score magnitude. The convergence residual is the
            // AUGMENTED stationarity `∇L − Sβ + ∇Φ`, so `∇Φ` is a first-class term
            // whose own numerical scale sets the achievable KKT floor: `∇Φ` is a
            // trace `½ tr(H_id⁻¹ Z_Jᵀ Ḣ Z_J)` formed from a FLOORED reduced-info
            // pseudo-inverse, so its components carry O(‖∇Φ‖·ε_floor) round-off
            // that the augmented residual cannot polish below. Scaling the KKT
            // tolerance by `max(grad, pen, ‖∇Φ‖)` (not just grad/pen) makes the
            // certificate reachable for coupled K-block Firth fits whose data
            // gradient is small but whose Firth score is O(1): otherwise the
            // augmented residual plateaus a few × above an unattainably tight
            // `inner_tol·(1+grad)` tol and the solve refuses just short of
            // convergence (gam#729/#715 — the residual stalled at ~8.8e-6 against a
            // ~1e-6 tol). No-op when the term is condition-gated (∇Φ=0).
            let firth_score_inf = head_jeffreys_term
                .as_ref()
                .map(|(grad_phi, _hphi)| grad_phi.iter().map(|v| v.abs()).fold(0.0_f64, f64::max))
                .unwrap_or(0.0);
            let residual_tol = inner_tol * (1.0 + grad_inf.max(pen_inf).max(firth_score_inf));
            // Arm the Jeffreys second-order endgame completion (gam#979) once
            // the residual enters the convergence band; latched (never
            // un-armed) so the endgame model cannot oscillate between the
            // divided-difference and exact Hessians across cycles.
            if residual.is_finite() && residual <= JEFFREYS_COMPLETION_RESIDUAL_BAND * residual_tol
            {
                jeffreys_completion_endgame = true;
            }
            let block_stationarity_tolerances = block_gradient_norms
                .iter()
                .zip(&block_penalty_norms)
                .map(|(grad_norm, penalty_norm)| inner_tol * (1.0 + grad_norm.max(*penalty_norm)))
                .collect::<Vec<_>>();
            // Active-set-projected stationarity residual vector (multiplier
            // mass of every pinned bound row already subtracted). Lifted out of
            // the per-block norm reduction so the constrained-stationary
            // certificate below can also test its component in the *range* of
            // the penalized Hessian (gam#553 penalty-null-space acceptance).
            let projected_residual_vec =
                exact_newton_joint_projected_stationarity_vector_from_gradient(
                    gradient,
                    &states,
                    specs,
                    &s_lambdas,
                    ridge,
                    options.ridge_policy,
                    &block_constraints,
                    Some(cached_active_sets.as_slice()),
                )?;
            let block_stationarity_norms = {
                let mut offset = 0usize;
                states
                    .iter()
                    .map(|state| {
                        let start = offset;
                        let end = start + state.beta.len();
                        offset = end;
                        projected_residual_vec
                            .slice(ndarray::s![start..end])
                            .iter()
                            .map(|x: &f64| x.abs())
                            .fold(0.0_f64, f64::max)
                    })
                    .collect::<Vec<_>>()
            };
            // Per-block stationarity must be judged on the IDENTIFIED (range-space)
            // residual, not the raw active-set-projected residual (gam#979). On the
            // survival I-spline time block the unpenalized affine baseline direction
            // is a genuine ker(H_pen) gauge mode: the raw per-block residual keeps
            // the full gradient component along it (the measured ~28 plateau at λ≈1e7
            // that the absolute tol can never reach), so the raw gate falsely rejects
            // a solve that IS stationary on every identifiable direction — the
            // residual mass it sees is the free gauge the outer IFT projects out
            // (gam#553). Use the range-projected per-block residual when a penalty
            // null space exists; fall back to the raw per-block residual when it does
            // not (there range == whole space, so they coincide and the strict gate
            // is unchanged for every well-identified family).
            //
            // PERF (gam#1082): the range projection eigendecomposes the FULL P·M
            // joint penalized Hessian — an O((P·M)³) cost. The two certificates
            // that consume the range-projected gate (the residual-stall and
            // relative-plateau exits below) only fire under rare preconditions
            // (`tr_clamped_during_stall` after a long no-improve streak; a latched
            // objective plateau). Computing the eigh EVERY inner cycle therefore
            // added a redundant O(p³) eigendecomposition per cycle to every
            // penalized family carrying a null space (every tp-smooth model) — the
            // multinomial smooth-by-factor wall-clock regression.
            //
            // The eigh is deferred to `range_projected_block_stationarity_small`
            // (called ONLY inside a certificate branch whose cheap precondition has
            // already passed via short-circuit `&&`), so on the convergence-tail
            // it runs at most once per accepted exit rather than once per cycle.
            let range_projected_block_stationarity_small = || -> bool {
                projected_residual_range_space_per_block_inf(
                    &projected_residual_vec,
                    &joint_hessian_source,
                    &ranges,
                    &s_lambdas,
                    ridge,
                    options.ridge_policy,
                    total_p,
                )
                .unwrap_or_else(|| block_stationarity_norms.clone())
                .iter()
                .zip(&block_stationarity_tolerances)
                .all(|(norm, tol)| {
                    norm.is_finite()
                        && tol.is_finite()
                        && *norm <= RESIDUAL_STALL_BLOCK_GRADIENT_FACTOR * *tol
                })
            };
            // gam#1082 perf: a per-cycle #979 divergence-trace logging block
            // lived here and computed — EVERY inner cycle for the first 40
            // cycles, purely to feed two `log::info!` lines — a FULL O((P·M)³)
            // eigendecomposition (`projected_residual_range_space_inf`), a
            // penalty-matrix min-eigenvalue, and per-penalty quadratic forms.
            // On any penalized family with a penalty null space (every
            // `select=TRUE` double-penalty tp-smooth model, including the
            // multinomial smooth-by-factor fit) the eigh's `nullity > 0` branch
            // actually ran, so each outer REML evaluation paid up to 40
            // redundant O(p³) eigendecompositions inside its inner joint-Newton.
            // That diagnostic instrumentation — not the outer iteration count —
            // was the dominant wall-clock cost (the #1082 overrun the outer
            // rel-cost decouple could not touch, because the cost is
            // per-inner-cycle, not per-outer-iteration). The trace has served
            // its #979 purpose and is removed from the production hot path; every
            // convergence-relevant quantity (`residual`, `block_stationarity_norms`,
            // and the lazily-evaluated range-space gate above) is still computed
            // where the gate actually consumes it.
            let near_convergence = residual <= 10.0 * residual_tol;
            // Augmented-objective change: `(quad(new) − Φ_gated(new)) −
            // (quad(old) − Φ_gated(old))`. `lastobjective` is quadratic-only and
            // `old_objective` already carries `−old_phi`, so subtract the accepted
            // β's `new_phi` here to keep both endpoints on the Φ-augmented merit
            // (gam#826/#872). On a skippable cycle both phis are 0 ⇒ identical to
            // the bare quadratic change.
            let signed_obj_change = (lastobjective - new_phi) - old_objective;
            let objective_change = signed_obj_change.abs();

            // Per-cycle observability for the convergence test. Surfaces
            // WHICH criterion is binding (proposed step, accepted step,
            // residual, objective change) at every iteration so CI logs
            // distinguish "Newton hasn't proposed a small step yet"
            // (algorithm still working) from "step is small but residual
            // won't drop below tol" (tolerance scaling problem). Without
            // this, the only visible signal is the objective itself,
            // which is insufficient to choose the right algorithmic
            // remedy.
            //
            // gam#979 discriminator: the PER-BLOCK projected stationarity
            // breakdown. The aggregate `residual` alone cannot distinguish a
            // genuinely-coupled stall from one block dragging the others — for
            // the survival marginal↔logslope grind the question "is the total
            // residual dominated by a single block (the multiplicative
            // z·exp(logslope) coupling channel), or spread evenly (global
            // conditioning)?" is answerable only from the split. `block_resid`
            // is already computed above for the convergence test, so surfacing
            // it per cycle is free; reading it across a 75 s repro under
            // RUST_LOG=info tells whether the slowdown is a single stuck block
            // (curvature/coupling channel) or an evenly slow descent
            // (conditioning) — without it the four #979 candidates are not
            // separable from the timeline.
            let block_resid_sig = block_stationarity_norms
                .iter()
                .map(|n| format!("{n:.3e}"))
                .collect::<Vec<_>>()
                .join(",");
            log::info!(
                "[PIRLS/joint-Newton convergence] cycle {:>3} | step_inf={:.3e} (tol={:.3e}) | accepted_step_inf={:.3e} | residual={:.3e} (tol={:.3e}) | per_block_resid=[{}] | obj_change={:.3e} (tol={:.3e}) | beta_inf={:.3e}",
                cycle,
                step_inf,
                step_tol,
                accepted_step_inf,
                residual,
                residual_tol,
                block_resid_sig,
                objective_change,
                objective_tol,
                beta_inf,
            );

            // gam#1082 perf: a tightly-gated `#1040 inner-conditioning probe`
            // lived here. Once the inner joint-Newton stalled (residual stuck
            // above tol for `RESIDUAL_STALL_NO_IMPROVE_CYCLES` cycles), it
            // eigendecomposed the FULL P·M penalized Hessian (O((P·M)³)) plus an
            // O(p²) Rayleigh-quotient loop EVERY cycle thereafter, purely to feed
            // one `log::info!`. The gate's whole point is "the solve is
            // grinding" — exactly the regime where it then fires on EVERY one of
            // the remaining (up to `inner_max_cycles`) cycles, turning a stall
            // into an O(p³)-per-cycle crawl (a dominant face of the #1082
            // multinomial wall-clock overrun: the cost is per-stalled-cycle, not
            // per-outer-iteration). The diagnostic is removed from the hot path;
            // the inner solve's own stall handling (trust-region clamp,
            // Newton-decrement and range-space convergence certificates) governs
            // termination, and the cheap per-cycle convergence line above already
            // surfaces residual/step/per-block-residual for observability.

            if verbose_cycle || near_convergence {
                log::info!(
                    "[PIRLS/JN] cyc={:>3}/{} obj={:.6e} -loglik={:.6e} pen={:.3e} Δobj={:+.3e} |δ|∞={:.3e} accepted_|δ|∞={:.3e} resid={:.3e} (tol={:.3e}) obj_tol={:.3e} step_tol={:.3e} |β|∞={:.3e} attempts={} t={:.3}s",
                    cycle,
                    inner_max_cycles,
                    lastobjective,
                    -current_log_likelihood,
                    current_penalty,
                    signed_obj_change,
                    step_inf,
                    accepted_step_inf,
                    residual,
                    residual_tol,
                    objective_tol,
                    step_tol,
                    beta_inf,
                    line_search_attempts,
                    cycle_started.elapsed().as_secs_f64(),
                );
            } else {
                log::info!(
                    "[PIRLS/JN] cyc={:>3}/{} obj={:.6e} Δobj={:+.3e} |δ|∞={:.3e} resid={:.3e} attempts={} t={:.3}s",
                    cycle,
                    inner_max_cycles,
                    lastobjective,
                    signed_obj_change,
                    accepted_step_inf,
                    residual,
                    line_search_attempts,
                    cycle_started.elapsed().as_secs_f64(),
                );
            }

            // Divergence guard: a non-finite KKT residual, objective, or
            // log-likelihood means the inner joint Newton has diverged (NaN
            // mass propagating from a near-unidentified penalized block — the
            // binomial location-scale shared-basis log-σ deviation channel is
            // the canonical trigger, gam#554). Every convergence and
            // residual-stall exit below is gated on finite `<=` comparisons,
            // which a NaN residual silently defeats; left unguarded the loop
            // then grinds the full `inner_loop_hard_ceiling` on every outer
            // ρ-eval and every startup seed, which is the multi-hour "hang".
            // Treat it as immediate non-convergence so the outer optimizer
            // rejects this point cleanly instead of burning the budget.
            if !residual.is_finite()
                || !lastobjective.is_finite()
                || !current_log_likelihood.is_finite()
            {
                log::warn!(
                    "[PIRLS/joint-Newton convergence] cycle {:>3} | divergence guard: non-finite inner state (residual={:.3e}, objective={:.3e}, -loglik={:.3e}); returning unconverged so the outer optimizer rejects this ρ evaluation instead of running to inner_max_cycles.",
                    cycle,
                    residual,
                    lastobjective,
                    -current_log_likelihood,
                );
                converged = false;
                break;
            }

            // KKT convergence: a small post-step residual is the
            // canonical optimality certificate for the penalized
            // objective. ‖∇L(β) − Sβ‖∞ ≤ residual_tol means the
            // iterate is at a KKT point to numerical precision and
            // further iteration cannot reduce it; the step magnitude
            // is irrelevant once the residual signal has fired.
            //
            // Tying convergence to a small step instead would refuse
            // to recognise quadratic-rate single-shot convergence:
            // exact Newton on an exact quadratic produces one full
            // step that lands at the optimum, so ‖delta‖∞ equals the
            // initial distance ‖β* − β₀‖∞ no matter how exact the
            // model is. Pairing a residual check with a step-size
            // requirement structurally rejects this entirely-correct
            // cycle-0 termination, leaving inner_max_cycles=1 callers
            // unable to certify convergence on a problem that was
            // solved exactly in one Newton step.
            if joint_inner_kkt_converged(residual, residual_tol) {
                converged = true;
                break;
            }
            // Identified-subspace (range-space) KKT certificate.
            //
            // The strict certificate above tests the FULL stationarity residual
            // ‖∇L − Sβ‖∞. On a genuinely rank-deficient penalized inner problem
            // — a degenerate small-n transformation-normal CTM/Box-Cox fit whose
            // joint Hessian carries an *unidentified* direction the
            // canonical-gauge pass cannot attribute to a single block (the same
            // structural null root-caused for the joint-Newton panic at
            // `solve_joint_newton_step_on_spectral_range`) — the stationarity
            // gradient keeps a fixed nonzero component inside ker(H_pen). The
            // spectral Newton step drops exactly that component (range-restricted
            // Moore–Penrose step: every null direction hits the `continue` branch
            // in the accumulation loop), so β converges on the identified
            // subspace and the step exhausts, yet the FULL residual never reaches
            // `residual_tol`. The strict test then runs the whole cycle budget
            // "non-converged" on an iterate that is, in fact, the optimum on the
            // only identifiable directions.
            //
            // The principled certificate is stationarity on range(H_pen): the
            // residual restricted to the curved (identified) subspace is at
            // tolerance while the leftover mass is provably confined to
            // ker(H_pen) — an unidentified direction with neither curvature nor
            // constraint. That null component is dropped by the spectral step
            // here and projected out of the KKT residual by the outer IFT
            // pseudo-inverse `U_S·H_proj⁻¹·U_Sᵀ` before the envelope correction
            // (see the gam#553 note and `projected_residual_range_space_inf`), so
            // it cannot bias the outer gradient.
            //
            // The remaining requirement is to prove we are AT the
            // range-restricted optimum rather than mid-descent, so this does not
            // short-circuit a genuinely nonlinear CTM fit that is still moving β.
            // There are two independent, equally-rigorous proofs of that, and
            // EITHER suffices once `range_residual ≤ residual_tol` has fired:
            //   (a) the full Newton step is exhausted (`step_inf ≤ step_tol`):
            //       the well-identified case, where the range-restricted step
            //       collapses to zero and the leftover ker(H_pen) component is
            //       already dropped by the spectral step, so the FULL step is
            //       small too; OR
            //   (b) the objective has stopped changing
            //       (`objective_change ≤ objective_tol`): the joint objective
            //       (−loglik + ½βᵀSβ) is a function of the IDENTIFIED coordinates
            //       ONLY — moving β along an unidentified direction in ker(H_pen)
            //       = ker(H_L) ∩ ker(S) changes neither the likelihood nor the
            //       penalty by construction — so a flat objective proves no
            //       identified-direction descent remains regardless of how large
            //       the FULL step is.
            // Proof (b) is the certificate that the constant-scale AFT (#736) and
            // the degenerate CTM (#733/#734) need: their unidentified cross-block
            // null (the time_transform polynomial/affine deviation aliased into
            // threshold/log_sigma) keeps the Levenberg-damped, trust-region-clamped
            // FULL step perpetually nonzero — `step_inf` never reaches `step_tol`
            // — even though the identified fit is exactly at its optimum (zero
            // range-space residual, frozen objective). Tying the certificate ONLY
            // to the full step (proof (a)) therefore burned the entire 200/84-cycle
            // budget on an iterate that is already optimal on every identifiable
            // direction, and the inner solve was rejected by the FULL-residual KKT
            // check. Adding proof (b) certifies on the identified subspace without
            // loosening anything for a genuinely-identified fit: there
            // `projected_residual_range_space_inf` returns `None` (nullity == 0 ⇒
            // range == whole space), so this branch is dormant and the strict
            // full-residual path above governs unchanged.
            //
            // Newton-decrement convergence certificate (gam#1040 / gam#1088).
            //
            // The strict / identified-subspace / constrained certificates all
            // gate on the penalized stationarity residual ‖∇L − Sβ‖∞ reaching
            // `residual_tol`. On a weakly-identified (near-flat) carrying block
            // — the survival marginal↔logslope alias, the binomial link-wiggle
            // block, the gaussian/binomial location-scale μ block — that residual
            // can stall ORDERS above tol (`g` is O(1e2) along a direction whose
            // penalized curvature `γ` is tiny) while every step the trust region
            // admits is clamped, so neither the residual nor the step-norm gate
            // ever closes and the loop grinds to the cycle ceiling, the outer
            // REML rejects ρ after ρ, and the fit times out (the #1040/#1088
            // benchmark hangs). Yet the ACHIEVABLE objective improvement is
            // `g²/(2γ)` — the Newton decrement — and on such a direction it is
            // far below `objective_tol`: no step the local quadratic model can
            // resolve lowers the penalized objective by more than `objective_tol`.
            // By the Conn–Gould–Toint stopping criterion (*Trust-Region Methods*,
            // Thm 6.4.6) the iterate is then the penalized optimum to within
            // tolerance, on the entire identifiable subspace — the residual's
            // un-resolvable mass lives on near-null directions the outer IFT
            // pseudo-inverse projects out (gam#553). The decrement is read off
            // the SAME D-whitened seed spectrum the step is built from (range
            // modes only; the null space contributes none), so it is exactly the
            // model decrease of the unconstrained modified-Newton step. A genuine
            // defect (real curvature AND large gradient) yields a LARGE decrement,
            // so this never certifies a non-converged iterate.
            //
            // Precondition (gam#1082): the original gate required the LAST cycle's
            // `objective_change ≤ objective_tol` to "confirm we are AT the plateau,
            // not one big step away." That precondition is the multinomial
            // smooth-by-factor blocker: the coupled-softmax select=TRUE gauge mode
            // is a NEAR-null (weak-but-above-`KKT_REFUSAL_RANK_TOL` curvature), so
            // the iterate keeps DRIFTING along it with a small but nonzero
            // `objective_change` every cycle (exactly the gam#979 survival
            // signature) — `objective_change ≤ objective_tol` never holds, the
            // decrement certificate never fires, and the solve crawls to
            // `inner_max_cycles` paying one ~p³ Newton-step eigh per cycle (the
            // eu-stack-profiled #1082 blow-up). But the decrement bound is itself
            // the correct, curvature-aware stopping test: by Conn–Gould–Toint Thm
            // 6.4.6 `decrement ≤ objective_tol` ALONE certifies the iterate is the
            // penalized optimum to tolerance — no model-resolvable step (gauge
            // drift included) lowers the objective by more than tol. So the
            // objective-flat precondition is replaced by the RESIDUAL-STALL window
            // (`cycles_since_residual_improved ≥ DECREMENT_STALL_WINDOW`): the
            // certificate fires once the raw residual has stopped descending and
            // the decrement confirms no resolvable improvement remains. This reuses
            // the EXACT degeneracy classification the Newton step uses (the
            // decrement skips every `|γ_k| ≤ null_cutoff` mode), so it catches the
            // near-null gauge direction the raw-`H_pen` range projection's absolute
            // `1e-10·λ_max` cutoff misses — without ever accepting a genuinely
            // curved (large-decrement) unconverged iterate. A still-progressing
            // solve never reaches the stall window (its residual keeps improving,
            // resetting the counter).
            const DECREMENT_STALL_WINDOW: usize = 3;
            if cycles_since_residual_improved >= DECREMENT_STALL_WINDOW
                && let Some(decrement) = joint_spectrum
                    .as_ref()
                    .map(|spectrum| spectrum.newton_decrement())
                && decrement.is_finite()
                && decrement <= objective_tol
                // Conditioning-robust safety (gam#1449): the raw decrement above
                // excludes every `|γ_k| ≤ null_cutoff = max(rank_tol·λ_max,
                // numerical_floor)` mode. On a badly-scaled penalized Hessian
                // `rank_tol·λ_max` can swallow a mode with small-but-REAL curvature
                // AND real signal — a weakly-identified direction, not gauge —
                // whose achievable improvement `c²/(2|γ|)` the raw decrement then
                // silently ignores. Require that improvement, measured over the
                // genuinely-curved band (above the machine-rank `numerical_floor`,
                // a conditioning-robust cutoff that does NOT scale with λ_max), to
                // ALSO be within tolerance before certifying. The genuine numerical
                // null space (below `numerical_floor`) still contributes nothing,
                // and the step is unchanged; this only HARDENS the stopping test so
                // a weakly-identified real mode blocks premature certification.
                && let Some(weak_decrement) = joint_spectrum
                    .as_ref()
                    .map(|spectrum| spectrum.weakly_identified_decrement())
                && weak_decrement.is_finite()
                && weak_decrement <= objective_tol
            {
                // Audit witness (#1082): the residual mass this certificate
                // EXCLUDES as gauge-null. The decrement bound is sound only when
                // that excluded mass truly lies on penalty-null directions; if it
                // is large the certificate may have discarded a weakly-identified
                // real mode (the `null_cutoff = rank_tol·λ_max` ill-conditioning
                // edge), so emit it at WARN to keep the decision auditable.
                let excluded_null_residual = joint_spectrum
                    .as_ref()
                    .map(|spectrum| spectrum.null_residual_inf())
                    .unwrap_or(0.0);
                if excluded_null_residual > residual_tol.max(1e-6) {
                    log::warn!(
                        "[PIRLS/joint-Newton convergence] cycle {cycle:>3} | Newton-decrement \
                         certificate fired with LARGE excluded near-null residual \
                         ={excluded_null_residual:.3e} (> tol={residual_tol:.3e}); the stopping \
                         rule treated this mass as free gauge. Sound iff it lies on genuine \
                         penalty-null directions — flagged for joint-stationarity audit (#1082)."
                    );
                }
                log::info!(
                    "[PIRLS/joint-Newton convergence] cycle {:>3} | Newton-decrement certificate (gam#1040/#1088/#1082): \
                     residual={:.3e} (tol={:.3e}) stalled above tol for {} cycles on a weakly-identified block (last \
                     |Δobjective|={:.3e}, drifting along a near-null gauge mode), but the unconstrained modified-Newton \
                     step's predicted objective decrease (Newton decrement ½gᵀH⁻¹g over identified modes, the SAME \
                     |γ_k|≤null_cutoff degeneracy classification the Newton step uses)={:.3e} ≤ objective_tol={:.3e} \
                     — no model-resolvable step lowers the penalized objective by more than tolerance, so the \
                     iterate is the REML optimum on the identifiable subspace (Conn–Gould–Toint Thm 6.4.6); \
                     the un-resolvable residual mass lies on near-null directions the outer IFT projects out.",
                    cycle,
                    residual,
                    residual_tol,
                    cycles_since_residual_improved,
                    objective_change,
                    decrement,
                    objective_tol,
                );
                // Record the residual this exit certified on so the terminal
                // line reports a finite certified residual (#1040 truthfulness):
                // the converged status is earned by the decrement bound, and the
                // finite stationarity residual at this iterate is the honest
                // certificate witness.
                if residual.is_finite() {
                    min_certified_residual = min_certified_residual.min(residual);
                }
                converged = true;
                break;
            }

            // Gauge-drift identified-subspace KKT certificate (gam#979 large-scale
            // survival-MS stall). The certificate just below requires EITHER
            // `step_inf ≤ step_tol` OR `objective_change ≤ objective_tol` as a
            // precondition before it will even look at the range-projected
            // residual. On the survival I-spline time block at large scale that
            // precondition is a Catch-22: the block carries a genuine ker(H_pen)
            // gauge mode (the unpenalized affine baseline — constant + linear time
            // trend — that the joint design does not gauge-fix out), so the
            // constrained QP keeps taking a small but NONZERO step (`step_inf` ~1e-4,
            // never ≤ step_tol) that drifts the iterate ALONG that near-null
            // direction, and because the direction has a tiny-but-nonzero curvature
            // the merit keeps changing by `objective_change` ~0.3 per cycle (never
            // ≤ objective_tol). Neither precondition is ever met, so the existing
            // exit cannot fire even though the iterate IS already stationary on the
            // entire identifiable subspace — the inner solve grinds to its hard
            // cycle ceiling, the outer rejects the ρ-eval, and the fit times out
            // (the measured cycles 8→20 trace: residual ~1.19e3, step_inf ~1e-4,
            // obj_change ~0.34, beta_inf frozen at 2.269).
            //
            // The honest stationarity test for that regime drops the
            // step/objective precondition and instead demands BOTH range-projected
            // measures be at tolerance simultaneously: the GLOBAL identified-subspace
            // residual `range_residual ≤ residual_tol` AND every BLOCK's
            // range-projected stationarity small. The range projection drops exactly
            // the ker(H_pen) gauge mass (the same mass the outer IFT pseudo-inverse
            // projects out, gam#553), so when both pass the iterate is the REML
            // optimum on the identifiable subspace by definition — the residual,
            // step, and objective drift that remain live purely in the unidentified
            // null and carry no outer-correctness information. The double
            // (global + per-block) range gate cannot be satisfied by a genuinely
            // non-stationary iterate: a real un-converged identifiable direction
            // shows up in BOTH the global range residual and its block's
            // range-projected component, so this never accepts a non-optimum on the
            // identified subspace (it is strictly stronger than the single-gate exit
            // below, only without the gauge-defeated step/objective precondition).
            // Cheap precondition gating the O((P·M)³) range-projection eigh
            // (gam#1082 perf discipline): only attempt this certificate once the
            // raw residual has stopped improving for a few consecutive cycles —
            // i.e. the iterate is no longer making descent progress in the
            // identifiable subspace and the remaining motion is the gauge drift
            // this exit exists to certify through. A healthy, fast-converging fit
            // never trips this window (it converges via the strict residual / step
            // certificates first), so it pays zero extra eigendecompositions; only
            // a genuinely stalled solve reaches it, where one eigh per cycle on the
            // short convergence tail is negligible against the alternative of
            // grinding to the hard cycle ceiling.
            //
            // Tolerance (gam#1082): the range-projected residual is read off an
            // O((P·M)³) eigendecomposition of the joint penalized Hessian and
            // reconstructed by summing the residual's coordinates along every
            // range-space eigenvector. On a large joint design (the multinomial
            // smooth-by-factor fit is ~382-dim, K−1 coupled blocks × one global
            // smooth + one smooth per group level, each select=TRUE with a
            // wiggliness AND a null-space-shrinkage penalty) that reconstruction
            // carries O(p·ε·‖r‖) round-off, so the identified-subspace residual
            // FLOORS a few × above the (already tiny, `inner_tol·(1+…)`-scaled)
            // `residual_tol`. Demanding the strict 1× `residual_tol` here is
            // therefore unreachable for a genuinely range-stationary iterate whose
            // remaining mass is pure ker(H_pen) gauge drift — exactly the
            // multinomial regime, where the gauge mode keeps the objective drifting
            // (so the sibling obj-plateau / step-or-obj exits below never fire) and
            // the inner solve grinds to `inner_max_cycles`, each cycle paying one
            // full Newton-step eigh (the #1082 wall-clock blow-up the profile
            // pins on `WhitenedHessianSpectrum::decompose`). The honest, mature
            // identified-subspace tolerance is the SAME `4×residual_tol` the
            // relative-objective-plateau gauge exit below already uses to certify
            // the identical mathematical condition (range-space stationarity); the
            // 1× here was an unjustified asymmetry below the eigh-reconstruction
            // floor. The gauge-drift precondition (raw residual stalled ≥ window
            // AND per-block range-stationarity) is strictly stronger than the
            // plateau exit's, so widening the global range tolerance to match it
            // cannot accept a non-optimum on the identified subspace.
            const GAUGE_DRIFT_STALL_WINDOW: usize = 3;
            const RANGE_RESIDUAL_EIGH_FLOOR_FACTOR: f64 = 4.0;
            if cycles_since_residual_improved >= GAUGE_DRIFT_STALL_WINDOW
                && let Some(range_residual) = projected_residual_range_space_inf(
                    &projected_residual_vec,
                    &joint_hessian_source,
                    &ranges,
                    &s_lambdas,
                    ridge,
                    options.ridge_policy,
                    total_p,
                )
                && range_residual <= RANGE_RESIDUAL_EIGH_FLOOR_FACTOR * residual_tol
                && range_projected_block_stationarity_small()
            {
                log::info!(
                    "[PIRLS/joint-Newton convergence] cycle {:>3} | gauge-drift identified-subspace KKT certificate (gam#979): total residual={:.3e} > tol={:.3e}, step_inf={:.3e} (step_tol={:.3e}) and |Δobjective|={:.3e} (obj_tol={:.3e}) both still nonzero from drift along an unidentified ker(H_pen) gauge mode (an unpenalized baseline/gauge direction the joint design does not fix out), but the range-space (identified-subspace) residual={:.3e} ≤ {:.3e} (= {}×tol, the eigh-reconstruction floor) AND every block's range-projected stationarity is at tolerance — the iterate is stationary on the entire identifiable subspace; the remaining residual/step/objective drift lives purely in the gauge null the outer IFT projects out (gam#553).",
                    cycle,
                    residual,
                    residual_tol,
                    step_inf,
                    step_tol,
                    objective_change,
                    objective_tol,
                    range_residual,
                    RANGE_RESIDUAL_EIGH_FLOOR_FACTOR * residual_tol,
                    RANGE_RESIDUAL_EIGH_FLOOR_FACTOR,
                );
                if range_residual.is_finite() {
                    min_certified_residual = min_certified_residual.min(range_residual);
                }
                converged = true;
                break;
            }

            // Unlike the constrained-stationary path below, this fires on a pure
            // identifiability null without requiring the `linearized_rel ≥ 0.5`
            // constraint-multiplier signature, which a structural rank-deficiency
            // need not produce.
            if (step_inf <= step_tol || objective_change <= objective_tol)
                && let Some(range_residual) = projected_residual_range_space_inf(
                    &projected_residual_vec,
                    &joint_hessian_source,
                    &ranges,
                    &s_lambdas,
                    ridge,
                    options.ridge_policy,
                    total_p,
                )
                && range_residual <= residual_tol
            {
                log::info!(
                    "[PIRLS/joint-Newton convergence] cycle {:>3} | identified-subspace KKT certificate: total residual={:.3e} > tol={:.3e} but its range-space (identified-subspace) component={:.3e} ≤ tol={:.3e}, step_inf={:.3e} (step_tol={:.3e}), |Δobjective|={:.3e} (obj_tol={:.3e}); the leftover residual lies in the unidentified penalized-Hessian null space ker(H_pen) (dropped by the range-restricted spectral step and projected out by the outer IFT pseudo-inverse) — the iterate is stationary on the entire identifiable subspace (proof: {}).",
                    cycle,
                    residual,
                    residual_tol,
                    range_residual,
                    residual_tol,
                    step_inf,
                    step_tol,
                    objective_change,
                    objective_tol,
                    if step_inf <= step_tol {
                        "full Newton step exhausted"
                    } else {
                        "objective frozen on the identified subspace while the unidentified null keeps the full step nonzero"
                    },
                );
                converged = true;
                break;
            }
            // Noise-floor KKT certificate.
            //
            // Reading the joint stationarity residual ‖∇L(β) − Sβ‖_∞ at finite
            // precision picks up rounding mass from the X'WX assembly and the
            // per-block penalty contraction. For well-conditioned problems
            // that floor sits well below `residual_tol`, so the strict path
            // fires and this branch is dormant. For tightly converged inner
            // states where the Newton iterate is already at the analytic
            // optimum but every additional step changes the objective by less
            // than `objective_tol` and the recomputed residual lands just
            // above `residual_tol` due to arithmetic noise, the strict path
            // alone refuses to certify convergence — even though no further
            // useful descent direction exists. Burning hundreds of identical
            // descent cycles past that point neither tightens the inner
            // optimum (the noise floor sets a hard lower bound on ‖rhs‖) nor
            // gives the outer optimizer more hyperparameter information; it
            // just causes the outer wrapper to reject every seed as
            // "inner did not converge" and downstream callers to mark the
            // analytic outer Hessian as unavailable.
            //
            // Combining two independent post-step signals — objective change
            // within scale-aware tolerance AND residual within the same KKT
            // tolerance — supplies the missing certificate without weakening
            // the envelope-theorem requirement. A residual above tolerance
            // can be a free Hessian-null gradient component, not an active
            // multiplier, so it must not be accepted by an objective-flatness
            // rule.
            //
            // Distinct from the strict path because the strict path is silent
            // on objective change;
            // distinct from the trust-region floor certificate at the head
            // of the cycle because that one fires only when the trust radius
            // has collapsed to its 1e-12 floor with all attempts rejected,
            // whereas this branch fires when the trust region is still open
            // but each accepted step is no longer producing detectable
            // objective progress.
            let objective_change = signed_obj_change.abs();
            if objective_change.is_finite() {
                geometric_tail_history.push_back(objective_change);
                while geometric_tail_history.len() > GEOMETRIC_TAIL_WINDOW {
                    geometric_tail_history.pop_front();
                }
            }
            if objective_change <= objective_tol && residual <= residual_tol {
                log::info!(
                    "[PIRLS/joint-Newton convergence] cycle {:>3} | noise-floor KKT certificate: residual={:.3e} <= tol={:.3e}, |Δobjective|={:.3e} <= obj_tol={:.3e}",
                    cycle,
                    residual,
                    residual_tol,
                    objective_change,
                    objective_tol,
                );
                converged = true;
                break;
            }

            // Constrained-stationary certificate.
            //
            // The inner Newton system is `Hδ = -g`, solved over the
            // active-constraint-aware subspace (the QP step path).  When
            // the *unprojected* gradient `g` carries a large Lagrange-
            // multiplier component pointing into the constraint —
            // i.e. some β coordinates are pinned at the bound or against
            // the family's structural constraint surface — the linear
            // solve correctly DOES NOT try to eliminate that component,
            // because doing so would push β infeasibly.  The signature of
            // this state is precise and entirely local to the most recent
            // accepted step:
            //
            //   • `‖g + Hδ‖∞ / ‖g‖∞ ≥ 0.5` — the linear solve neutralised
            //     ≤ 50 % of g; the remainder is structurally outside the
            //     solver's range, i.e. it's a Lagrange multiplier of the
            //     active constraints, not a defect of the linear solve.
            //   • `|actual − pred| / max(|pred|, …) ≤ 1e-3` — the local
            //     quadratic Newton model agrees with the actual objective
            //     change to roundoff, so the Hessian and gradient are
            //     correct AT this β.  The "stuck" residual is not noise
            //     in the linearisation; it's a real multiplier.
            //   • `|Δobjective| ≤ objective_tol` — the objective has
            //     ceased moving meaningfully.
            //   • `|δ|∞ ≤ step_tol` — the accepted feasible Newton step is
            //     exhausted. Objective flatness alone is not a terminal
            //     signal on large survival fits: a step of O(1e-2..1e-1)
            //     can still continue reducing the KKT residual after the
            //     objective first crosses tolerance.
            //
            // Together these four are the rigorous certificate that
            // Newton has reached a constrained-stationary point: further
            // cycles would reproduce the same plateau (the diagnostic in
            // PIRLS/JN/math shows `‖g+Hδ‖/‖g‖` constant near 1 cycle
            // after cycle, the very signature this certificate names).
            //
            // The 0.5 threshold on `linearized_rel` is conservative —
            // an unconstrained Newton step has `linearized_rel ≈ 1e-12`;
            // a step deliberately constrained to a (k-1)-dim subspace
            // leaves the orthogonal Lagrange direction in the residual
            // and `linearized_rel ≈ |λ|/|g| > 0`, typically 0.9+ in
            // practice when the multiplier dominates.  Anything ≥ 0.5
            // is unambiguously in the constrained-stationary regime;
            // unconstrained Newton with `linearized_rel ≥ 0.5` would
            // have already failed the trust-region's scalar model test
            // and been rejected upstream.
            if let Some(math) = last_joint_math.as_ref() {
                let linearized_rel = math.linearized_rel();
                let scalar_model_relerr = math.scalar_model_relative_error();
                let geometric_tail_bound = if geometric_tail_history.len() == GEOMETRIC_TAIL_WINDOW
                {
                    let values = geometric_tail_history.iter().copied().collect::<Vec<_>>();
                    let mut max_ratio = 0.0_f64;
                    let mut valid = true;
                    for pair in values.windows(2) {
                        let prev = pair[0];
                        let next = pair[1];
                        if prev <= 0.0 || next < 0.0 || !prev.is_finite() || !next.is_finite() {
                            valid = false;
                            break;
                        }
                        let ratio = next / prev;
                        if !ratio.is_finite() || ratio >= 1.0 {
                            valid = false;
                            break;
                        }
                        max_ratio = max_ratio.max(ratio);
                    }
                    if valid {
                        Some(objective_change / (1.0 - max_ratio).max(1.0e-12))
                    } else {
                        None
                    }
                } else {
                    None
                };
                let certificate_decision = constrained_stationary_certificate_decision(
                    math,
                    objective_change,
                    objective_tol,
                    step_tol,
                    geometric_tail_bound,
                    residual,
                    residual_tol,
                );
                if !matches!(
                    certificate_decision,
                    ConstrainedStationaryCertificate::NotCandidate
                ) {
                    // The `linearized_rel >= 0.5` signal is necessary but not
                    // sufficient. It proves either (a) g carries a Lagrange
                    // multiplier of an active constraint that the QP's active
                    // set already represents — in which case the *projected*
                    // residual is at tolerance — or (b) H is rank-deficient
                    // in the direction of g, so Hδ ≈ 0 along the null
                    // direction regardless of whether g is a multiplier or a
                    // real defect. Case (b) is the survival marginal-slope
                    // pathology at large scale: H σ_min ≈ 1e-12 and Newton
                    // genuinely cannot move g, but the residual is NOT a
                    // captured multiplier — it's an unresolved KKT defect in
                    // the H-null subspace.
                    //
                    // The projected residual computed at the top of this
                    // block (line ~12055) already subtracts the multiplier
                    // mass of every row in `cached_active_sets`. If that
                    // residual is at tolerance, case (a) holds and the
                    // certificate is honest. If it's still orders of
                    // magnitude above tolerance, case (b) holds: certifying
                    // here would hand the unified evaluator a
                    // `kkt_residual` with norm ≈ ‖g‖ which then gets
                    // amplified by H⁻¹_proj in the cost/gradient IFT
                    // corrections, contaminating the envelope formula and
                    // triggering the "envelope-gradient consistency"
                    // tripwire downstream. Bail with `converged = false` so
                    // the outer optimizer rejects this ρ cleanly, exactly
                    // as it would on any other non-converged inner exit.
                    let cert_residual_factor = 1.0;
                    if matches!(
                        certificate_decision,
                        ConstrainedStationaryCertificate::Accept
                    ) {
                        log::info!(
                            "[PIRLS/joint-Newton convergence] cycle {:>3} | constrained-stationary certificate: \
                             linear-solve neutralised {:.1}% of g (the remaining {:.1}% is a Lagrange multiplier \
                             of the active constraint set, not an unresolved gradient); \
                             scalar Newton model agrees with reality to relerr={:.3e} (Hessian+gradient are correct \
                             at this β); projected residual={:.3e} ≤ {:.1}×tol={:.3e} (multipliers captured by active set); \
                             |Δobjective|={:.3e}, geometric_tail_bound={:.3e}, obj_tol={:.3e}; further cycles cannot reduce the \
                             multiplier mass and would reproduce this plateau indefinitely; \
                             active-set multiplier mass will be projected out of the KKT residual \
                             before the outer IFT correction is assembled",
                            cycle,
                            (1.0 - linearized_rel) * 100.0,
                            linearized_rel * 100.0,
                            scalar_model_relerr,
                            residual,
                            cert_residual_factor,
                            cert_residual_factor * residual_tol,
                            objective_change,
                            geometric_tail_bound.unwrap_or(objective_change),
                            objective_tol,
                        );
                        converged = true;
                        break;
                    }
                    // Penalty-null-space acceptance (gam#553). The phantom-
                    // multiplier refusal fires when the active-set-projected
                    // residual is above tolerance, but that residual can be
                    // confined to `ker(H_pen)` — the polynomial null space of a
                    // penalized smooth (TP / Bernstein trend) that the censored
                    // location-scale / custom-family data does not pin down in
                    // the time_transform / log_sigma channel. Along that
                    // direction there is neither curvature nor a constraint, so
                    // it is a genuinely free gauge direction and the iterate is
                    // stationary on the entire identifiable (range) subspace.
                    // The downstream outer IFT trace removes exactly this
                    // null-space component via the projected pseudo-inverse, so
                    // only a *range-space* residual biases the envelope gradient
                    // (the precise concern of the "do NOT soft-accept" note
                    // below). Accept iff the range-space residual is at
                    // tolerance — preserving outer-gradient correctness while no
                    // longer aborting a well-posed fit on a data-unconstrained
                    // null direction.
                    if let Some(range_residual) = projected_residual_range_space_inf(
                        &projected_residual_vec,
                        &joint_hessian_source,
                        &ranges,
                        &s_lambdas,
                        ridge,
                        options.ridge_policy,
                        total_p,
                    ) && range_residual <= cert_residual_factor * residual_tol
                    {
                        log::info!(
                            "[PIRLS/joint-Newton convergence] cycle {:>3} | penalty-null-space certificate (gam#553): \
                             total projected residual={:.3e} > tol={:.3e} but its range-space (curved-subspace) \
                             component={:.3e} ≤ {:.1}×tol={:.3e}; the remaining residual lies in the data-unconstrained \
                             penalty null space ker(H_pen) (a free polynomial-trend gauge direction, not a defect) and is \
                             projected out of the KKT residual by the outer IFT pseudo-inverse before the envelope \
                             correction; |Δobjective|={:.3e}, obj_tol={:.3e}",
                            cycle,
                            residual,
                            cert_residual_factor * residual_tol,
                            range_residual,
                            cert_residual_factor,
                            cert_residual_factor * residual_tol,
                            objective_change,
                            objective_tol,
                        );
                        converged = true;
                        break;
                    }
                    // Constrained exact-fixed-point acceptance (gam#797).
                    //
                    // We reach here only with the iterate ALREADY proven stationary
                    // (objective + step exhausted, `linearized_rel >= 0.5` so the
                    // residual is multiplier/null mass, `scalar_relerr <= 1e-3` so
                    // the quadratic model is exact), the strict/range-space/noise
                    // certificates having declined. For a CONSTRAINED block the
                    // remaining residual can be a genuine active-constraint Lagrange
                    // multiplier that the active-set QP under-identified (it reports
                    // only rows it drove tight during a non-degenerate step, so a
                    // monotone derivative-guard row tight at the optimum but never
                    // explicitly stepped is missing), leaving the cone projection
                    // unable to decompose `r = A_activeᵀ λ` and the residual stuck
                    // far above tol on an iterate that is EXACTLY the constrained
                    // optimum (the `active_set_incomplete` refusal; gam#797 survival
                    // marginal/logslope/time blocks).
                    //
                    // When (a) the joint Newton has reached a numerical FIXED POINT
                    // — the accepted step and objective change are both at the
                    // machine-epsilon floor relative to the iterate, so no further
                    // progress is mathematically possible — (b) the local quadratic
                    // model is exact (`scalar_relerr` tiny), and (c) the design
                    // carries linear inequality constraints AND `H_pen` has NO
                    // numerical null space (so the residual is an active-constraint
                    // multiplier, NOT an H-null/rank-deficient defect, which the
                    // range-space certificate above already handles), the iterate is
                    // a bona fide constrained KKT point. The active-constraint
                    // multiplier mass is projected out of the KKT residual by the
                    // unified evaluator's active-constraint-aware IFT correction
                    // before the envelope gradient, exactly as for an explicitly
                    // captured multiplier, so certifying here is correct. Gated
                    // strictly on a fixed point with no H-null, so a genuinely
                    // non-converged or rank-deficient iterate is never accepted.
                    let any_block_constrained = block_constraints.iter().any(|c| c.is_some());
                    let beta_scale = states
                        .iter()
                        .flat_map(|s| s.beta.iter().copied())
                        .map(f64::abs)
                        .fold(0.0_f64, f64::max)
                        .max(1.0);
                    let fixed_point_floor = 64.0 * f64::EPSILON * beta_scale;
                    let objective_floor = 64.0 * f64::EPSILON * (1.0 + lastobjective.abs());
                    let at_numerical_fixed_point = accepted_step_inf.is_finite()
                        && accepted_step_inf <= fixed_point_floor
                        && objective_change <= objective_floor
                        && scalar_model_relerr <= 1e-3;
                    if any_block_constrained && at_numerical_fixed_point {
                        // Materialize H_pen = H + S(λ) (+ model ridge) and count its
                        // numerical null space at the shared rank tolerance: nullity == 0
                        // ⇒ the stuck residual is NOT an H-null/rank-deficient defect
                        // (that case is handled by the range-space certificate above) but
                        // a genuine active-constraint multiplier.
                        let hpen_nullity = materialize_joint_hessian_source(
                            &joint_hessian_source,
                            total_p,
                            "constrained fixed-point nullity check",
                        )
                        .ok()
                        .map(|mut h_pen| {
                            let model_diagonal_ridge =
                                if options.ridge_policy.include_quadratic_penalty && ridge > 0.0 {
                                    ridge
                                } else {
                                    0.0
                                };
                            add_joint_penalty_to_matrix(
                                &mut h_pen,
                                &ranges,
                                &s_lambdas,
                                model_diagonal_ridge,
                                None,
                            );
                            symmetrize_dense_in_place(&mut h_pen);
                            symmetric_penalized_hessian_nullity(&h_pen)
                        })
                        .unwrap_or(None);
                        if hpen_nullity == Some(0) {
                            log::info!(
                                "[PIRLS/joint-Newton convergence] cycle {:>3} | constrained fixed-point certificate:                                  accepted_step_inf={:.3e} ≤ {:.3e} and |Δobjective|={:.3e} ≤ {:.3e} (numerical fixed point),                                  scalar_relerr={:.3e}, linearized_rel={:.3e}; H_pen has no numerical null space so the                                  residual={:.3e} is an active-constraint Lagrange multiplier (the QP under-identified the                                  binding rows), projected out of the KKT residual by the active-constraint-aware IFT                                  correction before the envelope gradient — the iterate is a constrained KKT point",
                                cycle,
                                accepted_step_inf,
                                fixed_point_floor,
                                objective_change,
                                objective_floor,
                                scalar_model_relerr,
                                linearized_rel,
                                residual,
                            );
                            converged = true;
                            break;
                        }
                    }
                    // Still-converging guard (gam#787 duchon centers≥20). The
                    // certificates above all declined, so the iterate would be
                    // refused as a multiplier/null plateau. But the
                    // `linearized_rel ≥ 0.5` + flat-objective signature that
                    // routed us here ALSO holds for a logslope block whose
                    // objective is already at its Φ-bounded floor while the KKT
                    // residual is still polishing by a STEADY geometric factor
                    // each cycle. Refusing there rejects the seed a few cycles
                    // short of `residual_tol` (→ outer seed-rejection → raise).
                    // If the residual is in steady geometric descent over the
                    // recent window, the direction is genuinely converging, not
                    // plateaued: keep iterating (bounded by the inner cycle cap)
                    // rather than refuse. The genuine plateau (flat/oscillating
                    // residual above tol) fails this test and refuses as before.
                    if residual_in_steady_geometric_descent(&residual_descent_history) {
                        log::info!(
                            "[PIRLS/joint-Newton convergence] cycle {:>3} | certificate declined but residual in steady geometric descent (history={:?}, residual={:.3e}, tol={:.3e}); continuing to convergence rather than refusing as a plateau",
                            cycle,
                            residual_descent_history,
                            residual,
                            residual_tol,
                        );
                        continue;
                    }
                    // EARLY-CYCLE CARVE-OUT (gam#826/#872). The phantom-multiplier
                    // refusal asserts that the residual is a captured Lagrange
                    // multiplier / H-null mass that Newton genuinely cannot move —
                    // a claim that requires EVIDENCE of a plateau. The candidate
                    // conditions above (objective + step exhausted, linearized_rel ≥
                    // 0.5) are ALSO satisfied transiently when a single Newton step
                    // is small because the augmented (Firth) curvature `H_Φ` is
                    // legitimately large in the `∇Φ` direction at an oversmoothed
                    // cycle-0 seed: the step `(H+Sλ+H_Φ)⁻¹(∇L−Sβ+∇Φ)` is tiny (high
                    // curvature ⇒ short step) and ONE step undershoots the
                    // nonquadratic Firth optimum, so `step_inf` and `|Δobj|` look
                    // exhausted while the residual is still O(‖∇Φ‖) ≫ tol. Refusing
                    // there at cycle 0 (no descent history yet) aborts the coupled
                    // binomial location-scale / flexible-linkwiggle fit before the
                    // inner has taken the handful of cycles it needs to walk the
                    // curved Firth basin to its optimum. When the residual is still
                    // ORDERS above tol and we lack a full descent window to prove a
                    // genuine plateau, keep iterating — the inner cycle cap and the
                    // residual-stall / trust-region-floor guards still bound the
                    // loop and diagnose a true non-convergence. A genuine multiplier
                    // plateau (residual flat across the window) is caught once the
                    // history fills, exactly as before. The threshold is the same
                    // `RESIDUAL_DESCENT_WINDOW` the descent test uses, so this only
                    // defers the refusal until there is enough history to make it,
                    // never weakens it.
                    let residual_far_above_tol = residual.is_finite()
                        && residual_tol.is_finite()
                        && residual > cert_residual_factor * residual_tol;
                    if residual_far_above_tol
                        && residual_descent_history.len() < RESIDUAL_DESCENT_WINDOW
                    {
                        log::info!(
                            "[PIRLS/joint-Newton convergence] cycle {:>3} | constrained-stationary refusal DEFERRED: residual={:.3e} ≫ tol={:.3e} but only {} descent samples (< {} window) — too early to prove a multiplier/null plateau vs a high-curvature Firth-basin transient; continuing",
                            cycle,
                            residual,
                            residual_tol,
                            residual_descent_history.len(),
                            RESIDUAL_DESCENT_WINDOW,
                        );
                        continue;
                    }
                    // UNCONSTRAINED MODEL-STATIONARY ACCEPTANCE (gam#826/#808/#715).
                    //
                    // The phantom-multiplier refusal asserts the residual is a
                    // captured Lagrange multiplier of an active constraint that
                    // the QP could not decompose. That diagnosis is categorically
                    // IMPOSSIBLE when there is no active constraint at all: a
                    // residual cannot be a phantom multiplier of a constraint that
                    // does not exist. For a fully UNCONSTRAINED coupled fit
                    // (multinomial softmax; the location-scale flat blocks) on a
                    // near-flat Fisher surface (`diag(p)−ppᵀ → 0`, or the
                    // high-curvature/low-curvature `log_sigma` block) the
                    // Firth-augmented stationarity residual `‖∇L−Sβ+∇Φ‖` floors
                    // LEGITIMATELY above `4·residual_tol`: the absolute curvature
                    // is tiny so `residual_tol = inner_tol·(1+grad/pen/firth)` is
                    // tiny too, yet the Newton/dogleg step exhausts before the
                    // residual drops below that band — `residual_tol` is scaled by
                    // the gradient magnitude and does not see the flat-Fisher
                    // absolute-curvature floor. The well-conditioned spectrum keeps
                    // the conditioning-keyed Levenberg gate (`COND_NEWTON_SAFETY`)
                    // off, so neither LM nor the cond-armed dogleg engages, and
                    // every seed is refused as `phantom_multiplier_with_well_
                    // conditioned_H`.
                    //
                    // When the model itself certifies stationarity — the standard
                    // trust-region "predicted decrease ≈ 0" criterion, here the
                    // `at_numerical_fixed_point` flag (accepted step at the
                    // machine-eps floor, |Δobj| at the eps floor, scalar model
                    // exact to relerr ≤ 1e-3) — AND no further progress is being
                    // made (the steady-geometric-descent test above declined) AND
                    // we have a full descent window (the early-cycle deferral above
                    // passed, so this is a proven plateau not a Firth-basin
                    // transient), an unconstrained iterate is a bona fide
                    // first-order optimum: the quadratic model says no step can
                    // reduce the residual further, and there is no constraint whose
                    // multiplier the residual could otherwise represent. The
                    // residual that remains lives where the model is flat
                    // (vanishing curvature), so it carries no `gᵀ∂β/∂ρ` envelope
                    // contribution the outer IFT could not already neutralise
                    // through its penalty-projected pseudo-inverse. Accept.
                    //
                    // This does NOT regress #729 (coupled Dirichlet): that fit
                    // converges to a genuine `residual < residual_tol` and exits
                    // via the strict KKT certificate long before this branch, and
                    // even if reached it has a curved (non-flat) Fisher surface so
                    // its model is not at a fixed point with a residual stuck above
                    // tol. It does NOT mask a real non-convergence: a still-moving
                    // iterate fails `at_numerical_fixed_point` (its step / |Δobj|
                    // are above the eps floor), and a rank-deficient H-null defect
                    // is the CONSTRAINED concern the fixed-point certificate above
                    // already handles via its nullity check.
                    // The certificate-candidate conditions that routed us into
                    // this block already PROVE model stationarity for the
                    // unconstrained case: `objective_exhausted` + `step_inf ≤
                    // step_tol` (the model's minimizer is at this β), `scalar_relerr
                    // ≤ 1e-3` (the quadratic model is exact), and `linearized_rel ≥
                    // 0.5` (‖g+Hδ‖ ≈ ‖g‖, so `Hδ ≈ 0` — the residual lives in the
                    // flat/near-null subspace of H, exactly a flat-Fisher direction
                    // for an unconstrained fit). We do NOT additionally require the
                    // far stricter machine-eps `at_numerical_fixed_point` here: on a
                    // flat Fisher surface the dogleg keeps taking a small step at
                    // the `step_tol` floor every cycle, so `accepted_step_inf` floors
                    // a hair above `64·eps·|β|` and the eps-fixed-point flag never
                    // sets even though the model is stationary. The `step_tol` floor
                    // (`inner_tol·(1+|β|∞)`) is the principled stationarity gate; the
                    // eps floor is for the constrained-multiplier certificate, where
                    // a tighter proof is warranted because a wrong accept biases the
                    // constraint-aware IFT kernel.
                    let any_active_set_rows = cached_active_sets
                        .iter()
                        .any(|maybe| maybe.as_ref().is_some_and(|rows| !rows.is_empty()));
                    let unconstrained_fit = !any_block_constrained && !any_active_set_rows;
                    if unconstrained_fit {
                        log::info!(
                            "[PIRLS/joint-Newton convergence] cycle {:>3} | unconstrained model-stationary certificate (gam#826/#808/#715): \
                             no active constraint (active_set_rows_total=0) so the residual={:.3e} cannot be a phantom multiplier; \
                             the iterate is a numerical fixed point (accepted_step_inf={:.3e}, |Δobjective|={:.3e}, scalar_relerr={:.3e}) \
                             on a flat Fisher surface where residual_tol={:.3e} sits below the absolute-curvature floor; \
                             linearized_rel={:.3e}, |Δobjective| exhausted and residual not in steady descent → genuine first-order optimum, accepting",
                            cycle,
                            residual,
                            accepted_step_inf,
                            objective_change,
                            scalar_model_relerr,
                            residual_tol,
                            linearized_rel,
                        );
                        converged = true;
                        break;
                    }
                    // Structured per-block + per-spectrum refusal report.
                    // The legacy one-line refusal log printed only aggregate
                    // numbers (linearized_rel, scalar_relerr, residual,
                    // |Δobj|) and was not actionable on models with many
                    // blocks: it could not identify WHICH smooth carried
                    // the unresolved mass, nor whether H_pen was genuinely
                    // rank-deficient (the "polynomial null space slipped
                    // past absorption" pathology). Cost: one dense
                    // materialize + symmetric eigh on H_pen at this β,
                    // sub-millisecond for typical p, executed once per
                    // refusal (the loop breaks immediately after).
                    let report = compute_kkt_refusal_report(
                        cycle,
                        &states,
                        specs,
                        &s_lambdas,
                        &ranges,
                        cached_joint_gradient.as_ref(),
                        &cached_active_sets,
                        &block_constraints,
                        Some(&joint_hessian_source),
                        total_p,
                        ridge,
                        options.ridge_policy,
                        accepted_step_inf,
                        step_inf,
                        joint_trust_radius,
                        residual_tol,
                        objective_tol,
                        step_tol,
                        objective_change,
                        residual,
                        Some(&math),
                    );
                    log::warn!(
                        "{}",
                        report.format_structured_log(cert_residual_factor * residual_tol)
                    );
                    last_kkt_refusal_report = Some(report);
                    converged = false;
                    break;
                }
            }

            // INVESTIGATION NOTE — do NOT soft-accept here.
            //
            // The outer objective is V(ρ) = f(β*(ρ), ρ), where β*(ρ)
            // satisfies g(β*,ρ)=∇_β f=0.  The envelope/IFT gradient used
            // by the outer optimizer is
            //
            //   dV/dρ_j = ∂f/∂ρ_j
            //
            // only at g=0.  At a non-stationary β, the actual chain rule is
            //
            //   d f(β(ρ),ρ)/dρ_j = ∂f/∂ρ_j + gᵀ ∂β/∂ρ_j.
            //
            // A soft certificate based only on small Δf discards the second
            // term without proving it is small.  The projected pseudo-inverse
            // in the outer trace path removes null-space components of g, but
            // any range-space component still contributes gᵀ∂β/∂ρ and gives
            // ARC/BFGS a biased outer gradient.  The `[PIRLS/JN/math]` line
            // above now prints the actual Newton identity:
            //
            //   old_kkt = ‖g‖∞,
            //   linearized_next = ‖g + Hδ‖∞ = ‖Hδ-rhs‖∞,
            //   new_kkt = ‖g(β+δ)‖∞,
            //   scalar_model relerr = |actual-pred|/max(1,|pred|).
            //
            // That is the proof surface. The diagnostic reports the measured
            // linear solve residual, post-step KKT residual, scalar model
            // error, and step sizes directly; downstream analysis should use
            // those numbers rather than this solver attaching labels.

            // Residual-stall early-exit. The strict and noise-floor
            // certificates above require the KKT residual to land within
            // a small multiple of residual_tol. On survival marginal-slope
            // at large scale the residual oscillates in a band that is
            // orders of magnitude above tol without trending down while
            // the unconstrained proposal has |prop|∞ in the 10³–10⁶ range,
            // the TR clamps it, and each clamped step moves β by O(1)
            // without driving ‖∇L − Sβ‖∞ closer to KKT.
            //
            // Spending the remaining cycle budget on this pattern hits
            // inner_max_cycles "non-converged", which then routes the
            // outer optimizer through the first-order bridge with a stale
            // same-ρ inner mode and a gradient of magnitude 10⁷ that kills
            // BFGS line search at iter 0 (the failure mode pinned in the
            // commit messages of 6578e884 and 1c181d1f).
            //
            // Track the best residual seen so far and the number of
            // cycles since any meaningful improvement (≥ 10 % drop). Once
            // the inner has burned at least RESIDUAL_STALL_MIN_CYCLES
            // without progress, the accepted step kept hitting the
            // trust-region clamp, AND every block is already inside a
            // loose stationarity band, return `converged = false` with
            // the current finite β. The per-block gate is essential for
            // block-metric trust regions: an aggregate residual plateau
            // dominated by one near-singular block must not hide an
            // unresolved marginal block that can still make progress under
            // its own radius.
            if residual.is_finite() {
                if residual < RESIDUAL_STALL_IMPROVEMENT_FACTOR * best_residual_seen {
                    best_residual_seen = residual;
                    cycles_since_residual_improved = 0;
                    tr_clamped_during_stall = false;
                } else {
                    cycles_since_residual_improved =
                        cycles_since_residual_improved.saturating_add(1);
                    if last_accepted_hit_joint_trust_boundary {
                        tr_clamped_during_stall = true;
                    }
                }
            }
            if cycle + 1 >= RESIDUAL_STALL_MIN_CYCLES
                && cycles_since_residual_improved >= RESIDUAL_STALL_NO_IMPROVE_CYCLES
                && tr_clamped_during_stall
                && range_projected_block_stationarity_small()
            {
                // Penalty-null-space certificate at the STALL exit (gam#1040).
                // The survival marginal-slope joint block carries free gauge
                // directions (the #892 flexible-regime warp family) with no
                // curvature and no constraint: the optimizer drifts along them
                // with zero objective change, the Newton step never shrinks to
                // step_tol (nothing pins it), so the constrained-stationary
                // certificate's step-exhausted precondition is UNSATISFIABLE
                // and every full-budget solve used to exit here unconverged —
                // the outer REML then rejects ρ-evaluation after ρ-evaluation
                // and cycles for hours (#1040: matern/duchon/measure-jet all
                // time out; binary-MS, which has no such direction, fits in
                // seconds). Stationarity on the identifiable subspace is the
                // honest convergence statement: if the projected residual's
                // component in the RANGE of H_pen is at tolerance, the stalled
                // mass lives in ker(H_pen) — exactly what the outer IFT
                // projects out before the envelope correction (gam#553) — and
                // the iterate is accepted. A residual with genuine range-space
                // mass (a real defect) still exits unconverged below.
                if objective_change <= objective_tol
                    && let Some(range_residual) = projected_residual_range_space_inf(
                        &projected_residual_vec,
                        &joint_hessian_source,
                        &ranges,
                        &s_lambdas,
                        ridge,
                        options.ridge_policy,
                        total_p,
                    )
                    && range_residual <= 4.0 * residual_tol
                {
                    log::info!(
                        "[PIRLS/joint-Newton convergence] cycle {:>3} | residual-stall range-space certificate (gam#1040): \
                         total projected residual={:.3e} > tol={:.3e} stalled for {} cycles, but its range-space component={:.3e} \
                         ≤ 4×tol={:.3e} and |Δobjective|={:.3e} ≤ obj_tol={:.3e}; the stalled mass is a free \
                         ker(H_pen) gauge direction the outer IFT pseudo-inverse projects out — accepting as stationary \
                         on the identifiable subspace.",
                        cycle,
                        residual,
                        residual_tol,
                        cycles_since_residual_improved,
                        range_residual,
                        4.0 * residual_tol,
                        objective_change,
                        objective_tol,
                    );
                    // Record the residual this exit actually certified on
                    // (#1040 inner-report truthfulness): the converged status is
                    // earned by `range_residual ≤ 4×tol` on the identifiable
                    // subspace, so the terminal line must report that finite
                    // certified residual — not the `inf` stall-tracker sentinel,
                    // which a cycle-1 certificate exit (head KKT non-finite, so
                    // the head-of-cycle `min` update was skipped) would otherwise
                    // leave unset, printing `converged=true … best_residual_inf=inf`.
                    if range_residual.is_finite() {
                        min_certified_residual = min_certified_residual.min(range_residual);
                    }
                    converged = true;
                    break;
                }
                let last_math_summary = last_joint_math
                    .as_ref()
                    .map(|math| {
                        format!(
                            "last_newton_math={{old_kkt={:.3e}, linearized_next={:.3e}, actual={:+.3e}, pred={:+.3e}, rho={:+.3e}, scalar_relerr={:.3e}, step_inf={:.3e}, proposal_inf={:.3e}}}",
                            math.old_kkt_inf,
                            math.linearized_next_kkt_inf,
                            math.actual_reduction,
                            math.predicted_reduction,
                            math.trust_ratio,
                            math.scalar_model_relative_error(),
                            math.step_inf,
                            math.proposal_inf,
                        )
                    })
                    .unwrap_or_else(|| "last_newton_math=<none>".to_string());
                log::warn!(
                    "[PIRLS/joint-Newton convergence] cycle {:>3} | residual-stall early-exit: residual={:.3e} best_seen={:.3e} no_improve_cycles={} accepted_step_inf={:.3e} trust_radius={:.3e} block_stationarity_inf={:?} {}; returning unconverged with finite β so the outer optimizer rejects this ρ evaluation before inner_max_cycles.",
                    cycle,
                    residual,
                    best_residual_seen,
                    cycles_since_residual_improved,
                    accepted_step_inf,
                    joint_trust_radius,
                    block_stationarity_norms,
                    last_math_summary,
                );
                // Record a structured KKT-refusal report at the stall iterate so
                // the bubbled IntegrationFailed error carries the per-block
                // residual breakdown + H_pen spectrum instead of the opaque
                // "no joint Newton math snapshot" string (gam#979/#1040). This is
                // the dominant non-convergence exit for the survival
                // marginal-slope monotone-cone DGP; without a report the cause of
                // the abort is invisible past serialization.
                cycles_done = cycle + 1;
                let report = compute_kkt_refusal_report(
                    cycle,
                    &states,
                    specs,
                    &s_lambdas,
                    &ranges,
                    cached_joint_gradient.as_ref(),
                    &cached_active_sets,
                    &block_constraints,
                    Some(&joint_hessian_source),
                    total_p,
                    ridge,
                    options.ridge_policy,
                    accepted_step_inf,
                    step_inf,
                    joint_trust_radius,
                    residual_tol,
                    objective_tol,
                    step_tol,
                    objective_change,
                    residual,
                    last_joint_math.as_ref(),
                );
                last_kkt_refusal_report = Some(report);
                converged = false;
                break;
            }

            // KKT convergence: small residual plus EITHER a small
            // Newton step (tight quadratic-rate convergence, lets β
            // polish to machine precision), confirmed stagnation
            // (`accepted_step_inf <= step_tol` AND `objective_change
            // <= objective_tol`, the rank-deficient null-mode case),
            // OR a stricter stationarity certificate where both the
            // residual and objective change are an additional factor of
            // `inner_tol` below their scale-aware tolerances. The last
            // branch is deliberately stricter than the public tolerance:
            // it handles machine-precision null directions where β can
            // still move by about `step_tol` but the KKT residual and
            // objective are already over-polished. Using objective
            // stagnation alone is not sufficient; the residual guard is
            // what preserves first-order correctness.
            let superconverged_residual_tol = inner_tol * residual_tol;
            let superconverged_objective_tol = inner_tol * objective_tol;
            let superconverged_stationarity = residual <= superconverged_residual_tol
                && objective_change <= superconverged_objective_tol;
            if residual <= residual_tol
                && (step_inf <= step_tol
                    || (accepted_step_inf <= step_tol && objective_change <= objective_tol)
                    || superconverged_stationarity)
            {
                log::info!(
                    "[JN-EXIT] cycle={cycle} reason=plateau_objective_flat residual={residual:.3e} residual_tol={residual_tol:.3e} obj_change={objective_change:.3e} objective_tol={objective_tol:.3e} consecutive_flat={} accepted_step_inf={accepted_step_inf:.3e} step_tol={step_tol:.3e}",
                    obj_flat_streak.streak(),
                );
                // This branch certifies on `residual ≤ residual_tol`; record it
                // so the terminal line reports the finite certified residual
                // rather than the `inf` stall sentinel (#1040 truthfulness).
                if residual.is_finite() {
                    min_certified_residual = min_certified_residual.min(residual);
                }
                converged = true;
                break;
            }
            // Scale-invariant objective-plateau exit (gam#1040). The flatness
            // predicate is RELATIVE — `objective_tol = inner_tol·(1+|obj|)` —
            // so it fires identically whether the survival NLL objective is
            // O(1) or O(1e4); a fixed absolute ε never trips at the ~6e4
            // magnitude of a marginal-slope survival fit. When the objective
            // has been relative-flat for the full `FlatStreak` window the
            // iterate has stopped moving in value. On a genuinely flat REML
            // valley along the weakly-identified time-wiggle ρ the Newton
            // step is tiny because the gradient is tiny (not because the
            // trust region truncated it), so the `tr_clamped_during_stall`
            // precondition of the residual-stall range-space certificate
            // above is UNSATISFIED and that exit never fires — the loop used
            // to grind to `inner_loop_hard_ceiling` every outer eval, which
            // is the #1040 hang (outer REML rejects ρ after ρ for hours).
            // The honest convergence statement is identical to the tr-clamped
            // path: if the projected residual's component in range(H_pen) is
            // at tolerance, the un-moved mass lives in ker(H_pen) — the free
            // gauge directions the outer IFT pseudo-inverse projects out
            // (gam#553) — and the iterate IS the REML optimum on the
            // identifiable subspace. Report converged.
            let plateau_verdict = obj_flat_streak.note(objective_change <= objective_tol);
            if plateau_verdict == crate::solver::loop_guard::LoopVerdict::Plateaued
                && range_projected_block_stationarity_small()
                && let Some(range_residual) = projected_residual_range_space_inf(
                    &projected_residual_vec,
                    &joint_hessian_source,
                    &ranges,
                    &s_lambdas,
                    ridge,
                    options.ridge_policy,
                    total_p,
                )
                && range_residual <= 4.0 * residual_tol
            {
                log::info!(
                    "[JN-EXIT] cycle={cycle} reason=relative_objective_plateau (gam#1040): \
                     |Δobjective|={objective_change:.3e} ≤ obj_tol={objective_tol:.3e} for {} \
                     consecutive cycles (scale-invariant rel-flat streak); total projected \
                     residual={residual:.3e} > tol={residual_tol:.3e} but its range-space \
                     component={range_residual:.3e} ≤ 4×tol={:.3e} — the un-moved mass is a free \
                     ker(H_pen) gauge direction the outer IFT projects out; accepting as stationary \
                     on the identifiable subspace.",
                    obj_flat_streak.streak(),
                    4.0 * residual_tol,
                );
                // Certified on `range_residual ≤ 4×tol`; record it so the
                // terminal report carries this finite certified residual
                // instead of the `inf` stall sentinel (#1040 truthfulness).
                if range_residual.is_finite() {
                    min_certified_residual = min_certified_residual.min(range_residual);
                }
                converged = true;
                break;
            }
            // Carry the KKT-stationarity / objective-stagnation signals
            // into the next cycle so the line-search-failure path above
            // can recognise a true KKT optimum on a rank-deficient null
            // mode. See that path for the full rationale.
            last_cycle_residual_below_tol = residual <= residual_tol;
            last_cycle_obj_change_below_tol = objective_change <= objective_tol;

            // Flat-residual stall early-exit (gam#1040/#979/#370/#859).
            //
            // The `tr_clamped_during_stall` residual-stall exit above only fires
            // when the accepted step kept hitting the trust-region boundary. A
            // distinct but equally terminal stall reaches neither it nor any
            // acceptance certificate: the KKT residual stops improving (no ≥10%
            // drop for the full `RESIDUAL_STALL_NO_IMPROVE_CYCLES` window) while
            // the accepted steps stay strictly INSIDE the trust region (so
            // `tr_clamped_during_stall` never latches) and the objective keeps
            // drifting just above `objective_tol` (so the relative-objective
            // plateau exit's flat streak never completes). This is the measured
            // "[joint-newton-tr] cycles 1000+" wall on the binomial location-scale
            // / bms-flex / CTN inner solves: without an exit the loop grinds the
            // remaining budget to `inner_loop_hard_ceiling` on every outer
            // ρ-evaluation, then hands the outer optimizer a non-converged result
            // anyway.
            //
            // Reaching this point means every acceptance certificate above already
            // DECLINED this cycle — the residual is above `residual_tol`, its
            // range-space component is above tolerance (so the iterate is NOT
            // stationary on the identifiable subspace), and there is no
            // constrained-multiplier signature. The honest action mirrors the
            // `tr_clamped` `converged=false` exit: stop and return the current
            // finite β as NON-converged so the outer optimizer rejects this ρ
            // cleanly. This is purely a termination/perf guard — it certifies
            // nothing (`converged=false`) and so cannot bias the envelope
            // gradient; it only rejects the same non-optimum sooner. The `≥10%
            // drop` reset of `cycles_since_residual_improved` keeps a
            // geometrically-descending solve (residual dropping by a steady factor
            // each cycle) from ever reaching the window — only a genuinely flat
            // residual does.
            if residual.is_finite()
                && residual > residual_tol
                && cycle + 1 >= RESIDUAL_STALL_MIN_CYCLES
                && cycles_since_residual_improved >= RESIDUAL_STALL_NO_IMPROVE_CYCLES
            {
                log::warn!(
                    "[PIRLS/joint-Newton convergence] cycle {:>3} | flat-residual stall early-exit (gam#1040): residual={:.3e} (tol={:.3e}) best_seen={:.3e} stalled {} cycles with steps inside the trust region (tr_clamped={}) and no acceptance certificate satisfied; the residual is neither trending toward KKT nor stationary on the identifiable subspace, so returning unconverged with finite β instead of grinding to inner_max_cycles={}.",
                    cycle,
                    residual,
                    residual_tol,
                    best_residual_seen,
                    cycles_since_residual_improved,
                    tr_clamped_during_stall,
                    inner_max_cycles,
                );
                cycles_done = cycle + 1;
                converged = false;
                break;
            }

            // NOTE: there is deliberately NO wall-clock-driven "adaptive
            // early-exit" here. A convergence verdict that fires when a cycle's
            // wall-clock happens to fall below a fraction of a running EMA is
            // non-deterministic — under CPU contention (a parallel sweep) the
            // same fit accepts at a different iterate than it does run alone,
            // which cascades into a different outer seed and a different
            // continuation-pre-warm fire/collapse decision (gam#979's
            // "collapses sequentially, fires in parallel" instability). It also
            // accepts iterates up to 10× outside the real KKT/objective
            // tolerance, biasing the REML/LAML criterion the inner residual
            // feeds. Convergence is certified ONLY by the mathematical tests
            // above (KKT residual / Newton step / objective change at their
            // scale-aware tolerances); whether convergence is *reachable within
            // the cycle budget* is judged by the deterministic descent-rate
            // guard alongside the residual-stall detector above.
        }

        // Explicit terminal verdict for the joint-Newton inner solve.
        //
        // The per-cycle `[PIRLS/JN] cyc=N/MAX … resid=… (tol=…)` line prints
        // the KKT/step/objective gaps at every cycle but never states which
        // criterion *terminated* the loop, so the final visible line on a
        // budget-exhausted solve looks identical to an ordinary mid-run cycle
        // (gam#744). A reader scanning a sweep log cannot tell a fit that
        // reached a stationary point from one that simply ran out of cycles
        // with the residual still orders of magnitude above tolerance and only
        // the objective stalled. Emit one authoritative line, on every exit
        // path, naming the terminating condition: `converged` is the honest
        // status the result carries downstream, `budget_exhausted` distinguishes
        // "ran the full cap" from an early certificate/divergence exit, and the
        // residual/step/objective stall flags say *why*. A budget-exhausted,
        // non-converged exit is logged at WARN so it is impossible to miss even
        // when per-cycle INFO is filtered out; a clean convergence is INFO.
        {
            let budget_exhausted = cycles_done >= inner_max_cycles;
            // Hard convergence-truthfulness invariant (#1040): a converged exit
            // is, by construction, certified on a finite stationarity residual
            // ≤ tol (every `converged = true` path above is gated on a finite
            // residual / range-space check and records it into
            // `min_certified_residual`). If — through any path — `converged` is
            // set without a finite certified residual on record, the solve has
            // NOT actually certified convergence; reporting `converged=true …
            // best_residual_inf=inf` is the self-contradicting status #1040
            // flags. The honest status is then non-converged: downgrade it so
            // the outer REML/LAML evaluation rejects this ρ rather than
            // consuming a phantom optimum certified on no finite residual.
            if !crate::solver::loop_guard::inner_convergence_is_truthful(
                converged,
                min_certified_residual,
            ) {
                log::warn!(
                    "[PIRLS/joint-Newton terminal] cycle {cycles_done}/{inner_max_cycles}: a converged \
                     exit fired without any finite certified stationarity residual on record \
                     (min_certified_residual is non-finite) — this would report \
                     converged=true with best_residual_inf=inf, a convergence-truthfulness \
                     violation (#1040). Downgrading to non-converged so the outer optimizer \
                     rejects this evaluation."
                );
                converged = false;
            }
            let terminator = if converged {
                "KKT/certificate-converged"
            } else if budget_exhausted {
                "budget-exhausted (max cycles reached)"
            } else {
                "early-exit non-converged (divergence/stall guard)"
            };
            // `solve_wall` (whole inner-solve elapsed) + `cycles` make the
            // per-solve cost explicit on ONE line: gam#979's "outer
            // multiplication" candidate is read off by counting these terminal
            // lines across a repro and summing their wall-times, and the
            // overhead candidate by comparing `solve_wall / cycles` against the
            // [joint-newton-tr] phase splits. Together with the per-cycle
            // `per_block_resid` (which block stalls) and the existing TR line
            // (ρ gain-ratio + decision: model infidelity vs TR throttling), a
            // single RUST_LOG=info run separates all four #979 candidates.
            //
            // Report `min_certified_residual` (the smallest stationarity residual
            // the solve actually computed) rather than the stall-tracker
            // `best_residual_seen`: the latter is only written at the post-step
            // residual site, so a head-of-cycle / pre-line-search certificate exit
            // (cycle-0 KKT exit on already-stationary data) left it at the sentinel
            // `inf` and the line read `converged=true … best_residual_inf=inf`, a
            // self-contradicting status (#1040 inner-report truthfulness). A
            // converged exit always certified on a finite residual ≤ tol, so the
            // reported residual is finite whenever `converged` (every converged=true
            // path is gated on a `≤ tol` check of a residual recorded above).
            let reported_residual_below_tol = last_cycle_residual_below_tol
                || (converged && min_certified_residual <= last_residual_tol);
            let verdict = format!(
                "[PIRLS/joint-Newton terminal] converged={} terminator={} cycles={}/{} \
                 solve_wall={:.3}s best_residual_inf={:.3e} (tol={:.3e}) last_residual_below_tol={} \
                 last_obj_change_below_tol={} objective={:.6e}; this is the status the inner \
                 solve reports to the outer REML/LAML evaluation — a non-converged exit \
                 (residual ≫ tol with only the objective stalled) is rejected, not accepted",
                converged,
                terminator,
                cycles_done,
                inner_max_cycles,
                inner_started.elapsed().as_secs_f64(),
                min_certified_residual,
                last_residual_tol,
                reported_residual_below_tol,
                last_cycle_obj_change_below_tol,
                lastobjective,
            );
            if converged {
                log::info!("{verdict}");
            } else {
                log::warn!("{verdict}");
            }
        }

        // If joint Newton converged, skip the blockwise loop entirely.
        if converged {
            let penalty_value = total_quadratic_penalty(
                &states,
                &s_lambdas,
                ridge,
                options.ridge_policy,
                joint_bundle,
                Some(specs),
            );
            let (block_logdet_h, block_logdet_s) = blockwise_logdet_terms_with_workspace(
                family,
                specs,
                &mut states,
                block_log_lambdas,
                options,
                cached_joint_workspace.clone(),
            )?;
            // The IFT/outer KKT residual must be the AUGMENTED stationarity
            // `∇L − Sβ + ∇Φ` the inner Newton actually drove to zero — NOT the bare
            // `∇L − Sβ`. With the Firth term armed, `∇L − Sβ = −∇Φ` at the
            // converged β, so the bare residual's null-space component equals ∇Φ
            // (O(‖∇Φ‖), e.g. 2.49 for the coupled Dirichlet). The outer evaluator's
            // range-projected IFT validity gate (`projected_into_reduced_range`)
            // then sees that ‖∇Φ‖ of "unresolved mass outside the reduced range"
            // and rejects EVERY seed at outer startup validation ("no candidate
            // seeds passed", gam#729/#715). Folding ∇Φ into the gradient makes the
            // residual the genuinely-near-zero augmented stationarity the inner
            // certified, so the gate passes. No-op when the term is
            // condition-gated/unavailable (∇Φ=0).
            let augmented_joint_gradient: Option<Array1<f64>> = match (
                cached_joint_gradient.as_ref(),
                joint_jeffreys_subspace.as_ref(),
            ) {
                (Some(gradient), Some(z_joint)) => {
                    match custom_family_joint_jeffreys_term(
                        family, &states, specs, &ranges, z_joint,
                    )? {
                        Some((_phi, grad_phi, _hphi)) if grad_phi.len() == gradient.len() => {
                            Some(gradient + &grad_phi)
                        }
                        _ => None,
                    }
                }
                _ => None,
            };
            let ift_gradient = augmented_joint_gradient
                .as_ref()
                .or(cached_joint_gradient.as_ref());
            let kkt_residual = exact_newton_joint_kkt_residual_for_ift_from_cached_gradient(
                family,
                specs,
                &states,
                &s_lambdas,
                ridge,
                options.ridge_policy,
                Some(cached_active_sets.as_slice()),
                ift_gradient,
            )?;
            let kkt_residual =
                require_projected_kkt_residual(kkt_residual, "joint-Newton converged exit")?;
            // Thread the cert tolerance + free subspace rank through to
            // the unified evaluator's certificate so the outer
            // optimiser's InnerStatus carrier sees honest numbers
            // instead of NaN / None.
            let active_set_rows_total: usize = cached_active_sets
                .iter()
                .map(|maybe| maybe.as_ref().map(|v| v.len()).unwrap_or(0))
                .sum();
            let free_rank_at_cert = total_p.saturating_sub(active_set_rows_total);
            let kkt_residual = kkt_residual.with_metadata(last_residual_tol, free_rank_at_cert);
            // Build the joint active-constraint block for the unified
            // evaluator's constraint-aware kernel
            // `K_T = K_S − K_S Aᵀ (A K_S Aᵀ)⁻¹ A K_S`. Returns `None` when
            // the family has no declared inequality constraints, or when
            // no rows are currently active at the cert point; in either
            // case the consumer-side `with_active_constraints` helper
            // degrades back to the bare penalty-projected pseudo-inverse.
            let active_constraints = {
                let block_constraints = collect_block_linear_constraints(family, &states, specs)?;
                assemble_active_constraint_block(
                    &block_constraints,
                    &cached_active_sets,
                    &ranges,
                    total_p,
                )
                .map(std::sync::Arc::new)
            };
            return Ok(BlockwiseInnerResult {
                block_states: states,
                active_sets: normalize_active_sets(cached_active_sets),
                log_likelihood: current_log_likelihood,
                penalty_value,
                cycles: cycles_done,
                converged,
                block_logdet_h,
                block_logdet_s,
                s_lambdas,
                joint_workspace: cached_joint_workspace.clone(),
                kkt_residual: Some(kkt_residual),
                active_constraints,
            });
        }
        if cycles_done >= inner_max_cycles {
            if !converged {
                // Engine-level diagnostic. Emit measured quantities only:
                // objective movement, coefficient scale, per-block dimensions,
                // per-block β and gradient scales, the unprojected stationarity
                // norm at exit, the Hessian source shape, and the last accepted
                // Newton identity diagnostics. The outer error path has no
                // access to these internals, so this line is the complete
                // numerical record needed to decide the next fix.
                let block_grad_norms: Vec<f64> = match cached_joint_gradient.as_ref() {
                    Some(joint_grad) => {
                        let mut acc = 0usize;
                        states
                            .iter()
                            .map(|s| {
                                let n = s.beta.len();
                                let end = (acc + n).min(joint_grad.len());
                                let nrm = if acc < end {
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
                let block_widths: Vec<usize> = states.iter().map(|s| s.beta.len()).collect();
                let block_beta_inf: Vec<f64> = states
                    .iter()
                    .map(|s| s.beta.iter().map(|x: &f64| x.abs()).fold(0.0_f64, f64::max))
                    .collect();
                let descent_total = initial_joint_objective - lastobjective;
                let beta_inf_final = states
                    .iter()
                    .flat_map(|s| s.beta.iter().copied())
                    .map(f64::abs)
                    .fold(0.0_f64, f64::max);
                let block_diag_default =
                    !family.exact_newton_joint_hessian_beta_dependent() && specs.len() >= 2;
                let exit_unprojected_kkt_inf = cached_joint_gradient
                    .as_ref()
                    .and_then(|joint_grad| {
                        exact_newton_joint_stationarity_vector_from_gradient(
                            joint_grad,
                            &states,
                            specs,
                            &s_lambdas,
                            ridge,
                            options.ridge_policy,
                        )
                        .ok()
                    })
                    .map(|residual| {
                        residual
                            .iter()
                            .map(|x: &f64| x.abs())
                            .fold(0.0_f64, f64::max)
                    })
                    .unwrap_or(f64::NAN);
                let last_math_summary = last_joint_math
                    .as_ref()
                    .map(|math| {
                        format!(
                            "last_newton_math={{old_kkt={:.3e}, linearized_next={:.3e}, actual={:+.3e}, pred={:+.3e}, rho={:+.3e}, scalar_relerr={:.3e}, step_inf={:.3e}, proposal_inf={:.3e}}}",
                            math.old_kkt_inf,
                            math.linearized_next_kkt_inf,
                            math.actual_reduction,
                            math.predicted_reduction,
                            math.trust_ratio,
                            math.scalar_model_relative_error(),
                            math.step_inf,
                            math.proposal_inf,
                        )
                    })
                    .unwrap_or_else(|| "last_newton_math=<none>".to_string());
                log::warn!(
                    "[PIRLS/joint-Newton] cycle={} budget-exhausted without KKT: objective_start={:.6e} objective_end={:.6e} objective_drop={:+.3e} beta_inf={:.3e} exit_unprojected_kkt_inf={:.3e} total_p={} total_n={} block_widths={:?} block_beta_inf={:?} block_grad_inf={:?} block_diag_hessian_default={} {}; rejecting this outer REML/LAML evaluation",
                    cycles_done,
                    initial_joint_objective,
                    lastobjective,
                    descent_total,
                    beta_inf_final,
                    exit_unprojected_kkt_inf,
                    total_p,
                    total_joint_n,
                    block_widths,
                    block_beta_inf,
                    block_grad_norms,
                    block_diag_default,
                    last_math_summary,
                );
                if coupled_exact_joint_required {
                    // Budget-exhaustion error MUST carry `block_residual_inf=…`
                    // so the carrying block survives the bubble through the
                    // outer optimiser. If no in-cycle cert refusal produced
                    // a structured report we build one here from the cached
                    // joint gradient + states. `joint_hessian_source` is
                    // per-cycle so the H_pen spectrum fields degrade to
                    // NaN/empty; per-block residual data is fully present.
                    let block_diag = if let Some(report) = last_kkt_refusal_report.as_ref() {
                        report.format_bubbled_error()
                    } else {
                        let block_constraints =
                            collect_block_linear_constraints(family, &states, specs)?;
                        let report = compute_kkt_refusal_report(
                            cycles_done,
                            &states,
                            specs,
                            &s_lambdas,
                            &ranges,
                            cached_joint_gradient.as_ref(),
                            &cached_active_sets,
                            &block_constraints,
                            None,
                            total_p,
                            ridge,
                            options.ridge_policy,
                            f64::NAN,
                            f64::NAN,
                            f64::NAN,
                            last_residual_tol,
                            f64::NAN,
                            f64::NAN,
                            f64::NAN,
                            exit_unprojected_kkt_inf,
                            last_joint_math.as_ref(),
                        );
                        report.format_bubbled_error()
                    };
                    return Err(format!(
                        "coupled exact-joint inner solve exhausted the joint Newton budget without KKT convergence after {cycles_done} cycle(s) — {block_diag}"
                    ));
                }
            }
            let penalty_value = total_quadratic_penalty(
                &states,
                &s_lambdas,
                ridge,
                options.ridge_policy,
                joint_bundle,
                Some(specs),
            );
            let (block_logdet_h, block_logdet_s) = blockwise_logdet_terms_with_workspace(
                family,
                specs,
                &mut states,
                block_log_lambdas,
                options,
                cached_joint_workspace.clone(),
            )?;
            let active_constraints = {
                let local_ranges = block_param_ranges(specs);
                let local_total_p = local_ranges.last().map(|(_, end)| *end).unwrap_or(0);
                let block_constraints = collect_block_linear_constraints(family, &states, specs)?;
                assemble_active_constraint_block(
                    &block_constraints,
                    &cached_active_sets,
                    &local_ranges,
                    local_total_p,
                )
                .map(std::sync::Arc::new)
            };
            return Ok(BlockwiseInnerResult {
                block_states: states,
                active_sets: normalize_active_sets(cached_active_sets),
                log_likelihood: current_log_likelihood,
                penalty_value,
                cycles: cycles_done,
                converged,
                block_logdet_h,
                block_logdet_s,
                s_lambdas,
                joint_workspace: cached_joint_workspace.clone(),
                kkt_residual: None,
                active_constraints,
            });
        }
        if coupled_exact_joint_required {
            // Bubble the structured KKT refusal report (per-block residual
            // breakdown + H_pen spectrum + diagnosis) so the cause of the
            // refusal survives serialization through the outer optimizer,
            // the seed-validation cascade, and gamfit. When the cert refused
            // inside the cycle loop we already computed a `KktRefusalReport`
            // at the refusing iterate; reuse it verbatim. If a different
            // early-exit path reaches this branch, build the same structured
            // report from the last Newton math snapshot rather than routing
            // through a second diagnostic string format.
            let block_diag = last_kkt_refusal_report
                .as_ref()
                .map(KktRefusalReport::format_bubbled_error)
                .unwrap_or_else(|| {
                    "structured KKT refusal report unavailable: no joint Newton math snapshot"
                        .to_string()
                });
            return Err(format!(
                "coupled exact-joint inner solve exited the joint Newton path before convergence — {block_diag}"
            ));
        }
        // Otherwise fall through to blockwise iteration below.
    }

    let mut cached_eval = match cached_eval {
        Some(eval) => eval,
        None => family.evaluate(&states)?,
    };
    lastobjective = -cached_eval.log_likelihood + current_penalty;

    // Divergence-detection state for the blockwise loop.
    //
    // Some family parameterizations (e.g. BernoulliMarginalSlopeFamily with
    // linkwiggle + scorewarp) carry a near-null direction in the joint
    // Hessian when the link-deviation basis's empirical anchor — fixed at
    // the rigid-pilot η₀ when the basis is constructed — drifts during
    // PIRLS as the location/spatial blocks update η₀. The Newton step
    // becomes dominated by that null direction and is clamped at
    // MAX_NEWTON_STEP every cycle while β grows linearly along it; the
    // log-likelihood stays frozen, only the penalty changes (slowly).
    // Without an early-exit the loop runs to inner_max_cycles producing
    // the same -loglik over and over, which at large scale (each cycle
    // ~0.5s) burns ~50s per ρ-cost call and stacks up to a 2400s timeout.
    //
    // Detect the pattern and bail with `converged = false` so the cost
    // call returns Err / +∞, BFGS κ-optim backs off the divergent ρ
    // region, and the outer loop progresses instead of grinding.

    // Per-block trust-region radius in the block's penalized-Hessian metric.
    // Updated each cycle by `update_joint_trust_region_radius` (the same
    // function the joint-Newton path uses) on a real model-vs-truth rho
    // computed from each block's penalized quadratic. Using the curvature
    // metric here avoids the same starvation mechanism fixed in the joint
    // path: one near-null coordinate in a block must not raw-rescale every
    // other coordinate in that block. The η-overflow safety half of the
    // previous static `MAX_NEWTON_STEP = 20.0` is owned by the family's
    // `max_feasible_step_size` barrier check, called by the line search below;
    // this variable handles only the algorithmic trust-region half. The
    // initial seed value is the family-declared safe step for a fresh fit; the
    // function then adapts it freely (clamped to [1e-12, 1e6] by the function
    // itself, same as the joint path).
    const BLOCK_NEWTON_STEP_INITIAL: f64 = 20.0;
    let mut block_max_step: Vec<f64> = vec![BLOCK_NEWTON_STEP_INITIAL; specs.len()];

    let mut prev_log_likelihood_for_divergence_check = cached_eval.log_likelihood;
    // Frozen-loglik streak rides the shared window discipline
    // (loop_guard::FlatStreak, #968); the frozen-loglik predicate and the
    // clamped-step side condition below stay local — they are policy about
    // what counts as flat, which this loop rightly owns.
    let mut frozen_loglik_streak =
        crate::solver::loop_guard::FlatStreak::new(DIVERGENCE_FROZEN_LOGLIK_CYCLES);
    // Coordinate descent visits each block in turn, so `max_proposed_step`
    // (the per-cycle max across blocks) only fires the cap on cycles where
    // the divergent block is the active one. On a near-null direction this
    // produces an alternation pattern (e.g. cap, cap, small, cap, small,
    // cap, …) and a strict "consecutive cycles where step is clamped"
    // requirement resets the counter every time another block's smaller
    // step dominates the per-cycle maximum. The frozen-loglik signal,
    // however, is a property of the joint state — it stays true across
    // every cycle of the alternation. Track frozen-loglik consecutively
    // and require that `step_clamped` was observed AT LEAST ONCE inside
    // the frozen run (rather than EVERY cycle).
    let mut clamped_step_in_frozen_run: bool = false;
    const DIVERGENCE_FROZEN_LOGLIK_CYCLES: usize = 8;

    let is_dynamic = family.block_geometry_is_dynamic();
    for cycle in 0..inner_max_cycles {
        // Fires at the top of each blockwise coordinate cycle so we can count
        // iterations from CI logs when a benchmark hangs inside the first
        // outer-eval. Emitted at info-level: same rationale as the joint-Newton
        // sibling above — silent-grind diagnosis without debug logs.
        log::info!(
            "[PIRLS/blockwise coord] cycle {:>3}/{} | -loglik {:.6e} | penalty {:.6e} | objective {:.6e}",
            cycle,
            inner_max_cycles,
            -cached_eval.log_likelihood,
            current_penalty,
            lastobjective,
        );
        let mut max_proposed_beta_step = 0.0_f64;
        let mut max_accepted_beta_step = 0.0_f64;
        let mut trust_boundary_hit_in_cycle = false;

        let mut objective_cycle_prev = lastobjective;
        // Reuse cached evaluation from end of previous cycle (or initial eval).
        // For dynamic families, the end-of-cycle evaluation is also reused here
        // instead of re-evaluating redundantly — the state hasn't changed since
        // the last cycle's final evaluate.
        let mut cycle_eval = std::mem::replace(
            &mut cached_eval,
            FamilyEvaluation {
                log_likelihood: 0.0,
                blockworking_sets: Vec::new(),
            },
        );
        if cycle_eval.blockworking_sets.len() != specs.len() {
            return Err(format!(
                "family returned {} block working sets, expected {}",
                cycle_eval.blockworking_sets.len(),
                specs.len()
            ));
        }
        // Track whether any block was modified this cycle (for dynamic families,
        // we only need to re-evaluate before block b if a previous block changed).
        let mut any_block_modified = false;
        for b in 0..specs.len() {
            if is_dynamic && any_block_modified {
                // Only re-evaluate if a previous block in this cycle actually
                // modified coefficients. Skips the redundant evaluate for the
                // first block (b=0) since cached_eval is still valid.
                refresh_all_block_etas(family, specs, &mut states)?;
                cycle_eval = family.evaluate(&states)?;
                if cycle_eval.blockworking_sets.len() != specs.len() {
                    return Err(format!(
                        "family returned {} block working sets, expected {}",
                        cycle_eval.blockworking_sets.len(),
                        specs.len()
                    ));
                }
            }

            let spec = &specs[b];
            let work = &cycle_eval.blockworking_sets[b];
            let linear_constraints = family.block_linear_constraints(&states, b, spec)?;
            let s_lambda = &s_lambdas[b];
            let updater = work.updater();
            let update = updater.compute_update_step(&BlockUpdateContext {
                family,
                states: &states,
                spec,
                block_idx: b,
                s_lambda,
                options,
                linear_constraints: linear_constraints.as_ref(),
                cached_active_set: cached_active_sets[b].as_deref(),
            })?;
            if let Some(active_set) = update.active_set {
                cached_active_sets[b] = Some(active_set);
            }
            let beta_new_raw = update.beta_new_raw;
            let beta_new = family.post_update_block_beta(&states, b, spec, beta_new_raw.clone())?;
            reject_constrained_post_update_repair(
                b,
                spec,
                &beta_new_raw,
                &beta_new,
                linear_constraints.as_ref(),
            )?;
            let beta_old = states[b].beta.clone();
            let raw_delta = &beta_new - &beta_old;
            // Per-block trust-region radius in the block's local
            // penalized-Hessian metric. The cap is the current value of
            // `block_max_step[b]`, updated below via
            // `update_joint_trust_region_radius` once we know rho.
            let block_cap = block_max_step[b];
            let (delta, step_metric_norm) = truncate_block_step_to_metric_radius(
                spec,
                work,
                s_lambda,
                raw_delta,
                block_cap,
                ridge,
                options.ridge_policy,
            )?;
            let step_hit_trust_boundary =
                joint_block_step_hit_trust_boundary(step_metric_norm, block_cap);
            trust_boundary_hit_in_cycle |= step_hit_trust_boundary;
            // Capture the objective at the start of this block update so
            // we can compute the true `actual_reduction` once the line
            // search has finished. `objective_cycle_prev` is the running
            // total: it advances inside the line search whenever a trial
            // is accepted, so we must snapshot it here.
            let obj_before_block = objective_cycle_prev;
            let old_block_penalty =
                block_quadratic_penalty(&beta_old, s_lambda, ridge, options.ridge_policy);
            let step_beta_inf = delta.iter().copied().map(f64::abs).fold(0.0, f64::max);
            max_proposed_beta_step = max_proposed_beta_step.max(step_beta_inf);
            if step_beta_inf <= inner_tol {
                continue;
            }

            // Damped update: require non-increasing penalized objective under dynamic geometry.
            // Precompute X * delta once so line-search eta updates are O(n) not O(np).
            // Reuse pre-allocated eta backup to avoid O(n) allocation per block per cycle.
            let eta_checkpoint = BlockEtaCheckpoint::capture_reuse(&states[b], &mut eta_backups[b]);
            let x_delta = if !is_dynamic {
                Some(spec.solver_design().matrixvectormultiply(&delta))
            } else {
                None
            };
            let mut accepted = false;
            // Barrier-aware step ceiling: families with natural log-barrier
            // terms (e.g. log(h') in transformation-normal) report the maximum
            // feasible step fraction so the line search never evaluates the
            // likelihood outside its domain.
            let barrier_ceiling = family
                .max_feasible_step_size(&states, b, &delta)?
                .unwrap_or(1.0);
            // Reuse trial_beta_buf to avoid allocation per backtracking trial.
            let mut trial_beta_buf = beta_old.clone();
            let mut accepted_bt: usize = usize::MAX;
            for bt in 0..8 {
                let alpha = (0.5f64.powi(bt)).min(barrier_ceiling);
                trial_beta_buf.assign(&beta_old);
                trial_beta_buf.scaled_add(alpha, &delta);
                let trial_beta =
                    family.post_update_block_beta(&states, b, spec, trial_beta_buf.clone())?;
                reject_constrained_post_update_repair(
                    b,
                    spec,
                    &trial_beta_buf,
                    &trial_beta,
                    linear_constraints.as_ref(),
                )?;
                states[b].beta = trial_beta;
                // Use precomputed X*delta when geometry is static and beta wasn't modified.
                if let Some(ref xd) = x_delta {
                    if states[b].beta == trial_beta_buf {
                        eta_checkpoint.restore_eta_with_step(&mut states[b], alpha, xd);
                    } else {
                        refresh_single_block_eta(family, specs, &mut states, b)?;
                    }
                } else {
                    refresh_single_block_eta(family, specs, &mut states, b)?;
                }
                let trial_block_penalty =
                    block_quadratic_penalty(&states[b].beta, s_lambda, ridge, options.ridge_policy);
                let trial_penalty = current_penalty - old_block_penalty + trial_block_penalty;
                let line_search_options = coefficient_line_search_options(
                    options,
                    objective_cycle_prev - trial_penalty + 1e-10,
                );
                let trial_ll =
                    match family.log_likelihood_only_with_options(&states, &line_search_options) {
                        Ok(value) => value,
                        Err(_) => {
                            states[b].beta.assign(&beta_old);
                            eta_checkpoint.restore_eta(&mut states[b]);
                            continue;
                        }
                    };
                let trialobjective = -trial_ll + trial_penalty;
                if trialobjective.is_finite() && trialobjective <= objective_cycle_prev + 1e-10 {
                    objective_cycle_prev = trialobjective;
                    current_penalty = trial_penalty;
                    accepted = true;
                    accepted_bt = bt as usize;
                    break;
                }
            }
            // Trust-region update for this block, using the same
            // `update_joint_trust_region_radius` strategy the
            // joint-Newton path uses. Predicted reduction is computed
            // from the per-block penalized quadratic model:
            //
            //   Q(β + αδ) ≈ Q(β) − α·rhs·δ + 0.5·α²·δ·H_pen·δ
            //   predicted_reduction(α) = α·(rhs·δ) − 0.5·α²·(δ·H_pen·δ)
            //
            // where `rhs = score − S·β (− ridge·β)` is the penalized
            // gradient (in maximize-direction) and `H_pen = H + S
            // (+ ridge·I)` is the penalized observed information.
            // Actual reduction is the true penalized objective change
            // measured by the line search; rho = actual / predicted is
            // the standard model-vs-truth ratio that drives the same
            // 0.25 / 0.75 grow-shrink rules `update_joint_trust_region_radius`
            // already implements for the joint path.
            let alpha_accepted = if accepted {
                0.5_f64.powi(accepted_bt as i32)
            } else {
                0.0
            };
            let (rhs_block, hpen_delta_full): (Array1<f64>, Array1<f64>) = match work {
                BlockWorkingSet::ExactNewton { gradient, .. } => {
                    let mut rhs = gradient - &s_lambda.dot(&beta_old);
                    if options.ridge_policy.include_quadratic_penalty && ridge > 0.0 {
                        rhs.scaled_add(-ridge, &beta_old);
                    }
                    let hpen = block_penalized_hessian_vector(
                        spec,
                        work,
                        s_lambda,
                        &delta,
                        ridge,
                        options.ridge_policy,
                    );
                    (rhs, hpen)
                }
                BlockWorkingSet::Diagonal {
                    working_response,
                    working_weights,
                } => {
                    // IRLS local-quadratic gradient and Hessian:
                    //   rhs = X^T W (z − Xβ) − Sβ
                    //   H_pen δ = X^T W X δ + Sδ
                    let solver_design = spec.solver_design();
                    let xb = solver_design.matrixvectormultiply(&beta_old);
                    let resid = working_response - &xb;
                    let w_resid = &resid * working_weights;
                    let mut rhs = solver_design.transpose_vector_multiply(&w_resid);
                    rhs -= &s_lambda.dot(&beta_old);
                    if options.ridge_policy.include_quadratic_penalty && ridge > 0.0 {
                        rhs.scaled_add(-ridge, &beta_old);
                    }
                    let hpen = block_penalized_hessian_vector(
                        spec,
                        work,
                        s_lambda,
                        &delta,
                        ridge,
                        options.ridge_policy,
                    );
                    (rhs, hpen)
                }
            };
            let rhs_dot_delta = rhs_block.dot(&delta);
            let delta_dot_hpen = delta.dot(&hpen_delta_full);
            let predicted_reduction = alpha_accepted * rhs_dot_delta
                - 0.5 * alpha_accepted * alpha_accepted * delta_dot_hpen;
            let actual_reduction = obj_before_block - objective_cycle_prev;
            let trust_update = update_joint_trust_region_radius(
                block_max_step[b],
                alpha_accepted * step_metric_norm,
                actual_reduction,
                predicted_reduction,
                obj_before_block,
            );
            block_max_step[b] = trust_update.radius;
            if !accepted {
                states[b].beta.assign(&beta_old);
                eta_checkpoint.restore_eta(&mut states[b]);
                if let BlockWorkingSet::ExactNewton { gradient, .. } = work {
                    let mut raw_descent = gradient - &s_lambda.dot(&beta_old);
                    if options.ridge_policy.include_quadratic_penalty && ridge > 0.0 {
                        raw_descent -= &beta_old.mapv(|v| ridge * v);
                    }
                    let (descent_dir, descent_metric_norm) = truncate_block_step_to_metric_radius(
                        spec,
                        work,
                        s_lambda,
                        raw_descent,
                        block_cap,
                        ridge,
                        options.ridge_policy,
                    )?;
                    trust_boundary_hit_in_cycle |=
                        joint_block_step_hit_trust_boundary(descent_metric_norm, block_cap);
                    let dir_norm = descent_dir.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
                    if dir_norm > inner_tol {
                        // Precompute X * descent_dir once for incremental eta updates.
                        let x_descent = if !is_dynamic {
                            Some(spec.solver_design().matrixvectormultiply(&descent_dir))
                        } else {
                            None
                        };
                        let descent_barrier_ceiling = family
                            .max_feasible_step_size(&states, b, &descent_dir)?
                            .unwrap_or(1.0);
                        for bt in 0..12 {
                            let alpha = (0.5f64.powi(bt)).min(descent_barrier_ceiling);
                            trial_beta_buf.assign(&beta_old);
                            trial_beta_buf.scaled_add(alpha, &descent_dir);
                            let trial_beta = family.post_update_block_beta(
                                &states,
                                b,
                                spec,
                                trial_beta_buf.clone(),
                            )?;
                            reject_constrained_post_update_repair(
                                b,
                                spec,
                                &trial_beta_buf,
                                &trial_beta,
                                linear_constraints.as_ref(),
                            )?;
                            states[b].beta = trial_beta;
                            if let Some(ref xd) = x_descent {
                                if states[b].beta == trial_beta_buf {
                                    eta_checkpoint.restore_eta_with_step(&mut states[b], alpha, xd);
                                } else {
                                    refresh_single_block_eta(family, specs, &mut states, b)?;
                                }
                            } else {
                                refresh_single_block_eta(family, specs, &mut states, b)?;
                            }
                            let trial_block_penalty = block_quadratic_penalty(
                                &states[b].beta,
                                s_lambda,
                                ridge,
                                options.ridge_policy,
                            );
                            let trial_penalty =
                                current_penalty - old_block_penalty + trial_block_penalty;
                            let line_search_options = coefficient_line_search_options(
                                options,
                                objective_cycle_prev - trial_penalty + 1e-10,
                            );
                            let trial_ll = match family
                                .log_likelihood_only_with_options(&states, &line_search_options)
                            {
                                Ok(value) => value,
                                Err(_) => {
                                    states[b].beta.assign(&beta_old);
                                    eta_checkpoint.restore_eta(&mut states[b]);
                                    continue;
                                }
                            };
                            let trialobjective = -trial_ll + trial_penalty;
                            if trialobjective.is_finite()
                                && trialobjective <= objective_cycle_prev + 1e-10
                            {
                                objective_cycle_prev = trialobjective;
                                current_penalty = trial_penalty;
                                accepted = true;
                                break;
                            }
                            states[b].beta.assign(&beta_old);
                            eta_checkpoint.restore_eta(&mut states[b]);
                        }
                    }
                }
            }
            if !accepted {
                states[b].beta.assign(&beta_old);
                eta_checkpoint.restore_eta(&mut states[b]);
            } else {
                let accepted_step = states[b]
                    .beta
                    .iter()
                    .zip(beta_old.iter())
                    .map(|(new, old)| (new - old).abs())
                    .fold(0.0_f64, f64::max);
                max_accepted_beta_step = max_accepted_beta_step.max(accepted_step);
                any_block_modified = true;
            }
            // Recycle the checkpoint's buffer back into the pre-allocated pool.
            eta_backups[b] = eta_checkpoint.into_buffer();
        }

        // For non-dynamic families, incremental eta updates within the block loop
        // maintain correct etas. Only refresh from scratch for dynamic-geometry families
        // where block interactions may require recomputation.
        if is_dynamic {
            refresh_all_block_etas(family, specs, &mut states)?;
        }
        cached_eval = family.evaluate(&states)?;
        current_penalty = total_quadratic_penalty(
            &states,
            &s_lambdas,
            ridge,
            options.ridge_policy,
            joint_bundle,
            Some(specs),
        );
        let objective = -cached_eval.log_likelihood + current_penalty;
        let objective_change = (objective - lastobjective).abs();
        lastobjective = objective;
        cycles_done = cycle + 1;

        // Divergence guard (mirrors the joint-Newton sibling, gam#554): a
        // non-finite objective / log-likelihood means a near-unidentified
        // penalized block has propagated NaN mass through the coordinate
        // descent. Every convergence and divergence-frozen exit below is a
        // finite `<=` comparison that NaN silently defeats, so without this
        // the loop grinds the full `inner_max_cycles` on every outer ρ-eval
        // and startup seed. Break unconverged so the outer optimizer rejects
        // this point immediately instead of burning the budget.
        if !objective.is_finite() || !cached_eval.log_likelihood.is_finite() {
            log::warn!(
                "[PIRLS/blockwise convergence] cycle {:>3} | divergence guard: non-finite inner state (objective={:.3e}, -loglik={:.3e}); returning unconverged so the outer optimizer rejects this ρ evaluation instead of running to inner_max_cycles.",
                cycle,
                objective,
                -cached_eval.log_likelihood,
            );
            converged = false;
            break;
        }

        // Scale-aware tolerances — see the matching joint-Newton path
        // above for the rationale. At large scale absolute step/residual
        // tolerances against `inner_tol = 1e-6` keep this loop spinning
        // long after the objective has gone flat.
        let beta_inf = states
            .iter()
            .flat_map(|s| s.beta.iter().copied())
            .map(f64::abs)
            .fold(0.0_f64, f64::max);
        let step_tol = inner_tol * (1.0 + beta_inf);
        let objective_tol = inner_tol * (1.0 + objective.abs());
        let residual_tol = objective_tol;
        // For single-block models the blockwise iteration IS the joint
        // iteration, so block-conditional convergence implies joint
        // convergence.  The exact_newton_joint_stationarity check can
        // stall at ~10x the tolerance due to numerical differences
        // between the block-conditional and joint gradient formulations,
        // causing 100s of wasted cycles on an already-converged solution.
        let exact_joint_stationarity_ok = if has_joint_exacthessian && specs.len() >= 2 {
            exact_newton_joint_stationarity_inf_norm(
                family,
                specs,
                &cached_eval,
                &states,
                &s_lambdas,
                ridge,
                options.ridge_policy,
                None,
            )?
            .map(|residual| residual <= residual_tol)
            .unwrap_or(true)
        } else {
            true
        };
        log::info!(
            "[PIRLS/blockwise convergence] cycle {:>3} | max_proposed_step={:.3e} (tol={:.3e}) | max_accepted_step={:.3e} | obj_change={:.3e} (tol={:.3e}) | beta_inf={:.3e} | joint_stationarity_ok={}",
            cycle,
            max_proposed_beta_step,
            step_tol,
            max_accepted_beta_step,
            objective_change,
            objective_tol,
            beta_inf,
            exact_joint_stationarity_ok,
        );

        // Divergence early-exit. See the rationale block at the top of
        // this loop. We treat "log-likelihood unchanged + Newton step
        // pinned at the trust-region cap" as a near-null direction
        // signature and break out unconverged once it persists for
        // DIVERGENCE_FROZEN_LOGLIK_CYCLES consecutive iterations. Tracking
        // log-likelihood (not objective) is essential: when the null mode
        // dominates, only the penalty drifts cycle-to-cycle, so
        // `objective_change` stays above tol while -loglik is genuinely
        // frozen.
        let loglik_change_for_divergence_check =
            (cached_eval.log_likelihood - prev_log_likelihood_for_divergence_check).abs();
        let loglik_frozen_tol_for_divergence_check =
            inner_tol * (1.0 + cached_eval.log_likelihood.abs());
        let step_clamped_for_divergence_check = trust_boundary_hit_in_cycle;
        let loglik_frozen =
            loglik_change_for_divergence_check <= loglik_frozen_tol_for_divergence_check;
        let frozen_verdict = frozen_loglik_streak.note(loglik_frozen);
        if loglik_frozen {
            if step_clamped_for_divergence_check {
                clamped_step_in_frozen_run = true;
            }
        } else {
            clamped_step_in_frozen_run = false;
        }
        prev_log_likelihood_for_divergence_check = cached_eval.log_likelihood;
        if frozen_verdict == crate::solver::loop_guard::LoopVerdict::Plateaued
            && clamped_step_in_frozen_run
        {
            log::warn!(
                "[PIRLS/blockwise convergence] divergence early-exit at cycle {} | -loglik={:.6e} frozen for {} consecutive cycles | max_proposed_step={:.3e} (trust-boundary hit observed in frozen run) | step_tol={:.3e}; near-null Hessian direction detected — returning unconverged so the outer optimizer backs off this region instead of running to inner_max_cycles.",
                cycle,
                -cached_eval.log_likelihood,
                frozen_loglik_streak.streak(),
                max_proposed_beta_step,
                step_tol,
            );
            converged = false;
            break;
        }

        // NOTE: there is deliberately NO wall-clock-driven "adaptive
        // early-exit" here — the same discipline the joint-Newton sibling loop
        // documents above. A verdict that fires when a cycle's wall-clock falls
        // below a fraction of a running EMA is non-deterministic: under CPU
        // contention (a parallel sweep) the same fit accepts at a different
        // iterate than it does run alone, and it accepts iterates up to 10×
        // outside the real KKT/objective tolerance, biasing the REML/LAML
        // criterion the inner residual feeds. Convergence is certified ONLY by
        // the exact stationarity gate below.
        if max_accepted_beta_step <= step_tol && objective_change <= objective_tol {
            if exact_joint_stationarity_ok || max_proposed_beta_step <= step_tol {
                converged = true;
            }
            break;
        }
    }

    // ── Polishing joint Newton step ──
    //
    // For block-coupled multi-block families (e.g. GAMLSS wiggle), Gauss-Seidel
    // blockwise iteration can reach step_inf < inner_tol while the joint KKT
    // residual (||Sβ − grad_ℓ||_∞) remains at ~10× inner_tol. This is because
    // each block is solved conditionally on other blocks' current values —
    // block-conditional stationarity does not imply joint stationarity when
    // the likelihood couples blocks off-diagonally.
    //
    // Once blockwise has placed β near the true joint optimum, a single (or
    // a few) damped joint Newton steps can tighten the joint residual to the
    // floor set by β magnitudes. This polishing phase is essential for the
    // outer REML gradient formula (which assumes exact β̂ stationarity); a
    // non-converged β̂ produces large envelope-theorem violations in the
    // analytic outer gradient.
    if use_joint_newton && !converged {
        polish_joint_newton_step(
            family,
            specs,
            options,
            &s_lambdas,
            ridge,
            joint_bundle,
            inner_tol,
            &cached_active_sets,
            &mut states,
            &mut cached_eval,
            &mut current_penalty,
            &mut converged,
        )?;
    }

    assemble_inner_blockwise_result(
        family,
        specs,
        states,
        block_log_lambdas,
        options,
        s_lambdas,
        ridge,
        joint_bundle,
        cached_active_sets,
        &cached_eval,
        converged,
        cycles_done,
        last_residual_tol,
    )
}

/// Polishing joint-Newton step for the blockwise fall-through path of
/// [`inner_blockwise_fit`].
///
/// For block-coupled multi-block families (e.g. GAMLSS wiggle), Gauss-Seidel
/// blockwise iteration can reach `step_inf < inner_tol` while the joint KKT
/// residual (`||Sβ − grad_ℓ||_∞`) remains at ~10× `inner_tol`. Once blockwise
/// has placed β near the joint optimum, a few damped joint-Newton steps tighten
/// the joint residual to the floor set by β magnitudes; this is essential for the
/// outer REML gradient formula (which assumes exact β̂ stationarity).
///
/// Behavior is identical to the inline loop it replaced: the `?`-propagation, the
/// per-iteration `break` exits (gradient/Hessian unavailable, non-finite delta,
/// solver failure, residual-tolerance reached, line-search failure) and the
/// inner backtracking-search `continue` are preserved verbatim. Mutates `states`,
/// `cached_eval`, `current_penalty`, and `converged` in place exactly as before.
pub(crate) fn polish_joint_newton_step<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    s_lambdas: &[Array2<f64>],
    ridge: f64,
    joint_bundle: Option<&crate::families::joint_penalty::JointPenaltyBundle>,
    inner_tol: f64,
    cached_active_sets: &[Option<Vec<usize>>],
    states: &mut Vec<ParameterBlockState>,
    cached_eval: &mut FamilyEvaluation,
    current_penalty: &mut f64,
    converged: &mut bool,
) -> Result<(), String> {
    let ranges_joint: Vec<(usize, usize)> = {
        let mut offset = 0;
        specs
            .iter()
            .map(|s| {
                let start = offset;
                offset += s.design.ncols();
                (start, offset)
            })
            .collect()
    };
    let total_p_joint: usize = ranges_joint.last().map_or(0, |r| r.1);
    let joint_mode_diagonal_ridge = if ridge > 0.0 && options.ridge_policy.include_quadratic_penalty
    {
        ridge
    } else {
        0.0
    };
    let trace_diagonal_ridge = joint_mode_diagonal_ridge + JOINT_TRACE_STABILITY_RIDGE;

    // Allow up to a few polishing steps. The blockwise endpoint is close
    // to optimum, so step sizes should be small and line search should
    // accept full steps quickly.
    const POLISH_MAX_ITER: usize = 16;
    for _polish_iter in 0..POLISH_MAX_ITER {
        // Re-evaluate at current β to get the joint gradient and Hessian.
        refresh_all_block_etas(family, specs, states)?;
        let eval_for_polish = family.evaluate(states)?;
        let grad_full =
            match exact_newton_joint_gradient_from_eval(&eval_for_polish, specs, states)? {
                Some(g) => g,
                None => break,
            };
        // Spec-aware joint Hessian: canonical coupled-curvature source
        // (see the joint-Newton availability gate). Families overriding
        // only `_with_specs` return `None` from the spec-less default.
        let h_joint_opt = family.exact_newton_joint_hessian_with_specs(states, specs)?;
        let Some(h_joint) = h_joint_opt else { break };
        let mut h_dense = match symmetrized_square_matrix(
            h_joint,
            total_p_joint,
            "joint polish Hessian shape mismatch",
        ) {
            Ok(matrix) => matrix,
            Err(_) => break,
        };
        let h_unpenalized_dense = h_dense.clone();
        add_joint_penalty_to_matrix(
            &mut h_dense,
            &ranges_joint,
            s_lambdas,
            trace_diagonal_ridge,
            joint_bundle,
        );
        let joint_polish_diagonal_ridge = stabilized_joint_solver_diagonal_ridge(
            family,
            &JointHessianSource::Dense(h_unpenalized_dense),
            &ranges_joint,
            s_lambdas,
            trace_diagonal_ridge,
            options.ridge_floor,
            joint_bundle,
        );
        if joint_polish_diagonal_ridge != trace_diagonal_ridge {
            for d in 0..h_dense.nrows() {
                h_dense[[d, d]] += joint_polish_diagonal_ridge - trace_diagonal_ridge;
            }
        }

        let mut beta_joint = Array1::<f64>::zeros(total_p_joint);
        for b in 0..specs.len() {
            let (start, end) = ranges_joint[b];
            beta_joint
                .slice_mut(ndarray::s![start..end])
                .assign(&states[b].beta);
        }
        let penalty_beta = apply_joint_block_penalty(
            &ranges_joint,
            s_lambdas,
            &beta_joint,
            joint_mode_diagonal_ridge,
            joint_bundle,
        );
        let rhs = &grad_full - &penalty_beta;

        // Respect constraints that block line search on the boundary.
        // Gauss-Seidel blockwise leaves the joint KKT residual at a floor
        // around |λ_k S_k β̂| for boundary-active components. The residual
        // magnitude on FREE components is a better measure of whether we
        // should keep polishing: if β_i is clipped at the boundary and
        // KKT multiplier μ_i > 0, then rhs[i] is the multiplier, not a
        // free-space gradient violation.
        let block_constraints_now = collect_block_linear_constraints(family, states, specs)?;
        let joint_constraints_now = assemble_joint_linear_constraints(
            &block_constraints_now,
            &ranges_joint,
            total_p_joint,
        )?;
        let mut active_mask: Vec<bool> = vec![false; total_p_joint];
        if let Some(ref constraints) = joint_constraints_now
            && let Ok(Some(bounds)) = extract_simple_lower_bounds(constraints, total_p_joint)
        {
            for (idx, (bound, beta_val)) in bounds
                .lower_bounds
                .iter()
                .zip(beta_joint.iter())
                .enumerate()
            {
                if *bound > f64::NEG_INFINITY && (*beta_val - *bound).abs() < 1e-12 {
                    active_mask[idx] = true;
                }
            }
        }
        let res_inf_free = rhs
            .iter()
            .zip(active_mask.iter())
            .filter(|(_, active)| !**active)
            .map(|(v, _)| v.abs())
            .fold(0.0_f64, f64::max);
        // Scale-aware residual tolerance — the joint stationarity
        // residual ‖∇ℓ − Sβ‖_∞ scales with |obj| (≈ O(n) at large-scale
        // scale), so the historical absolute `inner_tol = 1e-6` is
        // unachievable here even at the true minimum. Same rationale
        // as the joint-Newton convergence test above.
        let polish_obj = -cached_eval.log_likelihood + *current_penalty;
        let polish_residual_tol = inner_tol * (1.0 + polish_obj.abs());
        if res_inf_free <= polish_residual_tol {
            *converged = true;
            break;
        }

        // Solve constrained Newton system if simple bounds are present,
        // else unconstrained.
        let delta = if let Some(ref constraints) = joint_constraints_now {
            let warm = flatten_joint_active_set(cached_active_sets, &block_constraints_now);
            let lower_bounds_opt = extract_simple_lower_bounds(constraints, total_p_joint)
                .ok()
                .flatten();
            if let Some(bounds) = lower_bounds_opt.as_ref() {
                match solve_quadratic_with_simple_lower_bounds(
                    &h_dense,
                    &rhs,
                    &beta_joint,
                    bounds,
                    warm.as_deref(),
                ) {
                    Ok((beta_new, _active)) => &beta_new - &beta_joint,
                    Err(_) => break,
                }
            } else {
                match solve_quadratic_with_linear_constraints(
                    &h_dense,
                    &rhs,
                    &beta_joint,
                    constraints,
                    warm.as_deref(),
                ) {
                    Ok((beta_new, _active)) => &beta_new - &beta_joint,
                    Err(_) => break,
                }
            }
        } else {
            let solver = crate::linalg::utils::StableSolver::new("joint polish");
            match solver.solvevectorwithridge_retries(&h_dense, &rhs, JOINT_TRACE_STABILITY_RIDGE) {
                Some(d) => d,
                None => break,
            }
        };
        if !delta.iter().all(|v| v.is_finite()) {
            break;
        }
        // Keep polishing until the free-space joint residual is small; a
        // tiny delta alone is not a certificate of stationarity.
        // Damped line search with projection.
        let old_states: Vec<ParameterBlockState> = states.clone();
        let old_obj = -eval_for_polish.log_likelihood + *current_penalty;
        let mut accepted_polish = false;
        for bt in 0..10 {
            let alpha = 0.5f64.powi(bt);
            for b in 0..specs.len() {
                let (start, end) = ranges_joint[b];
                let mut trial_beta = old_states[b].beta.clone();
                trial_beta.scaled_add(alpha, &delta.slice(ndarray::s![start..end]));
                let projected =
                    family.post_update_block_beta(&old_states, b, &specs[b], trial_beta.clone())?;
                reject_constrained_post_update_repair(
                    b,
                    &specs[b],
                    &trial_beta,
                    &projected,
                    block_constraints_now[b].as_ref(),
                )?;
                states[b].beta.assign(&projected);
            }
            refresh_all_block_etas(family, specs, states)?;
            let trial_ll = match family.log_likelihood_only(states) {
                Ok(v) => v,
                Err(_) => {
                    for (b, s) in old_states.iter().enumerate() {
                        states[b] = s.clone();
                    }
                    refresh_all_block_etas(family, specs, states)?;
                    continue;
                }
            };
            let trial_penalty = total_quadratic_penalty(
                states,
                s_lambdas,
                ridge,
                options.ridge_policy,
                joint_bundle,
                Some(specs),
            );
            let trial_obj = -trial_ll + trial_penalty;
            if trial_obj.is_finite() && trial_obj <= old_obj + 1e-12 {
                *current_penalty = trial_penalty;
                *cached_eval = family.evaluate(states)?;
                accepted_polish = true;
                break;
            }
        }
        if !accepted_polish {
            // Restore and stop polishing.
            for (b, s) in old_states.iter().enumerate() {
                states[b] = s.clone();
            }
            refresh_all_block_etas(family, specs, states)?;
            break;
        }
    }
    Ok(())
}

/// Final result assembly for the blockwise / polish fall-through path of
/// [`inner_blockwise_fit`]. Computes the penalty value, the block log-dets, the
/// (converged-only) projected KKT residual for the IFT, and the active-constraint
/// block, then moves `states`, `s_lambdas`, and `cached_active_sets` into the
/// returned [`BlockwiseInnerResult`]. Behavior is identical to the inline code it
/// replaced — the `?`-propagation and the `converged`-gate on `kkt_residual` are
/// preserved verbatim.
pub(crate) fn assemble_inner_blockwise_result<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    mut states: Vec<ParameterBlockState>,
    block_log_lambdas: &[Array1<f64>],
    options: &BlockwiseFitOptions,
    s_lambdas: Vec<Array2<f64>>,
    ridge: f64,
    joint_bundle: Option<&crate::families::joint_penalty::JointPenaltyBundle>,
    cached_active_sets: Vec<Option<Vec<usize>>>,
    cached_eval: &FamilyEvaluation,
    converged: bool,
    cycles_done: usize,
    last_residual_tol: f64,
) -> Result<BlockwiseInnerResult, String> {
    // Reuse cached evaluation from the last cycle's end (or the initial eval if 0 cycles ran).
    let penalty_value = total_quadratic_penalty(
        &states,
        &s_lambdas,
        ridge,
        options.ridge_policy,
        joint_bundle,
        Some(specs),
    );

    let (block_logdet_h, block_logdet_s) =
        blockwise_logdet_terms(family, specs, &mut states, block_log_lambdas, options)?;
    let kkt_residual = if converged {
        match exact_newton_joint_gradient_from_eval(cached_eval, specs, &states)? {
            Some(gradient) => {
                let block_constraints = collect_block_linear_constraints(family, &states, specs)?;
                let local_total_p: usize = specs.iter().map(|spec| spec.design.ncols()).sum();
                let active_set_rows_total: usize = cached_active_sets
                    .iter()
                    .map(|maybe| maybe.as_ref().map(|v| v.len()).unwrap_or(0))
                    .sum();
                let free_rank_at_cert = local_total_p.saturating_sub(active_set_rows_total);
                exact_newton_joint_projected_kkt_residual_for_ift_from_gradient(
                    &gradient,
                    specs,
                    &states,
                    &s_lambdas,
                    ridge,
                    options.ridge_policy,
                    &block_constraints,
                    Some(cached_active_sets.as_slice()),
                )?
                .map(|r| r.with_metadata(last_residual_tol, free_rank_at_cert))
            }
            None => None,
        }
    } else {
        // Inner did not converge; no caller should trust an IFT correction
        // at a non-KKT iterate.
        None
    };

    let active_constraints = {
        let local_ranges = block_param_ranges(specs);
        let local_total_p = local_ranges.last().map(|(_, end)| *end).unwrap_or(0);
        let block_constraints = collect_block_linear_constraints(family, &states, specs)?;
        assemble_active_constraint_block(
            &block_constraints,
            &cached_active_sets,
            &local_ranges,
            local_total_p,
        )
        .map(std::sync::Arc::new)
    };
    Ok(BlockwiseInnerResult {
        block_states: states,
        active_sets: normalize_active_sets(cached_active_sets),
        log_likelihood: cached_eval.log_likelihood,
        penalty_value,
        cycles: cycles_done,
        converged,
        block_logdet_h,
        block_logdet_s,
        s_lambdas,
        joint_workspace: None,
        kkt_residual,
        active_constraints,
    })
}

/// Borrowed derivative provider for joint models that wraps closures with
/// non-`'static` lifetimes.
///
/// The closures borrow data from the calling stack frame (family, synced states,
/// specs), so we use borrowed closures with a non-`'static` lifetime.
/// Instead we borrow the closures and implement `HessianDerivativeProvider` directly.
///
/// # Sign convention
///
/// The unified evaluator passes `v_k = H⁻¹(A_k β̂)` to `hessian_derivative_correction`.
/// By the implicit function theorem, `dβ̂/dρ_k = −v_k`. The stored `compute_dh`
/// expects the actual perturbation direction `δβ`, so we negate `v_k` before calling it.
pub(crate) struct BorrowedJointDerivProvider<'a> {
    pub(crate) compute_dh: &'a DriftDerivFn<'a>,
    pub(crate) compute_dh_many: Option<&'a DriftDerivManyFn<'a>>,
    pub(crate) compute_d2h: &'a DriftSecondDerivFn<'a>,
    /// Optional batched second-derivative callback. The unified evaluator's
    /// outer-Hessian ρ-ρ pair loop precomputes all K(K+1)/2 (v_k, v_l, u_kl)
    /// triples and calls this once per outer Hessian assembly when set, so
    /// families that fuse the per-row D²H walk across pairs (e.g. survival
    /// marginal-slope which scans n rows once per outer eval) replace
    /// K(K+1)/2 separate row-walks with one. The default `None` falls back
    /// to the per-pair `compute_d2h` dispatch and preserves the historical
    /// dispatch cost.
    pub(crate) compute_d2h_many: Option<&'a DriftSecondDerivManyFn<'a>>,
    pub(crate) family_outer_hessian_operator:
        Option<Arc<dyn crate::solver::rho_optimizer::OuterHessianOperator>>,
}
