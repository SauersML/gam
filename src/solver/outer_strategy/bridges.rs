use super::*;

pub(crate) struct OuterFirstOrderBridge<'a> {
    obj: &'a mut dyn OuterObjective,
    layout: OuterThetaLayout,
    /// Outer-aware inner-PIRLS cap atomic. When `Some`, the bridge stores
    /// a coarsen-then-tighten cap into it on every accepted gradient eval
    /// (see `first_order_inner_cap_schedule`).
    ///
    /// The cap is a perf optimization for the GRADIENT inner solve only: at
    /// the accepted ρ the warm-start is excellent, so a small cap converges
    /// the inner Newton and a still-non-converged result is honestly rejected
    /// as infeasible. But the line-search COST probe (`eval_cost`) evaluates a
    /// DIFFERENT trial ρ whose warm-start is worse; the same small cap can stop
    /// the inner solve short of its fixed point, returning a non-converged
    /// `f64::INFINITY` cost for a point that is actually feasible. With every
    /// trial step then reporting `∞`, no Wolfe/ARC step satisfies descent, the
    /// optimizer never leaves the accepted ρ, and the gradient re-evaluated
    /// there is identical iter after iter — the frozen-|g| outer stall in
    /// gam#787 (bernoulli matern marginal-slope) and gam#808 (survival
    /// marginal-slope). The line-search cost MUST be the same converged-inner
    /// objective the analytic envelope gradient differentiates; a capped
    /// surrogate is a different objective. So `eval_cost` UNCAPS the inner solve
    /// (stores `0` = full `pirls_config.max_iterations`) before delegating, and
    /// `eval_grad`/`eval_hessian` restore the scheduled cap on the next call.
    outer_inner_cap: Option<InnerProgressFeedback>,
    /// Counts gradient evaluations for logging only. Inner-PIRLS scheduling
    /// uses `InnerProgressFeedback.accepted_iter` so rejected line-search
    /// probes do not relax the inner work budget.
    iter_count: usize,
    /// First observed `‖g‖` from `eval_grad`. Used by the schedule to
    /// compute the gradient-ratio (`last / initial`) — when the ratio
    /// drops, the optimizer is approaching convergence and the inner
    /// cap should lift to full so the cached β is at full tolerance.
    g_norm_initial: Option<f64>,
    /// `‖g‖` from the most recent eval. Stale by one outer iter relative
    /// to the cap that consumes it (the cap is set BEFORE the new eval),
    /// but for monotone-decreasing g_norm this is safe — it makes the
    /// cap conservatively LARGER than the truly-needed value, never
    /// smaller.
    last_g_norm: Option<f64>,
    /// Most recent derivative-evaluation point. Value-only line-search probes
    /// log their distance from this reference so hidden backtracking work is
    /// visible in STAGE traces.
    last_value_grad_rho: Option<Array1<f64>>,
    /// Exact memo for recent line-search value probes. BFGS can re-query the
    /// same rejected trial when switching Wolfe strategies; the SAE inner solve
    /// behind a Value probe is deterministic, so serving an identical rho from
    /// this memo preserves the objective while avoiding duplicate refinement
    /// work.
    value_probe_cache: Vec<ValueProbeCacheEntry>,
    /// Gradient-independent cost-stall convergence guard. `opt::Bfgs` only
    /// terminates on a small *projected gradient norm* (its stall exit ANDs
    /// gradient-smallness with cost-smallness), so on a fully-penalized
    /// (double-penalty) REML surface with a shallow, weakly-identified valley —
    /// where the REML score flatlines while `‖∇_ρ V‖` plateaus *above*
    /// tolerance — no opt-side exit ever fires and BFGS burns its entire
    /// `max_iterations` budget (each iteration spending many line-search +
    /// coordinate-rescue + jiggle probes) on every seed. That is the #1089
    /// pathology: a trivial n≈30..120 Gaussian fit emitting ~850k cost-only
    /// evaluations until a wall-clock budget kills it. This guard adds the
    /// missing mgcv-style score-change stop: it watches the accepted-iterate
    /// REML objective and, once it stops improving by more than a relative
    /// tolerance over a window of consecutive accepted outer steps, publishes
    /// the best-so-far iterate and signals BFGS to stop. The runner then
    /// classifies the run as *converged at the flat-valley floor* rather than
    /// non-converged — the remaining gradient lies along weakly-identified ρ
    /// directions that do not reduce the objective.
    cost_stall: Option<CostStallGuard>,
    /// Count of consecutive `eval_cost` calls that returned `Recoverable`
    /// without a single success in between. When every trial step in every
    /// search direction is infeasible (the inner solve refuses to converge at
    /// any neighboring ρ), BFGS would otherwise spend its full
    /// `max_iterations × line_search_budget` budget doing inner solves that
    /// all fail — the non-termination reported in issue #NaN-outer-loop.
    ///
    /// Once this counter exceeds [`PROBE_REFUSAL_FATAL_THRESHOLD`] and no
    /// gradient evaluation has ever been accepted on this seed (`iter_count ==
    /// 0`), the bridge escalates to `Fatal` so BFGS exits immediately via
    /// `ObjectiveFailed`. The seed loop treats that outcome as a rejected seed
    /// and moves on, keeping the cascade bounded.
    ///
    /// Reset to 0 on any successful cost evaluation so normal line-search
    /// noise (a few recoverable probes followed by an accepted step) never
    /// trips this guard.
    consecutive_probe_refusals: usize,
}

pub(crate) const VALUE_PROBE_CACHE_CAPACITY: usize = 256;

pub(crate) const VALUE_PROBE_REJECT_COST_FLOOR: f64 = 1.0e11;

/// Number of consecutive recoverable `eval_cost` failures (every line-search
/// probe infeasible) before the bridge escalates to `Fatal` and forces an
/// immediate BFGS exit. This guard fires only before the first accepted
/// gradient step (`iter_count == 0`): once BFGS has accepted at least one
/// outer iteration the current ρ is feasible and isolated probe refusals are
/// normal line-search noise, not a stuck loop.
///
/// The threshold covers one full StrongWolfe attempt (up to 20 probes)
/// plus one backtracking fallback (up to 50 probes) with a small margin,
/// so a SINGLE failed direction does not fire the guard. Two consecutive
/// direction failures (120 probes) always does — once both Wolfe and
/// backtracking exhausted two complete directions with no success, the
/// neighborhood is globally infeasible and further BFGS iterations are
/// pure waste.
pub(crate) const PROBE_REFUSAL_FATAL_THRESHOLD: usize = 150;

/// Tighter probe-refusal threshold used when the bridge has never seen a
/// `eval_grad` call of its own — i.e. the seed (cost, gradient) was supplied
/// via `with_initial_sample` so `last_value_grad_rho` is `None` and every
/// `trial_rho_distance` prints as NaN.  In this case the seed gradient is
/// already confirmed feasible externally; if even the first line-search
/// direction exhausts its Wolfe probes without success (≈ 20 probes), the
/// neighborhood IS globally infeasible and further iterations just repeat
/// the same expensive inner solve 150 more times.  One generous Wolfe
/// budget (25 probes) is enough to confirm the failure; 13 seeds ×
/// 150 probes × ~3 s each would otherwise cause an observed ~97 min hang.
pub(crate) const PROBE_REFUSAL_FATAL_THRESHOLD_NAN_SEED: usize = 25;

/// Sentinel prefix embedded in the [`ObjectiveEvalError::Fatal`] message the
/// bridge returns when [`PROBE_REFUSAL_FATAL_THRESHOLD`] fires. The seed-loop
/// runner matches this prefix and routes the failed seed to
/// `rejection_reasons` rather than propagating a fatal error.
pub(crate) const PROBE_REFUSAL_FATAL_SENTINEL: &str = "OUTER_PROBE_REFUSAL_FATAL";

/// Sentinel embedded in the [`ObjectiveEvalError::Fatal`] message the bridge
/// returns when [`CostStallGuard`] halts BFGS on a cost stall. `opt::Bfgs`
/// preserves the message verbatim in [`BfgsError::ObjectiveFailed`]; the
/// seed-loop runner recognizes this sentinel and rebuilds an outer result from
/// the published best iterate. Whether that result is reported `converged` is
/// NOT decided here — it is carried on the published [`CostStallExit`], gated on
/// the projected gradient norm at the best iterate clearing the same outer
/// gradient tolerance the genuine convergence path uses. A cost stall whose
/// residual gradient still exceeds that tolerance is a flat-valley stall, not a
/// stationary optimum, and is reported `converged = false`.
pub(crate) const COST_STALL_CONVERGED_SENTINEL: &str = "OUTER_COST_STALL_CONVERGED";

/// Verdict produced by folding one accepted outer iterate into
/// [`CostStallGuard::observe`].
pub(crate) enum CostStallVerdict {
    /// The objective is still improving (or the no-improvement window has not
    /// yet filled). Keep descending.
    Continue,
    /// The objective has stopped improving over the window AND the projected
    /// gradient norm at the best iterate clears the outer gradient tolerance:
    /// a genuine stationary optimum on a (legitimately) flat REML surface.
    Converged,
    /// The objective has stopped improving over the window but the projected
    /// gradient norm at the best iterate is still above the outer gradient
    /// tolerance: a weakly-identified flat-valley FLOOR with residual
    /// non-stationarity. Halting here is correct (no further cost progress is
    /// available), but the iterate is NOT a stationary optimum and must be
    /// reported `converged = false`.
    FlatValleyStall { residual_grad_norm: f64 },
}

/// Number of consecutive accepted outer iterates with negligible relative
/// objective improvement required before the cost-stall guard declares
/// convergence. Matches the spirit of `opt`'s own `StallPolicy { window: 3 }`
/// but, crucially, is gated on the cost alone (not on gradient smallness),
/// which is the condition `opt` never checks in isolation.
pub(crate) const COST_STALL_WINDOW: usize = 6;

/// Best iterate captured by a cost-stall convergence, handed from the bridge
/// (which is moved into `opt::Bfgs`) back to the seed-loop runner via the
/// guard's shared cell.
#[derive(Clone)]
pub(crate) struct CostStallExit {
    pub(crate) rho: Array1<f64>,
    pub(crate) value: f64,
    pub(crate) grad_norm: f64,
    /// Accepted outer iterates observed when the stall fired (for the runner's
    /// `OuterResult.iterations` field and logging).
    pub(crate) iterations: usize,
    /// Whether the best iterate is a genuine stationary optimum: `true` only
    /// when its projected gradient norm cleared the outer gradient tolerance
    /// (legitimately-flat REML surface). `false` for a flat-valley stall whose
    /// residual gradient remains above tolerance — the runner reports the
    /// rebuilt outer result as non-converged in that case.
    pub(crate) converged: bool,
}

/// Tracks the monotone best accepted-iterate REML objective and a
/// no-improvement streak, firing a gradient-independent convergence once the
/// objective has effectively stopped decreasing. See the `cost_stall` field
/// doc on [`OuterFirstOrderBridge`] for the full rationale (#1089).
pub(crate) struct CostStallGuard {
    /// Relative improvement floor: an accepted step counts as "no improvement"
    /// when `(best - cost) <= rel_tol * (1 + |best|)`. Derived from the outer
    /// convergence tolerance so it tracks the configured precision rather than
    /// a free-standing magic constant.
    rel_tol: f64,
    /// Consecutive accepted-step window with no improvement before declaring
    /// convergence.
    window: usize,
    /// Projected outer gradient-norm threshold that the best iterate must clear
    /// for a cost stall to count as a genuine stationary optimum. This is the
    /// SAME threshold the normal BFGS convergence path uses
    /// (`outer_gradient_tolerance(config).threshold(seed_cost, ‖g_0‖)`),
    /// evaluated once at seed. A cost stall above this threshold is a
    /// flat-valley stall, reported `converged = false`.
    grad_threshold: f64,
    best_value: f64,
    best_rho: Option<Array1<f64>>,
    best_grad_norm: f64,
    no_improve_streak: usize,
    accepted_iters: usize,
    /// Shared publication slot read by the seed-loop runner after
    /// `optimizer.run()` returns the sentinel error.
    exit: Arc<Mutex<Option<CostStallExit>>>,
}

impl CostStallGuard {
    pub(crate) fn new(
        rel_tol: f64,
        window: usize,
        grad_threshold: f64,
        exit: Arc<Mutex<Option<CostStallExit>>>,
    ) -> Self {
        Self {
            rel_tol,
            window,
            grad_threshold,
            best_value: f64::INFINITY,
            best_rho: None,
            best_grad_norm: f64::INFINITY,
            no_improve_streak: 0,
            accepted_iters: 0,
            exit,
        }
    }

    /// Fold one accepted-iterate `(ρ, cost, ‖g‖)` into the guard. Returns a
    /// [`CostStallVerdict`]: `Continue` while the score is still improving,
    /// `Converged` when the score has stalled AND the projected gradient norm
    /// at the best iterate clears the outer gradient tolerance (a genuine
    /// stationary optimum on a flat REML surface), or `FlatValleyStall` when
    /// the score has stalled but the residual gradient remains above tolerance
    /// (a weakly-identified flat valley that is NOT stationary). Either stalled
    /// verdict publishes the best iterate to the shared cell, tagged with its
    /// `converged` status.
    fn observe(&mut self, rho: &Array1<f64>, value: f64, grad_norm: f64) -> CostStallVerdict {
        if !value.is_finite() {
            // A non-finite accepted objective is the inner-solver's problem,
            // not a stall; reset so a later real descent is not falsely
            // credited as a no-improvement step.
            self.no_improve_streak = 0;
            return CostStallVerdict::Continue;
        }
        self.accepted_iters = self.accepted_iters.saturating_add(1);
        let improvement = self.best_value - value;
        let floor = self.rel_tol * (1.0 + self.best_value.abs());
        if value < self.best_value {
            self.best_value = value;
            self.best_rho = Some(rho.clone());
            self.best_grad_norm = grad_norm;
        }
        if improvement <= floor {
            self.no_improve_streak = self.no_improve_streak.saturating_add(1);
        } else {
            self.no_improve_streak = 0;
        }
        if self.no_improve_streak < self.window {
            return CostStallVerdict::Continue;
        }
        // Publish the best iterate. Prefer the recorded best; fall back to the
        // current point if (pathologically) none was stored.
        let best_rho = self.best_rho.clone().unwrap_or_else(|| rho.clone());
        let best_value = if self.best_value.is_finite() {
            self.best_value
        } else {
            value
        };
        let best_grad_norm = if self.best_grad_norm.is_finite() {
            self.best_grad_norm
        } else {
            grad_norm
        };
        // Convergence is STATIONARITY, not cost-flatness: a cost stall counts
        // as a converged optimum only when the projected gradient norm at the
        // best iterate clears the same outer gradient tolerance the genuine
        // BFGS convergence path checks. Otherwise it is a flat-valley floor
        // with residual non-stationarity, reported `converged = false`.
        let converged = best_grad_norm.is_finite() && best_grad_norm <= self.grad_threshold;
        if let Ok(mut slot) = self.exit.lock() {
            *slot = Some(CostStallExit {
                rho: best_rho,
                value: best_value,
                grad_norm: best_grad_norm,
                iterations: self.accepted_iters,
                converged,
            });
        }
        if converged {
            CostStallVerdict::Converged
        } else {
            CostStallVerdict::FlatValleyStall {
                residual_grad_norm: best_grad_norm,
            }
        }
    }
}

#[derive(Clone)]
pub(crate) struct ValueProbeCacheEntry {
    rho: Array1<f64>,
    outcome: CachedValueProbeOutcome,
}

#[derive(Clone)]
pub(crate) enum CachedValueProbeOutcome {
    Cost(f64),
    Recoverable(String),
    Fatal(String),
}

pub(crate) fn trial_rho_distance(reference: Option<&Array1<f64>>, trial: &Array1<f64>) -> f64 {
    let Some(reference) = reference else {
        return f64::NAN;
    };
    if reference.len() != trial.len() {
        return f64::NAN;
    }
    reference
        .iter()
        .zip(trial.iter())
        .map(|(a, b)| {
            let d = b - a;
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

pub(crate) fn same_outer_point(a: &Array1<f64>, b: &Array1<f64>) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

pub(crate) fn cached_value_probe_result(
    outcome: &CachedValueProbeOutcome,
) -> Result<f64, ObjectiveEvalError> {
    match outcome {
        CachedValueProbeOutcome::Cost(cost) => Ok(*cost),
        CachedValueProbeOutcome::Recoverable(message) => {
            Err(ObjectiveEvalError::recoverable(message.clone()))
        }
        CachedValueProbeOutcome::Fatal(message) => Err(ObjectiveEvalError::Fatal {
            message: message.clone(),
        }),
    }
}

pub(crate) fn cache_value_probe_result(
    result: &Result<f64, ObjectiveEvalError>,
) -> CachedValueProbeOutcome {
    match result {
        Ok(cost) => CachedValueProbeOutcome::Cost(*cost),
        Err(ObjectiveEvalError::Recoverable { message }) => {
            CachedValueProbeOutcome::Recoverable(message.clone())
        }
        Err(ObjectiveEvalError::Fatal { message }) => {
            CachedValueProbeOutcome::Fatal(message.clone())
        }
    }
}

pub(crate) fn value_probe_outcome_label(outcome: &CachedValueProbeOutcome) -> &'static str {
    match outcome {
        CachedValueProbeOutcome::Cost(_) => "cost",
        CachedValueProbeOutcome::Recoverable(_) => "recoverable",
        CachedValueProbeOutcome::Fatal(_) => "fatal",
    }
}

pub(crate) fn value_probe_reject_outcome(outcome: &CachedValueProbeOutcome) -> bool {
    match outcome {
        CachedValueProbeOutcome::Cost(cost) => *cost >= VALUE_PROBE_REJECT_COST_FLOOR,
        CachedValueProbeOutcome::Recoverable(_) | CachedValueProbeOutcome::Fatal(_) => true,
    }
}

pub(crate) fn remember_value_probe(
    cache: &mut Vec<ValueProbeCacheEntry>,
    rho: &Array1<f64>,
    outcome: CachedValueProbeOutcome,
) {
    if let Some(entry) = cache
        .iter_mut()
        .find(|entry| same_outer_point(&entry.rho, rho))
    {
        entry.outcome = outcome;
        return;
    }
    if cache.len() == VALUE_PROBE_CACHE_CAPACITY {
        cache.remove(0);
    }
    cache.push(ValueProbeCacheEntry {
        rho: rho.clone(),
        outcome,
    });
}

impl ZerothOrderObjective for OuterFirstOrderBridge<'_> {
    fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, ObjectiveEvalError> {
        // Per-axis line-search step caps now live natively in opt::Bfgs
        // (`with_axis_step_caps`), which shortens the BFGS direction before
        // line search instead of poisoning the Wolfe bracket with a
        // sentinel cost. This entry point can therefore stay honest: any
        // call that lands here is a real line-search probe, not a too-far
        // attempt the bridge needs to swat away.
        //
        // Uncap the inner solve for the line-search cost probe (see the field
        // doc on `outer_inner_cap`): the deciding cost MUST be the true
        // converged-inner objective the analytic gradient differentiates, not
        // the scheduled gradient-path cap which can stop a trial-ρ inner solve
        // short of its fixed point and report a spurious `∞`. `eval_grad`
        // restores the scheduled cap on the next call.
        if let Some(feedback) = self.outer_inner_cap.as_ref() {
            feedback
                .cap
                .store(SEED_SCREENING_UNCAPPED, Ordering::Relaxed);
        }
        self.layout
            .validate_point_len(x, "outer eval_cost failed")?;
        let trial_rho_distance = trial_rho_distance(self.last_value_grad_rho.as_ref(), x);
        let stage_start = std::time::Instant::now();
        if let Some(entry) = self
            .value_probe_cache
            .iter()
            .find(|entry| same_outer_point(&entry.rho, x))
        {
            let outcome_label = value_probe_outcome_label(&entry.outcome);
            log::info!(
                "[STAGE] outer eval start order=Value dim={} trial_rho_distance={:.3e} (first-order bridge, iter={}, cached=true)",
                x.len(),
                trial_rho_distance,
                self.iter_count
            );
            match &entry.outcome {
                CachedValueProbeOutcome::Cost(cost) => log::info!(
                    "[STAGE] outer eval end order=Value elapsed={:.3}s cost={:.6e} trial_rho_distance={:.3e} (first-order bridge, iter={}, cached=true)",
                    stage_start.elapsed().as_secs_f64(),
                    cost,
                    trial_rho_distance,
                    self.iter_count
                ),
                CachedValueProbeOutcome::Recoverable(_) | CachedValueProbeOutcome::Fatal(_) => {
                    log::info!(
                        "[STAGE] outer eval end order=Value elapsed={:.3}s outcome={} trial_rho_distance={:.3e} (first-order bridge, iter={}, cached=true)",
                        stage_start.elapsed().as_secs_f64(),
                        outcome_label,
                        trial_rho_distance,
                        self.iter_count
                    );
                }
            }
            return cached_value_probe_result(&entry.outcome);
        }
        log::info!(
            "[STAGE] outer eval start order=Value dim={} trial_rho_distance={:.3e} (first-order bridge, iter={})",
            x.len(),
            trial_rho_distance,
            self.iter_count
        );
        let result = self
            .obj
            .eval_with_order(x, OuterEvalOrder::Value)
            .map_err(|err| into_objective_error("outer eval_cost failed", err))
            .and_then(|eval| finite_cost_or_error("outer eval_cost failed", eval.cost));
        let cached_outcome = cache_value_probe_result(&result);
        remember_value_probe(&mut self.value_probe_cache, x, cached_outcome);
        match &result {
            Ok(cost) => {
                // A successful probe resets the consecutive-refusal counter: the
                // current ρ neighbourhood has at least one feasible point, so
                // isolated refusals on other directions are normal line-search
                // noise, not a globally-infeasible neighbourhood.
                self.consecutive_probe_refusals = 0;
                log::info!(
                    "[STAGE] outer eval end order=Value elapsed={:.3}s cost={:.6e} trial_rho_distance={:.3e} (first-order bridge, iter={})",
                    stage_start.elapsed().as_secs_f64(),
                    cost,
                    trial_rho_distance,
                    self.iter_count
                );
            }
            Err(ObjectiveEvalError::Recoverable { .. }) => {
                log::info!(
                    "[STAGE] outer eval end order=Value elapsed={:.3}s outcome=recoverable trial_rho_distance={:.3e} (first-order bridge, iter={})",
                    stage_start.elapsed().as_secs_f64(),
                    trial_rho_distance,
                    self.iter_count
                );
                // Non-termination guard (#NaN-outer-loop): when every
                // line-search probe is infeasible and BFGS has never
                // accepted a gradient step (`iter_count == 0`), the
                // neighbourhood around the seed is globally degenerate.
                // BFGS would otherwise spend its entire max_iterations ×
                // line_search_budget doing inner solves that all fail.
                // Escalate to Fatal so BFGS exits immediately; the seed
                // loop routes it as a rejected seed.
                self.consecutive_probe_refusals = self.consecutive_probe_refusals.saturating_add(1);
                // When the bridge seed (cost, gradient) was supplied via
                // `with_initial_sample` the bridge's own `eval_grad` is
                // never called, so `last_value_grad_rho` stays `None` and
                // every `trial_rho_distance` prints as NaN.  The seed IS
                // feasible (it was evaluated externally), but if every
                // line-search probe is Recoverable from the very first
                // direction, the neighbourhood is globally infeasible.
                // Use the tighter NaN-seed threshold so the guard fires
                // after one generous Wolfe budget instead of 150 probes
                // (which, at ~3 s each × 13 seeds, would produce an
                // observed ~97 min hang on real D=5120 LLM activations).
                let threshold = if self.last_value_grad_rho.is_none() {
                    PROBE_REFUSAL_FATAL_THRESHOLD_NAN_SEED
                } else {
                    PROBE_REFUSAL_FATAL_THRESHOLD
                };
                if self.iter_count == 0 && self.consecutive_probe_refusals >= threshold {
                    log::warn!(
                        "[OUTER] probe-refusal non-termination guard fired after {} consecutive \
                         infeasible cost probes with no accepted gradient step \
                         (nan_seed={}); escalating to Fatal to abort this seed \
                         (first-order bridge, iter={})",
                        self.consecutive_probe_refusals,
                        self.last_value_grad_rho.is_none(),
                        self.iter_count,
                    );
                    return Err(ObjectiveEvalError::Fatal {
                        message: format!(
                            "{PROBE_REFUSAL_FATAL_SENTINEL}: {consecutive} consecutive \
                             infeasible probes with no accepted outer step",
                            consecutive = self.consecutive_probe_refusals,
                        ),
                    });
                }
            }
            Err(ObjectiveEvalError::Fatal { .. }) => {
                log::info!(
                    "[STAGE] outer eval end order=Value elapsed={:.3}s outcome=fatal trial_rho_distance={:.3e} (first-order bridge, iter={})",
                    stage_start.elapsed().as_secs_f64(),
                    trial_rho_distance,
                    self.iter_count
                );
            }
        }
        result
    }
}

impl FirstOrderObjective for OuterFirstOrderBridge<'_> {
    fn eval_grad(&mut self, x: &Array1<f64>) -> Result<FirstOrderSample, ObjectiveEvalError> {
        self.layout.validate_point_len(x, "outer eval failed")?;
        // Drive the outer-aware inner-PIRLS cap from accepted outer
        // iterations, BEFORE invoking the inner solve. Cap stays fixed
        // within line-search cost probes (`eval_cost` never touches the
        // atomic). A cap of 0 means "no cap from this source"; the inner
        // solver still honors `pirls_max_iterations` and the screening cap.
        if let Some(feedback) = self.outer_inner_cap.as_ref() {
            let g_ratio = match (self.last_g_norm, self.g_norm_initial) {
                (Some(g), Some(g0)) if g0 > 0.0 => Some(g / g0),
                _ => None,
            };
            let snapshot = feedback.snapshot();
            let accepted_iter = feedback.accepted_iter.load(Ordering::Relaxed);
            let cap = first_order_inner_cap_schedule(accepted_iter, g_ratio, snapshot);
            let prev = feedback.cap.swap(cap, Ordering::Relaxed);
            if prev != cap {
                let ratio_str = match g_ratio {
                    Some(r) => format!("{:.3e}", r),
                    None => "n/a".to_string(),
                };
                let snap_str = match snapshot {
                    Some(s) => format!(
                        "last_iters={} converged={} ift_residual={} accept_rho={}",
                        s.last_iters,
                        s.last_converged,
                        match s.last_ift_residual {
                            Some(r) => format!("{:.3e}", r),
                            None => "n/a".to_string(),
                        },
                        match s.last_accept_rho {
                            Some(r) => format!("{:.3}", r),
                            None => "n/a".to_string(),
                        },
                    ),
                    None => "no-history".to_string(),
                };
                log::info!(
                    "[OUTER schedule] inner-PIRLS cap transition accepted_iter={} eval_count={} g_ratio={} {} prev={} new={} ({})",
                    accepted_iter,
                    self.iter_count,
                    ratio_str,
                    snap_str,
                    prev,
                    cap,
                    if cap == 0 { "uncapped" } else { "capped" }
                );
            }
        }
        let stage_start = std::time::Instant::now();
        log::info!(
            "[STAGE] outer eval start order=ValueAndGradient dim={} (first-order bridge, iter={})",
            x.len(),
            self.iter_count
        );
        let eval = self
            .obj
            .eval_with_order(x, OuterEvalOrder::ValueAndGradient)
            .map_err(|err| into_objective_error("outer eval failed", err))?;
        let eval = finite_outer_first_order_eval_or_error("outer eval failed", self.layout, eval)?;
        let g_norm = eval.gradient.iter().map(|v| v * v).sum::<f64>().sqrt();
        let gradient = eval.gradient;
        if self.g_norm_initial.is_none() && g_norm.is_finite() && g_norm > 0.0 {
            self.g_norm_initial = Some(g_norm);
        }
        if g_norm.is_finite() {
            self.last_g_norm = Some(g_norm);
        }
        self.last_value_grad_rho = Some(x.clone());
        // A successful gradient evaluation means the current ρ is feasible;
        // reset the consecutive-probe-refusal counter so the guard only fires
        // when ALL probes in EVERY subsequent direction fail.
        self.consecutive_probe_refusals = 0;
        self.value_probe_cache
            .retain(|entry| value_probe_reject_outcome(&entry.outcome));
        log::info!(
            "[STAGE] outer eval end order=ValueAndGradient elapsed={:.3}s cost={:.6e} |g|={:.3e} (first-order bridge, iter={})",
            stage_start.elapsed().as_secs_f64(),
            eval.cost,
            g_norm,
            self.iter_count,
        );
        // Push the (cost, ‖g‖) sample so the live progress chart shows the
        // BFGS outer descent. Recorded as a trial; `OuterAcceptObserver`
        // promotes the latest trial into the accepted series when BFGS's
        // Wolfe line-search accepts the step. Cheap: throttled internally.
        crate::solver::visualizer::record_outer_eval(eval.cost, g_norm);
        self.iter_count = self.iter_count.saturating_add(1);
        // Cost-stall halt (#1089). `eval_grad` is invoked by `opt::Bfgs` at
        // each accepted iterate (line-search COST probes go through `eval_cost`,
        // not here), so folding the objective in here counts accepted outer
        // steps. When the REML score has stopped improving over
        // `COST_STALL_WINDOW` consecutive accepted steps, halt BFGS by returning
        // a sentinel `Fatal` (an observer cannot stop `opt::Bfgs`; an error is
        // the only in-band way to halt it). The runner rebuilds the outer result
        // from the published best iterate — but whether that result is reported
        // CONVERGED is decided by the guard's STATIONARITY test, not by
        // cost-flatness alone: a stall whose projected gradient still exceeds the
        // outer gradient tolerance is a flat-valley floor (`converged = false`),
        // a stationary one is a real optimum (`converged = true`). Both share the
        // sentinel; the verdict rides on the published `CostStallExit.converged`.
        if let Some(guard) = self.cost_stall.as_mut() {
            match guard.observe(x, eval.cost, g_norm) {
                CostStallVerdict::Continue => {}
                CostStallVerdict::Converged => {
                    log::info!(
                        "[OUTER] cost-stall convergence: REML objective improved < {:.3e} \
                         (relative) over {} consecutive accepted outer steps AND the projected \
                         gradient cleared the outer tolerance (|g|={:.3e} <= {:.3e}); accepting \
                         best-so-far as a stationary optimum (value={:.6e}).",
                        guard.rel_tol,
                        guard.window,
                        guard.best_grad_norm,
                        guard.grad_threshold,
                        guard.best_value,
                    );
                    return Err(ObjectiveEvalError::Fatal {
                        message: COST_STALL_CONVERGED_SENTINEL.to_string(),
                    });
                }
                CostStallVerdict::FlatValleyStall { residual_grad_norm } => {
                    log::warn!(
                        "[OUTER] cost-stall FLAT-VALLEY STALL: REML objective improved < {:.3e} \
                         (relative) over {} consecutive accepted outer steps but the projected \
                         gradient is still ABOVE the outer tolerance (|g|={:.3e} > {:.3e}); \
                         halting on a weakly-identified ρ valley floor and reporting NON-CONVERGED \
                         (residual outer non-stationarity, value={:.6e}).",
                        guard.rel_tol,
                        guard.window,
                        residual_grad_norm,
                        guard.grad_threshold,
                        guard.best_value,
                    );
                    return Err(ObjectiveEvalError::Fatal {
                        message: COST_STALL_CONVERGED_SENTINEL.to_string(),
                    });
                }
            }
        }
        Ok(FirstOrderSample {
            value: eval.cost,
            gradient,
        })
    }
}

/// Outer gradient-decay ratio `‖g_now‖/‖g_initial‖` below which the outer is
/// treated as essentially converged: the inner cap is lifted entirely so the
/// cached β reaches full inner tolerance before the convergence guard runs.
pub(crate) const INNER_CAP_CONVERGENCE_OVERRIDE_RATIO: f64 = 0.01;

/// Floor on the adaptive inner-PIRLS cap. Any cap below this is below the
/// inner-Newton noise level and would reject usable warm-started steps.
pub(crate) const INNER_CAP_FLOOR: usize = 3;

/// Ceiling on the adaptive inner-PIRLS cap, set at the inner-Newton noise
/// floor at large scale; further iterations are pure waste once the warm
/// start is close.
pub(crate) const INNER_CAP_CEILING: usize = 64;

/// Adaptive inner-PIRLS cap schedule. Replaces the older hardcoded
/// iter-tier (3/5/10/20) and ratio-tier (0.50/0.20/0.05/0.01) schedule
/// with a cap driven by the inner solver's actual convergence behavior
/// — Eisenstat-Walker style for the inner Newton.
///
/// Inputs:
/// - `iter_count`: outer iter index, used only as a fallback when no
///   inner-progress feedback has arrived yet (first 1-2 outer iters).
/// - `g_ratio`: outer gradient-norm decay `‖g_now‖ / ‖g_initial‖`. When
///   this drops below 1% the outer is essentially converged; we lift
///   the cap fully so the cached β is at full inner tolerance and the
///   convergence guard does not have to re-pay a full inner solve.
/// - `last`: snapshot from `InnerProgressFeedback`. When present and
///   the previous solve converged, we set the cap to `last_iters + 2`
///   (a small margin in case ρ moved enough to need a couple more
///   iters); when the previous solve hit the cap, we double — a
///   geometric backoff that recovers from too-tight a cap without
///   thrashing.
///
/// A cap of 0 means "no cap from this source"; the inner solver still
/// honors `pirls_max_iterations` and the screening cap. The cap is
/// floored at 3 (anything less is below noise) and ceilinged at 64
/// (the inner noise floor at large scale; further iters would be
/// pure waste).
pub(crate) fn first_order_inner_cap_schedule(
    iter_count: usize,
    g_ratio: Option<f64>,
    last: Option<InnerProgressSnapshot>,
) -> usize {
    // Convergence override: when the outer is essentially converged the
    // cached β must be at full inner tolerance. This belt-and-suspenders
    // path is independent of inner-progress history because the outer
    // re-evaluation guard pays a full inner solve anyway — uncapping
    // here just avoids one wasted iter at low cap before the guard.
    if matches!(g_ratio, Some(r) if r < INNER_CAP_CONVERGENCE_OVERRIDE_RATIO) {
        return 0;
    }

    // Adaptive path: drive the cap from the inner solver's prior iter
    // count rather than a hardcoded tier.
    if let Some(snap) = last {
        let next = if snap.last_converged {
            // Converged in `last_iters` last time; pick a small margin
            // for ρ-step variability. The IFT predictor's residual
            // tells us how close the warm-start was to the KKT point:
            //   residual < 0.01  → next solve starts essentially AT the
            //                      KKT β, so +1 iter of margin suffices.
            //   residual < 0.10  → +2 (default, current behavior).
            //   residual ≥ 0.10  → predictor was poor (or fell back to
            //                      flat); the inner Newton has more
            //                      recovery work, so +4 to be safe.
            //   None             → no signal yet → +2 (default).
            // This wires the [IFT-QUALITY] feedback directly into the
            // adaptive cap, replacing the previous fixed +2.
            let mut margin = match snap.last_ift_residual {
                Some(r) if r < 0.01 => 1usize,
                Some(r) if r >= 0.10 => 4usize,
                _ => 2usize,
            };
            // LM model fidelity (commit 6445c079): if the previous
            // solve's accepted gain ratio was poor (model overstating
            // predicted reduction), the inner Newton's quadratic model
            // is unreliable. Bump margin by +2 — even a fast-converged
            // previous iter (small `last_iters`) provides weaker
            // evidence about the next solve's required effort when the
            // model is mis-calibrated. Threshold 0.5 is the textbook
            // "good agreement" cutoff for trust-region gain ratios.
            if matches!(snap.last_accept_rho, Some(r) if r < 0.5) {
                margin = margin.saturating_add(2);
            }
            snap.last_iters.saturating_add(margin)
        } else {
            // Hit the cap. Geometric backoff so we don't thrash on a
            // marginally-too-tight cap, but enforce floor of
            // last_iters+4 to actually grow.
            //
            // LM-fidelity escalation: if the previous solve's accepted
            // gain ratio was VERY poor (`accept_rho < 0.3`), the LM
            // model is severely mis-calibrated — doubling the cap may
            // not give the inner Newton enough headroom to find a
            // usable trust radius. Triple instead of doubling so we
            // don't waste another cycle hitting the cap. The 0.3
            // threshold is tighter than the +2-margin trigger (0.5)
            // because here we ALREADY know the iter budget was
            // insufficient AND the model was poor — both signals
            // pointing the same way.
            let multiplier = if matches!(snap.last_accept_rho, Some(r) if r < 0.3) {
                3
            } else {
                2
            };
            snap.last_iters
                .saturating_mul(multiplier)
                .max(snap.last_iters.saturating_add(4))
        };
        return next.clamp(INNER_CAP_FLOOR, INNER_CAP_CEILING);
    }

    // No feedback yet (first outer iter, or right after a screening
    // bundle reset). Coarse iter-count fallback for the first 1-2
    // outer iters so the cold-start cap is shallow even before the
    // adaptive signal kicks in.
    match iter_count {
        0 => 3,
        1 => 5,
        _ => 10,
    }
}

#[cfg(test)]
mod inner_cap_schedule_tests;

pub(crate) struct OuterSecondOrderBridge<'a> {
    obj: &'a mut dyn OuterObjective,
    layout: OuterThetaLayout,
    hessian_source: HessianSource,
    /// When the evaluator returns `HessianResult::Operator(op)` and the
    /// operator advertises an exact dense route, the bridge may materialize the
    /// operator into a dense K×K matrix so the dense ARC path can run an exact
    /// factorization instead of operator-CG.
    materialize_operator_max_dim: usize,
    /// Counts gradient/Hessian evaluations so that progress is visible even
    /// when the upstream `opt` solver does not emit per-iteration logs of its
    /// own. Emitted at INFO from `eval_grad` and `eval_hessian` (the calls
    /// that gate one optimizer step); skipped on `eval_cost` so linesearch
    /// trial points do not flood the log. Also drives the outer-aware
    /// inner-PIRLS cap schedule (see `first_order_inner_cap_schedule`).
    eval_count: usize,
    /// Outer-aware inner-PIRLS cap atomic. When `Some`, the bridge stores
    /// a coarsen-then-tighten cap into it on every accepted eval_grad /
    /// eval_hessian call. Mirrors the BFGS-side wiring in
    /// `OuterFirstOrderBridge`. Cap is NEVER touched in `eval_cost` so
    /// line-search probes within an outer iter see a stable inner
    /// tolerance (Wolfe / trust-region acceptance both assume constant
    /// cost noise within a bracket).
    outer_inner_cap: Option<InnerProgressFeedback>,
    /// First observed `‖g‖` from `eval_grad`/`eval_hessian`. Used by the
    /// schedule's gradient-ratio gate so the cap lifts when the optimizer
    /// is approaching convergence, not just when iter count says so.
    g_norm_initial: Option<f64>,
    /// `‖g‖` from the most recent eval. See `OuterFirstOrderBridge` for
    /// the staleness rationale: monotone-decreasing g_norm means the cap
    /// is conservatively LARGER than truly needed, never smaller.
    last_g_norm: Option<f64>,
    /// Most recent derivative-evaluation point, used to log value-probe
    /// displacement in line-search / trial-acceptance STAGE traces.
    last_value_grad_rho: Option<Array1<f64>>,
}

impl ZerothOrderObjective for OuterSecondOrderBridge<'_> {
    fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, ObjectiveEvalError> {
        // Uncap the inner solve for the ARC line-search / trial-acceptance cost
        // probe. Identical rationale to `OuterFirstOrderBridge::eval_cost`: the
        // deciding cost must be the true converged-inner objective the analytic
        // gradient/Hessian differentiate, never the scheduled gradient-path cap
        // (which at a trial ρ can stop the inner solve short and report a
        // spurious `∞`, freezing the ARC at constant cost / |g| — gam#808
        // survival marginal-slope, gam#787 bernoulli matern marginal-slope).
        // `eval_grad`/`eval_hessian` restore the scheduled cap on the next call.
        if let Some(feedback) = self.outer_inner_cap.as_ref() {
            feedback
                .cap
                .store(SEED_SCREENING_UNCAPPED, Ordering::Relaxed);
        }
        self.layout
            .validate_point_len(x, "outer eval_cost failed")?;
        let trial_rho_distance = trial_rho_distance(self.last_value_grad_rho.as_ref(), x);
        let stage_start = std::time::Instant::now();
        log::info!(
            "[STAGE] outer eval start order=Value dim={} trial_rho_distance={:.3e}",
            x.len(),
            trial_rho_distance
        );
        let eval = self
            .obj
            .eval_with_order(x, OuterEvalOrder::Value)
            .map_err(|err| into_objective_error("outer eval_cost failed", err))?;
        let cost = finite_cost_or_error("outer eval_cost failed", eval.cost)?;
        log::info!(
            "[STAGE] outer eval end order=Value elapsed={:.3}s cost={:.6e} trial_rho_distance={:.3e}",
            stage_start.elapsed().as_secs_f64(),
            cost,
            trial_rho_distance
        );
        Ok(cost)
    }
}

impl FirstOrderObjective for OuterSecondOrderBridge<'_> {
    fn eval_grad(&mut self, x: &Array1<f64>) -> Result<FirstOrderSample, ObjectiveEvalError> {
        self.layout.validate_point_len(x, "outer eval failed")?;
        if let Some(feedback) = self.outer_inner_cap.as_ref() {
            // The ARC bridge increments `eval_count` in BOTH `eval_grad` and
            // `eval_hessian`. ARC calls both per outer iter, so `eval_count
            // / 2` is the correct iter index for the schedule. Without this
            // divisor the schedule would lift to full inner-cap at ARC iter
            // 3 instead of iter 6.
            // Use the observer-fed accepted-iter counter (opt 0.5.0
            // OptimizerObserver) instead of `eval_count / 2`; the
            // observer increments only on rho-accepted steps, so the
            // schedule no longer relaxes the cap on rejected trials.
            let arc_iter = feedback.accepted_iter.load(Ordering::Relaxed);
            let g_ratio = match (self.last_g_norm, self.g_norm_initial) {
                (Some(g), Some(g0)) if g0 > 0.0 => Some(g / g0),
                _ => None,
            };
            let snapshot = feedback.snapshot();
            let cap = first_order_inner_cap_schedule(arc_iter, g_ratio, snapshot);
            let prev = feedback.cap.swap(cap, Ordering::Relaxed);
            if prev != cap {
                let ratio_str = match g_ratio {
                    Some(r) => format!("{:.3e}", r),
                    None => "n/a".to_string(),
                };
                let snap_str = match snapshot {
                    Some(s) => format!(
                        "last_iters={} converged={} ift_residual={} accept_rho={}",
                        s.last_iters,
                        s.last_converged,
                        match s.last_ift_residual {
                            Some(r) => format!("{:.3e}", r),
                            None => "n/a".to_string(),
                        },
                        match s.last_accept_rho {
                            Some(r) => format!("{:.3}", r),
                            None => "n/a".to_string(),
                        },
                    ),
                    None => "no-history".to_string(),
                };
                log::info!(
                    "[OUTER schedule] inner-PIRLS cap transition (ARC bridge) arc_iter={} g_ratio={} {} prev={} new={} ({})",
                    arc_iter,
                    ratio_str,
                    snap_str,
                    prev,
                    cap,
                    if cap == 0 { "uncapped" } else { "capped" }
                );
            }
        }
        let stage_start = std::time::Instant::now();
        log::info!(
            "[STAGE] outer eval start order=ValueAndGradient dim={}",
            x.len()
        );
        let eval = self
            .obj
            .eval_with_order(x, OuterEvalOrder::ValueAndGradient)
            .map_err(|err| into_objective_error("outer eval failed", err))?;
        let eval = finite_outer_first_order_eval_or_error("outer eval failed", self.layout, eval)?;
        self.eval_count += 1;
        let g_norm = eval.gradient.iter().map(|v| v * v).sum::<f64>().sqrt();
        if self.g_norm_initial.is_none() && g_norm.is_finite() && g_norm > 0.0 {
            self.g_norm_initial = Some(g_norm);
        }
        if g_norm.is_finite() {
            self.last_g_norm = Some(g_norm);
        }
        self.last_value_grad_rho = Some(x.clone());
        log::info!(
            "[STAGE] outer eval end order=ValueAndGradient elapsed={:.3}s cost={:.6e} |g|={:.3e}",
            stage_start.elapsed().as_secs_f64(),
            eval.cost,
            g_norm,
        );
        log::info!(
            "[OUTER] eval#{n} (grad) cost={cost:.6e} |g|={gnorm:.3e} rho=[{rho}]",
            n = self.eval_count,
            cost = eval.cost,
            gnorm = g_norm,
            rho = x
                .iter()
                .map(|v| format!("{v:.3}"))
                .collect::<Vec<_>>()
                .join(","),
        );
        // Live-chart trial sample (ARC bridge first-order entry). Mirrors
        // the eval_hessian site below; both run once per outer iter, so the
        // chart's x-coord progresses on every accepted-or-rejected eval and
        // the accepted line moves only on rho-acceptance.
        crate::solver::visualizer::record_outer_eval(eval.cost, g_norm);
        Ok(FirstOrderSample {
            value: eval.cost,
            gradient: eval.gradient,
        })
    }
}

impl SecondOrderObjective for OuterSecondOrderBridge<'_> {
    fn eval_hessian(&mut self, x: &Array1<f64>) -> Result<SecondOrderSample, ObjectiveEvalError> {
        self.layout.validate_point_len(x, "outer eval failed")?;
        if let Some(feedback) = self.outer_inner_cap.as_ref() {
            // Use the observer-fed accepted-iter counter (opt 0.5.0
            // OptimizerObserver) instead of `eval_count / 2`; the
            // observer increments only on rho-accepted steps, so the
            // schedule no longer relaxes the cap on rejected trials.
            let arc_iter = feedback.accepted_iter.load(Ordering::Relaxed);
            let g_ratio = match (self.last_g_norm, self.g_norm_initial) {
                (Some(g), Some(g0)) if g0 > 0.0 => Some(g / g0),
                _ => None,
            };
            let snapshot = feedback.snapshot();
            let cap = first_order_inner_cap_schedule(arc_iter, g_ratio, snapshot);
            let prev = feedback.cap.swap(cap, Ordering::Relaxed);
            if prev != cap {
                let ratio_str = match g_ratio {
                    Some(r) => format!("{:.3e}", r),
                    None => "n/a".to_string(),
                };
                let snap_str = match snapshot {
                    Some(s) => format!(
                        "last_iters={} converged={} ift_residual={} accept_rho={}",
                        s.last_iters,
                        s.last_converged,
                        match s.last_ift_residual {
                            Some(r) => format!("{:.3e}", r),
                            None => "n/a".to_string(),
                        },
                        match s.last_accept_rho {
                            Some(r) => format!("{:.3}", r),
                            None => "n/a".to_string(),
                        },
                    ),
                    None => "no-history".to_string(),
                };
                log::info!(
                    "[OUTER schedule] inner-PIRLS cap transition (ARC bridge) arc_iter={} g_ratio={} {} prev={} new={} ({})",
                    arc_iter,
                    ratio_str,
                    snap_str,
                    prev,
                    cap,
                    if cap == 0 { "uncapped" } else { "capped" }
                );
            }
        }
        let stage_start = std::time::Instant::now();
        log::info!(
            "[STAGE] outer eval start order=ValueGradientHessian dim={}",
            x.len()
        );
        let eval = self
            .obj
            .eval_with_order(x, OuterEvalOrder::ValueGradientHessian)
            .map_err(|err| into_objective_error("outer eval failed", err))?;
        let eval = finite_outer_eval_or_error("outer eval failed", self.layout, eval)?;
        self.eval_count += 1;
        let g_norm = eval.gradient.iter().map(|v| v * v).sum::<f64>().sqrt();
        if self.g_norm_initial.is_none() && g_norm.is_finite() && g_norm > 0.0 {
            self.g_norm_initial = Some(g_norm);
        }
        if g_norm.is_finite() {
            self.last_g_norm = Some(g_norm);
        }
        self.last_value_grad_rho = Some(x.clone());
        log::info!(
            "[STAGE] outer eval end order=ValueGradientHessian elapsed={:.3}s cost={:.6e} |g|={:.3e}",
            stage_start.elapsed().as_secs_f64(),
            eval.cost,
            g_norm,
        );
        log::info!(
            "[OUTER] eval#{n} (hess) cost={cost:.6e} |g|={gnorm:.3e} rho=[{rho}]",
            n = self.eval_count,
            cost = eval.cost,
            gnorm = g_norm,
            rho = x
                .iter()
                .map(|v| format!("{v:.3}"))
                .collect::<Vec<_>>()
                .join(","),
        );
        let hessian = build_bridge_hessian_for_source(
            self.hessian_source,
            eval.hessian,
            self.materialize_operator_max_dim,
        )?;
        Ok(SecondOrderSample {
            value: eval.cost,
            gradient: eval.gradient,
            hessian,
        })
    }
}

// =====================================================================
// opt 0.4 matrix-free TR adapter (Phase 6)
// =====================================================================
//
// `OuterToOptHessianOperator` wraps gam's `OuterHessianOperator` so it
// can be passed to `opt::MatrixFreeTrustRegion` via
// `opt::HessianValue::Operator`. The two traits have nearly identical
// surfaces — the adapter is just shape/error translation:
//
//   gam::OuterHessianOperator              opt::HessianOperator
//     dim()                       <-->       dim()
//     matvec(v) -> Array1         <-->       apply_into(v, &mut out)
//     mul_mat(X) -> Array2        <-->       apply_mat(X)
//     materialization_capability  <-->       materialization
//     materialize_dense           <-->       materialize_dense
//
// gam errors are `String`; opt errors are `ObjectiveEvalError`. We
// promote everything to `ObjectiveEvalError::Fatal` because operator
// failures inside a solver step are not generally recoverable —
// shrinking the trust radius would not fix a dimension mismatch.
//
// `OuterOperatorBridge` is the bridge that implements
// `opt::OperatorObjective` for `gam`'s outer objective — parallel to
// `OuterSecondOrderBridge` but produces `OperatorSample` whose
// Hessian is `HessianValue::Operator(_)` (or `Dense(_)` when the
// operator declares an exact materialization route).

/// `opt::OptimizerObserver` that increments
/// `InnerProgressFeedback.accepted_iter` on every accepted outer
/// step. Replaces the bridge-side `eval_count / 2` heuristic on
/// routes that see trial-and-rejection probing (ARC dense,
/// matrix-free TR). The bridge's inner-cap schedule reads
/// `accepted_iter` from the feedback channel instead of inferring
/// it from raw eval counts.
pub(crate) struct OuterAcceptObserver {
    feedback: InnerProgressFeedback,
}

impl OptimizerObserver for OuterAcceptObserver {
    fn on_step_accepted(&mut self, info: &StepInfo) {
        log::trace!(
            "outer step accepted iter={} step_norm={:.3e} predicted_decrease={:.3e} actual_decrease={:.3e}",
            info.iter,
            info.step_norm,
            info.predicted_decrease,
            info.actual_decrease,
        );
        self.feedback.accepted_iter.fetch_add(1, Ordering::Relaxed);
    }
}

pub(crate) struct OuterToOptHessianOperator(Arc<dyn OuterHessianOperator>);

impl HessianOperator for OuterToOptHessianOperator {
    fn dim(&self) -> usize {
        self.0.dim()
    }

    fn apply_into(&self, v: &Array1<f64>, out: &mut Array1<f64>) -> Result<(), ObjectiveEvalError> {
        // Forward to gam's `OuterHessianOperator::apply_into` (default
        // impl wraps `matvec`; backends with a native into-buffer
        // kernel override for true zero-alloc CG iterations).
        self.0
            .apply_into(v, out)
            .map_err(|message| ObjectiveEvalError::Fatal {
                message: format!("outer Hessian operator apply_into failed: {message}"),
            })
    }

    fn apply_mat(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>, ObjectiveEvalError> {
        self.0
            .mul_mat(x)
            .map_err(|message| ObjectiveEvalError::Fatal {
                message: format!("outer Hessian operator mul_mat failed: {message}"),
            })
    }

    fn materialization(&self) -> HessianMaterialization {
        match self.0.materialization_capability() {
            OuterHessianMaterialization::Unavailable => HessianMaterialization::Unavailable,
            OuterHessianMaterialization::RepeatedHvp => HessianMaterialization::RepeatedHvp,
            OuterHessianMaterialization::BatchedHvp => HessianMaterialization::BatchedHvp,
            OuterHessianMaterialization::Explicit => HessianMaterialization::Explicit,
        }
    }

    fn materialize_dense(&self) -> Result<Array2<f64>, ObjectiveEvalError> {
        self.0
            .materialize_dense()
            .map_err(|message| ObjectiveEvalError::Fatal {
                message: format!("outer Hessian operator materialization failed: {message}"),
            })
    }
}

/// Translate a gam `HessianResult` into an `opt::HessianValue` for
/// consumption by `MatrixFreeTrustRegion`. `Analytic` becomes
/// `Dense`; `Operator` is wrapped in the adapter; `Unavailable` is
/// preserved (the solver's `HessianFallbackPolicy` decides what
/// happens then).
pub(crate) fn hessian_result_to_value(hessian: HessianResult) -> HessianValue {
    match hessian {
        HessianResult::Analytic(h) => HessianValue::Dense(h),
        HessianResult::Operator(op) => {
            HessianValue::Operator(Arc::new(OuterToOptHessianOperator(op)))
        }
        HessianResult::Unavailable => HessianValue::Unavailable,
    }
}

/// Bridge that exposes gam's outer objective as an
/// `opt::OperatorObjective`. Used on the matrix-free trust-region
/// route; the dense-Hessian / first-order routes still use
/// `OuterSecondOrderBridge` / `OuterFirstOrderBridge`.
pub(crate) struct OuterOperatorBridge<'a> {
    obj: &'a mut dyn OuterObjective,
    layout: OuterThetaLayout,
    /// Inner-PIRLS cap atomic, mirroring the BFGS / ARC bridges.
    outer_inner_cap: Option<InnerProgressFeedback>,
    /// Counts gradient/Hessian evaluations for the inner-cap schedule
    /// and progress logs.
    eval_count: usize,
    /// First observed `‖g‖`. Used by the inner-cap schedule's
    /// gradient-ratio gate.
    g_norm_initial: Option<f64>,
    /// `‖g‖` from the most recent eval.
    last_g_norm: Option<f64>,
    /// Most recent derivative-evaluation point, used to log value-probe
    /// displacement in line-search STAGE traces.
    last_value_grad_rho: Option<Array1<f64>>,
}

impl ZerothOrderObjective for OuterOperatorBridge<'_> {
    fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, ObjectiveEvalError> {
        // Uncap the inner solve for the matrix-free TR line-search cost probe.
        // Identical rationale to the BFGS / ARC bridges: the deciding cost must
        // be the true converged-inner objective the analytic gradient/operator
        // Hessian differentiate, never the scheduled gradient-path cap (which at
        // a trial ρ can stop the inner solve short and report a spurious `∞`,
        // freezing the TR at constant cost / |g|). This is the route the
        // ψ-bearing matern bernoulli marginal-slope fit takes (gam#787);
        // `eval_value_grad_op` restores the scheduled cap on the next call.
        if let Some(feedback) = self.outer_inner_cap.as_ref() {
            feedback
                .cap
                .store(SEED_SCREENING_UNCAPPED, Ordering::Relaxed);
        }
        self.layout
            .validate_point_len(x, "outer eval_cost failed")?;
        let trial_rho_distance = trial_rho_distance(self.last_value_grad_rho.as_ref(), x);
        let stage_start = std::time::Instant::now();
        log::info!(
            "[STAGE] outer eval start order=Value dim={} trial_rho_distance={:.3e} (operator bridge)",
            x.len(),
            trial_rho_distance
        );
        let eval = self
            .obj
            .eval_with_order(x, OuterEvalOrder::Value)
            .map_err(|err| into_objective_error("outer eval_cost failed", err))?;
        let cost = finite_cost_or_error("outer eval_cost failed", eval.cost)?;
        log::info!(
            "[STAGE] outer eval end order=Value elapsed={:.3}s cost={:.6e} trial_rho_distance={:.3e} (operator bridge)",
            stage_start.elapsed().as_secs_f64(),
            cost,
            trial_rho_distance
        );
        Ok(cost)
    }
}

impl FirstOrderObjective for OuterOperatorBridge<'_> {
    fn eval_grad(&mut self, x: &Array1<f64>) -> Result<FirstOrderSample, ObjectiveEvalError> {
        self.layout.validate_point_len(x, "outer eval failed")?;
        let eval = self
            .obj
            .eval_with_order(x, OuterEvalOrder::ValueAndGradient)
            .map_err(|err| into_objective_error("outer eval failed", err))?;
        let eval = finite_outer_first_order_eval_or_error("outer eval failed", self.layout, eval)?;
        let g_norm = eval.gradient.iter().map(|v| v * v).sum::<f64>().sqrt();
        if self.g_norm_initial.is_none() && g_norm.is_finite() && g_norm > 0.0 {
            self.g_norm_initial = Some(g_norm);
        }
        if g_norm.is_finite() {
            self.last_g_norm = Some(g_norm);
        }
        self.last_value_grad_rho = Some(x.clone());
        Ok(FirstOrderSample {
            value: eval.cost,
            gradient: eval.gradient,
        })
    }
}

impl OperatorObjective for OuterOperatorBridge<'_> {
    fn eval_value_grad_op(
        &mut self,
        x: &Array1<f64>,
    ) -> Result<OperatorSample, ObjectiveEvalError> {
        self.layout.validate_point_len(x, "outer eval failed")?;
        // Drive the outer-aware inner-PIRLS cap, mirroring
        // OuterSecondOrderBridge::eval_grad / eval_hessian. Each
        // accepted outer iter calls eval_value_grad_op exactly once
        // (the matrix-free TR's inner CG uses HVPs, not full
        // evaluations), so we increment per call without the /2 the
        // ARC bridge needs.
        if let Some(feedback) = self.outer_inner_cap.as_ref() {
            let g_ratio = match (self.last_g_norm, self.g_norm_initial) {
                (Some(g), Some(g0)) if g0 > 0.0 => Some(g / g0),
                _ => None,
            };
            let snapshot = feedback.snapshot();
            let cap = first_order_inner_cap_schedule(self.eval_count, g_ratio, snapshot);
            let previous_cap = feedback.cap.swap(cap, Ordering::Relaxed);
            if previous_cap != cap {
                log::trace!("outer operator bridge updated inner cap from {previous_cap} to {cap}");
            }
        }
        let stage_start = std::time::Instant::now();
        log::info!(
            "[STAGE] outer eval start order=ValueGradientHessian dim={} (operator bridge)",
            x.len(),
        );
        let eval = self
            .obj
            .eval_with_order(x, OuterEvalOrder::ValueGradientHessian)
            .map_err(|err| into_objective_error("outer eval failed", err))?;
        let eval = finite_outer_eval_or_error("outer eval failed", self.layout, eval)?;
        self.eval_count += 1;
        let g_norm = eval.gradient.iter().map(|v| v * v).sum::<f64>().sqrt();
        if self.g_norm_initial.is_none() && g_norm.is_finite() && g_norm > 0.0 {
            self.g_norm_initial = Some(g_norm);
        }
        if g_norm.is_finite() {
            self.last_g_norm = Some(g_norm);
        }
        self.last_value_grad_rho = Some(x.clone());
        log::info!(
            "[STAGE] outer eval end elapsed={:.3}s cost={:.6e} |g|={:.3e} (operator bridge)",
            stage_start.elapsed().as_secs_f64(),
            eval.cost,
            g_norm,
        );
        Ok(OperatorSample {
            value: eval.cost,
            gradient: eval.gradient,
            hessian: hessian_result_to_value(eval.hessian),
        })
    }
}

// Helpers preserved across the Phase 6 rewrite. Both were previously
// shared with `run_operator_trust_region` (now deleted in favor of
// `opt::MatrixFreeTrustRegion`), but they remain in use by the dense
// ARC and BFGS arms of the seed loop.

#[inline]
pub(crate) fn project_to_bounds(
    x: &Array1<f64>,
    bounds: Option<&(Array1<f64>, Array1<f64>)>,
) -> Array1<f64> {
    match bounds {
        Some((lower, upper)) => {
            let mut out = x.clone();
            for idx in 0..out.len() {
                out[idx] = out[idx].clamp(lower[idx], upper[idx]);
            }
            out
        }
        None => x.clone(),
    }
}

/// Translate an `OuterEval`'s Hessian into the `Option<Array2<f64>>`
/// shape expected by `opt::SecondOrderSample`, enforcing the contract
/// implied by the planner's `HessianSource`.
///
/// For `HessianSource::Analytic` (the exact second-order route) a missing
/// or non-materializable Hessian is FATAL: returning `None` here would
/// invite `opt::SecondOrderCache::finite_difference_hessian` to silently
/// estimate the Hessian by finite-differencing the gradient, which (a)
/// throws away the analytic structure the route was selected for, and
/// (b) costs O(K) full outer evaluations per ARC iteration — at large-scale
/// scale, hours of work per silently-mis-routed step. The right
/// behavior on a planner/runtime mismatch is to surface it loudly so
/// the seed loop can either retry, demote the plan, or fail the seed.
///
/// Operator Hessians that *are* cheaply materializable (the operator's
/// `materialization_capability` reports `Explicit` / `BatchedHvp` and the
/// dimension is below `materialize_operator_max_dim`) are converted to
/// dense in-place so dense ARC can run an exact factorization. Operator
/// Hessians that are NOT cheaply materializable should never arrive
/// here: the seed loop routes those to `run_operator_trust_region`
/// before constructing the bridge. Reaching this branch on the analytic
/// route means the runtime contradicted the seed-time decision, which
/// is the same kind of mismatch we treat as fatal.
///
/// For `HessianSource::BfgsApprox`, `EfsFixedPoint`, and
/// `HybridEfsFixedPoint` we deliberately return `None`: those routes do
/// not consume an analytic Hessian and feed the Hessian into a
/// quasi-Newton/fixed-point update instead. (Today these `HessianSource`
/// variants don't actually drive `opt`'s second-order solvers, but the
/// match preserves the original behavior in case a future routing
/// reuses this bridge.)
pub(crate) fn build_bridge_hessian_for_source(
    source: HessianSource,
    hessian: HessianResult,
    materialize_operator_max_dim: usize,
) -> Result<Option<Array2<f64>>, ObjectiveEvalError> {
    match source {
        HessianSource::Analytic => match hessian {
            HessianResult::Analytic(h) => Ok(Some(h)),
            HessianResult::Operator(op)
                if op.materialization_capability().is_available()
                    && op.dim() <= materialize_operator_max_dim =>
            {
                op.materialize_dense()
                    .map(Some)
                    .map_err(|message| ObjectiveEvalError::Fatal {
                        message: format!(
                            "outer Hessian operator materialization failed: {message}"
                        ),
                    })
            }
            HessianResult::Operator(op) => Err(ObjectiveEvalError::Fatal {
                message: format!(
                    "outer plan declared HessianSource::Analytic but the runtime returned a \
                     non-materializable Hessian operator (dim={}, materialization={:?}); \
                     finite-difference Hessian estimation is not permitted on the analytic route",
                    op.dim(),
                    op.materialization_capability(),
                ),
            }),
            HessianResult::Unavailable => Err(ObjectiveEvalError::Fatal {
                message: "outer plan declared HessianSource::Analytic but the runtime returned \
                          HessianResult::Unavailable; finite-difference Hessian estimation is \
                          not permitted on the analytic route"
                    .to_string(),
            }),
        },
        HessianSource::BfgsApprox
        | HessianSource::EfsFixedPoint
        | HessianSource::HybridEfsFixedPoint => Ok(None),
    }
}

pub(crate) struct OuterFixedPointBridge<'a> {
    obj: &'a mut dyn OuterObjective,
    layout: OuterThetaLayout,
    barrier_config: Option<BarrierConfig>,
    fixed_point_tolerance: f64,
    /// Consecutive HybridEFS iterations whose ψ block was zeroed after
    /// exhausting backtracking. When this reaches
    /// [`MAX_CONSECUTIVE_PSI_STAGNATION`], the bridge surfaces the
    /// [`EFS_FIRST_ORDER_FALLBACK_MARKER`] error so the runner aborts the
    /// HybridEFS attempt and the fallback ladder routes to a joint
    /// gradient-based solver where ψ stationarity ∇_ψ V = 0 can be enforced.
    consecutive_psi_zero_iters: usize,
}

impl OuterFixedPointBridge<'_> {
    fn reject_nonstationary_tiny_psi_step(
        &self,
        step: &Array1<f64>,
        psi_indices: Option<&[usize]>,
        psi_gradient: Option<&Array1<f64>>,
        cost: f64,
    ) -> Result<(), ObjectiveEvalError> {
        let Some(psi_indices) = psi_indices else {
            return Ok(());
        };
        let Some(psi_gradient) = psi_gradient else {
            return Ok(());
        };
        let psi_step_inf = psi_indices
            .iter()
            .map(|&idx| step[idx].abs())
            .fold(0.0_f64, f64::max);
        let psi_grad_inf = psi_gradient.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        if psi_step_inf <= self.fixed_point_tolerance && psi_grad_inf > self.fixed_point_tolerance {
            return Err(ObjectiveEvalError::recoverable(format!(
                "{} HybridEFS ψ nonstationary: ||Δψ||∞={:.3e} <= tol={:.3e} \
                 but raw ||gψ||∞={:.3e} (rho_dim={}, psi_dim={}, n_params={}, cost={:.6e})",
                EFS_FIRST_ORDER_FALLBACK_MARKER,
                psi_step_inf,
                self.fixed_point_tolerance,
                psi_grad_inf,
                self.layout.rho_dim(),
                self.layout.psi_dim,
                self.layout.n_params,
                cost,
            )));
        }
        Ok(())
    }
}

/// Maximum number of α halvings for the cost line search wrapping the EFS
/// step.
///
/// The Wood–Fasiolo paper proves that the EFS update direction is an *ascent
/// direction* for REML/LAML on penalty-like coordinates, but full-step
/// monotonicity is not guaranteed — both the original Fellner–Schall paper
/// and the extension recommend step-length control. We backtrack the entire
/// θ vector by halving α ∈ {1, 1/2, …, 1/2⁸ ≈ 0.004}, accepting the first
/// trial point with a strictly lower cost. With 8 halvings the smallest
/// trial step is ≈ 0.4% of the raw EFS step in every coordinate, which is
/// enough to clear pathologies near the identifiability boundary while
/// staying inside one cache-warm Hessian factorization budget.
pub(crate) const MAX_EFS_BACKTRACK: usize = 8;

/// Step components below this threshold (in θ-space) are treated as zero
/// for backtracking purposes — there is no point line-searching a step of
/// magnitude `1e-12`, and skipping the trial keeps the convergence path
/// numerically clean (no spurious cost decreases from ULP noise).
pub(crate) const EFS_NEGLIGIBLE_STEP: f64 = 1e-12;

/// Maximum infinity-norm of the EFS step (in θ-space) at which we skip the
/// cost line search and trust the multiplicative formula's quadratic
/// convergence. Above this, we always backtrack.
///
/// At small step magnitudes the canonical formula `Δρ = log((d−t)/q_eff)`
/// is itself a Newton step on the REML stationarity equation, with
/// quadratic local convergence. Under Wood–Fasiolo's Loewner-order
/// assumptions on the penalty derivative, sufficiently small steps are
/// always descent on `V`, so the line search would add an inner P-IRLS
/// solve per outer iteration with essentially zero chance of finding a
/// halving that beats the full step. The threshold is set to ~exp(0.5)
/// ≈ 1.65× change in any single λ_i (well inside the local-convergence
/// regime) and gates only the line-search call — the step itself is
/// applied unchanged, so correctness is preserved.
pub(crate) const EFS_LINESEARCH_THRESHOLD: f64 = 0.5;

/// Relative tolerance for the descent condition `c < current_cost` during
/// EFS backtracking. Without this, ULP-level cost noise near a fixed point
/// can cause spurious backtracking even when the step is mathematically
/// correct. We accept any trial whose cost is within
/// `EFS_COST_DESCENT_TOL · |current_cost|` of the current value.
pub(crate) const EFS_COST_DESCENT_TOL: f64 = 1e-12;

/// Maximum number of consecutive HybridEFS iterations whose ψ block was
/// zeroed before the bridge bails out and triggers a solver switch.
///
/// On hard problems (Matérn additive at large scale, Duchon60, anisotropic
/// joint penalties) a single zeroed-ψ iteration after exhausted backtracking
/// is already strong evidence the EFS ψ direction is not descent-correlated
/// at the current iterate; continuing on ρ alone with Δψ = 0 cannot enforce
/// ∇_ψ V = 0 and burns outer iterations on a non-stationary direction.
/// Bail out immediately so the fallback ladder routes to a joint
/// gradient-based solver (BFGS / L-BFGS) where ψ stationarity is part of
/// the optimality condition.
pub(crate) const MAX_CONSECUTIVE_PSI_STAGNATION: usize = 1;

impl FixedPointObjective for OuterFixedPointBridge<'_> {
    fn eval_step(&mut self, x: &Array1<f64>) -> Result<FixedPointSample, ObjectiveEvalError> {
        self.layout.validate_point_len(x, "outer EFS eval failed")?;
        let eval = match self.obj.eval_efs(x) {
            Ok(eval) => eval,
            Err(err @ EstimationError::GradientUnavailable { .. })
                if requests_immediate_first_order_fallback(&err.to_string()) =>
            {
                log::warn!(
                    "[STAGE] EFS -> gradient fallback: gradient unavailable at \
                     fixed-point dispatch; retrying with fixed-point disabled \
                     (rho_dim={}, psi_dim={}, n_params={})",
                    self.layout.rho_dim(),
                    self.layout.psi_dim,
                    self.layout.n_params,
                );
                return Err(ObjectiveEvalError::recoverable(format!(
                    "outer EFS eval failed: {err}"
                )));
            }
            Err(err) => return Err(into_objective_error("outer EFS eval failed", err)),
        };
        self.layout
            .validate_efs_eval(&eval, "outer EFS eval failed")?;
        if !eval.cost.is_finite() {
            return Err(ObjectiveEvalError::recoverable(
                "outer EFS eval failed: objective returned a non-finite cost".to_string(),
            ));
        }
        // Reject non-finite EFS step components at the bridge boundary with
        // full diagnostic context (which coord, its value, and whether it is
        // a ρ or ψ coord). Without this, a NaN/Inf step flows into the
        // hybrid-EFS backtrack loop, which halves it via `NaN * 0.5^k = NaN`
        // until backtracking exhausts, then silently zeros the ψ block and
        // applies only the ρ step — masking the analytic-gradient bug that
        // produced the NaN. The opt crate's FixedPoint::run also detects
        // this downstream (opt 0.2.2 lib.rs:4949) but surfaces only the bare
        // `NonFiniteStep` variant with no context, which is not actionable.
        if let Some((idx, value)) = eval.steps.iter().enumerate().find(|(_, v)| !v.is_finite()) {
            let psi_indices = eval.psi_indices.as_deref();
            let coord_kind = match psi_indices {
                Some(indices) if indices.contains(&idx) => "ψ",
                Some(_) => "ρ/τ",
                None => "ρ",
            };
            return Err(ObjectiveEvalError::recoverable(format!(
                "outer EFS eval failed: non-finite {coord_kind} step at coord {idx} \
                 (step[{idx}]={value}, rho_dim={}, psi_dim={}, n_params={}, cost={:.6e})",
                self.layout.rho_dim(),
                self.layout.psi_dim,
                self.layout.n_params,
                eval.cost,
            )));
        }
        if let Some(ref barrier_cfg) = self.barrier_config
            && let Some(ref beta) = eval.beta
        {
            // Scale-free precondition check for EFS. Wood–Fasiolo's
            // multiplicative log-λ update is derived under the
            // assumption that the inner Hessian is ≈ X'WX + S. A log
            // barrier adds τ/(β_j−l_j)² to the Hessian diagonal at the
            // constrained coords; when the tightest slack is much
            // smaller than the typical slack, that diagonal becomes
            // locally dominant and the EFS direction is no longer
            // guaranteed-ascent. Comparing slack *ratios* is
            // dimensionless — independent of τ, β scale, and the
            // inner-Hessian magnitude — which is exactly the regime
            // change EFS cannot represent. The earlier criterion
            // `barrier_curvature_is_significant(β, ref_diag=1.0, 0.01)`
            // was dimensionful and depended on three quantities the
            // bridge has no way to set correctly.
            //
            // Two principled triggers, each catching a distinct
            // failure mode of the EFS precondition:
            //  • `ratio = 0.1`        — asymmetric concentration:
            //    the worst slack is ≥10× tighter than the median.
            //    Catches the common "one coefficient hits its bound
            //    while others stay healthy" case.
            //  • `saturation = 1.0`   — absolute saturation:
            //    `max_j τ/Δ_j² ≥ 1`, i.e. at least one barrier-
            //    diagonal entry has reached the natural unit penalty
            //    scale. Catches the symmetric near-boundary regime
            //    that ratio-only checks would let through (median Δ
            //    also small, so min/median ratio stays near 1, but
            //    EFS's "ignore the barrier diagonal" assumption is
            //    still violated everywhere on the active set).
            const LOCAL_CONCENTRATION_RATIO: f64 = 0.1;
            const BARRIER_CURVATURE_SATURATION: f64 = 1.0;
            const BARRIER_CURVATURE_RELATIVE_THRESHOLD: f64 = 0.05;
            if let Some(hessian_scale) = eval.inner_hessian_scale
                && hessian_scale.is_finite()
                && hessian_scale > 0.0
                && barrier_cfg.barrier_curvature_is_significant(
                    beta,
                    hessian_scale,
                    BARRIER_CURVATURE_RELATIVE_THRESHOLD,
                )
            {
                return Err(ObjectiveEvalError::recoverable(format!(
                    "{} EFS barrier curvature significant relative to inner Hessian \
                         (rho_dim={}, psi_dim={}, n_params={}, cost={:.6e}, ref_diag={:.3e})",
                    EFS_FIRST_ORDER_FALLBACK_MARKER,
                    self.layout.rho_dim(),
                    self.layout.psi_dim,
                    self.layout.n_params,
                    eval.cost,
                    hessian_scale,
                )));
            }
            if barrier_cfg.barrier_curvature_locally_concentrated(
                beta,
                LOCAL_CONCENTRATION_RATIO,
                BARRIER_CURVATURE_SATURATION,
            ) {
                return Err(ObjectiveEvalError::recoverable(format!(
                    "{} EFS barrier curvature locally concentrated \
                         (rho_dim={}, psi_dim={}, n_params={}, cost={:.6e})",
                    EFS_FIRST_ORDER_FALLBACK_MARKER,
                    self.layout.rho_dim(),
                    self.layout.psi_dim,
                    self.layout.n_params,
                    eval.cost,
                )));
            }
        }
        let status = FixedPointStatus::Continue;

        let raw_step = Array1::from_vec(eval.steps);
        let psi_indices = eval.psi_indices.clone();
        self.reject_nonstationary_tiny_psi_step(
            &raw_step,
            psi_indices.as_deref(),
            eval.psi_gradient.as_ref(),
            eval.cost,
        )?;
        let max_step_abs = raw_step.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);
        let current_cost = eval.cost;
        if self.fixed_point_step_converged(x, &raw_step, psi_indices.as_deref()) {
            if psi_indices.is_some() {
                self.consecutive_psi_zero_iters = 0;
            }
            return Ok(FixedPointSample {
                value: current_cost,
                step: raw_step,
                status: FixedPointStatus::Stop,
            });
        }

        // Negligible raw step — the iteration is at (or numerically
        // indistinguishable from) a fixed point. Pass it through so the
        // outer step-norm convergence check fires; no point evaluating the
        // cost at x + 1e-30·s to chase ULP-level "improvements".
        if max_step_abs < EFS_NEGLIGIBLE_STEP {
            if psi_indices.is_some() {
                self.consecutive_psi_zero_iters = 0;
            }
            return Ok(FixedPointSample {
                value: current_cost,
                step: raw_step,
                status,
            });
        }

        // Small-step fast path. The canonical Wood–Fasiolo formula is
        // locally quadratically convergent, so once we are inside the
        // multiplicative-Newton basin (`||Δθ||∞ < EFS_LINESEARCH_THRESHOLD`)
        // a halving is essentially never accepted over the full step. Skip
        // the inner P-IRLS solve we'd otherwise burn on backtracking. When a
        // barrier is configured, every accepted rho-step must still pass
        // through the barrier-aware cost because feasibility can change even
        // under a small smoothing-parameter move. For hybrid runs we still
        // need to reset the ψ-stagnation counter.
        if self.barrier_config.is_none() && max_step_abs < EFS_LINESEARCH_THRESHOLD {
            if psi_indices.is_some() {
                self.consecutive_psi_zero_iters = 0;
            }
            return Ok(FixedPointSample {
                value: current_cost,
                step: raw_step,
                status,
            });
        }

        // ── Stage 1: full-vector cost backtracking ──
        //
        // Wood–Fasiolo gives ascent in the EFS direction but not full-step
        // monotonicity, so backtrack α ∈ {1, 1/2, …} on the *whole* step
        // vector (not just ψ). This is a uniform requirement: even on the
        // pure-ρ path, the additive log-λ formula is exact only at the
        // fixed point and is otherwise just a Newton-flavoured Wood–Fasiolo
        // surrogate that benefits from line search at large iterations.
        if let Some(scaled) = self.efs_backtrack(x, &raw_step, current_cost, MAX_EFS_BACKTRACK)? {
            if psi_indices.is_some() {
                self.consecutive_psi_zero_iters = 0;
            }
            return Ok(FixedPointSample {
                value: current_cost,
                step: scaled,
                status,
            });
        }

        // ── Stage 2 (hybrid only): ψ-zeroed retry ──
        //
        // Full-vector backtracking exhausted means *every* α we tried gave
        // a worse cost. On the hybrid path, the most common cause is a
        // bad ψ direction polluting an otherwise-good ρ step (preconditioned
        // gradient step on a near-singular ψ-ψ Gram matrix overshoots).
        // Try the ρ/τ block alone with the same backtracking schedule. If
        // that succeeds, we make progress on ρ this iteration; the ψ
        // stagnation counter advances and triggers the joint-solver
        // fallback once it crosses MAX_CONSECUTIVE_PSI_STAGNATION.
        if let Some(psi_idx) = psi_indices.as_ref() {
            let mut rho_only = raw_step.clone();
            for &i in psi_idx {
                rho_only[i] = 0.0;
            }
            let max_rho_abs = rho_only.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);
            if max_rho_abs >= EFS_NEGLIGIBLE_STEP
                && let Some(scaled) =
                    self.efs_backtrack(x, &rho_only, current_cost, MAX_EFS_BACKTRACK)?
            {
                self.consecutive_psi_zero_iters = self.consecutive_psi_zero_iters.saturating_add(1);
                log::info!(
                    "[HYBRID-EFS] full-vector backtrack exhausted; ρ/τ-only step \
                         accepted. Consecutive ψ-zero iters = {}",
                    self.consecutive_psi_zero_iters,
                );
                if self.consecutive_psi_zero_iters >= MAX_CONSECUTIVE_PSI_STAGNATION {
                    log::info!(
                        "[STAGE] HybridEFS -> joint gradient (BFGS/L-BFGS) fallback: \
                             {} consecutive ψ-zero iterations after exhausted backtracking \
                             (rho_dim={}, psi_dim={}, n_params={}, cost={:.6e})",
                        self.consecutive_psi_zero_iters,
                        self.layout.rho_dim(),
                        self.layout.psi_dim,
                        self.layout.n_params,
                        current_cost,
                    );
                    return Err(ObjectiveEvalError::recoverable(format!(
                        "{} HybridEFS ψ stagnation: {} consecutive iterations \
                             exhausted backtracking and zeroed ψ step \
                             (rho_dim={}, psi_dim={}, n_params={}, cost={:.6e})",
                        EFS_FIRST_ORDER_FALLBACK_MARKER,
                        self.consecutive_psi_zero_iters,
                        self.layout.rho_dim(),
                        self.layout.psi_dim,
                        self.layout.n_params,
                        current_cost,
                    )));
                }
                return Ok(FixedPointSample {
                    value: current_cost,
                    step: scaled,
                    status,
                });
            }
            // ρ/τ-only backtracking also failed — surface the joint-solver
            // fallback marker so the runner abandons EFS for this attempt.
            log::info!(
                "[STAGE] HybridEFS -> joint gradient fallback: ρ/τ-only step also \
                 failed all {} halvings (rho_dim={}, psi_dim={}, n_params={}, \
                 cost={:.6e})",
                MAX_EFS_BACKTRACK,
                self.layout.rho_dim(),
                self.layout.psi_dim,
                self.layout.n_params,
                current_cost,
            );
            return Err(ObjectiveEvalError::recoverable(format!(
                "{} HybridEFS step rejected after {} halvings on full vector \
                 and {} halvings on ρ/τ-only fallback \
                 (rho_dim={}, psi_dim={}, n_params={}, cost={:.6e})",
                EFS_FIRST_ORDER_FALLBACK_MARKER,
                MAX_EFS_BACKTRACK,
                MAX_EFS_BACKTRACK,
                self.layout.rho_dim(),
                self.layout.psi_dim,
                self.layout.n_params,
                current_cost,
            )));
        }

        // Pure-EFS path with full backtracking exhausted: there is no ψ
        // block to escape to. Surface the same fallback marker so the
        // runner switches to a gradient-based solver instead of looping.
        log::info!(
            "[STAGE] EFS -> gradient fallback: no α ∈ {{1, …, 2^-{}}} decreased the \
             cost (rho_dim={}, n_params={}, cost={:.6e})",
            MAX_EFS_BACKTRACK,
            self.layout.rho_dim(),
            self.layout.n_params,
            current_cost,
        );
        Err(ObjectiveEvalError::recoverable(format!(
            "{} EFS step rejected after {} halvings on pure-ρ vector \
             (rho_dim={}, n_params={}, cost={:.6e})",
            EFS_FIRST_ORDER_FALLBACK_MARKER,
            MAX_EFS_BACKTRACK,
            self.layout.rho_dim(),
            self.layout.n_params,
            current_cost,
        )))
    }
}

impl OuterFixedPointBridge<'_> {
    /// Backtrack the cost along `raw_step` by halving α ∈ {1, 1/2, …, 2^-k}
    /// up to `max_halvings` times. Returns `Some(α·raw_step)` for the first
    /// α that yields a strictly lower finite cost, `None` if every trial
    /// failed or evaluation errored. Eval errors at trial points are
    /// treated as step rejection (a common pathology in inner solves at
    /// over-aggressive λ jumps), not propagated.
    fn efs_backtrack(
        &mut self,
        x: &Array1<f64>,
        raw_step: &Array1<f64>,
        current_cost: f64,
        max_halvings: usize,
    ) -> Result<Option<Array1<f64>>, ObjectiveEvalError> {
        // Relaxed Armijo: accept any trial within ULP noise of the current
        // cost. Pure `<` rejects ULP-noise dithering on flat regions of V
        // and forces unnecessary halvings.
        let cost_floor = current_cost + EFS_COST_DESCENT_TOL * current_cost.abs().max(1.0);
        let mut alpha = 1.0_f64;
        for bt in 0..=max_halvings {
            let trial_step = raw_step * alpha;
            let trial = x + &trial_step;
            match self.obj.eval_cost(&trial) {
                Ok(c) if c.is_finite() && c <= cost_floor => {
                    if bt > 0 {
                        log::debug!(
                            "[EFS] backtrack accepted at α=2^-{bt}={alpha:.4e} \
                             after {bt} halvings (cost: {current_cost:.6e} → {c:.6e})"
                        );
                    }
                    return Ok(Some(trial_step));
                }
                Ok(c) => {
                    log::trace!(
                        "[EFS] backtrack α=2^-{bt}={alpha:.4e}: trial cost {c:.6e} \
                         not below current {current_cost:.6e}, halving"
                    );
                }
                Err(err) => {
                    log::trace!(
                        "[EFS] backtrack α=2^-{bt}={alpha:.4e}: trial eval failed \
                         ({err}), halving"
                    );
                }
            }
            alpha *= 0.5;
        }
        Ok(None)
    }

    fn fixed_point_step_converged(
        &self,
        x: &Array1<f64>,
        step: &Array1<f64>,
        psi_indices: Option<&[usize]>,
    ) -> bool {
        if x.len() != step.len() {
            return false;
        }
        for idx in 0..step.len() {
            let scale = match psi_indices {
                Some(indices) if indices.contains(&idx) => x[idx].abs().max(1.0),
                _ => 1.0,
            };
            let normalized = step[idx].abs() / scale;
            if !normalized.is_finite() || normalized > self.fixed_point_tolerance {
                return false;
            }
        }
        true
    }
}

pub(crate) fn solution_into_outer_result(
    solution: Solution,
    converged: bool,
    plan_used: OuterPlan,
) -> OuterResult {
    let mut result = OuterResult::new(
        solution.final_point,
        solution.final_value,
        solution.iterations,
        converged,
        plan_used,
    );
    result.final_grad_norm = solution.final_gradient_norm;
    result.final_gradient = solution.final_gradient;
    result.final_hessian = solution.final_hessian;
    result
}

pub(crate) fn outer_result_with_gradient_norm(
    rho: Array1<f64>,
    final_value: f64,
    iterations: usize,
    final_grad_norm: Option<f64>,
    converged: bool,
    plan_used: OuterPlan,
) -> OuterResult {
    let mut result = OuterResult::new(rho, final_value, iterations, converged, plan_used);
    result.final_grad_norm = final_grad_norm;
    result
}

pub(crate) fn outer_result_with_gradient(
    rho: Array1<f64>,
    final_value: f64,
    iterations: usize,
    final_grad_norm: Option<f64>,
    final_gradient: Option<Array1<f64>>,
    converged: bool,
    plan_used: OuterPlan,
) -> OuterResult {
    let mut result = outer_result_with_gradient_norm(
        rho,
        final_value,
        iterations,
        final_grad_norm,
        converged,
        plan_used,
    );
    result.final_gradient = final_gradient;
    result
}

use crate::inference::diagnostics::format_top_abs as format_top_abs_components;

pub(crate) fn bfgs_line_search_failure_message(
    context: &str,
    solution: &Solution,
    max_attempts: usize,
    failure_reason: impl std::fmt::Debug,
) -> String {
    let grad_norm = solution
        .final_gradient_norm
        .or_else(|| {
            solution
                .final_gradient
                .as_ref()
                .map(|gradient| gradient.iter().map(|v| v * v).sum::<f64>().sqrt())
        })
        .unwrap_or(f64::NAN);
    let gradient_detail = solution
        .final_gradient
        .as_ref()
        .map(|gradient| format_top_abs_components(gradient, "top_abs_gradient", 6))
        .unwrap_or_else(|| "top_abs_gradient=<unavailable>".to_string());
    format!(
        "{context}: BFGS line search failed; reason={failure_reason:?} \
         max_attempts={max_attempts} iterations={} final_value={:.6e} \
         |g|={:.3e} func_evals={} grad_evals={} {} {}",
        solution.iterations,
        solution.final_value,
        grad_norm,
        solution.func_evals,
        solution.grad_evals,
        format_top_abs_components(&solution.final_point, "top_abs_rho", 6),
        gradient_detail,
    )
}
