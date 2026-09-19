// [#780 line-count gate] The inner solve behind the undamped evidence log-determinant: the
// non-PD evidence-row test, the frozen collapse-prevention gates, the exact-A saddle descent and
// the gate-frozen inner convergence loop, as a further `impl SaeManifoldTerm` block. Included
// from `construction_quasi_laplace.rs`.

impl SaeManifoldTerm {
    pub(crate) fn is_undamped_evidence_row_non_pd(err: &ArrowSchurError) -> bool {
        matches!(
            err,
            ArrowSchurError::PerRowFactorFailed { reason, .. }
                if reason.contains("H_tt is non-PD at base ridge")
                    && reason.contains("evidence mode preserves the genuine Cholesky")
        )
    }

    /// Drive the inner `(t, β)` Newton solve to the KKT/step-converged optimum
    /// and return the final UNDAMPED (`ridge = 0`) joint-Hessian factor cache.
    ///
    /// The Laplace normaliser `½log|H|` is only the correct penalized quasi-Laplace criterion at
    /// the inner optimum `(t̂, β̂)`, so the criterion must refine the inner state
    /// until either the KKT gradient or the undamped Newton step meets tolerance
    /// before factoring. Crucially, **at the converged optimum the per-row
    /// `H_tt^(i)` blocks are PD**, so the undamped (`ridge = 0`) factorization
    /// succeeds; an off-optimum iterate (e.g. the initial seed, or a state
    /// stopped after only `inner_max_iter` steps) can have an indefinite /
    /// rank-deficient per-row block (`p_out = 1` → rank-1 `JᵀJ`, softmax
    /// assignment-sparsity negative logit curvature) that surfaces
    /// `PerRowFactorFailed` from the undamped `factor_one_row`. Both the dense
    /// (`penalized_quasi_laplace_criterion_with_cache`) and the streaming
    /// (`penalized_quasi_laplace_criterion_streaming_exact_with_lane`) criterion paths route through this same
    /// driver, so they converge to the identical inner state (#847).
    ///
    /// ⚠ #2509 — a shared inner state is NOT a shared log-determinant. #2330
    /// Phase-2 changed which OPERATOR each lane factors at that shared state:
    /// dense prices the exact observed information `A = B + ΔC`
    /// (`exact_observed_information_log_dets`), streaming prices the Arrow–Schur
    /// majorizer `B` (`streaming_exact_arrow_log_det_with_lane_and_system`). The #847 bit-identity
    /// claim held for `B` against `B` and does not survive that migration.
    /// Freeze the collapse-prevention gates for one criterion evaluation,
    /// returning whether they were ALREADY frozen so the caller can restore.
    ///
    /// One place, because the set has to be the same set. Before #2515 the freeze
    /// refreshed two gates — decoder repulsion and barrier coactivation — while
    /// `assemble_arrow_schur_scaled` refreshes THREE when unfrozen, the third being
    /// the #2343 amplitude barrier. So the amplitude gate was the only one that
    /// never got refreshed at the entry state at all: inside the frozen window the
    /// assembler skips it, and the freeze did not do it either, leaving it carrying
    /// whatever the PREVIOUS evaluation left behind. That is the same
    /// value-versus-gradient desync #1625 and #2343 each fixed for their own gate,
    /// reintroduced by the freeze that was supposed to prevent it.
    ///
    /// The list is exhaustive against its consumer by construction: this is the
    /// only producer, `assemble_arrow_schur_scaled`'s `if !streaming_gates_frozen`
    /// block is the only consumer, and a gate added to one without the other is a
    /// gate whose frozen value is not the entry state's.
    fn freeze_collapse_prevention_gates(&mut self) -> bool {
        let gates_were_frozen = self.streaming_gates_frozen;
        if !gates_were_frozen {
            self.refresh_decoder_repulsion_gate();
            self.refresh_barrier_coactivation_gate();
            self.refresh_amplitude_barrier_gate();
            self.streaming_gates_frozen = true;
        }
        gates_were_frozen
    }

    pub(crate) fn converge_inner_for_undamped_logdet(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        rho_fixed: &mut SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
        loss: &mut SaeManifoldLoss,
        criterion_fixed_point: &mut bool,
        options: &ArrowSolveOptions,
        refine_progress_extension: bool,
    ) -> Result<ArrowFactorCache, String> {
        // ONE CRITERION EVALUATION = ONE OBJECTIVE (#2228 Zeno ratchet). The
        // collapse-prevention gates (decoder repulsion, barrier coactivation)
        // historically re-froze at EVERY assembly, so each accepted refine /
        // terminal-Newton move slightly changed the objective being priced —
        // the stationary point walked away from the solver ~1.5% in ‖g‖ per
        // polish∘re-entry cycle (measured on the tier-0 fixtures: 54
        // consecutive committed Newton steps with monotonically RISING entry
        // ‖g‖ 1.01e-4 → 1.16e-4 against a 6.07e-5 band, then budget
        // refusal). Freezing the gates ONCE for the whole evaluation is the
        // same discipline the streaming fit already trusts
        // (`streaming_gates_frozen`, chunk-size-invariance pinned) and is
        // exactly what value/gradient consistency (#1026/#1625) wants at the
        // evaluation scope rather than per assembly. A NEW evaluation (new ρ,
        // or an evidence re-entry) still re-freezes from its own entry state,
        // so a settled state re-prices identically — the #2253 idempotence
        // certificate is preserved, and V(ρ) still tracks routing changes
        // across ρ moves.
        let gates_were_frozen = self.freeze_collapse_prevention_gates();
        let out = self.converge_inner_for_undamped_logdet_gate_frozen(
            target,
            rho,
            rho_fixed,
            registry,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
            loss,
            criterion_fixed_point,
            options,
            refine_progress_extension,
        );
        self.streaming_gates_frozen = gates_were_frozen;
        out
    }

    /// #2080 — descend one refused exact-A saddle at the evidence root.
    ///
    /// `directions` are what
    /// [`Self::exact_observed_information_log_dets_with_saddle_directions`] collects:
    /// unit vectors in the
    /// joint `(t, β)` cache layout, each with its basin curvature `μ < −floor`.
    /// Each is turned downhill (`gᵀd ≤ 0`) and the penalized objective is minimized
    /// along it by [`Self::minimize_objective_along`] with the curvature term `−μ`,
    /// so a direction whose decrease is second-order (`slope ≈ 0` at a KKT point)
    /// is searched from the step at which `½·|μ|·α²` reaches the material floor.
    ///
    /// Every refused direction is tried in turn, from the state the directions
    /// before it left, against that state's own gradient and objective. A
    /// direction is kept only when its committed, re-evaluated decrease clears
    /// `SAE_MANIFOLD_INNER_OBJECTIVE_STALL_REL_TOL · (1 + |f|)`, the floor every
    /// inner mover commits against. One pass therefore descends every refused mode
    /// that can be descended, instead of paying a refine and a dense
    /// re-materialization per mode. `true` means at least one direction committed
    /// and the caller converges again. Every commit lowers one gate-frozen
    /// objective by more than the floor, so repeated passes end. `false` means no
    /// refused direction realizes a material decrease, and the state is as it was
    /// found.
    fn descend_exact_a_saddle(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        cache: &ArrowFactorCache,
        directions: &[(Array1<f64>, f64)],
    ) -> Result<bool, String> {
        let total_t = cache.delta_t_len();
        let mut committed = 0usize;
        let mut last_floor = f64::NAN;
        for (vector, curvature) in directions {
            if vector.len() != total_t + cache.k {
                return Err(format!(
                    "SaeManifoldTerm::descend_exact_a_saddle: direction length {} != joint \
                     dimension {}",
                    vector.len(),
                    total_t + cache.k,
                ));
            }
            // The gradient and objective of the CURRENT state: an earlier direction
            // may already have moved it.
            let system = self.assemble_arrow_schur(target, rho, registry)?;
            let mut gradient = Array1::<f64>::zeros(total_t + cache.k);
            let mut offset = 0usize;
            for row in &system.rows {
                if offset + row.gt.len() > total_t {
                    break;
                }
                for (axis, &value) in row.gt.iter().enumerate() {
                    gradient[offset + axis] = value;
                }
                offset += row.gt.len();
            }
            if offset != total_t || system.gb.len() != cache.k {
                return Err(format!(
                    "SaeManifoldTerm::descend_exact_a_saddle: the assembled gradient has {offset} \
                     row coordinates and border width {}, but the evidence cache has {total_t} \
                     and {}",
                    system.gb.len(),
                    cache.k,
                ));
            }
            for (index, &value) in system.gb.iter().enumerate() {
                gradient[total_t + index] = value;
            }
            drop(system);
            let base_objective = self.penalized_objective_total(target, rho, registry, 1.0)?;
            if !base_objective.is_finite() {
                break;
            }
            let material_floor =
                SAE_MANIFOLD_INNER_OBJECTIVE_STALL_REL_TOL * (1.0 + base_objective.abs());
            last_floor = material_floor;
            let along = gradient.dot(vector);
            let direction = if along > 0.0 { -vector } else { vector.clone() };
            let slope = along.abs();
            let snapshot = self.snapshot_mutable_state();
            let line = self.minimize_objective_along(
                target,
                rho,
                registry,
                direction.view(),
                total_t,
                base_objective,
                slope,
                -*curvature,
                material_floor,
                &snapshot,
            )?;
            if !(line.alpha > 0.0 && base_objective - line.value > material_floor) {
                continue;
            }
            if let Err(err) = self.apply_newton_step(
                direction.slice(s![..total_t]),
                direction.slice(s![total_t..]),
                line.alpha,
            ) {
                self.restore_mutable_state(&snapshot).map_err(|restore_err| {
                    format!(
                        "SaeManifoldTerm::descend_exact_a_saddle: committed step application \
                         failed ({err}); restoring the pre-descent state also failed \
                         ({restore_err})"
                    )
                })?;
                return Err(format!(
                    "SaeManifoldTerm::descend_exact_a_saddle: committed step application: {err}"
                ));
            }
            let committed_objective =
                match self.penalized_objective_total(target, rho, registry, 1.0) {
                    Ok(value) => value,
                    Err(err) => {
                        self.restore_mutable_state(&snapshot).map_err(|restore_err| {
                            format!(
                                "SaeManifoldTerm::descend_exact_a_saddle: committed objective \
                                 evaluation failed ({err}); restoring the pre-descent state \
                                 also failed ({restore_err})"
                            )
                        })?;
                        return Err(format!(
                            "SaeManifoldTerm::descend_exact_a_saddle: committed objective \
                             evaluation: {err}"
                        ));
                    }
                };
            let decrease = base_objective - committed_objective;
            if !(committed_objective.is_finite() && decrease > material_floor) {
                self.restore_mutable_state(&snapshot)
                    .map_err(|err| format!("SaeManifoldTerm::descend_exact_a_saddle: {err}"))?;
                continue;
            }
            committed += 1;
            log::info!(
                "[SAE-SADDLE] descended a refused exact-A direction: basin curvature \
                 {curvature:.6e}, slope {slope:.6e}, α={:.6e}, objective \
                 {base_objective:.10e} → {committed_objective:.10e} (decrease {decrease:.6e}, \
                 floor {material_floor:.6e}, {} objective evaluations)",
                line.alpha,
                line.objective_evaluations,
            );
        }
        if committed == 0 {
            log::info!(
                "[SAE-SADDLE] none of {} refused exact-A direction(s) realizes a decrease above \
                 the material floor {last_floor:.6e}; the saddle stays refused",
                directions.len(),
            );
        }
        Ok(committed > 0)
    }

    fn converge_inner_for_undamped_logdet_gate_frozen(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        rho_fixed: &mut SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
        loss: &mut SaeManifoldLoss,
        criterion_fixed_point: &mut bool,
        options: &ArrowSolveOptions,
        refine_progress_extension: bool,
    ) -> Result<ArrowFactorCache, String> {
        // `inner_max_iter == 0` is a genuine FREEZE of the inner `(t, β)` state
        // — a verbatim warm-start reuse, not a convergence request (gam#577/#579,
        // #850). The convergence/refinement loop below MUST NOT run even one
        // Newton step in that case (the old `inner_max_iter.max(1)` floor moved
        // β off the seed), so we factor exactly once at the frozen iterate and
        // return that undamped cache without invoking the stationarity gate.
        // The caller has already run
        // `run_joint_fit_arrow_schur_for_quasi_laplace(..., 0, ...)`,
        // which under the `max_iter == 0` freeze (gam#577/#579, #850) runs ONLY
        // the β-neutral basis refresh and returns the loss without touching β —
        // it skips the rank-reduction, frame activation, re-seed guards, and the
        // #1026 decoder-LSQ polish that would otherwise refit β off the seed — so
        // `self` is at the warm-start β here.
        if inner_max_iter == 0 {
            let mut sys = self
                .assemble_arrow_schur(target, rho, registry)
                .map_err(|err| {
                    format!("SaeManifoldTerm::penalized_quasi_laplace_criterion: {err}")
                })?;
            // #1095/#2228 — same decoupling as the stall / gradient-stationary
            // acceptance paths. This frozen warm-start criterion log-det is read from
            // the ridge-0 factor below, which is non-PD BY CONSTRUCTION on an
            // over-parametrized chart (a rank-1 radial null per row). Per-row
            // spectral deflation only fires when `row_gauge_deflation.is_some()`, and
            // the decoded-derivative gauge predicate (`decoded_motion_is_rounding_zero`) can
            // leave it None on exactly the flat axis that carries the null — so
            // force the evidence system to opt into per-row spectral discovery: the
            // null is unit-stiffness deflated (`log 1 = 0`, ρ-independent) and the
            // frozen log-det is finite, instead of refusing a rescuable warm-start
            // reuse. A full-rank block has no sub-floor eigenvalue and is untouched.
            Self::ensure_row_gauge_deflation_for_quasi_laplace(&mut sys);
            let factored =
                solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, options).map_err(|err| {
                    format!("SaeManifoldTerm::penalized_quasi_laplace_criterion: {err}")
                })?;
            // The frozen-state Newton step (factored.0, factored.1) is discarded
            // — only the undamped factor cache (factored.2) is consumed for the
            // log-det / selected-inverse traces; β stays at the warm-start seed.
            return Ok(factored.2);
        }
        let mut total_inner_iter = inner_max_iter;
        let accepted_base_refine_iter = inner_max_iter.max(1).saturating_mul(16).max(64);
        let value_probe_base_refine_iter = inner_max_iter.max(1).saturating_mul(4).max(16);
        let base_refine_iter = if refine_progress_extension {
            accepted_base_refine_iter
        } else {
            value_probe_base_refine_iter
        };
        let progress_refine_iter = if refine_progress_extension {
            inner_max_iter.max(1).saturating_mul(64).max(256)
        } else {
            base_refine_iter
        };
        let mut previous_refine_grad_norm: Option<f64> = None;
        // #2234 — one progress-gated extra refinement window (see the budget
        // escalation at the non-convergence refusal below). 0 until granted.
        let mut budget_escalation_extra = 0usize;
        // #2228 certificate-metric-keyed escalation state: the ½λ²/scale
        // decrement certificate measured at the last budget-limit hit, and a
        // pure anti-runaway cap on how many certificate-paid windows one
        // evaluation may earn (the geometric-progress gate below is the real
        // bound; the cap only guards against a certificate oscillating around
        // the progress threshold).
        let mut last_limit_certificate: Option<f64> = None;
        let mut certificate_escalations = 0usize;
        // #2080 -- the polish-paid window granted at the budget-exhaustion branch
        // below is gated on `terminal_newton_polish_armed`, a FLAG that any
        // materially descending refine round re-arms. Its two sibling lanes both
        // carry real counters (the certificate lane's cap right here, and the
        // progress lane's `budget_escalation_extra == 0` one-shot); this one does
        // not. Each grant resets the effective limit to
        // `total_inner_iter + refine_limit`, so arm -> grant -> descend -> re-arm
        // extends ONE criterion evaluation without bound while `criterion_calls`
        // stays flat -- which is why the probe-budget fixtures time out instead of
        // failing their `<= 64` assertion. A flag is not a budget.
        //
        // Past the cap the evaluation falls through to the final-gate certificate
        // and then to the typed non-convergence refusal the outer already maps to
        // +inf, so nothing is silently accepted.
        let mut polish_escalations = 0usize;
        const POLISH_ESCALATION_ANTI_RUNAWAY_CAP: usize = 2;
        const CERTIFICATE_ESCALATION_PROGRESS: f64 = 0.7;
        const CERTIFICATE_ESCALATION_ANTI_RUNAWAY_CAP: usize = 8;
        // #1051 — objective-stagnation convergence. On an ill-conditioned
        // penalised bilinear fit (the euclidean / Duchon decoder × latent
        // coordinate system on a trivial shape), the inner Newton crawls: each
        // refine round lowers the penalised objective by a shrinking amount while
        // the KKT gradient and the undamped step stay above their relative
        // tolerances (the near-singular Schur amplifies the step in the
        // weakly-identified decoder direction). The grad-OR-step gate then never
        // fires and the solve is rejected as "did not converge". A Newton/LM
        // iterate whose objective has stopped decreasing is diagnosed as a
        // numerical stall. It is not a stationary envelope root unless the raw
        // or quotient KKT residual also meets its gate, so a persistent stall is
        // refused instead of ranked.
        //
        // ONE SCALAR: the stall detector prices `penalized_objective_total` —
        // the exact scalar the inner Armijo line search descends and the KKT
        // gradient differentiates — NOT the native-terms-only `loss.total()`.
        // The KKT gradient carries the registry analytic penalties, decoder
        // repulsion, and the Jeffreys separation barrier; a trajectory
        // descending the full objective by trading data-fit against those
        // terms shows a flat or non-monotone `loss.total()` (spurious stall),
        // and vice versa. Progress, descent, and stationarity must be measured
        // on the same function.
        let entry_loss_total = self
            .penalized_objective_total(target, rho, registry, 1.0)
            .map_err(|err| format!("SaeManifoldTerm::penalized_quasi_laplace_criterion: {err}"))?;
        let mut previous_loss_total = entry_loss_total;
        let mut refine_rounds: usize = 0;
        // Consecutive stall rounds. Once this reaches
        // `SAE_MANIFOLD_INNER_OBJECTIVE_STALL_MIN_ROUNDS` without a KKT
        // certificate, returning `Err` is the same "did not converge" signal that
        // `is_recoverable_value_probe_refusal` already handles, so the outer
        // BFGS treats it as an INFINITY probe and tries a different ρ instead
        // of looping forever burning the extended progress budget.  Without
        // this counter the stagnation handler fell through when the undamped
        // factor failed and the loop kept extending via `saw_refine_progress`
        // from earlier rounds, accumulating minutes of wasted work (#1094).
        let mut consecutive_objective_stalls: usize = 0;
        // #2228 — the ½λ²/scale-MINIMIZING iterate seen across the inner
        // solve (captured in the polish, where the decrement is computed per
        // step). ACCEPTANCE KEYS ON THE CERTIFICATE, NOT ‖g‖ — the ‖g‖-min
        // and ½λ²/scale-min iterates DIFFER near an indefinite mode, and the
        // stall acceptance is priced on ½λ²/scale, so that is the honest
        // best-seen. Read ONLY at the terminal give-up exits (FINAL-GATE +
        // the non-convergence refusals); the continuation never reads it, so
        // the iterating trajectory is byte-identical (unlike the prior
        // restore-in-polish variants). The band is UNCHANGED: the decrement
        // certificate floors at ~ε (quadratic in g), 8 orders under the 1e-8
        // band, so a plateau above the band is a solver stall — reported
        // honestly at best-seen, never accepted past the band.
        let mut best_seen: Option<(f64, f64, SaeManifoldMutableState)> = None;
        let refine_started = std::time::Instant::now();
        // #2267 — name this phase to the process monitor. gam-sae registered no
        // monitor scope, so every heartbeat of an SAE fit read `instrumented_threads=0
        // active=<idle>` whatever the fit was doing. The guard lives to every exit of
        // this function.
        let criterion_scope = gam_runtime::process_monitor::track_scope(format!(
            "sae criterion inner converge inner_max_iter={inner_max_iter}"
        ));
        // #2228 Stage-2 / #2132 — whether the terminal exact-Newton polish
        // (`terminal_exact_newton_polish`) is armed for the NEXT objective-stall
        // plateau. Re-armed by any materially-descending refine round, so a
        // long solve that alternates MM plateaus with real descent gets one
        // polish per plateau instead of a fixed ration (the measured K=3
        // planted-circle fit descended 5793 → 4347 across three plateaus and
        // was refused at the third purely because a 2-invocation budget was
        // spent — at a point 100× LESS stationary than the plateaus the budget
        // had rescued). Runaway is impossible by construction: invoking the
        // polish disarms it, and only an intervening materially-descending
        // round re-arms, so a plateau the polish cannot unlock refuses on its
        // second visit with the polish disarmed.
        let mut terminal_newton_polish_armed = true;
        // #2762 — whether the gauge-orbit block descent is armed for the NEXT
        // objective-stall fixed-point claim of THIS loop. Same discipline as the
        // polish above and as the joint fit's own arming: consulting disarms,
        // and only a materially-descending refine round re-arms, so a plateau
        // the block cannot unlock refuses on its second visit.
        let mut gauge_block_armed = true;
        loop {
            let mut sys = self
                .assemble_arrow_schur(target, rho, registry)
                .map_err(|err| {
                    format!("SaeManifoldTerm::penalized_quasi_laplace_criterion: {err}")
                })?;
            // Evidence-only factorization: the Newton step (Δt, Δβ) is discarded
            // and only the factor cache is consumed — the exact undamped log-det
            // and the selected-inverse traces. As ρ sweeps to extremes (e.g. a
            // wide ARD-α sweep), H_tt is genuinely PD but can be ill-conditioned;
            // the standard Direct guard rejects that to protect Newton-step
            // accuracy, but the log-det is exact from diag(L) regardless of the
            // condition number and the traces only need the (PD) factor. So
            // tolerate the ill-conditioning rejection here (a genuine non-PD pivot
            // still errors). The cache stays undamped at ridge=0, so
            // `ArrowFactorCache::arrow_log_det` remains exact.
            // The exact KKT stationarity residual is the joint gradient
            // ‖g‖ = √(Σ_i ‖g_t^(i)‖² + ‖g_β‖²), read straight off the assembled
            // system. Unlike the Newton step Δ = H⁻¹g, the gradient is
            // factorisation-independent: it is NOT amplified by an inverse, so a
            // genuinely stationary but ill-conditioned fit (tiny g, possibly large
            // Δ in a flat direction) is correctly recognised as converged. The
            // positive-definite evidence Direct factor below documents that
            // its Δ may be inaccurate in exactly those flat directions, so using Δ
            // alone as the convergence gate would falsely reject healthy fits.
            let grad_norm_sq: f64 = Self::system_grad_norm_sq(&sys);
            let grad_norm = grad_norm_sq.sqrt();
            let lambda_smooth = rho_fixed.lambda_smooth_vec()?;
            let quotient_grad_norm =
                self.quotient_gradient_norm_from_system(&sys, grad_norm_sq, &lambda_smooth);
            let iterate_scale = self.inner_iterate_scale();
            // Scaled KKT-gradient tolerance for stationarity. Convergence is
            // accepted only on raw or quotient gradient stationarity; the Newton
            // step can collapse along the chart gauge before the quotient
            // residual is small, so it never gates convergence (it is only
            // computed — and logged — at the accepted stationary factorization).
            let grad_tolerance = SAE_MANIFOLD_INNER_GRAD_REL_TOL * iterate_scale;
            if !grad_norm_sq.is_finite() {
                return Err(format!(
                    "SaeManifoldTerm::penalized_quasi_laplace_criterion: undamped inner KKT residual is non-finite \
                     at the inner optimum (‖g‖²={grad_norm_sq}); the joint Hessian \
                     factorisation is degenerate at this ρ"
                ));
            }
            // #2080 criterion-cost restructure — the Laplace normaliser ½log|H|
            // is the penalized quasi-Laplace criterion ONLY at the inner KKT optimum, so the FULL
            // undamped Direct factorization (dense border β-Schur assembly
            // `O(n·q·k²)` plus the `O(k³)` border Cholesky / eigen-floor, with
            // `k = border_dim = Σ_k M_k·p`) is taken exactly ONCE — at the
            // stationary iterate whose cache is returned. Historically it was
            // ALSO taken on every non-stationary refine round and immediately
            // discarded: the pre-stationarity Newton step Δ = H⁻¹g was never
            // applied (the refinement below re-enters `run_joint_fit_arrow_schur`
            // from the same state) and convergence is judged on the
            // factorisation-independent KKT gradient alone, so the dense border
            // factor bought nothing at a non-stationary iterate. That discarded
            // cubic factor was the dominant wide-`p` criterion cost (#2080).
            //
            // A non-stationary round needs exactly ONE bit from the
            // factorization: whether the undamped per-row H_tt blocks are PD —
            // the infeasible-ρ signal that drives the #2080 probe fast-refusal
            // and the refine-budget escalation below.
            // `probe_undamped_evidence_row_factors` surfaces that identical
            // verdict (same #1038 ordered Beta--Bernoulli self-term downdate, same gauge/spectral
            // deflation policy, same `factor_one_row` error text) at the
            // per-row-only `O(N·q³)` cost, never forming the border Schur.
            //
            // EXACTNESS: the refinement trajectory is unchanged (the same
            // sequence of `run_joint_fit_arrow_schur` calls runs between the
            // same assembled systems), the stationary iterate is unchanged, and
            // the returned cache is the factorization of the same system at
            // that iterate — identical to what the historical loop returned —
            // so the criterion VALUE is untouched. Only work whose result was
            // provably discarded is removed.
            let gradient_stationary =
                Self::quasi_laplace_kkt_stationary(grad_norm, quotient_grad_norm, grad_tolerance);
            // #2253 — a coarse KKT-band hit is only an admission signal, not the
            // differentiable root the IFT gradient assumes. A bounded evidence
            // chunk reports `fixed_point` only when a whole re-entry accepted no
            // strict Newton/proximal step and made no temperature/polish state
            // transition. A stationary-but-moving state therefore falls through
            // to the SAME progress-extension/refusal accounting as the ordinary
            // refinement path below; it cannot factor or return from this block.
            // No new tolerance or work budget is introduced: either the existing
            // progress-paid grant reaches the true no-descent recurrence, or the
            // existing non-convergence refusal wins.
            if gradient_stationary && *criterion_fixed_point {
                // #2228 — price the root, not wherever the band admitted the state
                // (`refine_accepted_root`). The refined root keeps this arm's witnesses: a
                // non-finite undamped step, or a failed quotient-step projection, refuses.
                if let Some(refined) = self.refine_accepted_root(
                    target,
                    Some(rho),
                    rho_fixed,
                    registry,
                    &lambda_smooth,
                    options,
                    inner_max_iter,
                    learning_rate,
                    ridge_ext_coord,
                    ridge_beta,
                    loss,
                    criterion_fixed_point,
                    &mut total_inner_iter,
                )? {
                    let step_norm_sq = refined.delta_t.dot(&refined.delta_t)
                        + refined.delta_beta.dot(&refined.delta_beta);
                    if !step_norm_sq.is_finite() {
                        return Err(format!(
                            "SaeManifoldTerm::penalized_quasi_laplace_criterion: undamped inner residual \
                             is non-finite at the refined root (‖Δ‖²={step_norm_sq}); the joint Hessian \
                             factorisation is degenerate at this ρ"
                        ));
                    }
                    let quotient_step_norm_sq = self.quotient_newton_step_norm_sq(
                        refined.delta_t.view(),
                        refined.delta_beta.view(),
                        step_norm_sq,
                        &lambda_smooth,
                    )?;
                    log::info!(
                        "[SAE-ACCEPT] kkt fixed point at the refined root: ‖Δ‖={:.6e} \
                         ‖Π⊥null Δ‖={:.6e} after {total_inner_iter} inner iterations",
                        step_norm_sq.sqrt(),
                        quotient_step_norm_sq.sqrt(),
                    );
                    drop(criterion_scope);
                    return Ok(refined.cache);
                }
                // #1095/#2228 — decouple this ACCEPT from undamped-factor success,
                // the same acceptance-local pattern as the stall path below. A
                // cleanly-fit over-parametrized chart (d_atom=2 on intrinsic 1-D
                // data) is gradient-STATIONARY — the tangent is fit and the rank-1
                // radial null contributes ZERO gradient — so it lands HERE rather
                // than the objective-stall path, yet its ridge-0 per-row H_tt is
                // non-PD by construction. Force the acceptance factor to opt into
                // per-row spectral discovery so the null is unit-stiffness deflated
                // (`log 1 = 0`, ρ-independent) and the criterion log-det is finite.
                // This does NOT touch the undamped #2080 probe: the probe runs only
                // in the non-stationary branch below, which THIS block never reaches
                // (every arm returns), and a non-stationary iteration never installs
                // this deflation — so `sys` stays undamped for the probe.
                Self::ensure_row_gauge_deflation_for_quasi_laplace(&mut sys);
                let (delta_t, delta_beta, cache): (Array1<f64>, Array1<f64>, ArrowFactorCache) =
                    match solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, options) {
                        Ok(factored) => factored,
                        Err(err) if Self::is_undamped_evidence_row_non_pd(&err) => {
                            // K>1: the softmax/ordered Beta--Bernoulli logit–coordinate Gauss-Newton
                            // cross-terms (H_zt = J_z^T J_t, assembled row-locally from
                            // the assignment JVP × basis JVP) can make a per-row H_tt
                            // indefinite at the TRUE KKT stationary point — when two
                            // atoms' decoders specialise in opposite directions the
                            // Schur complement of the logit block goes negative even
                            // though the priors and the full-joint GN term are PSD.
                            //
                            // The undamped criterion factor conditions that block the
                            // PRINCIPLED way: with per-row spectral discovery now
                            // force-enabled above (`row_gauge_deflation` installed),
                            // `factor_spectral_deflated_criterion_row` discovers the
                            // negative/flat eigen-direction — including the #1095/#2228
                            // radial null the decoded-derivative gauge predicate
                            // (`decoded_motion_is_rounding_zero`) would otherwise have excluded
                            // from the gauge list — and stiffens it to UNIT curvature
                            // (eigenvalue → +1), a ρ-INDEPENDENT log 1 = 0 evidence
                            // contribution (the quotient pseudo-determinant convention
                            // of the #1037 gauge and #1117 data-null deflations).
                            // Reaching THIS arm therefore no longer means "deflation was
                            // never enabled" (the old #1095 refusal, now fixed) — it
                            // means the deflation was ATTEMPTED and genuinely DECLINED
                            // (a non-finite block or a failed eigendecomposition), so
                            // the state is broken: surface the hard refusal and let the
                            // outer BFGS treat this ρ as an INFINITY probe
                            // (`is_recoverable_value_probe_refusal`). We must NOT
                            // ridge-damp here: a `+ridge·I` fallback injects a
                            // ρ-dependent ½·log|I + ridge·H_tt⁻¹| bias into the VALUE
                            // that the analytic ρ-gradient (built for the undamped
                            // Laplace log-det) never sees, desyncing the outer
                            // line-search — the multi-atom non-convergence #1117 removes.
                            return Err(format!(
                                "SaeManifoldTerm::penalized_quasi_laplace_criterion: stationary undamped \
                                 criterion factorization has a {} \
                                 that spectral unit-stiffness deflation could not \
                                 condition (‖g‖={grad_norm:.6e}, tol {grad_tolerance:.6e}); \
                                 {err}",
                                ProbeRefusalKind::non_pd_per_row_marker()
                            ));
                        }
                        Err(err) => {
                            return Err(format!(
                                "SaeManifoldTerm::penalized_quasi_laplace_criterion: {err}"
                            ));
                        }
                    };
                // Only the factor cache is consumed (the stationary Newton step Δ
                // is discarded), but the full solve above still computes Δ, so
                // the historical degenerate-factorisation witnesses stay armed at
                // the ACCEPTED iterate: a non-finite undamped step, or a failed
                // quotient-step projection, refuses exactly as before.
                let step_norm_sq: f64 = delta_t.iter().map(|&v| v * v).sum::<f64>()
                    + delta_beta.iter().map(|&v| v * v).sum::<f64>();
                if !step_norm_sq.is_finite() {
                    return Err(format!(
                        "SaeManifoldTerm::penalized_quasi_laplace_criterion: undamped inner residual is non-finite at \
                         the inner optimum (‖Δ‖²={step_norm_sq}, ‖g‖²={grad_norm_sq}); the joint \
                         Hessian factorisation is degenerate at this ρ"
                    ));
                }
                let quotient_step_norm_sq = self.quotient_newton_step_norm_sq(
                    delta_t.view(),
                    delta_beta.view(),
                    step_norm_sq,
                    &lambda_smooth,
                )?;
                log::info!(
                    "[SAE-ACCEPT] kkt fixed point: ‖g‖={grad_norm:.6e} \
                     ‖Π⊥null g‖={quotient_grad_norm:.6e} tol={grad_tolerance:.6e} \
                     ‖Δ‖={:.6e} ‖Π⊥null Δ‖={:.6e} after {total_inner_iter} inner iterations",
                    step_norm_sq.sqrt(),
                    quotient_step_norm_sq.sqrt(),
                );
                drop(criterion_scope);
                return Ok(cache);
            }
            // NON-stationary refine round: per-row-only undamped feasibility
            // probe in place of the historically-discarded full factorization
            // (see the #2080 block comment above). A coarse-KKT iterate that is
            // not yet idempotent skips this probe and flows directly into the
            // shared refinement accounting below: its factor feasibility is
            // already known from stationarity, but its state is not returnable.
            if !gradient_stationary {
                match probe_undamped_evidence_row_factors(&sys, options) {
                    Ok(()) => {}
                    Err(err) if Self::is_undamped_evidence_row_non_pd(&err) => {
                        // #2080 — a non-PD per-row H_tt block means the undamped
                        // Laplace log-det is undefined at this provisional inner
                        // state. The raw reduced-budget policy
                        // (`refine_progress_extension == false`) is retained only
                        // for focused diagnostics: it returns a typed refusal
                        // after this factor pass rather than grinding. Production
                        // ranking, line-search, seed-validation, and accepted
                        // lanes all use the full drive because any finite value
                        // they return selects the estimator; that drive may cross
                        // the transient indefinite state and only classifies the
                        // converged fixed point.
                        if !refine_progress_extension {
                            return Err(format!(
                                "SaeManifoldTerm::penalized_quasi_laplace_criterion: undamped evidence \
                             factorization hit a {} before KKT \
                             stationarity at an infeasible-ρ probe (‖g‖={grad_norm:.6e}, \
                             tol {grad_tolerance:.6e}); returning the typed infeasible \
                             refusal without grinding the probe refinement budget; {err}",
                                ProbeRefusalKind::non_pd_per_row_marker()
                            ));
                        }
                        let refine_limit = Self::refine_iteration_limit(
                            total_inner_iter,
                            base_refine_iter,
                            progress_refine_iter,
                            previous_refine_grad_norm,
                            grad_norm,
                        );
                        if total_inner_iter >= refine_limit {
                            // #1117/#1118 — pre-stationarity genuinely-indefinite
                            // non-gauge H_tt under K>1 ordered Beta--Bernoulli/softmax row-sharing. The
                            // logit × coordinate Gauss-Newton cross term H_zt = J_zᵀJ_t
                            // can drive a shared row's H_tt Schur complement NEGATIVE off
                            // the gauge orbit; the LM-escalated refinement above cannot
                            // always cross the indefinite basin into the PD region within
                            // the descent-extended budget.
                            //
                            // The undamped (ridge=0) criterion factor already conditions
                            // that block the PRINCIPLED way: `factor_spectral_deflated_
                            // evidence_row` discovers the negative/flat eigen-direction
                            // and stiffens it to UNIT curvature (eigenvalue → +1), a
                            // ρ-independent `log 1 = 0` criterion contribution — so a
                            // spectral-deflatable indefinite block factors fine (both
                            // here and in the stationary factorization above) and
                            // returns a finite, monotone-comparable value to the outer
                            // BFGS WITHOUT a ρ-dependent bias. Reaching THIS arm means
                            // even that spectral deflation declined (a non-finite block
                            // or a failed eigendecomposition): the iterate is genuinely
                            // broken, so we surface the hard refusal and let the outer
                            // BFGS treat this ρ as an INFINITY probe.
                            //
                            // We must NOT ridge-damp here: a `+ridge·I` evidence
                            // fallback injects a ρ-dependent ½·log|I + ridge·H_tt⁻¹|
                            // bias into the VALUE that the analytic ρ-gradient (built
                            // for the undamped Laplace log-det) never sees, desyncing
                            // the outer line-search — the multi-atom non-convergence this
                            // fix removes. K=1 (and any already-PD or spectral-deflatable
                            // K>1 row) never reaches this branch.
                            return Err(format!(
                                "SaeManifoldTerm::penalized_quasi_laplace_criterion: undamped evidence \
                             factorization hit a {} before KKT \
                             stationarity (‖g‖={grad_norm:.6e}, tol {grad_tolerance:.6e}) \
                             and the refinement budget was exhausted after \
                             {total_inner_iter} inner iterations; {err}",
                                ProbeRefusalKind::non_pd_per_row_marker()
                            ));
                        }
                        let remaining = refine_limit - total_inner_iter;
                        let refine_iter = inner_max_iter.max(1).min(remaining);
                        previous_refine_grad_norm = Some(grad_norm);
                        let refine = self.run_joint_fit_arrow_schur_for_quasi_laplace(
                            target,
                            rho_fixed,
                            registry,
                            refine_iter,
                            learning_rate,
                            ridge_ext_coord,
                            ridge_beta,
                        )?;
                        *loss = refine.loss;
                        *criterion_fixed_point = refine.fixed_point;
                        total_inner_iter += refine_iter;
                        continue;
                    }
                    Err(err) => {
                        return Err(format!(
                            "SaeManifoldTerm::penalized_quasi_laplace_criterion: {err}"
                        ));
                    }
                }
            }
            let refine_limit = Self::refine_iteration_limit(
                total_inner_iter,
                base_refine_iter,
                progress_refine_iter,
                previous_refine_grad_norm,
                grad_norm,
            );
            let effective_refine_limit = refine_limit
                .checked_add(budget_escalation_extra)
                .ok_or_else(|| {
                    "SaeManifoldTerm::penalized_quasi_laplace_criterion: inner-refinement budget overflow"
                        .to_string()
                })?;
            if total_inner_iter >= effective_refine_limit {
                // #2234 stall synthesis — PROGRESS-GATED budget escalation.
                // Two prior designs collide here: the #2080 wide-p hang fix makes
                // budget-limited solves refuse fast — so at any ρ whose inner
                // problem needs more than the budget, EVERY lane that lands here
                // returns infeasible evidence, the
                // line search sees cliffs in all directions, and the outer fit
                // freezes at a live gradient and refuses to mint (measured
                // fleet-wide 2026-07-10: gam-sae 126 test failures, ten-orders
                // cost-lane disagreement at one ρ). A solve whose latest KKT residual is
                // MEASURABLY DESCENDING is an unfinished
                // computation, not an infeasibility: grant it one additional
                // window of the same size and keep refining. The ordinary
                // nonstationary lane retains that single-window hang bound.
                //
                // The former UNBOUNDED `stationary_window_paid` grind (which
                // chased hook-injected motion) stays deleted; hooks are
                // quiescent inside the KKT band. But the sweep-first engine is
                // a NEW legitimate mover: an evidence re-entry at a KKT-band
                // iterate may commit one more strict (t, B) sweep decrease
                // before the joint sweep∘walk fixed point is reached, and
                // refusing at first budget exhaustion there refused genuinely
                // convergent fits (tier-0 K=2 fixtures: band entered, refused
                // at 512). A KKT-band state therefore earns the SAME single
                // bounded window a measurably-descending solve gets — one
                // window, once, so the joint fixed point can complete; a state
                // still moving after that is genuinely non-idempotent and
                // takes the typed refusal.
                // #2228 CERTIFICATE-METRIC-KEYED ESCALATION — the general form
                // of the single-window grant below, measured in the
                // certificate's own units. At every budget-limit hit, price
                // the affine-invariant decrement ½λ²/scale on the exact
                // deflated Hessian (one factor per limit hit — paid only at
                // limit boundaries, never per iteration):
                //   · at/below the stall band ⇒ the iterate IS the numerical
                //     stationary root: accept the cache right here (identical
                //     doctrine to the stall-branch/final-gate acceptances);
                //   · not certified, before any certificate-paid window ⇒ try the
                //     terminal exact-Newton polish first (#2267, below);
                //   · DECREASING geometrically since the last limit hit ⇒ the
                //     walk is converging in the certificate metric even where
                //     the objective-decrease and gradient tests cannot see it
                //     (the stiff-valley regime: tiny accepted steps, ‖g‖ may
                //     legitimately RISE); grant one more window. A fixed
                //     budget is an arbitrary refusal point in that regime —
                //     the measured tier-0/wheel failures parked at 1.0034× to
                //     2.25× over the gradient band with the certificate still
                //     improving every round;
                //   · stalled certificate ⇒ fall through to the historical
                //     branches (single objective-progress window, then the
                //     typed refusal, whose final gate re-checks the
                //     certificate one last time).
                if let Ok(limit_factor) =
                    self.factor_deflated_evidence_with_grad_norms(&mut sys, &lambda_smooth, options)
                {
                    let decrement_sq = Self::inner_certificate_decrement_sq(
                        &sys,
                        limit_factor.delta_t.view(),
                        limit_factor.delta_beta.view(),
                    );
                    let limit_scale = self
                        .penalized_objective_total(target, rho_fixed, registry, 1.0)
                        .map(|obj| obj.abs() + 1.0)
                        .unwrap_or(f64::INFINITY);
                    let predicted_relative_decrease = 0.5 * decrement_sq / limit_scale;
                    if Self::inner_decrement_certifies(predicted_relative_decrease) {
                        log::info!(
                            "[SAE-ACCEPT] limit-boundary decrement certificate: ‖g‖={grad_norm:.6e} \
                             (tol {grad_tolerance:.6e}) ½λ²/scale={predicted_relative_decrease:.6e} \
                             after {total_inner_iter} inner iterations"
                        );
                        let refined = self.refine_accepted_root(
                            target,
                            Some(rho),
                            rho_fixed,
                            registry,
                            &lambda_smooth,
                            options,
                            inner_max_iter,
                            learning_rate,
                            ridge_ext_coord,
                            ridge_beta,
                            loss,
                            criterion_fixed_point,
                            &mut total_inner_iter,
                        )?;
                        drop(criterion_scope);
                        return Ok(refined.map_or(limit_factor.cache, |factor| factor.cache));
                    }
                    // #2267 — try the superlinear finish before paying for the first
                    // majorized window it would replace. On the shipped example's K=8
                    // rung the certificate kept contracting, so the loop kept granting
                    // windows (inner_limit 3880 after 40 refine rounds and 1640 inner
                    // iterations, ‖g‖ trendless in [0.097, 7.09]) and the polish below,
                    // which only runs once those windows stop paying, never ran inside
                    // the 900 s contract (job 529628: zero `[SAE-NEWTON]` lines). The
                    // polish commits only Armijo decreases of the penalized objective
                    // and bails, leaving the state untouched, when no damping buys one.
                    // A bail therefore falls through to the certificate windows exactly
                    // as before, having cost one exact-Hessian materialization. It is
                    // tried once per evaluation (before any certificate-paid window) and
                    // shares the committed-phase cap and the arming of the polish below.
                    if !gradient_stationary
                        && certificate_escalations == 0
                        && terminal_newton_polish_armed
                        && polish_escalations < POLISH_ESCALATION_ANTI_RUNAWAY_CAP
                    {
                        terminal_newton_polish_armed = false;
                        if self.terminal_exact_newton_polish(
                            target,
                            rho_fixed,
                            registry,
                            &lambda_smooth,
                            grad_tolerance,
                            previous_loss_total.abs() + 1.0,
                            options,
                            64,
                            &mut best_seen,
                        )? {
                            polish_escalations += 1;
                            *criterion_fixed_point = false;
                            consecutive_objective_stalls = 0;
                            budget_escalation_extra = total_inner_iter
                                .saturating_sub(refine_limit)
                                .saturating_add(refine_limit.max(1));
                            log::debug!(
                                "SaeManifoldTerm::penalized_quasi_laplace_criterion: polish-paid \
                                 window {polish_escalations}/\
                                 {POLISH_ESCALATION_ANTI_RUNAWAY_CAP} at a budget-limit hit before \
                                 any certificate-paid window, after {total_inner_iter} inner \
                                 iterations"
                            );
                            continue;
                        }
                    }
                    let certificate_improving = last_limit_certificate.is_none_or(|previous| {
                        predicted_relative_decrease <= CERTIFICATE_ESCALATION_PROGRESS * previous
                    });
                    if certificate_improving
                        && certificate_escalations < CERTIFICATE_ESCALATION_ANTI_RUNAWAY_CAP
                    {
                        certificate_escalations += 1;
                        last_limit_certificate = Some(predicted_relative_decrease);
                        let escalation_window = refine_limit.max(1);
                        budget_escalation_extra = total_inner_iter
                            .saturating_sub(refine_limit)
                            .saturating_add(escalation_window);
                        log::debug!(
                            "SaeManifoldTerm::penalized_quasi_laplace_criterion: certificate-paid \
                             window {certificate_escalations} at fixed ρ — ½λ²/scale=\
                             {predicted_relative_decrease:.6e} still contracting (‖g‖=\
                             {grad_norm:.6e}, tol {grad_tolerance:.6e}) after {total_inner_iter} \
                             inner iterations; granting {escalation_window} more"
                        );
                        // Skip the loop-bottom refine accounting for this
                        // round; the widened limit re-enters normally.
                        continue;
                    }
                    last_limit_certificate = Some(predicted_relative_decrease);
                }
                if (refine_limit > base_refine_iter || gradient_stationary)
                    && budget_escalation_extra == 0
                {
                    let escalation_window = refine_limit.max(1);
                    // `refine_iteration_limit` is dynamic and may return a
                    // ceiling below the iterations already consumed.  Carry
                    // that overshoot into the extension before adding the one
                    // progress window; otherwise the subtraction below can
                    // underflow immediately after escalation.
                    budget_escalation_extra = total_inner_iter
                        .saturating_sub(refine_limit)
                        .checked_add(escalation_window)
                        .ok_or_else(|| {
                            "SaeManifoldTerm::penalized_quasi_laplace_criterion: escalated inner-refinement budget overflow"
                                .to_string()
                        })?;
                    log::debug!(
                        "SaeManifoldTerm::penalized_quasi_laplace_criterion: budget escalation at fixed ρ — \
                         ‖g‖={grad_norm:.6e} (tol {grad_tolerance:.6e}) still descending after \
                         {total_inner_iter} inner iterations; granting a progress-paid window of \
                         {escalation_window} iterations"
                    );
                } else if gradient_stationary {
                    // #2228 — the exact-A polish before refusing a non-idempotent band
                    // state. None of the three polish sites runs inside the band: the
                    // limit-boundary one needs a non-stationary gate, the stall branch
                    // needs an idempotent round, and the budget branch is this arm's
                    // `else`. So a state whose re-entries keep committing material
                    // decreases refused here without taking one exact step. Pool job
                    // 604252 at `5e8f45d7d`
                    // (`/scratch.global/sauer354/pool/sae2228/g5.seed2132.604252.txt`,
                    // C=4 K=4 softmax, refine rounds 77–80) read ‖g‖ ≈ 9.4e-4 against
                    // tol 2.9e-3 while every inner iteration took a logit-only step
                    // (‖Δ_logit‖ 6.6e-3 → 1.48, gᵀΔ 5e-7 → 5e-5, many of them rejected
                    // and routed to the proximal correction), and the penalized
                    // objective crept from −23.1902692 to −23.1904910 before this
                    // refusal. The softmax entropy prior's infimum lies at saturated
                    // gates. Tolerance 0 removes the phase's band exit, so it steps until
                    // its own ladder can buy no verifiable Armijo decrease. It mints
                    // nothing: acceptance still needs the next evidence re-entry to
                    // recur exactly.
                    if terminal_newton_polish_armed
                        && polish_escalations < POLISH_ESCALATION_ANTI_RUNAWAY_CAP
                    {
                        terminal_newton_polish_armed = false;
                        if self.terminal_exact_newton_polish(
                            target,
                            rho_fixed,
                            registry,
                            &lambda_smooth,
                            0.0,
                            previous_loss_total.abs() + 1.0,
                            options,
                            64,
                            &mut best_seen,
                        )? {
                            polish_escalations += 1;
                            *criterion_fixed_point = false;
                            consecutive_objective_stalls = 0;
                            budget_escalation_extra = total_inner_iter
                                .saturating_sub(refine_limit)
                                .saturating_add(refine_limit.max(1));
                            log::debug!(
                                "SaeManifoldTerm::penalized_quasi_laplace_criterion: polish-paid \
                                 window {polish_escalations}/\
                                 {POLISH_ESCALATION_ANTI_RUNAWAY_CAP} inside the KKT band after \
                                 {total_inner_iter} inner iterations"
                            );
                            continue;
                        }
                    }
                    let intensive = self.intensive_kkt_diagnostic(target, rho, registry);
                    return Err(format!(
                        "SaeManifoldTerm::penalized_quasi_laplace_criterion: {}; \
                         KKT entered its admission band (raw ‖g‖={grad_norm:.6e}, quotient \
                         ‖Π⊥null g‖={quotient_grad_norm:.6e}, tolerance {grad_tolerance:.6e}) \
                         but an evidence-only re-entry still made a strict state/objective move \
                         after {total_inner_iter} granted iterations ({intensive}). Refusing to \
                         differentiate a non-idempotent inner map.",
                        ProbeRefusalKind::inner_not_converged_marker()
                    ));
                } else {
                    // #2228 Stage-2, budget branch — the terminal exact-Newton
                    // phase exists precisely to convert "the budget died near
                    // the root" into convergence, but it historically lived
                    // only behind the STALL branch; a solve that exhausts the
                    // budget WITHOUT three stalled+idempotent rounds (measured
                    // tier-0: ‖g‖ = 8.0e-5 against a 6.1e-5 band after 128,
                    // one Newton step from the band) refused here without the
                    // phase ever running. Try it before refusing: every committed
                    // step is an Armijo decrease of the penalized objective
                    // (#2861), the phase bails cheaply at the first step no
                    // damping buys a decrease for, and on progress the loop
                    // resumes with fresh accounting — the loop-top KKT gate and
                    // the idempotence certificate remain the sole acceptance
                    // authority, exactly as at the stall branch.
                    if terminal_newton_polish_armed
                        && polish_escalations < POLISH_ESCALATION_ANTI_RUNAWAY_CAP
                    {
                        terminal_newton_polish_armed = false;
                        if self.terminal_exact_newton_polish(
                            target,
                            rho_fixed,
                            registry,
                            &lambda_smooth,
                            grad_tolerance,
                            previous_loss_total.abs() + 1.0,
                            options,
                            64,
                            &mut best_seen,
                        )? {
                            polish_escalations += 1;
                            *criterion_fixed_point = false;
                            consecutive_objective_stalls = 0;
                            budget_escalation_extra = total_inner_iter
                                .saturating_sub(refine_limit)
                                .saturating_add(refine_limit.max(1));
                            log::debug!(
                                "SaeManifoldTerm::penalized_quasi_laplace_criterion: polish-paid \
                                 window {polish_escalations}/\
                                 {POLISH_ESCALATION_ANTI_RUNAWAY_CAP} at fixed rho after \
                                 {total_inner_iter} inner iterations"
                            );
                            continue;
                        }
                    }
                    // FINAL-GATE decrement certificate — the #2253 doctrine at
                    // the refusal boundary itself. A stiff narrow valley can
                    // park the ambient ‖g‖ above the Euclidean tolerance while
                    // the exact deflated Hessian's own model predicts no
                    // resolvable descent (measured tier-0: 128 iterations of
                    // ~1.7e-6 quotient steps, ‖g‖ drifting 5.5e-5 → 8.0e-5
                    // against a 6.1e-5 band, then this refusal — and the polish
                    // above cannot contract what the objective's resolution
                    // cannot express). EVERY refusal lane consults the
                    // curvature certificate before refusing; paid only on the
                    // refusal path, and quadratic λ² scaling keeps genuine
                    // non-convergence refused unchanged.
                    if let Ok(DeflatedEvidenceFactor {
                        delta_t: final_dt,
                        delta_beta: final_db,
                        cache: final_cache,
                        ..
                    }) = self.factor_deflated_evidence_with_grad_norms(
                        &mut sys,
                        &lambda_smooth,
                        options,
                    ) {
                        let final_objective_scale = self
                            .penalized_objective_total(target, rho_fixed, registry, 1.0)
                            .map(|obj| obj.abs() + 1.0)
                            .unwrap_or(f64::INFINITY);
                        let newton_decrement_sq = Self::inner_certificate_decrement_sq(
                            &sys,
                            final_dt.view(),
                            final_db.view(),
                        );
                        let excursion_cert = 0.5 * newton_decrement_sq / final_objective_scale;
                        // #2228 — the acceptance verdict keys on the BEST-SEEN
                        // certificate, not the excursion the polish left. The
                        // band is UNCHANGED; a best-seen plateau ABOVE it is a
                        // solver stall, refused honestly below with the best-
                        // seen ‖g‖. When best-seen clears the band we certify
                        // THERE (restore + re-factor) — the continuation is
                        // over, so nothing consumes the restore.
                        let best_clears = best_seen.as_ref().is_some_and(|(c, _, _)| {
                            *c < excursion_cert && Self::inner_decrement_certifies(*c)
                        });
                        if best_clears {
                            let (best_cert, best_g, best_state) =
                                best_seen.as_ref().expect("best_clears gated on Some");
                            let excursion = self.snapshot_mutable_state();
                            self.restore_mutable_state(best_state)?;
                            let refactored = self
                                .assemble_arrow_schur(target, rho, registry)
                                .ok()
                                .and_then(|mut best_sys| {
                                    self.factor_deflated_evidence_with_grad_norms(
                                        &mut best_sys,
                                        &lambda_smooth,
                                        options,
                                    )
                                    .ok()
                                });
                            if let Some(best_factor) = refactored {
                                log::info!(
                                    "[SAE-ACCEPT] best-seen decrement certificate: ‖g‖ {grad_norm:.6e} \
                                     \u{2192} {best_g:.6e}, ½λ²/scale {excursion_cert:.6e} \
                                     \u{2192} {best_cert:.6e} after {total_inner_iter} iters"
                                );
                                let refined = self.refine_accepted_root(
                                    target,
                                    Some(rho),
                                    rho_fixed,
                                    registry,
                                    &lambda_smooth,
                                    options,
                                    inner_max_iter,
                                    learning_rate,
                                    ridge_ext_coord,
                                    ridge_beta,
                                    loss,
                                    criterion_fixed_point,
                                    &mut total_inner_iter,
                                )?;
                                drop(criterion_scope);
                                return Ok(refined.map_or(best_factor.cache, |factor| factor.cache));
                            }
                            // Re-factor at best-seen failed: restore the
                            // excursion so state + final_cache stay consistent,
                            // then fall through to the honest refusal below.
                            self.restore_mutable_state(&excursion)?;
                        } else if Self::inner_decrement_certifies(excursion_cert) {
                            log::info!(
                                "[SAE-ACCEPT] final-gate decrement certificate: ‖g‖={grad_norm:.6e} \
                                 (tol {grad_tolerance:.6e}) λ²={newton_decrement_sq:.6e} \
                                 ½λ²/scale={excursion_cert:.6e} after \
                                 {total_inner_iter} inner iterations"
                            );
                            let refined = self.refine_accepted_root(
                                target,
                                Some(rho),
                                rho_fixed,
                                registry,
                                &lambda_smooth,
                                options,
                                inner_max_iter,
                                learning_rate,
                                ridge_ext_coord,
                                ridge_beta,
                                loss,
                                criterion_fixed_point,
                                &mut total_inner_iter,
                            )?;
                            drop(criterion_scope);
                            return Ok(refined.map_or(final_cache, |factor| factor.cache));
                        }
                    }
                    // Inner solve did not converge; the returned Err carries
                    // the non-convergence diagnostic (gradient /
                    // quotient-gradient norms and the tolerance) to the caller.
                    // #2228 — report a CONSISTENT best-seen snapshot: recompute
                    // BOTH norms at the best-seen state, never the best-seen raw
                    // mixed with the excursion's stale quotient. Terminal give-up
                    // path, so the restore has no downstream state to corrupt.
                    let (grad_norm, quotient_grad_norm) = match best_seen.as_ref() {
                        Some((_, _, best_state)) => {
                            self.restore_mutable_state(best_state)?;
                            match self.assemble_arrow_schur(target, rho, registry) {
                                Ok(best_sys) => {
                                    let g2 = Self::system_grad_norm_sq(&best_sys);
                                    let q = self.quotient_gradient_norm_from_system(
                                        &best_sys,
                                        g2,
                                        &lambda_smooth,
                                    );
                                    (g2.sqrt(), q)
                                }
                                Err(_) => (grad_norm, quotient_grad_norm),
                            }
                        }
                        None => (grad_norm, quotient_grad_norm),
                    };
                    // gam#2080/#2627: this refusal has two regimes that need different
                    // fixes, and the raw norms alone do not separate them. Measured over
                    // the occurrences that print both: ~10/14 sit at `null_share < 0.5`
                    // (the solve is genuinely far from a KKT point in the directions it
                    // can move) and ~4/14 at `null_share > 0.5` (the remainder within a
                    // few x of tolerance — close in the directions that matter, held off
                    // by gauge content). Report the ratios that discriminate so any
                    // occurrence can be bucketed without parsing the norms pairwise.
                    //
                    // gam#2720 — WHAT `Π` PROJECTS OFF, since the field names used to
                    // say `gauge` and the span no longer contains one. It is
                    // `posterior_null_quotient_basis`: the decoder β-null and
                    // decoder-channel-null families, i.e. directions the PENALIZED
                    // objective is flat along. The chart reparametrisation orbit was
                    // removed from it — the priors are written on the chart coordinates
                    // and are not flat there — so these fields no longer describe it and
                    // no longer claim to. A reader diagnosing an orbit-dominated refusal
                    // wants `orbit_best_objective_drop` from the gauge-orbit block, which
                    // is emitted separately.
                    //
                    // gam#2715 — WHAT `null_share` IS, AND WHAT IT IS NOT. It is
                    // `1 − ‖Π⊥null g‖/‖g‖`, i.e. one minus the RETAINED fraction. It is
                    // NOT the share of the gradient lying in the null span: the two
                    // components are orthogonal, so norms add in QUADRATURE and
                    // `‖Π∥null g‖/‖g‖ = sqrt(1 − retained²)`, which is strictly larger.
                    // MEASURED at one refusal state: `null_share = 0.4999` while the
                    // removed span actually holds 0.8660 of the norm and 0.7500 of the
                    // energy — reading the field as "about half the gradient is removed"
                    // understates it badly, and has already misled a reader. So emit the
                    // projected component ITSELF next to the other two norms; then
                    // `‖g‖² = ‖Π∥‖² + ‖Π⊥‖²` is checkable from the message and no ratio
                    // has to be inferred from a field name. (Note `retained + null_share
                    // = 1` is an identity, so agreement between those two numbers is
                    // never evidence of anything.)
                    let null_share = if grad_norm > 0.0 {
                        1.0 - quotient_grad_norm / grad_norm
                    } else {
                        f64::NAN
                    };
                    let null_component = (grad_norm * grad_norm
                        - quotient_grad_norm * quotient_grad_norm)
                        .max(0.0)
                        .sqrt();
                    let quotient_over_tol = if grad_tolerance > 0.0 {
                        quotient_grad_norm / grad_tolerance
                    } else {
                        f64::INFINITY
                    };
                    let intensive = self.intensive_kkt_diagnostic(target, rho, registry);
                    return Err(format!(
                        "SaeManifoldTerm::penalized_quasi_laplace_criterion: {}; \
                         neither the KKT gradient ‖g‖={grad_norm:.6e} nor the quotient KKT gradient \
                         ‖Π⊥null g‖={quotient_grad_norm:.6e} met tolerance {grad_tolerance:.6e} \
                         after {total_inner_iter} inner iterations \
                         (‖Π∥null g‖={null_component:.6e}, null_share={null_share:.4}, \
                         quotient_over_tol={quotient_over_tol:.3e}, \
                         {intensive}). \
                         Refusing to rank an off-optimum Laplace criterion.",
                        ProbeRefusalKind::inner_not_converged_marker()
                    ));
                }
            }
            let refine_limit = refine_limit
                .checked_add(budget_escalation_extra)
                .ok_or_else(|| {
                    "SaeManifoldTerm::penalized_quasi_laplace_criterion: inner-refinement budget overflow"
                        .to_string()
                })?;
            let remaining = refine_limit.checked_sub(total_inner_iter).ok_or_else(|| {
                format!(
                    "SaeManifoldTerm::penalized_quasi_laplace_criterion: inner-refinement accounting mismatch \
                     ({total_inner_iter} iterations consumed past limit {refine_limit})"
                )
            })?;
            let refine_iter = inner_max_iter.max(1).min(remaining);
            previous_refine_grad_norm = Some(grad_norm);
            let refine_window_scope = gam_runtime::process_monitor::track_scope(format!(
                "sae refine window round={} refine_iter={refine_iter} inner_total={total_inner_iter}",
                refine_rounds + 1
            ));
            let refine = self.run_joint_fit_arrow_schur_for_quasi_laplace(
                target,
                rho_fixed,
                registry,
                refine_iter,
                learning_rate,
                ridge_ext_coord,
                ridge_beta,
            )?;
            drop(refine_window_scope);
            *loss = refine.loss;
            *criterion_fixed_point = refine.fixed_point;
            total_inner_iter += refine_iter;
            refine_rounds += 1;
            // #2472/#2762 — one line per refine round: the nominal progress
            // budget is `inner_max_iter x 64` (>= 256) TOTAL inner iterations,
            // and each round is a full
            // assembly + factorization + damped Newton sweep, so this loop is
            // where a criterion evaluation spends its wall clock. Without it a
            // running evaluation is indistinguishable from a hang. Report the
            // round ordinal and the current total-iteration limit separately:
            // printing the iteration limit as a round denominator made a
            // 1,920-iteration ceiling read as 1,920 thirty-iteration rounds.
            log::info!(
                "[SAE-REFINE] round={refine_rounds} refine_iter={refine_iter} \
                 inner_total={total_inner_iter} inner_limit={refine_limit} \
                 elapsed={:.1}s",
                refine_started.elapsed().as_secs_f64(),
            );
            // #1051 — objective-stagnation fixed point. A whole refine round that
            // failed to lower the penalised objective by a meaningful FRACTION of
            // the total since-entry reduction means the Newton/LM iterate is at
            // its numerical optimum: the remaining KKT residual lives in the
            // weakly-identified decoder / gauge directions the near-singular Schur
            // cannot resolve. Ranking the Laplace criterion at this fixed point is
            // correct (the only further motion is cosmetic flat-valley crawl), so
            // accept the current cache instead of refining until the budget dies.
            // Requires a few completed refine rounds (so the fraction baseline is
            // meaningful) but is NOT gated behind the full refine budget — the
            // whole point is to terminate the crawl long before that.
            // Same ONE-SCALAR contract as `entry_loss_total` above: the round's
            // progress is measured on the penalized objective the line search
            // descends, not the native-terms-only loss.
            let new_loss_total = self
                .penalized_objective_total(target, rho, registry, 1.0)
                .map_err(|err| {
                    format!("SaeManifoldTerm::penalized_quasi_laplace_criterion: {err}")
                })?;
            // `grad_norm` was read at this round's entry, before its refine iterations ran,
            // so it sits beside the post-round objective under its own name (#2228).
            log::info!(
                "[SAE-REFINE] round={refine_rounds} penalized_objective={new_loss_total:.10e} \
                 entry ‖g‖={grad_norm:.6e}",
            );
            // Two stagnation signals, both required: (1) the latest refine round
            // contributed a negligible FRACTION of the total objective reduction
            // achieved since entry — the fit has captured essentially all the
            // achievable improvement and is now crawling cosmetically along the
            // weakly-identified valley; (2) the absolute relative decrease is
            // itself tiny. The fraction test is scale- and rate-free (it fires
            // whether the crawl decays fast or slow), so it recognises the
            // over-smoothed / rank-deficient fixed point the bare relative floor
            // misses, while still never firing on a fit that is materially
            // improving round over round.
            let total_improvement = (entry_loss_total - new_loss_total).max(0.0);
            let round_improvement = (previous_loss_total - new_loss_total).max(0.0);
            let objective_scale = previous_loss_total.abs().max(new_loss_total.abs()) + 1.0;
            let relative_decrease = round_improvement / objective_scale;
            let captured_fraction = if total_improvement > 0.0 {
                round_improvement / total_improvement
            } else {
                0.0
            };
            let stalled = new_loss_total.is_finite()
                && relative_decrease.is_finite()
                && captured_fraction.is_finite()
                && relative_decrease < SAE_MANIFOLD_INNER_OBJECTIVE_STALL_REL_TOL
                && captured_fraction < SAE_MANIFOLD_INNER_OBJECTIVE_STALL_FRACTION;
            previous_loss_total = new_loss_total;
            if stalled
                && refine_rounds >= SAE_MANIFOLD_INNER_OBJECTIVE_STALL_MIN_ROUNDS
                && *criterion_fixed_point
            {
                let mut stall_polish_permitted = false;
                let mut stationary_sys = self
                    .assemble_arrow_schur(target, rho_fixed, registry)
                    .map_err(|err| {
                        format!("SaeManifoldTerm::penalized_quasi_laplace_criterion: {err}")
                    })?;
                // #1095/#2228 — diagnose the stalled state with the ridge-0
                // deflated factor. Only the raw/quotient KKT residual can accept;
                // the affine Newton decrement is reported but cannot mint an
                // envelope root. On a chart that is over-parametrized for its
                // intrinsic data dimension — d_atom=2 on an intrinsic 1-D circle,
                // so every per-row H_tt carries a rank-1 radial null — that
                // undamped per-row Cholesky is non-PD BY CONSTRUCTION, so without
                // spectral deflation `solve_arrow_newton_step_with_options` errors,
                // the whole `if let Ok(..)` is skipped, and a perfectly good fit is
                // refused to the non-convergence sentinel (#1095: public
                // sae_manifold_fit K=1 circle → GamError at every N).
                //
                // Ensure the stationary EVIDENCE system opts into per-row spectral
                // discovery (installing an empty-per-row `row_gauge_deflation` is
                // exactly the #974 low-rank-whiten seam): an intrinsic flat /
                // indefinite direction is then deflated to UNIT stiffness (log 1 = 0,
                // ρ-independent — the quotient pseudo-determinant convention the
                // gauge / #1273 / #974 deflations already use), so the ridge-0
                // factor is PD-by-deflation, the log-det is finite, and the affine
                // ½λ² below is measured on the IDENTIFIABLE subspace (the deflated
                // null direction contributes a bounded step, not a Schur-amplified
                // blow-up). A full-rank block has no eigenvalue below the spectral
                // floor and is returned bit-for-bit unchanged, so healthy fits are
                // untouched — this only makes acceptance REACHABLE on a
                // rank-deficient chart. The UNDAMPED (non-deflated) per-row verdict
                // remains the #2080 infeasible-ρ probe upstream
                // (`probe_undamped_evidence_row_factors` on the loop `sys`), which
                // this does not touch: it is a probe signal, not an acceptance gate.
                if let Ok(DeflatedEvidenceFactor {
                    delta_t: stationary_dt,
                    delta_beta: stationary_db,
                    cache: stationary_cache,
                    grad_norm: stationary_grad_norm,
                    quotient_grad_norm: stationary_quotient_grad_norm,
                }) = self.factor_deflated_evidence_with_grad_norms(
                    &mut stationary_sys,
                    &lambda_smooth,
                    options,
                ) {
                    if Self::quasi_laplace_kkt_stationary(
                        stationary_grad_norm,
                        stationary_quotient_grad_norm,
                        grad_tolerance,
                    ) {
                        log::info!(
                            "[SAE-ACCEPT] stall-branch kkt: ‖g‖={stationary_grad_norm:.6e} \
                             ‖Π⊥null g‖={stationary_quotient_grad_norm:.6e} tol={grad_tolerance:.6e} \
                             after {total_inner_iter} inner iterations"
                        );
                        let refined = self.refine_accepted_root(
                            target,
                            None,
                            rho_fixed,
                            registry,
                            &lambda_smooth,
                            options,
                            inner_max_iter,
                            learning_rate,
                            ridge_ext_coord,
                            ridge_beta,
                            loss,
                            criterion_fixed_point,
                            &mut total_inner_iter,
                        )?;
                        drop(criterion_scope);
                        return Ok(refined.map_or(stationary_cache, |factor| factor.cache));
                    }
                    // Affine-invariant stationarity certificate (#2226). The raw and
                    // quotient KKT gradient norms above are measured in the ambient
                    // Euclidean parameter metric, which lumps the heterogeneous
                    // logit / coordinate / decoder-coefficient blocks together with
                    // unit weight. The floor that norm can reach is set by the joint
                    // Hessian's conditioning and therefore by the float summation
                    // order, so NEON (arm64) and AVX (x86) plateau at slightly
                    // different values — a couple of digits apart on this K=1 circle,
                    // enough that arm64 parks above the absolute iterate-scaled
                    // tolerance x86 clears and the fixed point is hard-refused
                    // (issue #2226: `sae_manifold_fit(K=1, atom_topology="circle")`).
                    //
                    // The Newton decrement λ² = gᵀH⁻¹g = −gᵀΔ (Δ the exact undamped
                    // joint Newton step just factored above) is invariant to any
                    // affine reparametrisation of the iterate, and ½λ² is the
                    // quadratic model's predicted remaining decrease in the penalised
                    // objective. `sae_manifold_newton_directional_decrease` returns
                    // −gᵀΔ = λ² for the descent step Δ. We are already inside the
                    // objective-stall fixed point (both `relative_decrease` and
                    // `captured_fraction` fell below their floors above), so no step
                    // lowers the objective by a meaningful fraction of its scale; the
                    // model-predicted decrease ½λ² is then likewise below that scale,
                    // and we accept on that affine-invariant witness. Measuring the
                    // predicted decrease RELATIVE to the objective scale — the exact
                    // structure `relative_decrease` (round_improvement / objective_scale)
                    // uses — keeps this neither looser nor tighter than the stall gate
                    // that just fired: it can only accept when the model itself
                    // predicts no further meaningful descent, never a still-descending
                    // iterate (a large λ² leaves this below and falls through to the
                    // deterministic refine budget exactly as before).
                    let newton_decrement_sq = Self::inner_certificate_decrement_sq(
                        &stationary_sys,
                        stationary_dt.view(),
                        stationary_db.view(),
                    );
                    let predicted_relative_decrease = 0.5 * newton_decrement_sq / objective_scale;
                    log::debug!(
                        "SAE inner stall certificate: ‖g‖={stationary_grad_norm:.6e} \
                         ‖Π⊥null g‖={stationary_quotient_grad_norm:.6e} tol={grad_tolerance:.6e} \
                         λ²={newton_decrement_sq:.6e} ½λ²/scale={predicted_relative_decrease:.6e} \
                         obj_scale={objective_scale:.6e} accept_tol={SAE_MANIFOLD_INNER_OBJECTIVE_STALL_REL_TOL:.6e}"
                    );
                    // Affine-invariant ACCEPTANCE (#2253 doctrine, applied to the
                    // inner gate). ½λ² = ½·gᵀH⁻¹g is the exact quadratic model's
                    // predicted remaining decrease measured on the SAME deflated
                    // exact Hessian the outer adjoint consumes. When it falls at
                    // or below the stall detector's own no-meaningful-change
                    // band, NO step — in any direction, under any affine
                    // reparametrisation — lowers the penalized objective by an
                    // amount the criterion can resolve: the iterate IS the
                    // numerical stationary root on the identifiable subspace,
                    // regardless of where the ambient-metric ‖g‖ sits (a stiff
                    // narrow valley legitimately parks ‖g‖ orders above the
                    // Euclidean tolerance while λ² certifies optimality — the
                    // measured tier-0 refusal was ‖g‖ 1.0034× tol with
                    // ½λ²/scale = 5.9e-11 against a 1e-8 band). This mirrors the
                    // outer certify_outer_optimality Newton-decrement rescue
                    // verbatim and inherits its safety argument: the decrement
                    // scales quadratically with ‖g‖ at fixed direction, so a fit
                    // with genuinely available descent inflates λ² and falls
                    // through to the refine budget exactly as before. (The
                    // historical refusal here predates the outer rescue; keeping
                    // the inner gate blind to curvature while the outer gate
                    // trusts it was inconsistent, and no budget can close a gap
                    // that the objective's own resolution cannot express.)
                    if Self::inner_decrement_certifies(predicted_relative_decrease) {
                        log::info!(
                            "[SAE-ACCEPT] stall decrement certificate: ‖g‖={stationary_grad_norm:.6e} \
                             ½λ²/scale={predicted_relative_decrease:.6e} tol={grad_tolerance:.6e} \
                             after {total_inner_iter} inner iterations"
                        );
                        let refined = self.refine_accepted_root(
                            target,
                            None,
                            rho_fixed,
                            registry,
                            &lambda_smooth,
                            options,
                            inner_max_iter,
                            learning_rate,
                            ridge_ext_coord,
                            ridge_beta,
                            loss,
                            criterion_fixed_point,
                            &mut total_inner_iter,
                        )?;
                        drop(criterion_scope);
                        return Ok(refined.map_or(stationary_cache, |factor| factor.cache));
                    }
                    // #2267/#2283 — permitted at every armed plateau. What re-arms
                    // the polish is a materially descending refine round (the stall
                    // streak's `else` below), i.e. progress in the penalized objective,
                    // which is the currency the polish commits in since #2861. The
                    // gate-norm frontier certificate this replaces (#2653) additionally
                    // required the plateau's ‖g‖ to set a new low, while descent along a
                    // resolved negative-curvature mode generally RAISES ‖g‖, so it
                    // refused the second polish on a plateau the first polish had
                    // lowered the objective to. Termination does not rest on a frontier:
                    // invoking the polish disarms it, only material objective descent
                    // re-arms it, and every refine round spends the iteration budget
                    // the loop-top budget branch bounds.
                    stall_polish_permitted = true;
                    // Otherwise: a flat objective round is only a convergence
                    // shortcut when a certificate is stationary. Keep using the
                    // deterministic refinement budget: either later rounds reach
                    // stationarity, or the normal `total_inner_iter >=
                    // refine_limit` branch reports non-convergence without
                    // ranking an off-optimum Laplace criterion. Returning `Err`
                    // here was too strong for K=1 circle fits: one weakly
                    // identified round could abort a still-descending solve and
                    // poison the outer BFGS line search with a false value-probe
                    // refusal.
                }
                // #2228 Stage-2 — the objective has stalled but the KKT gate is
                // unmet: this is exactly the linear-rate crawl regime where the
                // MM/GN phase needs ~10³ more iterations it does not have. Hand
                // the iterate to the exact-Hessian terminal Newton phase; every
                // committed step is an Armijo decrease of the penalized objective
                // (#2861), so the refine loop resumes from a strictly lower
                // objective instead of refusing. The phase
                // mints nothing — acceptance stays with the loop-top KKT gate
                // and the idempotence certificate (the state moved, so
                // `criterion_fixed_point` is cleared and one evidence re-entry
                // must recur exactly before acceptance, same as any hook move).
                if terminal_newton_polish_armed && stall_polish_permitted {
                    terminal_newton_polish_armed = false;
                    if self.terminal_exact_newton_polish(
                        target,
                        rho_fixed,
                        registry,
                        &lambda_smooth,
                        grad_tolerance,
                        objective_scale,
                        options,
                        // Anti-runaway cap ONLY — every committed step is an
                        // Armijo decrease of the penalized objective (#2861), and
                        // the phase bails at the first step no damping on its
                        // ladder buys a decrease for, so the cap bounds only a
                        // phase that keeps buying resolvable decreases. Measured
                        // (tier-0 fixtures, host lane): at 12 the polish silently
                        // expired at ‖g‖ = 6.48e-5 against a 6.11e-5 band —
                        // refused 1.07× from convergence purely by cap. Near the
                        // marginally-indefinite root the steps contract slower than
                        // pure quadratic, so the cap must not impersonate a
                        // convergence bound.
                        64,
                        &mut best_seen,
                    )? {
                        *criterion_fixed_point = false;
                        consecutive_objective_stalls = 0;
                        continue;
                    }
                }
                // #2762 — THE THIRD FIXED-POINT CLAIM, and the one this issue's
                // refusals are actually raised at.
                //
                // The two claims inside `run_joint_fit_arrow_schur` (its
                // objective-stall shortcut and its no-strict-decrease exit) now
                // consult the gauge-orbit block, but this loop makes its OWN
                // fixed-point claim on top of theirs, over whole refine ROUNDS,
                // and it makes it at a state the joint fit may never have left
                // the block stationary at — the terminal refusal below reports
                // `best_seen`, the ½λ²/scale-minimizing iterate, which is a
                // different state from wherever the last joint fit stopped.
                //
                // Measured on `zz2015` after the two joint-fit sites landed: the
                // refusal still carried `orbit_best_objective_drop = 6.426e-3`,
                // relative `3.30e-7` — 33x the `1e-8` resolution this same branch
                // calls "no meaningful change". A stall over refine rounds is a
                // fixed point only if it is one in BOTH blocks, so ask the block
                // here too, on the same arming discipline as the polish above: a
                // materially-descending round re-arms it, so a plateau the block
                // cannot unlock refuses on its second visit with both movers
                // disarmed.
                if gauge_block_armed {
                    gauge_block_armed = false;
                    let orbit = self.descend_gauge_orbit_consuming_best_seen(
                        target,
                        rho_fixed,
                        registry,
                        &lambda_smooth,
                        &mut best_seen,
                        // Same bound as the joint fit's: the block returns at the
                        // first round that cannot commit a material decrease, so
                        // this caps a loop that terminates on its own. The refine
                        // loop has no per-iteration counter to borrow, so the
                        // block's own budget is the stall streak it is answering.
                        SAE_MANIFOLD_INNER_OBJECTIVE_STALL_MIN_ROUNDS.max(1),
                    )?;
                    if orbit.moved() {
                        *criterion_fixed_point = false;
                        consecutive_objective_stalls = 0;
                        log::debug!(
                            "SAE inner refine loop: gauge-orbit descent recovered {:.6e} over \
                             {} round(s) at the objective-stall fixed point (span dim {}, \
                             maxᵢ|gᵀvᵢ|={:.6e}, {} objective evaluations) after \
                             {total_inner_iter} inner iterations",
                            orbit.objective_decrease,
                            orbit.rounds,
                            orbit.dimension,
                            orbit.max_directional_derivative,
                            orbit.evaluations,
                        );
                        continue;
                    }
                }
                // Persistent objective-stall fixed point (`STALL_MIN_ROUNDS`
                // consecutive stalled rounds) without KKT stationarity. Surface
                // the typed refusal that the outer bridge treats as an infeasible
                // probe; a finite factor or objective floor is not an envelope
                // certificate. This also terminates the loop instead of burning
                // the extended progress budget indefinitely.
                consecutive_objective_stalls += 1;
                if consecutive_objective_stalls >= SAE_MANIFOLD_INNER_OBJECTIVE_STALL_MIN_ROUNDS {
                    // #2228 — recompute the raw ‖g‖ at the best-seen state so the
                    // reported residual is the best-seen iterate's, not the
                    // excursion's. Terminal give-up path; restore is safe.
                    let (grad_norm, quotient_grad_norm) = match best_seen.as_ref() {
                        Some((_, _, best_state)) => {
                            self.restore_mutable_state(best_state)?;
                            match self.assemble_arrow_schur(target, rho, registry) {
                                Ok(best_sys) => {
                                    let g2 = Self::system_grad_norm_sq(&best_sys);
                                    let q = self.quotient_gradient_norm_from_system(
                                        &best_sys,
                                        g2,
                                        &lambda_smooth,
                                    );
                                    (g2.sqrt(), q)
                                }
                                Err(_) => (grad_norm, quotient_grad_norm),
                            }
                        }
                        None => (grad_norm, quotient_grad_norm),
                    };
                    // gam#2674 — this message NAMED the quotient gradient and never
                    // emitted its value, so an occurrence could not be bucketed
                    // without re-running it under instrumentation. The sibling
                    // iteration-budget refusal above already reports
                    // `‖Π⊥null g‖` with `null_share` / `quotient_over_tol`;
                    // emit the identical fields here so the two terminal inner
                    // refusals are read the same way. See the sibling site above
                    // for what `null_share` is and is not (gam#2715): it is
                    // `1 − retained`, NOT the gauge's share of the gradient, and
                    // `‖Π∥null g‖` is emitted beside it so no ratio has to be
                    // inferred from a field name.
                    //
                    // CORRECTION (gam#2715) to this comment as first landed: it
                    // said `null_share` "is the share of ‖g‖ the projection
                    // removes, MEASURED at 0.87 and 0.93". Those 0.87/0.93 are
                    // `‖Π∥null g‖/‖g‖` from the #2674 solver-side probe, a
                    // DIFFERENT quantity — at that same state `null_share` reads
                    // 0.4999. The load-bearing part of that note survives and is
                    // restated correctly here: the removed directions carry
                    // directional derivatives 8x-10x the tolerance at this state
                    // (#2674), so they are NOT flat, which is what makes a small
                    // quotient an unsafe acceptance signal rather than a
                    // stationarity certificate.
                    //
                    // Scope: on the one fixture where the projection's rank has
                    // been measured (#2715, 332/332 calls) it is the rank-2
                    // chart-gauge orbit ALONE — both decoder-null families
                    // returned zero directions — so "chart-gauge/decoder-null"
                    // overstates what is actually being removed there.
                    //
                    // Diagnostic only: no gate, bound or trajectory moves.
                    let null_share = if grad_norm > 0.0 {
                        1.0 - quotient_grad_norm / grad_norm
                    } else {
                        f64::NAN
                    };
                    let null_component = (grad_norm * grad_norm
                        - quotient_grad_norm * quotient_grad_norm)
                        .max(0.0)
                        .sqrt();
                    let quotient_over_tol = if grad_tolerance > 0.0 {
                        quotient_grad_norm / grad_tolerance
                    } else {
                        f64::INFINITY
                    };
                    let intensive = self.intensive_kkt_diagnostic(target, rho, registry);
                    let orbit =
                        self.gauge_orbit_descent_diagnostic(target, rho, registry, &lambda_smooth);
                    return Err(format!(
                        "SaeManifoldTerm::penalized_quasi_laplace_criterion: {}; \
                         objective stalled for {consecutive_objective_stalls} consecutive refine \
                         rounds, but neither the raw KKT gradient ‖g‖={grad_norm:.6e} nor the \
                         quotient KKT gradient ‖Π⊥null g‖={quotient_grad_norm:.6e} met tolerance \
                         {grad_tolerance:.6e} (‖Π∥null g‖={null_component:.6e}, \
                         null_share={null_share:.4}, \
                         quotient_over_tol={quotient_over_tol:.3e}, {intensive}, {orbit}). Objective \
                         stagnation and a finite deflated factor are diagnostic only; refusing to \
                         rank or differentiate an off-optimum Laplace criterion.",
                        ProbeRefusalKind::inner_not_converged_marker()
                    ));
                }
            } else {
                // The stall streak broke (this round is materially descending or
                // the fraction baseline is not yet meaningful). Material descent
                // re-arms the terminal polish for the next plateau (#2132).
                consecutive_objective_stalls = 0;
                terminal_newton_polish_armed = true;
                gauge_block_armed = true;
            }
        }
    }
}
