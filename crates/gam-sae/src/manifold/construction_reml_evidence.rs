// [#780 line-count gate] The fixed-ρ quasi-Laplace evidence criterion and its
// evidence-pricing machinery (reml_criterion* entries, rank-charge ledger,
// deflated-factor evidence path) live in this sibling file as a second
// `impl SaeManifoldTerm` block, inlined via `include!` from construction.rs so
// it keeps the SAME module scope and private-field access. Keeps the tracked
// construction.rs under the 10k limit.

impl SaeManifoldTerm {
    /// Penalised quasi-Laplace evidence score for the SAE term at a FIXED ρ.
    ///
    /// #1421: this is NOT a true normalized-prior REML/evidence objective. The
    /// assignment priors (softmax entropy, JumpReLU) have NO finite normalizer:
    /// for softmax the reference-logit chart sends `P(ℓ)→0` as a free logit →±∞
    /// so `∫ e^{−λP} dℓ = ∞`, and JumpReLU's bounded penalty `0<P<λ` keeps
    /// `e^{−λP}` bounded below over an unbounded domain, also divergent. There is
    /// therefore no ρ-independent assignment-prior normalizer that can be dropped
    /// as a constant. The smoothing-penalty `−½log|λS|_+` term IS a genuine
    /// (proper-Gaussian) REML normalizer and is kept exactly; the rest is a
    /// penalized quasi-Laplace score (Laplace curvature term `½log|H|` around the
    /// inner optimum), which the engine minimizes over ρ.
    ///
    /// Runs the inner `(t, β)` arrow-Schur Newton solve to convergence at the
    /// supplied ρ (with NO in-loop ARD update — ρ is owned by the engine),
    /// then forms the Laplace/REML cost
    ///
    /// ```text
    /// V(ρ) = ℓ_pen(t̂, β̂; ρ) + ½ log|H(t̂, β̂; ρ)|
    ///        − ½ · p · (Σ_k rank S_k) · log λ_smooth
    /// ```
    ///
    /// where `ℓ_pen = loss.total()` is the penalised objective at the inner
    /// optimum and `½ log|H|` is the Laplace normaliser. `H` is the joint
    /// `(t, β)` Hessian assembled by the arrow-Schur system; its `H_tt` block
    /// carries `α = exp(log_ard)` on its diagonal, so as α grows `½ log|H|`
    /// rises while the `−½·n·log α` already inside `loss.ard` falls — their
    /// balance IS the effective-dof term that the deleted `α = n/‖t‖²` rule
    /// dropped, which is why the criterion needs no clamp to stay finite on a
    /// collapsing axis.
    ///
    /// The final `−½·p·rank(S)·log λ_smooth` term is the smoothing-penalty
    /// normaliser `−½ log|λ S|_+` restricted to its ρ-dependent part: `S_k` is
    /// shared across all `p` decoder output channels (the `⊗ I_p` Kronecker
    /// structure), so `log|λ S|_+ = p·rank(S)·log λ + p·log|S|_+`, and the
    /// `½ p·log|S|_+` piece is ρ-independent. The ρ-independent additive
    /// constants that ARE dropped here (they shift `V` by a constant and do not
    /// affect the ρ-argmin) are the `2π` Laplace constant and the base
    /// `½ p·log|S|_+` penalty logdet. #1421: NO assignment-prior normalizer is
    /// dropped, because none exists (softmax/JumpReLU priors are improper — see
    /// the doc on this function): the quasi-Laplace score simply omits a
    /// normalizer that is not a finite constant.
    ///
    /// Returns `(V, loss)` so the engine can both rank ρ and surface the inner
    /// loss breakdown.
    pub fn reml_criterion(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
    ) -> Result<(f64, SaeManifoldLoss), String> {
        self.reml_criterion_with_refine_policy(
            target,
            rho,
            registry,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
            true,
        )
    }

    pub(crate) fn reml_criterion_with_refine_policy(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
        refine_progress_extension: bool,
    ) -> Result<(f64, SaeManifoldLoss), String> {
        self.reml_criterion_with_refine_policy_and_lane(
            target,
            rho,
            registry,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
            refine_progress_extension,
            None,
        )
    }

    /// [`Self::reml_criterion_with_refine_policy`] with the #2080 surrogate lane
    /// threaded to the streaming `log|S|` evidence term. `lane = None` is the
    /// bit-identical SLQ path; on the dense (non-streaming) branch the lane is
    /// unused (the dense evidence has its own factor-cache log-det).
    pub(crate) fn reml_criterion_with_refine_policy_and_lane(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
        refine_progress_extension: bool,
        lane: Option<&mut SurrogateLaneState>,
    ) -> Result<(f64, SaeManifoldLoss), String> {
        // #976 evidence-ledger scope: one criterion evaluation = one per-atom
        // reseed budget. The joint-fit driver no longer clears the ledger on
        // evidence re-entries (each refine round used to get a fresh budget and
        // could fire an unguarded reseed once per round — the ‖g‖-spike /
        // progress-budget-collapse pathology), so the criterion entry owns the
        // clear.
        self.collapse_events.clear();
        let plan = self.streaming_plan().admitted_or_error(
            self.n_obs(),
            self.output_dim(),
            self.k_atoms(),
        )?;
        if plan.streaming {
            // #1225: streaming and dense MUST optimize the SAME mathematical
            // objective — the full REML criterion `loss.total() + extra_penalty +
            // ½ log|H| − Occam`. The streaming branch previously returned only
            // `loss.total() + extra_penalty_energy`, dropping the Laplace
            // normalizer `½ log|H|` and the Occam term, so large shapes (exactly
            // where streaming is needed) were ranked by penalized loss rather than
            // REML — and dense vs streaming disagreed on the objective. Route
            // through the streaming exact-logdet path, which assembles the same
            // chunk-by-chunk-bit-identical `½ log|H|_stream` and the same
            // `−Occam`/extra-penalty terms as the dense `reml_criterion_with_cache`
            // (different memory strategy, same objective).
            self.reml_criterion_streaming_exact_with_lane(
                target,
                rho,
                registry,
                inner_max_iter,
                learning_rate,
                ridge_ext_coord,
                ridge_beta,
                lane,
            )
        } else {
            let (v, loss, _cache) = self.reml_criterion_with_cache_refine_policy(
                target,
                rho,
                registry,
                inner_max_iter,
                learning_rate,
                ridge_ext_coord,
                ridge_beta,
                refine_progress_extension,
            )?;
            Ok((v, loss))
        }
    }

    /// As [`Self::reml_criterion`], but also returns the converged undamped
    /// `ArrowFactorCache` so callers (the EFS fixed-point step) can read the
    /// selected-inverse traces `(H⁻¹)_tt` / `(H⁻¹)_ββ` without re-factoring.
    /// The cache is the single shared O(K³) Direct factor; both the
    /// log-determinant criterion and the Fellner-Schall ρ-step consume it.
    pub fn reml_criterion_with_cache(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
    ) -> Result<(f64, SaeManifoldLoss, ArrowFactorCache), String> {
        self.reml_criterion_with_cache_refine_policy(
            target,
            rho,
            registry,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
            true,
        )
    }

    pub(crate) fn reml_criterion_with_cache_refine_policy(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
        refine_progress_extension: bool,
    ) -> Result<(f64, SaeManifoldLoss, ArrowFactorCache), String> {
        // #976 evidence-ledger scope (see `reml_criterion_with_refine_policy_
        // and_lane`): direct cache-lane callers also get a fresh per-evaluation
        // reseed budget here; the double clear when routed through the value
        // entry is an idempotent no-op.
        self.collapse_events.clear();
        let admission_plan = self.streaming_plan().admitted_or_error(
            self.n_obs(),
            self.output_dim(),
            self.k_atoms(),
        )?;
        if !admission_plan.direct_logdet_admitted() {
            // The cache-returning REML entry is used by the EFS/outer lanes that
            // need selected-inverse traces in addition to the scalar evidence.
            // Large SAE fits cannot form the dense `N · q · border_dim`
            // evidence slab (`q = K(1+d)`, `border_dim = Σ_k M_k · p`), so the
            // correct implementation is not to reject here and force callers
            // onto a value-only path.  Route through the streaming evidence
            // implementation instead: it reuses the converged per-row factor
            // cache for traces and recomputes the reduced-Schur logdet by
            // chunks / matrix-free matvecs, keeping peak memory at the admitted
            // streaming working set rather than the dense n·k·p floor.
            return self.reml_criterion_streaming_exact_with_cache(
                target,
                rho,
                registry,
                inner_max_iter,
                learning_rate,
                ridge_ext_coord,
                ridge_beta,
            );
        }
        // 1. Run the inner (t, β) Newton solve to its numerical fixed point at
        //    FIXED ρ. Evidence uses the idempotence polish rather than stopping
        //    at the first coarse-KKT-band hit: the value and its implicit
        //    derivative must describe the same differentiable root (#2253).
        let mut rho_fixed = rho.clone();
        let initial_fit = self.run_joint_fit_arrow_schur_for_evidence(
            target,
            &mut rho_fixed,
            registry,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
        )?;
        let mut loss = initial_fit.loss;
        let mut evidence_fixed_point = initial_fit.fixed_point;

        // 2. Drive the inner (t, β) solve to the KKT/step-converged optimum and
        //    take one final UNDAMPED factor there to obtain the joint Hessian
        //    log-determinant. We force ridge = 0 and the dense `Direct` Schur
        //    mode so `arrow_log_det_from_cache` returns the exact
        //    `log|H| = Σ_i log|H_tt^(i)| + log|Schur_β|` (it rejects damped
        //    factors and InexactPCG caches, which have no dense Schur factor).
        //    This is the same evidence convention the main GAM REML path uses.
        //    The shared `converge_inner_for_undamped_logdet` driver guarantees
        //    the per-row `H_tt^(i)` blocks are PD at the converged optimum so
        //    the undamped (`ridge = 0`) factorization succeeds — the streaming
        //    log-det path reuses the identical driver so both rank the same
        //    converged Laplace optimum and stay bit-identical.
        //
        //    #2080 COST NOTE — why the dense `log|Schur_β|` is NOT rank-updated
        //    across outer ρ probes from a cached factor. The tempting identity
        //    is the matrix-determinant / pencil form: with the smooth penalty
        //    entering the border block linearly in λ = e^ρ (block-diagonal
        //    `Σ_k λ_k · (S_k ⊗ I_p)` on the full-`B` layout, `Σ_k λ_k · S̃_k` on
        //    the framed layout — see `assemble_arrow_schur` /
        //    `construction_arrow_schur_assembly.rs`), a probe at ρ' would give
        //        S(ρ') = S(ρ) + Σ_k (e^{ρ'_k} − e^{ρ_k}) · P_k ,
        //    and `log|S(ρ')|` would follow exactly from the cached generalized
        //    eigendecomposition of the pencil `(S(ρ), P)`. That identity is an
        //    EXACT algebraic statement ONLY at a FIXED inner state `(t̂, β̂)`.
        //    The criterion is defined at the RE-CONVERGED inner optimum of each
        //    probed ρ (this driver refuses to rank an off-optimum Laplace
        //    value), and the converged state moves with ρ by the implicit-
        //    function law `dθ̂/dρ = −H⁻¹ · ∂g/∂ρ`, so every Gauss-Newton block
        //    of S — `H_ββ(t̂, β̂)` AND the eliminated `Σ_i H_βt H_tt⁻¹ H_tβ`
        //    downdate — changes DENSELY between probes, not by a low-rank or
        //    scaled-block term. A pencil update across probes would therefore
        //    be an approximation, which the exactness doctrine bans from this
        //    criterion. The one lane whose premise DOES hold — the frozen
        //    `inner_max_iter == 0` warm-start reuse, where `(t̂, β̂)` is pinned
        //    by contract — already factors exactly once per evaluation, so
        //    there is no second factorization for the identity to replace.
        //    The structural saving that IS exact — factoring the dense border
        //    Schur once per evaluation (at the stationary iterate) instead of
        //    once per refine round — lives inside
        //    `converge_inner_for_undamped_logdet`.
        let options = ArrowSolveOptions::direct()
            .with_ill_conditioning_tolerated()
            .with_schur_pd_floor(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR);
        let cache = self.converge_inner_for_undamped_logdet(
            target,
            rho,
            &mut rho_fixed,
            registry,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
            &mut loss,
            &mut evidence_fixed_point,
            &options,
            refine_progress_extension,
        )?;
        self.record_evidence_gauge_deflation_count(cache.gauge_deflated_directions)?;
        loss.evidence_gauge_deflated_directions = cache.gauge_deflated_directions;
        let log_det = arrow_log_det_from_cache(&cache).ok_or_else(|| {
            // Distinguish a GENUINE infeasibility — a probed ρ where the joint
            // Hessian is not PD so the Laplace evidence log-det is undefined —
            // from a real factorization defect. The cross-row IBP Woodbury
            // capacitance `C = I_R + D·Uᵀ H₀'⁻¹ U` can have det ≤ 0 at a ρ the
            // outer optimizer line-searches into (the indefinite basin adjacent
            // to the PD region); there the log-det legitimately does not exist.
            // That refusal must be RECOVERABLE (the outer BFGS should get +∞ and
            // steer back into the PD region), exactly like the "non-PD per-row
            // H_tt block" refusal — not a fatal `RemlOptimizationFailed` that
            // aborts the whole fit. See `is_recoverable_value_probe_refusal`.
            // (The old message claimed "no dense Schur factor", which is false
            // here — the Schur factor is present; the Woodbury correction is the
            // non-finite term.)
            if cache.cross_row_woodbury.is_some() && !cache.cross_row_woodbury_log_det().is_finite()
            {
                "SaeManifoldTerm::reml_criterion: cross-row IBP joint Hessian is non-PD at \
                 this ρ; evidence Laplace log-det undefined (infeasible ρ probe)"
                    .to_string()
            } else {
                "SaeManifoldTerm::reml_criterion: arrow_log_det_from_cache returned None \
                 (undamped joint Hessian log-det unavailable for the Laplace normaliser)"
                    .to_string()
            }
        })?;

        // 3. Smoothing-penalty Occam term `−½·Σ_k r_k·rank(S_k)·log λ_smooth`
        //    plus the profiled-frame evidence-dimension correction
        //    `+½·Σ_k r_k·(p−r_k)·log λ_smooth` (issue #972). On the full-`B` path
        //    (`r_k == p`, no frames) this is exactly the historical
        //    `½·p·(Σ rank S_k)·log λ_smooth`, so the small-model criterion is
        //    unchanged. The single seam is `reml_occam_term`, shared with the
        //    streaming path so both rank the identical Laplace dimension count.
        let occam = self.reml_occam_term(rho)?;

        // Extra penalized-objective energy with no native `loss.*` twin
        // (#671/#737, and the full-objective completion): all registry analytic
        // penalties (Isometry, SCAD/MCP, BlockOrthogonality, decoder-block
        // set), the decoder repulsion conditioner, and the Jeffreys separation
        // barrier. The inner solve descends all of them (they enter the KKT
        // gradient), so the Laplace criterion must add them to rank the SAME
        // penalized deviance — the envelope theorem the analytic outer gradient
        // relies on holds only then. See `reml_extra_penalty_value_total`.
        let extra_penalty_energy = self
            .reml_extra_penalty_value_total(registry)
            .map_err(|err| format!("SaeManifoldTerm::reml_criterion: {err}"))?;

        let v = {
            // #5/(B): replace the COORDINATE-block ½log|H_tt| in the Laplace
            // complexity with the honest BIC ½·d_eff·log n on each atom's realised
            // decoder rank. The decoder-scale mispricing (`½log(a²‖B‖²)` scale,
            // over-charging real atoms + rewarding a²‖B‖²→0) lives ENTIRELY in the
            // coordinate block (`H_tt ∝ ‖B‖²`); the β/Schur block is
            // ‖B‖-independent (ρ⁰ coupling) and stays. `d_eff` is rotation-
            // invariant, so it accepts a real rank-2 circle and neutralises a
            // vanishing atom — but does NOT distinguish clean-vs-blend (producer's
            // job).
            // Noise floor R = residual dispersion φ (per-fit, noise-relative — NOT a
            // hardcoded/self-relative floor). If it cannot be computed the vanishing-
            // atom detection silently degrades (R→0 keeps rank_eff≈rank), so surface
            // it loudly rather than hiding a re-admitted co-collapse.
            let residual = self.reconstruction_residual(target, rho)?;
            let disp = self
                .reconstruction_dispersion(&loss, &cache, rho, Some(residual.view()))
                .map_err(|e| {
                    format!(
                        "SaeManifoldTerm::reml_criterion: rank-charge dispersion is required: {e}"
                    )
                })?;
            let d_eff = self.per_atom_realised_rank_dof(rho, disp)?;
            // Occupancy-aware effective sample size N_eff,k = Σ_i a_{ik}², the #2a
            // per-atom BIC log-scale (same quantity `per_atom_realised_rank_dof` uses
            // internally for the MP edge; recomputed here — a cheap Σa² — to price the
            // charge in the same currency).
            let n_eff = self.per_atom_effective_sample_size();
            // #5 VETO — categorical Laplace-VALIDITY condition (blend-null null-license
            // fix, recov matrix 12484591): an atom with rank_eff==0 (⟺ d_eff==0)
            // reconstructs NOTHING. Its Laplace evidence is not "small" — it is INVALID:
            // the vanishing decoder makes the β-mode degenerate, and the β-Schur log-det
            // → −∞ is the approximation BREAKING DOWN, not a real reward (which is why a
            // zero-‖B‖ atom got "born" on a featureless blend-null residual while the
            // rank charge only neutralised — charge 0 — its coordinate block). Such an
            // atom is unbirthable: reject CATEGORICALLY (v → +∞) rather than pricing a
            // degenerate Laplace term. No tuned constant — a validity condition, not a
            // penalty. rank_eff is an integer MP count so ==0 is crisp; a real rank-2
            // circle (rank_eff=2) is untouched. This is #10's "make the degenerate class
            // unbirthable" at the birth gate. TRAILHEAD: the deeper fix is a floor on the
            // β-Schur decoder-curvature block (assemble_arrow_schur) so a vanishing β
            // doesn't drive its Schur log-det → −∞; deferred (touches the shipped Schur
            // path); the birth-gate veto here is the guard.
            //
            // #2b — RLCT justification (why the veto is a VALIDITY condition, not a
            // heuristic): the null atom (truth B*=0) sits at a singularity of the model
            // — the product form a²‖B‖² makes the Fisher information degenerate there —
            // and singular learning theory gives it real log-canonical threshold (RLCT)
            // λ=½: the leading zeta pole of ∫(a²‖B‖²)^s comes from the amplitude at s=½,
            // independent of M,p,d. So the null's asymptotic evidence cost is only
            // ½·ln n per e-fold, and NO Θ(log n) rank charge can separate a null birth
            // from a real one AT the singular point. The categorical veto (v→+∞ when
            // rank_eff==0) is therefore the only valid way to keep the degenerate class
            // unbirthable; a finite penalty could not.
            // #2a — occupancy-aware BIC/Laplace scale. The shared scalar helper
            // owns both the rank-zero veto and the exact replacement
            // `0.5 log|H| - 0.5 log|H_tt| + rank_charge`; dense, streaming, and
            // criterion-as-atoms assembly therefore cannot drift apart.
            let log_det_tt = coordinate_block_log_det(&cache)?;
            let laplace_complexity =
                rank_adjusted_laplace_complexity(log_det, log_det_tt, &d_eff, &n_eff)?;
            loss.total() + extra_penalty_energy + laplace_complexity - occam
        };
        Ok((v, loss, cache))
    }

    /// The #1037 quotient-dimension invariant: a Laplace normalizer `½log|H|` is
    /// only comparable across ρ at a COMMON quotient (gauge-deflation) dimension.
    /// The first observation pins the expected count; a later match is a no-op.
    ///
    /// A later observation that DIFFERS is, under the K>1 fit, a LEGITIMATE
    /// quotient-dimension event — an atom born, reseeded (the #976 collapse
    /// guards), or rank-reduced moves the number of gauge-flat rows. Because a
    /// deflated direction is lifted to unit stiffness and contributes the
    /// ρ-independent `log 1 = 0` to the evidence, re-anchoring the comparison to
    /// the new dimension is exactly evidence-preserving and keeps every future
    /// cross-ρ comparison consistent — the principled response, not an abort.
    ///
    /// The genuine pathology the guard still catches is a count that NEVER
    /// STABILIZES: re-anchors are bounded by the per-atom structural-event budget
    /// (`k·(reseed_budget+1)+1`), and a runaway quotient dimension past that
    /// bound refuses loudly. This supersedes the prior strict-constant guard and
    /// its ±1 flicker band (#1117) at root — the band was masking exactly the
    /// legitimate K>1 dimension changes this re-anchoring now handles.
    pub(crate) fn record_evidence_gauge_deflation_count(
        &mut self,
        count: usize,
    ) -> Result<(), String> {
        match self.expected_evidence_gauge_deflated_directions {
            Some(expected) if expected == count => Ok(()),
            Some(expected) => {
                // A change in the gauge-deflation count between two evidence
                // factorizations is a legitimate quotient-dimension event under
                // the K>1 fit: an atom can be born, reseeded (the #976 collapse
                // guards), or rank-reduced across the ρ-walk, and each such event
                // moves the number of gauge-flat rows. The #1037 invariant is
                // NOT "the count never changes" — it is "two Laplace normalizers
                // are only comparable at a COMMON quotient dimension". The
                // principled response to a legitimate change is therefore to
                // RE-ANCHOR the comparison to the new dimension (so every future
                // cross-ρ comparison within the optimization is consistent), not
                // to abort the fit. This is exactly evidence-preserving: each
                // gauge-deflated direction is lifted to unit stiffness and
                // contributes the ρ-independent `log 1 = 0` to `½log|H|`, so the
                // converged criterion value is identical whether a given row is
                // counted as deflated or not — only the BOOKKEEPING dimension
                // must agree across a comparison, and re-anchoring restores that.
                //
                // The genuine pathology the guard must still catch is a count
                // that NEVER STABILIZES — an OSCILLATING quotient dimension that
                // re-anchors without converging, signalling a truly ill-posed
                // evidence surface. But the deflation count is NOT a discrete
                // dictionary-level event count: it is the per-ROW-summed number of
                // near-null evidence directions across all N rows (#1217). On real
                // K≥2 activations it is an O(N) quantity that drifts SMOOTHLY and
                // monotonically as the conditioning improves over the ρ-walk
                // (e.g. 171→156→…→113 as smoothing increases) — a benign,
                // evidence-neutral change (each deflated direction contributes the
                // ρ-independent `log 1 = 0` to `½log|H|`, so re-anchoring never
                // moves the criterion value). Charging such a monotone drift
                // against a `k`-sized "structural event" budget was wrong: it
                // counts threshold crossings of a continuous per-row quantity, not
                // atom births/reseeds, so the budget tripped on a perfectly healthy
                // converging K=2 fit (#1217 regression from the #1189/#1190
                // basin-escape fixes, which shifted which rows sit near the
                // deflation floor).
                //
                // The principled discriminator is DIRECTION REVERSALS: a count
                // that drifts one way and settles is benign; a count that bounces
                // up and down without settling is the oscillating-quotient
                // pathology. We therefore charge the re-anchor budget ONLY on a
                // reversal of the change direction, and size the budget by the
                // number of distinct dictionary structural events (births/reseeds)
                // that can each legitimately flip the drift direction. A monotone
                // drift of any length re-anchors freely (it is consistently
                // re-anchored and evidence-neutral); a genuinely oscillating count
                // exhausts the reversal budget and refuses loudly.
                let delta_sign: i8 = if count > expected { 1 } else { -1 };
                let is_reversal = self.evidence_gauge_deflation_last_delta_sign != 0
                    && delta_sign != self.evidence_gauge_deflation_last_delta_sign;
                self.evidence_gauge_deflation_last_delta_sign = delta_sign;
                // A reversal alone is NOT the pathology — a BOUNDED flicker of a
                // few rows crossing the near-null deflation floor reverses
                // direction every step yet is the discretization jitter of a
                // continuous evidence spectrum, fully evidence-neutral (each
                // deflated direction contributes `log 1 = 0` either way). The
                // genuine "quotient dimension not stabilizing" pathology is a
                // WIDE-amplitude oscillation: a substantial FRACTION of the
                // dimension flipping back and forth. The count is an O(N) per-row
                // sum, so the discriminator must be the reversal AMPLITUDE
                // relative to the dimension level, not the bare reversal. Charge
                // the reversal budget only when a reversal's step exceeds a
                // relative jitter band; a converged-but-flickering fit (e.g.
                // 150<->147 on N=200, ~2% of the level) re-anchors freely while a
                // true runaway (e.g. 9<->2, ~80% of the level) still trips every
                // reversal and exhausts the budget. This was the second #795 root
                // cause: the single-planted-circle fit's per-row count flickers
                // 150<->147 near the deflation floor, so the bare-reversal guard
                // refused the simplest possible fit — with the isometry gauge ON
                // *or* OFF — long before the gauge magnitude mattered.
                let amplitude = expected.abs_diff(count);
                let level = expected.max(count);
                let jitter_band = (level / 4).max(2);
                if is_reversal && amplitude > jitter_band {
                    self.evidence_gauge_deflation_reanchors += 1;
                }
                let reversal_budget = self
                    .k_atoms()
                    .saturating_mul(
                        SAE_ATOM_COLLAPSE_RESEED_BUDGET
                            + SAE_DICTIONARY_COCOLLAPSE_RESEED_BUDGET
                            + 1,
                    )
                    .saturating_add(1);
                if self.evidence_gauge_deflation_reanchors > reversal_budget {
                    return Err(format!(
                        "SaeManifoldTerm::reml_criterion: row-gauge evidence deflation count \
                         oscillated (reversed direction {} times, last {expected}->{count}) within \
                         one optimization, exceeding the {reversal_budget}-reversal budget for {} \
                         atoms; the quotient dimension is not stabilizing, refusing to compare \
                         Laplace normalizers",
                        self.evidence_gauge_deflation_reanchors,
                        self.k_atoms()
                    ));
                }
                log::debug!(
                    "SaeManifoldTerm::reml_criterion: per-row evidence deflation count changed \
                     {expected}->{count} (a benign per-row conditioning drift across the ρ-walk; \
                     reversal {}/{reversal_budget}); re-anchoring the Laplace normalizer comparison \
                     to the new dimension",
                    self.evidence_gauge_deflation_reanchors
                );
                self.expected_evidence_gauge_deflated_directions = Some(count);
                Ok(())
            }
            None => {
                self.expected_evidence_gauge_deflated_directions = Some(count);
                Ok(())
            }
        }
    }

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
    /// The Laplace normaliser `½log|H|` is only the correct REML criterion at
    /// the inner optimum `(t̂, β̂)`, so the criterion must refine the inner state
    /// until either the KKT gradient or the undamped Newton step meets tolerance
    /// before factoring. Crucially, **at the converged optimum the per-row
    /// `H_tt^(i)` blocks are PD**, so the undamped (`ridge = 0`) factorization
    /// succeeds; an off-optimum iterate (e.g. the initial seed, or a state
    /// stopped after only `inner_max_iter` steps) can have an indefinite /
    /// rank-deficient per-row block (`p_out = 1` → rank-1 `JᵀJ`, softmax
    /// assignment-sparsity negative logit curvature) that surfaces
    /// `PerRowFactorFailed` from the undamped `factor_one_row`. Both the dense
    /// (`reml_criterion_with_cache`) and the streaming
    /// (`reml_criterion_streaming_exact`) evidence paths route through this same
    /// driver, so they converge to the identical inner state and their
    /// `ridge = 0` log-determinants stay bit-identical (#847).
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
        evidence_fixed_point: &mut bool,
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
        // `run_joint_fit_arrow_schur_for_evidence(..., 0, ...)`,
        // which under the `max_iter == 0` freeze (gam#577/#579, #850) runs ONLY
        // the β-neutral basis refresh and returns the loss without touching β —
        // it skips the rank-reduction, frame activation, re-seed guards, and the
        // #1026 decoder-LSQ polish that would otherwise refit β off the seed — so
        // `self` is at the warm-start β here.
        if inner_max_iter == 0 {
            let mut sys = self
                .assemble_arrow_schur(target, rho, registry)
                .map_err(|err| format!("SaeManifoldTerm::reml_criterion: {err}"))?;
            // #1095/#2228 — same decoupling as the stall / gradient-stationary
            // acceptance paths. This frozen warm-start evidence log-det is read from
            // the ridge-0 factor below, which is non-PD BY CONSTRUCTION on an
            // over-parametrized chart (a rank-1 radial null per row). Per-row
            // spectral deflation only fires when `row_gauge_deflation.is_some()`, and
            // the decoded-derivative gauge floor (`tangent·tangent > 1e-24`) can
            // leave it None on exactly the flat axis that carries the null — so
            // force the evidence system to opt into per-row spectral discovery: the
            // null is unit-stiffness deflated (`log 1 = 0`, ρ-independent) and the
            // frozen log-det is finite, instead of refusing a rescuable warm-start
            // reuse. A full-rank block has no sub-floor eigenvalue and is untouched.
            Self::ensure_row_gauge_deflation_for_evidence(&mut sys);
            let factored = solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, options)
                .map_err(|err| format!("SaeManifoldTerm::reml_criterion: {err}"))?;
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
        let mut saw_refine_progress = false;
        // #2234 — one progress-gated extra refinement window (see the budget
        // escalation at the non-convergence refusal below). 0 until granted.
        let mut budget_escalation_extra = 0usize;
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
            .map_err(|err| format!("SaeManifoldTerm::reml_criterion: {err}"))?;
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
        // #2228 Stage-2 — terminal exact-Newton invocations remaining for this
        // criterion evaluation. Each invocation runs a bounded superlinear
        // polish (`terminal_exact_newton_polish`) from the objective-stall
        // plateau; two invocations of ≤12 steps bound the extra cost while
        // covering a re-stall after an intervening refine round.
        let mut terminal_newton_rounds: usize = 2;
        loop {
            let mut sys = self
                .assemble_arrow_schur(target, rho, registry)
                .map_err(|err| format!("SaeManifoldTerm::reml_criterion: {err}"))?;
            // Evidence-only factorization: the Newton step (Δt, Δβ) is discarded
            // and only the factor cache is consumed — the exact undamped log-det
            // and the selected-inverse traces. As ρ sweeps to extremes (e.g. a
            // wide ARD-α sweep), H_tt is genuinely PD but can be ill-conditioned;
            // the standard Direct guard rejects that to protect Newton-step
            // accuracy, but the log-det is exact from diag(L) regardless of the
            // condition number and the traces only need the (PD) factor. So
            // tolerate the ill-conditioning rejection here (a genuine non-PD pivot
            // still errors). The cache stays undamped at ridge=0, so
            // `arrow_log_det_from_cache` remains exact.
            // The exact KKT stationarity residual is the joint gradient
            // ‖g‖ = √(Σ_i ‖g_t^(i)‖² + ‖g_β‖²), read straight off the assembled
            // system. Unlike the Newton step Δ = H⁻¹g, the gradient is
            // factorisation-independent: it is NOT amplified by an inverse, so a
            // genuinely stationary but ill-conditioned fit (tiny g, possibly large
            // Δ in a flat direction) is correctly recognised as converged. The
            // `with_ill_conditioning_tolerated` Direct factor below documents that
            // its Δ may be inaccurate in exactly those flat directions, so using Δ
            // alone as the convergence gate would falsely reject healthy fits.
            let grad_norm_sq: f64 = Self::system_grad_norm_sq(&sys);
            let grad_norm = grad_norm_sq.sqrt();
            let lambda_smooth = rho_fixed.lambda_smooth_vec();
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
                    "SaeManifoldTerm::reml_criterion: undamped inner KKT residual is non-finite \
                     at the inner optimum (‖g‖²={grad_norm_sq}); the joint Hessian \
                     factorisation is degenerate at this ρ"
                ));
            }
            // #2080 criterion-cost restructure — the Laplace normaliser ½log|H|
            // is the REML criterion ONLY at the inner KKT optimum, so the FULL
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
            // verdict (same #1038 IBP self-term downdate, same gauge/spectral
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
                Self::evidence_kkt_stationary(grad_norm, quotient_grad_norm, grad_tolerance);
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
            if gradient_stationary && *evidence_fixed_point {
                // #1095/#2228 — decouple this ACCEPT from undamped-factor success,
                // the same acceptance-local pattern as the stall path below. A
                // cleanly-fit over-parametrized chart (d_atom=2 on intrinsic 1-D
                // data) is gradient-STATIONARY — the tangent is fit and the rank-1
                // radial null contributes ZERO gradient — so it lands HERE rather
                // than the objective-stall path, yet its ridge-0 per-row H_tt is
                // non-PD by construction. Force the acceptance factor to opt into
                // per-row spectral discovery so the null is unit-stiffness deflated
                // (`log 1 = 0`, ρ-independent) and the evidence log-det is finite.
                // This does NOT touch the undamped #2080 probe: the probe runs only
                // in the non-stationary branch below, which THIS block never reaches
                // (every arm returns), and a non-stationary iteration never installs
                // this deflation — so `sys` stays undamped for the probe.
                Self::ensure_row_gauge_deflation_for_evidence(&mut sys);
                let (delta_t, delta_beta, cache): (Array1<f64>, Array1<f64>, ArrowFactorCache) =
                    match solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, options) {
                        Ok(factored) => factored,
                        Err(err) if Self::is_undamped_evidence_row_non_pd(&err) => {
                            // K>1: the softmax/IBP logit–coordinate Gauss-Newton
                            // cross-terms (H_zt = J_z^T J_t, assembled row-locally from
                            // the assignment JVP × basis JVP) can make a per-row H_tt
                            // indefinite at the TRUE KKT stationary point — when two
                            // atoms' decoders specialise in opposite directions the
                            // Schur complement of the logit block goes negative even
                            // though the priors and the full-joint GN term are PSD.
                            //
                            // The undamped evidence factor conditions that block the
                            // PRINCIPLED way: with per-row spectral discovery now
                            // force-enabled above (`row_gauge_deflation` installed),
                            // `factor_spectral_deflated_evidence_row` discovers the
                            // negative/flat eigen-direction — including the #1095/#2228
                            // radial null the decoded-derivative gauge floor
                            // (`tangent·tangent > 1e-24`) would otherwise have excluded
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
                                "SaeManifoldTerm::reml_criterion: stationary undamped \
                                 evidence factorization has a non-PD per-row H_tt block \
                                 that spectral unit-stiffness deflation could not \
                                 condition (‖g‖={grad_norm:.6e}, tol {grad_tolerance:.6e}); \
                                 {err}"
                            ));
                        }
                        Err(err) => {
                            return Err(format!("SaeManifoldTerm::reml_criterion: {err}"));
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
                        "SaeManifoldTerm::reml_criterion: undamped inner residual is non-finite at \
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
                log::debug!(
                    "SAE evidence factor accepted at KKT stationarity: ‖g‖={grad_norm:.6e} \
                     ‖Π⊥gauge g‖={quotient_grad_norm:.6e} tol={grad_tolerance:.6e} \
                     ‖Δ‖={:.6e} ‖Π⊥gauge Δ‖={:.6e} after {total_inner_iter} inner iterations",
                    step_norm_sq.sqrt(),
                    quotient_step_norm_sq.sqrt(),
                );
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
                        // Laplace log-det is UNDEFINED at this ρ: the ρ is
                        // infeasible. For a PROBE (line-search value / FD /
                        // seed-validation lane, `refine_progress_extension == false`)
                        // the caller only needs a typed infeasible verdict so the
                        // outer search steers back into the PD region — refining the
                        // inner solve to try to CROSS the indefinite basin is the
                        // accepted-iterate's job, not a probe's. Grinding the probe
                        // refine budget (up to `4×inner_max_iter`, and historically
                        // the accepted `16×/64×` via `reml_criterion_with_cache`) on
                        // every overshooting line-search / FD probe is exactly the
                        // wide-`p` outer REML hang (#2080). Return the typed refusal
                        // after this single diagnostic factor pass;
                        // `is_recoverable_value_probe_refusal` maps it to the finite
                        // infeasibility wall.
                        if !refine_progress_extension {
                            return Err(format!(
                                "SaeManifoldTerm::reml_criterion: undamped evidence \
                             factorization hit a non-PD per-row H_tt block before KKT \
                             stationarity at an infeasible-ρ probe (‖g‖={grad_norm:.6e}, \
                             tol {grad_tolerance:.6e}); returning the typed infeasible \
                             refusal without grinding the probe refinement budget; {err}"
                            ));
                        }
                        let refine_limit = Self::refine_iteration_limit(
                            total_inner_iter,
                            base_refine_iter,
                            progress_refine_iter,
                            previous_refine_grad_norm,
                            grad_norm,
                            saw_refine_progress,
                        );
                        if total_inner_iter >= refine_limit {
                            // #1117/#1118 — pre-stationarity genuinely-indefinite
                            // non-gauge H_tt under K>1 IBP/softmax row-sharing. The
                            // logit × coordinate Gauss-Newton cross term H_zt = J_zᵀJ_t
                            // can drive a shared row's H_tt Schur complement NEGATIVE off
                            // the gauge orbit; the LM-escalated refinement above cannot
                            // always cross the indefinite basin into the PD region within
                            // the descent-extended budget.
                            //
                            // The undamped (ridge=0) evidence factor already conditions
                            // that block the PRINCIPLED way: `factor_spectral_deflated_
                            // evidence_row` discovers the negative/flat eigen-direction
                            // and stiffens it to UNIT curvature (eigenvalue → +1), a
                            // ρ-INDEPENDENT log 1 = 0 evidence contribution — so a
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
                                "SaeManifoldTerm::reml_criterion: undamped evidence \
                             factorization hit a non-PD per-row H_tt block before KKT \
                             stationarity (‖g‖={grad_norm:.6e}, tol {grad_tolerance:.6e}) \
                             and the refinement budget was exhausted after \
                             {total_inner_iter} inner iterations; {err}"
                            ));
                        }
                        let remaining = refine_limit - total_inner_iter;
                        let refine_iter = inner_max_iter.max(1).min(remaining);
                        saw_refine_progress |=
                            Self::refine_round_made_progress(previous_refine_grad_norm, grad_norm);
                        previous_refine_grad_norm = Some(grad_norm);
                        let refine = self.run_joint_fit_arrow_schur_for_evidence(
                            target,
                            rho_fixed,
                            registry,
                            refine_iter,
                            learning_rate,
                            ridge_ext_coord,
                            ridge_beta,
                        )?;
                        *loss = refine.loss;
                        *evidence_fixed_point = refine.fixed_point;
                        total_inner_iter += refine_iter;
                        continue;
                    }
                    Err(err) => {
                        return Err(format!("SaeManifoldTerm::reml_criterion: {err}"));
                    }
                }
            }
            let refine_limit = Self::refine_iteration_limit(
                total_inner_iter,
                base_refine_iter,
                progress_refine_iter,
                previous_refine_grad_norm,
                grad_norm,
                saw_refine_progress,
            );
            let effective_refine_limit = refine_limit
                .checked_add(budget_escalation_extra)
                .ok_or_else(|| {
                    "SaeManifoldTerm::reml_criterion: inner-refinement budget overflow".to_string()
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
                // cost-lane disagreement at one ρ). A solve that is MEASURABLY
                // DESCENDING (`saw_refine_progress`) is an unfinished
                // computation, not an infeasibility: grant it one additional
                // window of the same size and keep refining. The ordinary
                // nonstationary lane retains that single-window hang bound. A
                // coarse-KKT state that is still non-idempotent is different:
                // returning it is the #2253 value/gradient mismatch, so every
                // FURTHER window remains available only while the EXISTING
                // round-to-round KKT predicate proves fresh progress. The first
                // non-decreasing round reaches the typed refusal below; no new
                // tolerance or fixed iteration budget is introduced.
                let stationary_window_paid = gradient_stationary
                    && budget_escalation_extra > 0
                    && Self::refine_round_made_progress(previous_refine_grad_norm, grad_norm);
                if (saw_refine_progress && budget_escalation_extra == 0)
                    || stationary_window_paid
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
                            "SaeManifoldTerm::reml_criterion: escalated inner-refinement budget overflow"
                                .to_string()
                        })?;
                    log::debug!(
                        "SaeManifoldTerm::reml_criterion: budget escalation at fixed ρ — \
                         ‖g‖={grad_norm:.6e} (tol {grad_tolerance:.6e}) still descending after \
                         {total_inner_iter} inner iterations; granting a progress-paid window of \
                         {escalation_window} iterations"
                    );
                } else if gradient_stationary {
                    return Err(format!(
                        "SaeManifoldTerm::reml_criterion: inner solve did not converge at fixed ρ; \
                         KKT entered its admission band (raw ‖g‖={grad_norm:.6e}, quotient \
                         ‖Π⊥gauge g‖={quotient_grad_norm:.6e}, tolerance {grad_tolerance:.6e}) \
                         but an evidence-only re-entry still made a strict state/objective move \
                         after {total_inner_iter} granted iterations. Refusing to differentiate \
                         a non-idempotent inner map."
                    ));
                } else {
                    // Inner solve did not converge in reml_criterion; the returned
                    // Err below carries the non-convergence diagnostic (gradient /
                    // quotient-gradient norms and the tolerance) to the caller. The
                    // historical quotient-Newton-step figures are no longer printed:
                    // the pre-stationarity Newton step was ALWAYS diagnostic-only
                    // (convergence is judged on the factorisation-independent KKT
                    // gradient), and the #2080 restructure above no longer pays the
                    // full dense factorization that produced it at non-stationary
                    // rounds.
                    return Err(format!(
                        "SaeManifoldTerm::reml_criterion: inner solve did not converge at fixed ρ; \
                         neither the KKT gradient ‖g‖={grad_norm:.6e} nor the quotient KKT gradient \
                         ‖Π⊥gauge g‖={quotient_grad_norm:.6e} met tolerance {grad_tolerance:.6e} \
                         after {total_inner_iter} inner iterations. Refusing to rank an \
                         off-optimum Laplace criterion."
                    ));
                }
            }
            let refine_limit = refine_limit
                .checked_add(budget_escalation_extra)
                .ok_or_else(|| {
                    "SaeManifoldTerm::reml_criterion: inner-refinement budget overflow".to_string()
                })?;
            let remaining = refine_limit.checked_sub(total_inner_iter).ok_or_else(|| {
                format!(
                    "SaeManifoldTerm::reml_criterion: inner-refinement accounting mismatch \
                     ({total_inner_iter} iterations consumed past limit {refine_limit})"
                )
            })?;
            let refine_iter = inner_max_iter.max(1).min(remaining);
            saw_refine_progress |=
                Self::refine_round_made_progress(previous_refine_grad_norm, grad_norm);
            previous_refine_grad_norm = Some(grad_norm);
            let refine = self.run_joint_fit_arrow_schur_for_evidence(
                target,
                rho_fixed,
                registry,
                refine_iter,
                learning_rate,
                ridge_ext_coord,
                ridge_beta,
            )?;
            *loss = refine.loss;
            *evidence_fixed_point = refine.fixed_point;
            total_inner_iter += refine_iter;
            refine_rounds += 1;
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
                .map_err(|err| format!("SaeManifoldTerm::reml_criterion: {err}"))?;
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
                && *evidence_fixed_point
            {
                let mut stationary_sys = self
                    .assemble_arrow_schur(target, rho_fixed, registry)
                    .map_err(|err| format!("SaeManifoldTerm::reml_criterion: {err}"))?;
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
                    if Self::evidence_kkt_stationary(
                        stationary_grad_norm,
                        stationary_quotient_grad_norm,
                        grad_tolerance,
                    ) {
                        return Ok(stationary_cache);
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
                    let newton_decrement_sq = sae_manifold_newton_directional_decrease(
                        &stationary_sys,
                        stationary_dt.view(),
                        stationary_db.view(),
                    )
                    .max(0.0);
                    let predicted_relative_decrease = 0.5 * newton_decrement_sq / objective_scale;
                    log::debug!(
                        "SAE inner stall certificate: ‖g‖={stationary_grad_norm:.6e} \
                         ‖Π⊥gauge g‖={stationary_quotient_grad_norm:.6e} tol={grad_tolerance:.6e} \
                         λ²={newton_decrement_sq:.6e} ½λ²/scale={predicted_relative_decrease:.6e} \
                         obj_scale={objective_scale:.6e} accept_tol={SAE_MANIFOLD_INNER_OBJECTIVE_STALL_REL_TOL:.6e}"
                    );
                    // A small model-predicted decrease is diagnostic evidence of
                    // an ill-conditioned crawl, not a KKT root. The analytic outer
                    // gradient differentiates the stationary envelope, so only the
                    // raw/quotient KKT gate above may return this cache.
                    // A flat objective round is only a convergence shortcut when
                    // the KKT certificate above is stationary. If not, keep using
                    // the deterministic refinement budget: either later rounds
                    // reach stationarity, or the normal `total_inner_iter >=
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
                // the iterate to the exact-Hessian terminal Newton phase; a
                // committed step strictly contracts ‖g‖, so the refine loop
                // resumes with fresh progress instead of refusing. The phase
                // mints nothing — acceptance stays with the loop-top KKT gate
                // and the idempotence certificate (the state moved, so
                // `evidence_fixed_point` is cleared and one evidence re-entry
                // must recur exactly before acceptance, same as any hook move).
                if terminal_newton_rounds > 0 {
                    terminal_newton_rounds -= 1;
                    if self.terminal_exact_newton_polish(
                        target,
                        rho_fixed,
                        registry,
                        &lambda_smooth,
                        grad_tolerance,
                        objective_scale,
                        options,
                        12,
                    )? {
                        *evidence_fixed_point = false;
                        consecutive_objective_stalls = 0;
                        saw_refine_progress = true;
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
                    return Err(format!(
                        "SaeManifoldTerm::reml_criterion: inner solve did not converge at fixed ρ; \
                         objective stalled for {consecutive_objective_stalls} consecutive refine \
                         rounds, but neither the raw KKT gradient ‖g‖={grad_norm:.6e} nor its \
                         quotient met tolerance {grad_tolerance:.6e}. Objective stagnation and a \
                         finite deflated factor are diagnostic only; refusing to rank or \
                         differentiate an off-optimum Laplace criterion."
                    ));
                }
            } else {
                // The stall streak broke (this round is materially descending or
                // the fraction baseline is not yet meaningful).
                consecutive_objective_stalls = 0;
            }
        }
    }

    /// The empty per-row `ArrowRowGaugeDeflation` that opts a system into per-row
    /// spectral discovery (the #974 low-rank-whiten seam). An intrinsic flat /
    /// indefinite `H_tt` direction is then deflated to UNIT stiffness
    /// (`log 1 = 0`, ρ-independent, the quotient pseudo-determinant convention),
    /// so the ridge-0 factor is PD-by-deflation and the evidence log-det finite;
    /// a full-rank block has no sub-floor eigenvalue and is untouched.
    ///
    /// Shared by the acceptance-site installer
    /// [`Self::ensure_row_gauge_deflation_for_evidence`] and by the two
    /// fixed-decoder assembler `.or_else` fallbacks in
    /// `construction_arrow_schur_assembly`, which keep their `low_rank_whiten`
    /// gate (this fn only mints the value they conditionally install).
    pub(crate) fn empty_row_gauge_deflation(n: usize) -> ArrowRowGaugeDeflation {
        ArrowRowGaugeDeflation::new(vec![Vec::new(); n])
    }

    /// Force an EVIDENCE/ACCEPTANCE system to opt into per-row spectral discovery
    /// by installing [`Self::empty_row_gauge_deflation`] when none is present
    /// (#1095/#2228): the frozen warm-start reuse and the two stationary /
    /// objective-stall diagnostic factorizations. Idempotent — an already-gauged
    /// system (rotation/phase gauge, #1273/#974 metric-null) is left untouched.
    ///
    /// CRITICAL INVARIANT: this MUST only ever run on a system that is about to
    /// be FACTORED for an accepted evidence log-det, never on the loop `sys` fed
    /// to `probe_undamped_evidence_row_factors` — the #2080 infeasible-ρ probe is
    /// contractually the UNDAMPED (non-deflated) per-row verdict (#2080/#2228).
    pub(crate) fn ensure_row_gauge_deflation_for_evidence(sys: &mut ArrowSchurSystem) {
        if sys.row_gauge_deflation.is_none() {
            let n_rows = sys.rows.len();
            sys.set_row_gauge_deflation(Self::empty_row_gauge_deflation(n_rows));
        }
    }

    /// The exact KKT stationarity residual `‖g‖² = Σ_i ‖g_t^(i)‖² + ‖g_β‖²` read
    /// straight off an assembled system. Unlike the Newton step `Δ = H⁻¹g`, the
    /// gradient is factorisation-independent — it is NOT amplified by an inverse,
    /// so a genuinely stationary but ill-conditioned fit (tiny `g`, possibly
    /// large `Δ` in a flat direction) is correctly recognised as converged.
    pub(crate) fn system_grad_norm_sq(sys: &ArrowSchurSystem) -> f64 {
        sys.rows
            .iter()
            .map(|row| row.gt.iter().map(|&v| v * v).sum::<f64>())
            .sum::<f64>()
            + sys.gb.iter().map(|&v| v * v).sum::<f64>()
    }

    /// The sole acceptance gate for a differentiable inner-envelope root.
    /// Objective stagnation, a finite deflated factor, or a small Newton
    /// decrement may diagnose conditioning but cannot substitute for raw or
    /// quotient KKT stationarity.
    pub(crate) fn evidence_kkt_stationary(
        grad_norm: f64,
        quotient_grad_norm: f64,
        tolerance: f64,
    ) -> bool {
        tolerance.is_finite()
            && tolerance >= 0.0
            && ((grad_norm.is_finite() && grad_norm <= tolerance)
                || (quotient_grad_norm.is_finite() && quotient_grad_norm <= tolerance))
    }

    /// Install the per-row spectral deflation on an ACCEPTANCE system, take its
    /// undamped (ridge-0) evidence factorization, and read back both KKT residual
    /// norms (raw and quotient) off the SAME assembled system. This is the
    /// objective-stall diagnostic factorization (#1095/#2228/#1094): the returned
    /// [`DeflatedEvidenceFactor`] carries the finite deflated cache plus the
    /// discarded Newton step retained for the affine Newton-decrement diagnostic
    /// (#2226). Only its KKT residual fields can authorize acceptance. A solve failure surfaces as `Err`,
    /// exactly the `if let Ok(..)` guard the caller uses to fall through to the
    /// persistent-stall counter.
    fn factor_deflated_evidence_with_grad_norms(
        &self,
        sys: &mut ArrowSchurSystem,
        lambda_smooth: &[f64],
        options: &ArrowSolveOptions,
    ) -> Result<DeflatedEvidenceFactor, String> {
        Self::ensure_row_gauge_deflation_for_evidence(sys);
        let (delta_t, delta_beta, cache) =
            solve_arrow_newton_step_with_options(sys, 0.0, 0.0, options)
                .map_err(|err| err.to_string())?;
        let grad_norm_sq = Self::system_grad_norm_sq(sys);
        let grad_norm = grad_norm_sq.sqrt();
        let quotient_grad_norm =
            self.quotient_gradient_norm_from_system(sys, grad_norm_sq, lambda_smooth);
        Ok(DeflatedEvidenceFactor {
            delta_t,
            delta_beta,
            cache,
            grad_norm,
            quotient_grad_norm,
        })
    }

    pub(crate) fn refine_iteration_limit(
        total_inner_iter: usize,
        base_refine_iter: usize,
        progress_refine_iter: usize,
        previous_grad_norm: Option<f64>,
        grad_norm: f64,
        saw_refine_progress: bool,
    ) -> usize {
        // Flat affine-gauge valleys can keep crawling productively after the
        // historical base budget. Extend only when the measured KKT residual has
        // shown a real finite round-to-round drop; true stalls end at the base
        // work budget (#968/#1029). Value-order probes pass the base budget as
        // their progress budget, so this branch cannot make probes expensive.
        //
        // #2230 COST-PROPORTIONAL EXTENSION: `saw_refine_progress` is the
        // LATEST-round verdict, not a sticky historical OR. The historical
        // `|=` accumulation meant ONE gradient drop anywhere granted the
        // 16×/64× extended budget for the rest of the evaluation — an
        // oscillating or stalled tail then ground the full extended budget on
        // every criterion eval (the #1094 "kept extending via
        // saw_refine_progress from earlier rounds" pathology, and the
        // dominant per-eval cost of the measured multi-hour outer churn).
        // Under the per-round contract each extension round must PAY for
        // itself with a monotone KKT-residual decrease; the first
        // non-decreasing round drops the limit back to the base budget and
        // the evaluation concludes (stall acceptance or typed refusal),
        // bounding every eval at base + the genuinely-descending tail.
        if total_inner_iter < base_refine_iter {
            return base_refine_iter;
        }
        let making_progress =
            saw_refine_progress && Self::refine_round_made_progress(previous_grad_norm, grad_norm);
        if making_progress && grad_norm.is_finite() {
            progress_refine_iter
        } else {
            base_refine_iter
        }
    }

    pub(crate) fn refine_round_made_progress(
        previous_grad_norm: Option<f64>,
        grad_norm: f64,
    ) -> bool {
        previous_grad_norm
            .is_some_and(|prev| prev.is_finite() && grad_norm.is_finite() && grad_norm < prev)
    }

    /// #2228 Stage-2 TERMINAL NEWTON PHASE — the superlinear tail the majorized
    /// Gauss–Newton inner loop is missing.
    ///
    /// The MM/GN inner solver is guaranteed descent but converges LINEARLY with
    /// contraction rate → 1 exactly where real data puts it: high residual (the
    /// GN data block drops first-order residual curvature) and huge near-flat
    /// bands (t-reparameterization, penalty-flat frame orientation). Measured
    /// stable-tail contraction on the production repro is 0.9965–0.9979 per
    /// iteration, i.e. ~1,800–3,000 uninterrupted iterations to close the gap
    /// from the objective-stall plateau (‖g‖ ≈ 1.4) to the KKT band — against a
    /// ~1e3 refine budget. The stall detector fires precisely when the MM phase
    /// has entered that crawl (gain ratio ≈ 1, relative decrease below the
    /// stall floor): from there, Newton on the EXACT Hessian is locally
    /// quadratic and closes the same gap in O(10) steps, making the strict KKT
    /// contract REACHABLE instead of loosened.
    ///
    /// Everything here reuses machinery that already exists for the outer IFT:
    /// the exact joint Hessian apply `A·v = B·v + ΔC·v`
    /// ([`Self::apply_exact_hessian`]) and its B-preconditioned, gauge-fixed
    /// quotient GMRES ([`Self::solve_exact_stationarity`], via the same
    /// [`Self::outer_gradient_arrow_solver`] quotient the ρ-gradient adjoint
    /// uses). Indefiniteness of `A` (measured K=1-circle μ = −1.66e-3) is
    /// handled by the acceptance test, not the solve: GMRES does not require
    /// SPD, and a step is committed ONLY when it strictly contracts the joint
    /// KKT residual while holding the penalized objective within the same
    /// no-meaningful-change band the stall detector just certified
    /// (`SAE_MANIFOLD_INNER_OBJECTIVE_STALL_REL_TOL × objective_scale`). A step
    /// failing that after backtracking restores the snapshot bit-for-bit, so
    /// this phase can never worsen the state the refine loop would otherwise
    /// have refused; every internal failure degrades to `Ok(false)` (fall
    /// through to the historical stall accounting), never to a new error class.
    ///
    /// Returns `Ok(true)` when at least one Newton step was committed (the
    /// caller re-enters the refine loop, whose existing raw/quotient KKT gate +
    /// idempotence certificate remain the SOLE acceptance authority — this
    /// phase mints nothing).
    fn terminal_exact_newton_polish(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho_fixed: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        lambda_smooth: &[f64],
        grad_tolerance: f64,
        objective_scale: f64,
        options: &ArrowSolveOptions,
        max_steps: usize,
    ) -> Result<bool, String> {
        let mut made_progress = false;
        let mut prev_grad_norm = f64::INFINITY;
        for _ in 0..max_steps {
            let mut sys = self
                .assemble_arrow_schur(target, rho_fixed, registry)
                .map_err(|err| format!("SaeManifoldTerm::terminal_exact_newton_polish: {err}"))?;
            let grad_norm_sq = Self::system_grad_norm_sq(&sys);
            if !grad_norm_sq.is_finite() {
                break;
            }
            let grad_norm = grad_norm_sq.sqrt();
            let quotient_grad_norm =
                self.quotient_gradient_norm_from_system(&sys, grad_norm_sq, lambda_smooth);
            if Self::evidence_kkt_stationary(grad_norm, quotient_grad_norm, grad_tolerance) {
                // In the band: hand back to the refine loop, whose gate +
                // idempotence certificate decide acceptance.
                return Ok(true);
            }
            if !(grad_norm < prev_grad_norm) {
                break;
            }
            prev_grad_norm = grad_norm;
            // Ridge-0 deflated evidence factor = the B-preconditioner for the
            // exact-pencil GMRES (identical to the outer IFT's preconditioner).
            let Ok(factor) =
                self.factor_deflated_evidence_with_grad_norms(&mut sys, lambda_smooth, options)
            else {
                break;
            };
            let cache = factor.cache;
            let Ok(solver) = self.outer_gradient_arrow_solver(&cache, lambda_smooth) else {
                break;
            };
            // Newton step on the exact Hessian: A Δ = −g on the gauge quotient.
            let mut rhs_t = Array1::<f64>::zeros(cache.delta_t_len());
            let mut offset = 0usize;
            for row in &sys.rows {
                for (axis, &g) in row.gt.iter().enumerate() {
                    rhs_t[offset + axis] = -g;
                }
                offset += row.gt.len();
            }
            let rhs = SaeArrowVector {
                t: rhs_t,
                beta: sys.gb.mapv(|v| -v),
            };
            let Ok(newton) = self.solve_exact_stationarity(rho_fixed, target, &cache, &solver, &rhs)
            else {
                break;
            };
            let pre_obj = self
                .penalized_objective_total(target, rho_fixed, registry, 1.0)
                .map_err(|err| format!("SaeManifoldTerm::terminal_exact_newton_polish: {err}"))?;
            let obj_rise_budget = SAE_MANIFOLD_INNER_OBJECTIVE_STALL_REL_TOL * objective_scale;
            let snapshot = self.snapshot_mutable_state();
            let mut accepted = false;
            let mut alpha = 1.0_f64;
            for _ in 0..8 {
                if self
                    .apply_newton_step(newton.t.view(), newton.beta.view(), alpha)
                    .is_err()
                {
                    self.restore_mutable_state(&snapshot)?;
                    break;
                }
                let trial_obj = self
                    .penalized_objective_total(target, rho_fixed, registry, 1.0)
                    .unwrap_or(f64::INFINITY);
                let trial_grad = self
                    .assemble_arrow_schur(target, rho_fixed, registry)
                    .ok()
                    .map(|trial_sys| Self::system_grad_norm_sq(&trial_sys).sqrt());
                // Gradient-norm merit (this is root-finding at a stalled
                // objective, so ‖g‖ is the monotone quantity) with the
                // objective held to the stall detector's own no-change band.
                let obj_ok = trial_obj.is_finite() && trial_obj <= pre_obj + obj_rise_budget;
                let grad_ok = trial_grad.is_some_and(|g| g.is_finite() && g < grad_norm);
                if obj_ok && grad_ok {
                    accepted = true;
                    made_progress = true;
                    break;
                }
                self.restore_mutable_state(&snapshot)?;
                alpha *= 0.5;
            }
            if !accepted {
                break;
            }
            log::debug!(
                "SAE terminal Newton step committed: ‖g‖ {grad_norm:.6e} → next round \
                 (α={alpha:.3e}, tol {grad_tolerance:.6e})"
            );
        }
        Ok(made_progress)
    }

    pub(crate) fn outer_gradient_arrow_solver<'a>(
        &'a self,
        cache: &'a ArrowFactorCache,
        penalized_gram_scale: &[f64],
    ) -> Result<DeflatedArrowSolver<'a>, OuterGradientError> {
        let Err(conditioning_err) = Self::outer_gradient_conditioning_error(cache) else {
            return Ok(DeflatedArrowSolver::plain(cache));
        };
        let Some(max_pivot) = arrow_factor_max_pivot(cache) else {
            return Err(conditioning_err);
        };
        if !(max_pivot.is_finite() && max_pivot > 0.0) {
            return Err(conditioning_err);
        }

        // The conditioning gate has already flagged a near-singular joint Hessian
        // (`conditioning_err`). Below we attempt to attribute that flatness to the
        // closed-form gauge orbit (chart step gauges) plus the penalty-aware
        // decoder-null directions and deflate it. When NO such deflatable
        // direction can be recovered, the flat subspace is genuinely
        // non-identifiable -- a degenerate direction OUTSIDE the gauge orbit -- a
        // diagnosis distinct from the raw pivot-ratio conditioning trip.
        // Surfacing the gauge-degenerate case as its own
        // [`OuterGradientError::NonIdentifiable`] preserves that typed evidence
        // when the derivative is refused.
        let non_identifiable_err = OuterGradientError::NonIdentifiable {
            reason: format!(
                "near-singular joint Hessian with no deflatable gauge/decoder-null \
                 direction (max pivot {max_pivot:.3e})"
            ),
        };

        let full_len = cache.delta_t_len() + cache.k;
        let mut raw_gauges = Vec::new();
        for gauge in self
            .dense_step_gauge_vectors()
            .map_err(OuterGradientError::internal)?
        {
            if gauge.len() != full_len {
                continue;
            }
            let norm_sq = gauge.iter().map(|v| v * v).sum::<f64>();
            if !(norm_sq.is_finite() && norm_sq > 1.0e-24) {
                continue;
            }
            raw_gauges.push(gauge);
        }
        // #2253: everything pushed above comes from `dense_step_gauge_vectors`
        // — the closed-form CHART gauge orbit (circle/torus phase, and the
        // translation/scale orbits of the linear/euclidean/duchon/poincaré
        // patches). These are EXACT criterion symmetries (global motion +
        // decoder compensation), flat by construction, unlike the empirical
        // decoder-null candidates admitted below (which the Rayleigh floor
        // exists to screen). Remember the boundary so the exact-gauge subspace
        // can be deflated UNCONDITIONALLY, keeping the deflation COUNT stable
        // across the ρ-walk.
        let n_exact_raw = raw_gauges.len();
        // #1051/#1273: admit the penalty-aware decoder-β null directions as
        // additional deflation candidates. A rank-deficient decoder design
        // (e.g. a euclidean-1D line in a p=2 ambient: decoder column rank 1 of
        // 3) puts a genuine near-null direction of the joint Hessian in the β
        // block, OUTSIDE the closed-form chart gauge orbit. #1273: probing the
        // RAW unit-β basis `e_j` produced an INCOMPLETE candidate set — the
        // true flat direction is the penalised null of `G_k + λ_smooth·S_k`,
        // not an axis-aligned coordinate, so the outer gate rejected trial ρ
        // with a pivot ratio (5.3e-16 < 1e-12) that the inner gate (which
        // already uses `joint_decoder_beta_null_directions(λ_smooth)`) accepts. Use
        // the SAME penalty-aware null directions here, evaluated at the smooth
        // scale the Schur factor used, so the outer and inner gates agree.
        // These full (n·q + beta_dim)-length vectors drop into the same
        // Gram-Schmidt + Rayleigh + Faddeev-Popov path below; the Rayleigh
        // floor still keeps only genuinely flat (sub-floor) directions, so a
        // well-conditioned decoder is unaffected.
        for dir in self
            .joint_decoder_beta_null_directions(penalized_gram_scale)
            .map_err(OuterGradientError::internal)?
        {
            if dir.len() == full_len {
                raw_gauges.push(dir);
            }
        }
        // #1051/#1273: also admit the decoder COLUMN-SPAN null (an unrealised
        // ambient output channel of a rank-deficient decoder), which the
        // channel-free basis-null above structurally cannot represent. The
        // rank-1-decoder-line geometry (e.g. a 1-D euclidean line in p=2
        // ambient: decoder column rank 1 of 2) puts the joint Hessian's
        // sub-floor pivot entirely in one output channel; without this
        // candidate the outer gate had nothing to deflate it with and rejected
        // the trial ρ. The Rayleigh floor below still prunes any candidate that
        // is not genuinely flat against the cached Hessian.
        for dir in self
            .decoder_channel_null_directions()
            .map_err(OuterGradientError::internal)?
        {
            if dir.len() == full_len {
                raw_gauges.push(dir);
            }
        }
        if raw_gauges.is_empty() {
            return Err(non_identifiable_err);
        }

        let mut gauge_span: Vec<Array1<f64>> = Vec::new();
        // Exact chart gauges (raw indices `< n_exact_raw`) are processed first,
        // so their Gram-Schmidt survivors occupy the FRONT of `gauge_span`;
        // `exact_basis_count` records that contiguous prefix.
        let mut exact_basis_count = 0usize;
        for (raw_idx, mut gauge) in raw_gauges.into_iter().enumerate() {
            for basis in &gauge_span {
                let coeff = gauge.dot(basis);
                for i in 0..gauge.len() {
                    gauge[i] -= coeff * basis[i];
                }
            }
            let norm_sq = gauge.iter().map(|v| v * v).sum::<f64>();
            if !(norm_sq.is_finite() && norm_sq > 1.0e-24) {
                continue;
            }
            let inv_norm = norm_sq.sqrt().recip();
            for value in gauge.iter_mut() {
                *value *= inv_norm;
            }
            if raw_idx < n_exact_raw {
                exact_basis_count += 1;
            }
            gauge_span.push(gauge);
        }
        if gauge_span.is_empty() {
            return Err(non_identifiable_err);
        }

        let span_rank = gauge_span.len();
        let mut h_span = Array2::<f64>::zeros((span_rank, span_rank));
        for col in 0..span_rank {
            let h_gauge = match apply_cached_arrow_hessian(
                cache,
                gauge_span[col].slice(s![..cache.delta_t_len()]),
                gauge_span[col].slice(s![cache.delta_t_len()..]),
            ) {
                Ok(value) => value,
                // #1451: a shape/dimension mismatch or non-finite intermediate
                // from the Hessian apply is an internal-invariant defect and MUST
                // propagate; a genuine numeric failure on a finite,
                // correctly-shaped input keeps the typed conditioning class.
                Err(err) => {
                    return Err(OuterGradientError::classify_arrow_solver_error(
                        &err,
                        conditioning_err.clone(),
                    ));
                }
            };
            let h_flat = flatten_arrow_parts(h_gauge.t.view(), h_gauge.beta.view());
            for row in 0..span_rank {
                h_span[[row, col]] = gauge_span[row].dot(&h_flat);
            }
        }
        for row in 0..span_rank {
            for col in 0..row {
                let sym = 0.5 * (h_span[[row, col]] + h_span[[col, row]]);
                h_span[[row, col]] = sym;
                h_span[[col, row]] = sym;
            }
        }
        // #1451: a non-finite entry in the projected gauge Hessian is an
        // internal-invariant defect (a NaN/Inf intermediate leaked into the
        // span), not a conditioning failure — it MUST propagate rather than be
        // masked behind a degraded descent. Guard finiteness BEFORE the eigh so a
        // genuine decomposition failure on a finite, correctly-shaped matrix keeps
        // the typed conditioning class.
        if !h_span.iter().all(|v| v.is_finite()) {
            return Err(OuterGradientError::internal(format!(
                "outer_gradient_arrow_solver: non-finite entry in projected gauge \
                 Hessian (h_span is {span_rank}x{span_rank})"
            )));
        }
        let (evals, evecs) = h_span
            .eigh(Side::Lower)
            .map_err(|_| conditioning_err.clone())?;
        let strict_gauge_floor = SAE_OUTER_GRADIENT_GAUGE_RAYLEIGH_FACTOR * max_pivot;
        let mut orthonormal: Vec<Array1<f64>> = Vec::new();
        for eig_idx in 0..evals.len() {
            let rayleigh = evals[eig_idx];
            if !(rayleigh.is_finite() && rayleigh <= strict_gauge_floor) {
                continue;
            }
            let mut direction = Array1::<f64>::zeros(full_len);
            for basis_idx in 0..span_rank {
                let coeff = evecs[[basis_idx, eig_idx]];
                for row in 0..full_len {
                    direction[row] += coeff * gauge_span[basis_idx][row];
                }
            }
            let norm_sq = direction.iter().map(|v| v * v).sum::<f64>();
            if !(norm_sq.is_finite() && norm_sq > 1.0e-24) {
                continue;
            }
            let inv_norm = norm_sq.sqrt().recip();
            for value in direction.iter_mut() {
                *value *= inv_norm;
            }
            orthonormal.push(direction);
        }
        // #2253: deflate the EXACT chart-gauge subspace unconditionally. A
        // borderline gauge eigenvalue can flicker across `strict_gauge_floor`
        // as ρ moves; for the empirical decoder-null candidates that screen is
        // the point, but for the exact chart gauges (circle/torus phase orbit,
        // patch translation/scale) it changes the deflation COUNT by ±1 and
        // re-anchors ½log|H|, desyncing the fixed-ρ criterion gradient from the
        // value (the K=1 circle non-stationary stall). The exact-gauge subspace
        // is `gauge_span[0..exact_basis_count]` (flat by construction); add any
        // of its directions the floor loop dropped, orthogonalized against what
        // was already kept, so the deflation dimension is ρ-stable. When the
        // floor already kept a gauge, its residual here is ~0 and it is not
        // double-counted.
        for exact_idx in 0..exact_basis_count {
            let mut direction = gauge_span[exact_idx].clone();
            for kept in &orthonormal {
                let coeff = direction.dot(kept);
                for row in 0..direction.len() {
                    direction[row] -= coeff * kept[row];
                }
            }
            let norm_sq = direction.iter().map(|v| v * v).sum::<f64>();
            if !(norm_sq.is_finite() && norm_sq > 1.0e-24) {
                continue;
            }
            let inv_norm = norm_sq.sqrt().recip();
            for value in direction.iter_mut() {
                *value *= inv_norm;
            }
            orthonormal.push(direction);
        }
        if orthonormal.is_empty() {
            // The joint factor is ill-conditioned, but no direction in the
            // analytically known gauge/decoder-null span is actually flat at the
            // rank-revealing Rayleigh threshold. The unreliable direction lies
            // outside the quotient we can justify, so refuse the derivative
            // instead of projecting an arbitrary least-curvature candidate.
            return Err(non_identifiable_err);
        }

        // Quotient-geometry gauge fixing: add stiffness only along the closed-form
        // gauge orbit (Faddeev-Popov style). Components orthogonal to that orbit
        // are identical to the original inverse solve, while gauge components are
        // bounded at the Hessian scale `max_pivot`.
        // #1451: a shape/length mismatch or non-finite stiffness/intermediate in
        // the deflated-solver assembly is an internal-invariant defect and MUST
        // propagate; a genuine near-singular gauge Woodbury/back-solve keeps the
        // typed conditioning class.
        DeflatedArrowSolver::from_orthonormal_gauges(cache, orthonormal, max_pivot)
            .map_err(|err| OuterGradientError::classify_arrow_solver_error(&err, conditioning_err))
    }

    pub(crate) fn outer_gradient_conditioning_error(
        cache: &ArrowFactorCache,
    ) -> Result<(), OuterGradientError> {
        let pivot = arrow_factor_min_pivot(cache);
        let Some(min_pivot) = pivot.min_pivot else {
            return Err(OuterGradientError::IllConditioned {
                reason: "joint Hessian numerically singular (no cached Cholesky pivots)"
                    .to_string(),
            });
        };
        let Some(max_pivot) = arrow_factor_max_pivot(cache) else {
            return Err(OuterGradientError::IllConditioned {
                reason: "joint Hessian numerically singular (no cached Cholesky pivot scale)"
                    .to_string(),
            });
        };
        let ratio = min_pivot / max_pivot;
        if min_pivot.is_finite()
            && max_pivot.is_finite()
            && max_pivot > 0.0
            && ratio.is_finite()
            && ratio >= SAE_OUTER_GRADIENT_PIVOT_RATIO_FLOOR
        {
            return Ok(());
        }
        Err(OuterGradientError::IllConditioned {
            reason: format!(
                "joint Hessian numerically singular (min/max pivot ratio {ratio:.3e} < floor {floor:.3e}; min pivot {min_pivot:.3e}, max pivot {max_pivot:.3e})",
                floor = SAE_OUTER_GRADIENT_PIVOT_RATIO_FLOOR,
            ),
        })
    }

    /// Smoothing-penalty Occam normalizer `−½ Σ_k r_k·rank(S_k)·log λ_smooth`
    /// (issue #972; #1556 per-atom λ).
    ///
    /// This is the `log λ`-dependent part of the penalty log-determinant
    /// `−½ log|λ_k S_k|_+` summed over the `r_k` penalized decoder channels: the
    /// `S_k` roughness penalty acts on `r_k` coordinate channels (`r_k == p` on
    /// the full-`B` path, the smaller frame rank when a Grassmann frame is
    /// active), each contributing `rank(S_k)` penalized directions, so the
    /// `λ_k`-normalizer is `½ r_k·rank(S_k)·log λ_k`.
    ///
    /// The profiled frame ORIENTATION `U_k` is NOT penalized by `λ_k` — the
    /// isotropic `⊗ I_{r_k}` penalty is invariant to rotating the frame, so the
    /// `r_k(p−r_k)` Grassmann directions are flat directions of the penalty and
    /// their Laplace curvature comes from the DATA fit, carrying NO `log λ_k`
    /// dependence. The historical `−½ r_k(p−r_k)·log λ_k` "frame evidence
    /// dimension" term therefore attached a `log λ_k` factor to a
    /// λ-INDEPENDENT geometric dimension (e.g. `p=896, r=1, rank S=1`:
    /// `0.5·(1−895)=−447`, i.e. `+447·log λ` pushed into the smoothing selection
    /// from an unpenalized orientation) and is dropped. On the full-`B` path
    /// `r_k == p` so `frame_dim = r_k(p−r_k) = 0` and this is bit-for-bit
    /// unchanged; only frame-active fits change, toward the correct normalizer.
    /// A genuine frame-orientation evidence correction, if wanted, is a SEPARATE
    /// (λ-independent) Laplace term built from the actual frame Hessian.
    pub(crate) fn reml_occam_term(&self, rho: &SaeManifoldRho) -> Result<f64, String> {
        let mut acc = 0.0_f64;
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let rank_s = Self::symmetric_rank(&atom.smooth_penalty)?;
            // Penalized decoder dimension: `r_k` coordinate channels carry the
            // `S_k` roughness penalty (full-`B` path ⇒ `r_k == p`).
            let penalized_channel_dim = atom.border_frame_rank() * rank_s;
            // The SAME clamped log-strength the penalty/Hessian channels
            // exponentiate: past the ±700 band λ_eff is constant, so the
            // normalizer must be too — a raw coordinate would let the criterion
            // drift linearly while the model it scores is frozen. (The outer
            // walk is bounded at |ρ| ≤ 30, so the band is unreachable in
            // production; this keeps the two conventions identical anyway.)
            let log_lambda = SaeManifoldRho::clamped_log_strength(rho.log_lambda_smooth[atom_idx]);
            acc += 0.5 * (penalized_channel_dim as f64) * log_lambda;
        }
        // `V = … − occam`, so the net occam SUBTRACTS the penalty normalizer.
        Ok(acc)
    }

    /// Per-atom derivative `∂(occam)/∂log λ_smooth[k]` (#1556): atom `k`'s entry
    /// is `½·r_k·rank(S_k)` inside the ±700 clamp band and `0` outside it
    /// (where [`Self::reml_occam_term`] reads the clamped, constant
    /// log-strength), matching the per-atom Occam term exactly. The
    /// unpenalized-frame `frame_dim` term carries no `log λ` dependence and is
    /// absent from both. Returns one entry per atom in atom order.
    pub(crate) fn reml_occam_log_lambda_smooth_derivative(
        &self,
        rho: &SaeManifoldRho,
    ) -> Result<Vec<f64>, String> {
        let mut out = Vec::with_capacity(self.atoms.len());
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let rank_s = Self::symmetric_rank(&atom.smooth_penalty)?;
            let penalized_channel_dim = atom.border_frame_rank() * rank_s;
            let raw = rho.log_lambda_smooth[atom_idx];
            let inside_band = SaeManifoldRho::clamped_log_strength(raw) == raw;
            out.push(if inside_band {
                0.5 * (penalized_channel_dim as f64)
            } else {
                0.0
            });
        }
        Ok(out)
    }

    /// Streaming criterion that RETURNS the converged arrow-factor cache — the
    /// per-row factored Hessian (matrix-free, feasible at massive K; the dense
    /// `border_dim²` Schur is NEVER formed here), so the EFS hyperparameter lane
    /// can take its matrix-free ARD / smoothness traces off this cache in the
    /// streaming regime instead of hard-erroring on the dense evidence path. The
    /// log-determinant is the chunked matrix-free `streaming_exact_arrow_log_det`.
    /// Convenience over [`Self::reml_criterion_streaming_exact_with_cache_and_lane`]
    /// with no #2080 surrogate lane (bit-identical SLQ evidence).
    pub fn reml_criterion_streaming_exact_with_cache(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
    ) -> Result<(f64, SaeManifoldLoss, ArrowFactorCache), String> {
        self.reml_criterion_streaming_exact_with_cache_and_lane(
            target,
            rho,
            registry,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
            None,
        )
    }

    /// [`Self::reml_criterion_streaming_exact_with_cache`] with the #2080 surrogate
    /// lane threaded to the streaming `log|S|` term (`None` = bit-identical SLQ).
    pub fn reml_criterion_streaming_exact_with_cache_and_lane(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
        lane: Option<&mut SurrogateLaneState>,
    ) -> Result<(f64, SaeManifoldLoss, ArrowFactorCache), String> {
        let mut rho_fixed = rho.clone();
        let initial_fit = self.run_joint_fit_arrow_schur_for_evidence(
            target,
            &mut rho_fixed,
            registry,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
        )?;
        let mut loss = initial_fit.loss;
        let mut evidence_fixed_point = initial_fit.fixed_point;
        // Drive the inner (t, β) state to the SAME KKT/step-converged optimum the
        // dense `reml_criterion_with_cache` reaches before factoring. At that
        // optimum the per-row `H_tt^(i)` blocks are PD, so the undamped
        // (`ridge_t = 0`) streaming factorization in `streaming_exact_arrow_log_det`
        // succeeds — without this, a state stopped after only `inner_max_iter`
        // steps can leave a rank-deficient / indefinite row block (`p_out = 1` →
        // rank-1 `JᵀJ`, softmax negative-logit curvature) that surfaces
        // `PerRowFactorFailed` at base ridge 0. Sharing the driver also keeps the
        // streaming and dense log-determinants bit-identical (#847).
        let options = ArrowSolveOptions::direct()
            .with_ill_conditioning_tolerated()
            .with_schur_pd_floor(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR);
        // The converged arrow-factor cache is the per-row factored Hessian
        // (matrix-free, feasible at massive K — the dense border_dim² Schur is
        // never materialised here); it is RETURNED so the EFS lane can take its
        // matrix-free ARD/smoothness traces off it. The log-determinant itself is
        // recomputed chunk-by-chunk in `streaming_exact_arrow_log_det` to bound
        // peak memory (bit-identical to the dense path, #847).
        let mut converged_cache = self.converge_inner_for_undamped_logdet(
            target,
            rho,
            &mut rho_fixed,
            registry,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
            &mut loss,
            &mut evidence_fixed_point,
            &options,
            true,
        )?;
        // #9: accumulate the per-atom Grams + N_eff + log_det_tt in the same
        // log-det pass. These are required by the canonical rank-charge criterion.
        let mut rank_inputs = StreamingRankInputs::default();
        let log_det = self.streaming_exact_arrow_log_det_with_lane(
            target,
            rho,
            registry,
            Some(&mut rank_inputs),
            lane,
        )?;
        // The returned row-factor cache and the external matrix-free log|S|
        // estimate are one evidence operator. Stamp the authoritative joint
        // value onto the cache so from-probes theta-adjoint consumers can verify
        // that their selected-inverse bundle differentiates a live log-det,
        // exactly as dense caches do through their Schur-factor path.
        converged_cache.joint_hessian_log_det = Some(log_det);
        converged_cache.schur_factor_is_undamped = true;
        let occam = self.reml_occam_term(rho)?;
        // Extra penalized-objective energy (#671/#737 + full-objective
        // completion: registry penalties + repulsion + separation barrier),
        // matching the full-batch `reml_criterion_with_cache` path so streaming
        // and dense criteria rank the identical penalized objective.
        let extra_penalty_energy = self
            .reml_extra_penalty_value_total(registry)
            .map_err(|err| format!("SaeManifoldTerm::reml_criterion_streaming_exact: {err}"))?;
        let v = {
            let ri = rank_inputs;
            // #9/#5 streaming rank charge: replace the coordinate-block ½log|H_tt|
            // (= log_det_tt/2, exposed by the log-det pass) with Σ ½·d_eff·log n on
            // each atom's realised decoder rank, priced through the SAME
            // `rank_dof_from_grams` MP hard count as the dense path off the
            // chunk-accumulated Grams. The β/Schur block (the ‖B‖-independent part
            // of log_det) is untouched — bit-identical dense↔streaming by design.
            let residual = self.reconstruction_residual(target, rho)?;
            let disp = self
                .reconstruction_dispersion(
                    &loss,
                    &converged_cache,
                    rho,
                    Some(residual.view()),
                )
                .map_err(|e| {
                    format!(
                        "SaeManifoldTerm::reml_criterion_streaming_exact: rank-charge dispersion is required: {e}"
                    )
                })?;
            let d_eff = self.rank_dof_from_grams(&ri.grams, &ri.n_eff, rho, disp)?;
            // #5 VETO (streaming): categorical Laplace-validity condition — a
            // rank_eff==0 (d_eff==0) atom reconstructs nothing, so its evidence is
            // INVALID (degenerate β-mode / β-Schur log-det → −∞), not payable. Reject
            // categorically (v → +∞). Same guard as the dense path; see the dense
            // reml_criterion for the full rationale + β-Schur-floor trailhead.
            let laplace_complexity =
                rank_adjusted_laplace_complexity(log_det, ri.log_det_tt, &d_eff, &ri.n_eff)?;
            loss.total() + extra_penalty_energy + laplace_complexity - occam
        };
        Ok((v, loss, converged_cache))
    }

    /// Value-only streaming criterion — the cache-returning
    /// [`Self::reml_criterion_streaming_exact_with_cache`] with the cache dropped.
    pub fn reml_criterion_streaming_exact(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
    ) -> Result<(f64, SaeManifoldLoss), String> {
        self.reml_criterion_streaming_exact_with_lane(
            target,
            rho,
            registry,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
            None,
        )
    }

    /// [`Self::reml_criterion_streaming_exact`] with the #2080 surrogate lane
    /// threaded to the streaming `log|S|` term (`None` = bit-identical SLQ).
    pub fn reml_criterion_streaming_exact_with_lane(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
        lane: Option<&mut SurrogateLaneState>,
    ) -> Result<(f64, SaeManifoldLoss), String> {
        let (cost, loss, _cache) = self.reml_criterion_streaming_exact_with_cache_and_lane(
            target,
            rho,
            registry,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
            lane,
        )?;
        Ok((cost, loss))
    }

    /// Value-only streaming reduced-Schur evidence log-det via the historical SLQ
    /// lane — convenience over [`Self::streaming_exact_arrow_log_det_with_lane`]
    /// with `lane = None` (bit-identical to the pre-#2080 SLQ path).
    pub fn streaming_exact_arrow_log_det(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        rank_inputs: Option<&mut StreamingRankInputs>,
    ) -> Result<f64, String> {
        self.streaming_exact_arrow_log_det_with_lane(target, rho, registry, rank_inputs, None)
    }

    /// Assemble the one whole-row matrix-free evidence system at the current
    /// fitted state. The dense reduced Schur is never formed: the returned
    /// system retains only the structured shared-block and row-cross operators.
    ///
    /// This single source of truth is consumed both by the rational
    /// log-determinant and by #2230's exact-stationarity IFT solve, ensuring the
    /// value and assignment-strength residual cannot reassemble different
    /// operators. Optional rank inputs are accumulated from the same full chunk.
    pub(crate) fn assemble_full_matrix_free_evidence_system(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        mut rank_inputs: Option<&mut StreamingRankInputs>,
    ) -> Result<ArrowSchurSystem, String> {
        let n_total = self.n_obs();
        let full_logits = self.assignment.logits.slice(s![0..n_total, ..]).to_owned();
        let full_coords: Vec<Array2<f64>> = self
            .assignment
            .coords
            .iter()
            .map(|coord| coord.as_matrix().slice(s![0..n_total, ..]).to_owned())
            .collect();
        let mut full_chunk = self.materialize_chunk(
            full_logits,
            full_coords,
            self.chunk_frozen_logits(0, n_total),
        )?;
        if let Some(weights) = self.row_loss_weights.as_deref() {
            full_chunk.row_loss_weights = Some(weights[0..n_total].to_vec());
        }
        if let Some(inputs) = rank_inputs.as_deref_mut() {
            full_chunk.accumulate_decoder_gram(&mut inputs.grams);
            let assignments = full_chunk.assignment.assignments();
            for atom in 0..inputs.n_eff.len() {
                let support = SupportMeasure::from_assignment_matrix(assignments.view(), atom)
                    .expect("streaming full-rank chunk assignment shape must match atoms");
                inputs.n_eff[atom] += support.fisher_n();
            }
        }
        let mut system = full_chunk
            .assemble_arrow_schur_scaled(target, rho, registry, 1.0)
            .map_err(|error| {
                format!("SaeManifoldTerm::streaming_exact_arrow_log_det: {error}")
            })?;
        // The exact-stationarity inverse consumes this system with the factor
        // cache emitted from it. Persist the completed row/registry fingerprint
        // now so the stale-pair guard compares two identities from the same
        // assembled operator instead of the constructor sentinel `0`.
        system.refresh_row_hessian_fingerprint();
        Ok(system)
    }

    /// Streaming reduced-Schur evidence `log|H| = Σ log|H_tt| + log|S|` with the
    /// #2080 surrogate lane threaded to the `log|S|` term. `lane = None` runs the
    /// bit-identical SLQ path; `lane = Some(state)` builds-or-reuses the frozen
    /// derived-rank rational surrogate (matrix-free, desync-safe) instead.
    pub fn streaming_exact_arrow_log_det_with_lane(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        mut rank_inputs: Option<&mut StreamingRankInputs>,
        lane: Option<&mut SurrogateLaneState>,
    ) -> Result<f64, String> {
        if target.dim() != (self.n_obs(), self.output_dim()) {
            return Err(format!(
                "SaeManifoldTerm::streaming_exact_arrow_log_det: target must be ({}, {}); got {:?}",
                self.n_obs(),
                self.output_dim(),
                target.dim()
            ));
        }
        // #9: when the rank charge is on, accumulate the per-atom Grams + effective
        // sample sizes chunk-additively alongside the log-det (single pass), and
        // hand back the coordinate-block `log_det_tt` (= 2·htt_half). Zero cost /
        // untouched when `None`.
        if let Some(ri) = rank_inputs.as_deref_mut() {
            ri.grams = self.empty_decoder_gram_accumulator();
            ri.n_eff = vec![0.0; self.k_atoms()];
            ri.log_det_tt = 0.0;
        }
        let plan = self.streaming_plan().admitted_or_error(
            self.n_obs(),
            self.output_dim(),
            self.k_atoms(),
        )?;
        if plan.estimated_dense_schur_bytes > plan.in_core_budget_bytes {
            // #988 memory-matrix-free evidence route. The dense k×k reduced Schur
            // (≈8 GB at the K=32k manifold border) does NOT fit the in-core
            // budget, so estimate log|S| via Stochastic Lanczos Quadrature on the
            // matrix-free `schur_matvec` apply (`gam_solve::arrow_schur::
            // matrix_free_arrow_evidence_log_det`) instead of assembling +
            // Cholesky-factoring the dense Schur. Peak memory is the per-row block
            // storage the inner PCG already holds, not the extra O(k²) dense S.
            //
            // Valid for the NON-IBP (softmax / JumpReLU) evidence, whose exact
            // log-det is `log_det_tt + log_det_schur` with NO cross-row Woodbury
            // correction. The IBP cross-row term additionally needs
            // `log det(I_R + D Uᵀ H₀'⁻¹ U)`, which has no matrix-free route yet, so
            // it keeps refusing (loudly, pointing at the dense resident path).
            if ibp_assignment_third_channels_weighted(
                &self.assignment,
                rho,
                false,
                self.row_loss_weights.as_deref(),
            )?
            .is_some()
            {
                return Err(format!(
                    "SaeManifoldTerm::streaming_exact_arrow_log_det: predicted dense reduced Schur \
                     {} bytes exceeds budget {} bytes and the exact cross-row IBP Woodbury evidence \
                     has no matrix-free log-det route yet; route IBP-active large-K fits through the \
                     dense resident ArrowFactorCache::arrow_log_det",
                    plan.estimated_dense_schur_bytes, plan.in_core_budget_bytes
                ));
            }
            let options = ArrowSolveOptions::direct()
                .with_ill_conditioning_tolerated()
                .with_schur_pd_floor(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR);
            // Assemble the WHOLE system once (a single "chunk" over all rows) so the
            // matrix-free reduced-Schur apply `v ↦ S·v` can iterate every row; the
            // per-row block storage is exactly what the inner solve already holds.
            let sys = self.assemble_full_matrix_free_evidence_system(
                target,
                rho,
                registry,
                rank_inputs.as_deref_mut(),
            )?;
            // #2080: the reduced-Schur `log|S|` term. `lane = None` runs the
            // bit-identical SLQ estimate; `lane = Some(state)` swaps in the frozen
            // derived-rank rational surrogate (matrix-free, value+ρ-gradient one
            // functional). `log_det_tt` (the Σ log|H_tt| coordinate block) is exact
            // on the shared factorization either way.
            let (log_det_tt, log_det_schur) = matrix_free_arrow_evidence_log_det_surrogate(
                &sys,
                0.0,
                0.0,
                &options,
                SCHUR_SLQ_LOGDET_PROBES,
                SCHUR_SLQ_LOGDET_LANCZOS_STEPS,
                SCHUR_SLQ_LOGDET_SEED,
                lane,
            )
            .map_err(|err| {
                format!(
                    "SaeManifoldTerm::streaming_exact_arrow_log_det: matrix-free evidence log-det: {err:?}"
                )
            })?;
            if !log_det_schur.is_finite() {
                return Err(format!(
                    "SaeManifoldTerm::streaming_exact_arrow_log_det: matrix-free reduced-Schur \
                     log|S| non-finite ({log_det_schur})"
                ));
            }
            if let Some(ri) = rank_inputs.as_deref_mut() {
                ri.log_det_tt = log_det_tt;
            }
            return Ok(log_det_tt + log_det_schur);
        }
        let n_total = self.n_obs();
        let chunk_size = plan.chunk_size.min(n_total.max(1));
        // #972 / #977 T1: the reduced β-Schur is over the FACTORED border when
        // frames are active (each chunk inherits the frames via
        // `materialize_chunk`, so every `chunk_schur` is `border_dim²`), matching
        // the dense path's factored log-det. Full-`B` ⇒ `border_dim == beta_dim`.
        let border_dim = if self.frames_active() {
            self.factored_border_dim()
        } else {
            self.beta_dim()
        };
        let mut schur_acc = Array2::<f64>::zeros((border_dim, border_dim));
        let mut log_det_tt = 0.0_f64;
        // #1038 cross-row IBP Woodbury accumulators. `M = Uᵀ H₀'⁻¹ U` is
        // chunk-additive in `M0 = Σ Uᵢᵀ Aᵢ⁻¹ Uᵢ` and `W = Σ Bᵢᵀ Aᵢ⁻¹ Uᵢ`
        // (`A = H₀'` block-diagonal, `U` row-supported), closed against the
        // GLOBAL reduced Schur `S = schur_acc` after the loop. `None` for every
        // non-IBP (softmax / JumpReLU) term, where the streaming log-det is
        // exactly the bare `log_det_tt + log_det_schur` as before.
        let mut wood_m0: Option<Array2<f64>> = None;
        let mut wood_w: Option<Array2<f64>> = None;
        let mut wood_d: Option<Array1<f64>> = None;
        let options = ArrowSolveOptions::direct()
            .with_ill_conditioning_tolerated()
            .with_schur_pd_floor(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR);
        let mut start = 0usize;
        while start < n_total {
            let end = (start + chunk_size).min(n_total);
            let penalty_scale = (end - start) as f64 / n_total as f64;
            let chunk_logits = self.assignment.logits.slice(s![start..end, ..]).to_owned();
            let chunk_coords: Vec<Array2<f64>> = self
                .assignment
                .coords
                .iter()
                .map(|coord| coord.as_matrix().slice(s![start..end, ..]).to_owned())
                .collect();
            let mut chunk = self.materialize_chunk(
                chunk_logits,
                chunk_coords,
                self.chunk_frozen_logits(start, end),
            )?;
            // #1117 — rank deficiency is removed at the basis layer at fit entry
            // (`reduce_atoms_to_data_supported_rank`), so each chunk inherits the
            // already-reduced full-rank atoms via `materialize_chunk`; there are
            // no global deflation projectors to propagate.
            // #991: chunk terms inherit the row's design honesty weight slice
            // (global mean-1 normalization preserved — NOT re-normalized per
            // chunk — so the per-chunk sums reconstruct the global weighted
            // objective exactly).
            if let Some(w) = self.row_loss_weights.as_deref() {
                chunk.row_loss_weights = Some(w[start..end].to_vec());
            }
            if let Some(ri) = rank_inputs.as_deref_mut() {
                chunk.accumulate_decoder_gram(&mut ri.grams);
                let asg = chunk.assignment.assignments();
                for k in 0..ri.n_eff.len() {
                    let support = SupportMeasure::from_assignment_matrix(asg.view(), k)
                        .expect("streaming chunk assignment shape must match atoms");
                    ri.n_eff[k] += support.fisher_n();
                }
            }
            let z_chunk = target.slice(s![start..end, ..]);
            let sys = chunk
                .assemble_arrow_schur_scaled(z_chunk, rho, registry, penalty_scale)
                .map_err(|err| format!("SaeManifoldTerm::streaming_exact_arrow_log_det: {err}"))?;
            let mut streaming = StreamingArrowSchur::from_system(&sys, sys.rows.len().max(1));
            let (chunk_log_det_tt, chunk_schur, chunk_wood) = streaming
                .reduced_schur_log_det_tt_woodbury(0.0, 0.0, &options)
                .map_err(|err| format!("SaeManifoldTerm::streaming_exact_arrow_log_det: {err}"))?;
            log_det_tt += chunk_log_det_tt;
            for row in 0..border_dim {
                for col in 0..border_dim {
                    schur_acc[[row, col]] += chunk_schur[[row, col]];
                }
            }
            if chunk_wood.is_some() && chunk_size < n_total {
                // The cross-row IBP empirical mass `M_k = Σ_i z_ik` couples ALL
                // rows, so the per-row `H₀'` diagonal (`score_derivative_k(M_k)`)
                // and the column coefficient `d_k = w·s'_k(M_k)` are only exact
                // when every row is assembled together — a SINGLE chunk. Under a
                // genuine multi-chunk pass each chunk would see a partial mass and
                // the Woodbury (and the bare per-row log-det) would be inexact, so
                // refuse loudly and route to the dense resident path rather than
                // return a silently-wrong evidence. The streaming log-det only
                // runs when the dense reduced Schur fits budget, so the single-
                // chunk regime is the common case; this guards the rest.
                return Err(
                    "SaeManifoldTerm::streaming_exact_arrow_log_det: exact cross-row IBP \
                     Woodbury evidence requires a single-chunk pass (the empirical mass \
                     M_k = Σ_i z_ik couples all rows); this shape needs >1 chunk. Route \
                     IBP-active large-n fits through the dense resident \
                     ArrowFactorCache::arrow_log_det."
                        .to_string(),
                );
            }
            if let Some(cw) = chunk_wood {
                wood_m0 = Some(match wood_m0.take() {
                    Some(mut acc) => {
                        acc += &cw.m0;
                        acc
                    }
                    None => cw.m0,
                });
                wood_w = Some(match wood_w.take() {
                    Some(mut acc) => {
                        acc += &cw.w;
                        acc
                    }
                    None => cw.w,
                });
                // `D = diag(d_k)` is per-atom; identical across chunks for a
                // single-chunk evidence pass (the regime the streaming log-det
                // runs in — the dense reduced Schur must fit budget here), where
                // it equals the global mass-derived `cross_row_d`.
                wood_d = Some(cw.d);
            }
            start = end;
        }
        let log_det_schur = StreamingArrowSchur::reduced_schur_log_det(&schur_acc, &options)
            .map_err(|err| format!("SaeManifoldTerm::streaming_exact_arrow_log_det: {err}"))?;
        let mut total = log_det_tt + log_det_schur;
        // #1038/#1225: close the exact cross-row IBP Woodbury correction
        // `log det(I_R + D Uᵀ H₀'⁻¹ U)` so the streaming evidence equals the
        // dense `arrow_log_det_from_cache` (which adds the SAME term). Without
        // it the streaming criterion would silently drop the entire cross-row
        // coupling and disagree with the dense path by exactly `log|C|`.
        if let (Some(m0), Some(w), Some(d)) = (wood_m0, wood_w, wood_d) {
            let correction = streaming_cross_row_woodbury_log_det(
                &schur_acc,
                &m0,
                &w,
                &d,
                options.schur_pd_floor,
            )
            .map_err(|err| format!("SaeManifoldTerm::streaming_exact_arrow_log_det: {err}"))?
            .ok_or_else(|| {
                "SaeManifoldTerm::reml_criterion: cross-row IBP joint Hessian is non-PD at \
                     this ρ; evidence Laplace log-det undefined (infeasible ρ probe)"
                    .to_string()
            })?;
            total += correction;
        }
        if let Some(ri) = rank_inputs.as_deref_mut() {
            ri.log_det_tt = log_det_tt;
        }
        Ok(total)
    }

    /// Per-atom decoder-smoothness penalty quadratic form (#1556): entry `k` is
    /// the λ-free `<B_k, ½(S_k+S_kᵀ)·B_k> = Σ_oc B_k[:,oc]ᵀ S_k B_k[:,oc]`, the
    /// per-atom denominator of atom `k`'s λ_smooth Fellner-Schall update. The sum
    /// over atoms is `βᵀ(⊕_k S_k ⊗ I_p)β`, the un-scaled total penalty energy.
    /// `S_k` is symmetrised defensively (as the assembler does); the per-atom
    /// `½(S+Sᵀ)·B_k` GEMMs ride the multi-GPU batched smoothness GEMM with an
    /// exact per-atom CPU fallback.
    pub(crate) fn decoder_smoothness_quadratic_form_per_atom(&self) -> Vec<f64> {
        let sb_inputs: Vec<(ArrayView2<'_, f64>, ArrayView2<'_, f64>)> = self
            .atoms
            .iter()
            .map(|atom| (atom.smooth_penalty.view(), atom.decoder_coefficients.view()))
            .collect();
        let sb_all = batched_smooth_sb(&sb_inputs, true);
        let mut per_atom = vec![0.0_f64; self.atoms.len()];
        for (atom_idx, (atom, sb)) in self.atoms.iter().zip(sb_all.iter()).enumerate() {
            per_atom[atom_idx] = (&atom.decoder_coefficients * sb).sum();
        }
        per_atom
    }

    /// Per-atom effective penalized dof of the decoder smoothness penalty
    /// (#1556): entry `k` is `tr(S_β⁻¹ · M_k)` with `M_k = (λ_smooth[k]·S_k) ⊗ I`
    /// and `S_β⁻¹ = (H⁻¹)_ββ` the Schur-complement inverse, each atom scaled by
    /// its OWN `lambda_smooth[atom_idx]`. Built on
    /// [`ArrowFactorCache::schur_inverse_apply`]: column `(k,μ,oc)` of `M_k` is
    /// `λ_k·S_k[:,μ] ⊗ e_oc` (sparse), so we apply `S_β⁻¹` to that K-vector and
    /// read back `result[col]`. The total edf is the sum of the returned vector
    /// (a uniform/broadcast λ reproduces the historical global trace).
    ///
    /// At `K ≥ SMOOTHNESS_DOF_HUTCHINSON_MIN_ATOMS` this delegates to the
    /// matrix-free Hutchinson estimator (the exact `K·M·p`-solve trace is
    /// infeasible at that scale); below it the exact column solve is used
    /// unchanged.
    pub(crate) fn decoder_smoothness_effective_dof_per_atom(
        &self,
        cache: &ArrowFactorCache,
        lambda_smooth: &[f64],
    ) -> Result<Vec<f64>, ArrowSchurError> {
        let p = self.output_dim();
        let frames_active = self.frames_active();
        let (offsets, out_dim): (Vec<usize>, Box<dyn Fn(usize) -> usize>) = if frames_active {
            let ranks: Vec<usize> = self.atoms.iter().map(|a| a.border_frame_rank()).collect();
            (
                self.factored_beta_offsets(),
                Box::new(move |k: usize| ranks[k]),
            )
        } else {
            (self.beta_offsets(), Box::new(move |_k: usize| p))
        };
        let k = cache.k;
        if self.atoms.len() >= Self::SMOOTHNESS_DOF_HUTCHINSON_MIN_ATOMS {
            // Massive-K: `Σ_k M_k·r_k` exact solves is infeasible — estimate every
            // atom's trace matrix-free with one `S_β⁻¹` solve per Hutchinson probe.
            return self
                .decoder_smoothness_effective_dof_per_atom_hutchinson(
                    k,
                    &offsets,
                    out_dim.as_ref(),
                    lambda_smooth,
                    Self::SMOOTHNESS_DOF_HUTCHINSON_PROBES,
                    Self::SMOOTHNESS_DOF_HUTCHINSON_SEED,
                    |rhs| {
                        cache
                            .schur_inverse_apply(rhs)
                            .map_err(|e| format!("schur_inverse_apply: {e:?}"))
                    },
                )
                .map_err(|reason| ArrowSchurError::SchurFactorFailed { reason });
        }
        let mut per_atom = vec![0.0_f64; self.atoms.len()];
        let mut m_col = Array1::<f64>::zeros(k);
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let s = &atom.smooth_penalty;
            let m = atom.basis_size();
            let off = offsets[atom_idx];
            let r = out_dim(atom_idx);
            let lambda = lambda_smooth[atom_idx];
            let mut trace = 0.0_f64;
            for mu in 0..m {
                for oc in 0..r {
                    let col = off + mu * r + oc;
                    m_col.fill(0.0);
                    for nu in 0..m {
                        let s_nu_mu = 0.5 * (s[[nu, mu]] + s[[mu, nu]]);
                        m_col[off + nu * r + oc] = lambda * s_nu_mu;
                    }
                    let z = cache.schur_inverse_apply(m_col.view())?;
                    trace += z[col];
                }
            }
            per_atom[atom_idx] = trace;
        }
        Ok(per_atom)
    }

    /// Per-atom effective penalized dof via the deflated solver (#1556): entry
    /// `k` is `tr((H⁻¹)_ββ · M_k)` for `M_k = (λ_smooth[k]·S_k) ⊗ I`, each atom
    /// scaled by its OWN `lambda_smooth[atom_idx]`. The total is the sum.
    pub(crate) fn decoder_smoothness_effective_dof_with_solver_per_atom(
        &self,
        cache: &ArrowFactorCache,
        solver: &DeflatedArrowSolver<'_>,
        lambda_smooth: &[f64],
    ) -> Result<Vec<f64>, String> {
        let p = self.output_dim();
        // #972 / #977 T1: the cache's β block is the FACTORED border when frames
        // are active (`cache.k == factored_border_dim`), so the smoothness edf
        // trace `tr((H⁻¹)_ββ · M)` is taken over the same factored layout, with
        // `M = ⊕_k (λ_k S_k) ⊗ I_{r_k}` at the factored offsets (the `U_kᵀU_k = I`
        // collapse means the per-coordinate-channel penalty is `λ_k S_k`, exactly
        // as in the full-`B` `⊗ I_p` case but with `r_k` channels). On the
        // full-`B` path `frames_active` is false: `out_dim_k = p`, the offsets
        // are `beta_offsets`, and this is bit-for-bit the historical trace.
        let frames_active = self.frames_active();
        let (offsets, out_dim): (Vec<usize>, Box<dyn Fn(usize) -> usize>) = if frames_active {
            let ranks: Vec<usize> = self.atoms.iter().map(|a| a.border_frame_rank()).collect();
            (
                self.factored_beta_offsets(),
                Box::new(move |k: usize| ranks[k]),
            )
        } else {
            (self.beta_offsets(), Box::new(move |_k: usize| p))
        };
        let k = cache.k;
        // The t-RHS is identically zero for every β-only smoothness solve; build
        // it once instead of re-zeroing a delta_t_len()-sized buffer per column.
        let zero_t = Array1::<f64>::zeros(cache.delta_t_len());
        if self.atoms.len() >= Self::SMOOTHNESS_DOF_HUTCHINSON_MIN_ATOMS {
            // Massive-K matrix-free path: one deflated `(H⁻¹)_ββ` solve per
            // Hutchinson probe estimates ALL per-atom traces, replacing the
            // `Σ_k M_k·r_k` deflated solves that form the `O(K³·M·p)` wall.
            return self.decoder_smoothness_effective_dof_per_atom_hutchinson(
                k,
                &offsets,
                out_dim.as_ref(),
                lambda_smooth,
                Self::SMOOTHNESS_DOF_HUTCHINSON_PROBES,
                Self::SMOOTHNESS_DOF_HUTCHINSON_SEED,
                |rhs| Ok(solver.solve(zero_t.view(), rhs)?.beta),
            );
        }
        let mut per_atom = vec![0.0_f64; self.atoms.len()];
        let mut m_col = Array1::<f64>::zeros(k);
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let s = &atom.smooth_penalty;
            let m = atom.basis_size();
            let off = offsets[atom_idx];
            let r = out_dim(atom_idx);
            let lambda = lambda_smooth[atom_idx];
            let mut trace = 0.0_f64;
            for mu in 0..m {
                for oc in 0..r {
                    let col = off + mu * r + oc;
                    // M[:,col] = λ_k · S_k[:,mu] ⊗ e_oc (nonzero at off+ν·r+oc).
                    m_col.fill(0.0);
                    for nu in 0..m {
                        let s_nu_mu = 0.5 * (s[[nu, mu]] + s[[mu, nu]]);
                        m_col[off + nu * r + oc] = lambda * s_nu_mu;
                    }
                    let z = solver.solve(zero_t.view(), m_col.view())?.beta;
                    trace += z[col];
                }
            }
            per_atom[atom_idx] = trace;
        }
        Ok(per_atom)
    }

    pub(crate) fn assignment_log_strength_hessian_trace(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        solver: &DeflatedArrowSolver<'_>,
    ) -> Result<f64, String> {
        let k_atoms = self.k_atoms();
        // #1038 softmax: `H` carries the DENSE entropy block, and since the
        // entropy curvature scales linearly with `λ_sparse = exp(ρ)`,
        // `∂H/∂ρ = H_entropy` (the full dense per-row block, not just its
        // diagonal). The trace `½ tr(H⁻¹ ∂H/∂ρ)` must therefore contract the
        // dense `∂H/∂ρ` against the per-row selected-inverse BLOCK, mirroring the
        // dense `log|H|` and θ-adjoint — a diagonal-only contraction would
        // desync the ρ-gradient from the criterion. The assembled majorizer
        // `D = diag(Σ_j|H_kj|)` is itself DIAGONAL (#1419), so the contraction
        // reduces to `½ Σ_slot (H⁻¹)_{slot,slot}·D_atom`. On the dense `None`
        // layout the logit slot equals the atom position; on the compact
        // softmax top-`k` layout (#1408/#1409) the slots are the row's active
        // atoms — the SAME `D_atom` (full-`K` abs-row-sum) the assembly wrote.
        if let AssignmentMode::Softmax {
            temperature,
            sparsity,
        } = self.assignment.mode
        {
            if k_atoms <= 1 {
                return Ok(0.0);
            }
            let inv_tau = 1.0 / temperature;
            let scale = rho.lambda_sparse() * sparsity * inv_tau * inv_tau;
            let penalty = gam_terms::analytic_penalties::SoftmaxAssignmentSparsityPenalty::new(
                k_atoms,
                temperature,
            );
            // Softmax uses the reduced K−1 free-logit chart on the dense layout
            // (last reference logit fixed); the compact layout carries one slot
            // per active atom. The diagonal selected inverse gives each slot's
            // (H⁻¹)_{slot,slot}.
            let assignment_dim = self.assignment.assignment_coord_dim();
            // Kept-subspace inverse diagonal: the deflated inverse assigns
            // `1/λ̃ = 1` to each per-row UNIT-stiffness direction `vᵢ`, so a raw
            // diagonal `D` contraction would spuriously add `½ Σ_i vᵢᵀ D vᵢ` (a
            // ρ-independent direction must add 0). `latent_inverse_diagonal_kept`
            // removes that per-row deflated diagonal centrally.
            let inv_diag = solver
                .latent_inverse_diagonal_kept()
                .map_err(|err| format!("assignment_log_strength_hessian_trace: {err}"))?;
            let row_loss_w = self.row_loss_weights.as_deref();
            let mut trace = 0.0_f64;
            for row in 0..self.n_obs() {
                let row_base = cache.row_offsets[row];
                // #991 — the softmax prior curvature written to `htt` carries the
                // row's design weight `w_row` (via the `scale·w_row` the majorizer
                // sites fold in), so its ρ-trace must carry the SAME `w_row`.
                let w_row = row_loss_w.map_or(1.0, |w| w[row]);
                // ∂(scale·D)/∂ρ = scale·D (linear in λ_sparse = eᵖ) — the SAME
                // operator the assembly and θ-adjoint differentiate.
                match self.last_row_layout {
                    Some(ref layout) => {
                        // #1410: the compact adjoint reads `D_kk` only for this
                        // row's `≤ top_k` active atoms, so compute those entries
                        // directly from the softmax row `a` via the active-only
                        // Gershgorin helper — no full-`K` `row_logits` copy and no
                        // full-`K` `d` vector. `a` itself is the irreducible `O(K)`
                        // softmax normalisation, computed once per row and shared
                        // across the row's active slots.
                        let a = crate::assignment::softmax_row(
                            self.assignment.logits.row(row),
                            temperature,
                        );
                        let a = a.as_slice().expect("softmax row must be contiguous");
                        let m = softmax_majorizer_log_mean(a);
                        // #Bug1: only FREE-logit atoms carry a compact logit slot; the
                        // softmax reference atom (last active) has none — matching the
                        // dense branch which sums only the K−1 free logit slots.
                        for (j, &atom) in layout.logit_atoms[row].iter().enumerate() {
                            let d_atom =
                                active_softmax_gershgorin_majorizer_entry(a, atom, m, scale);
                            trace += inv_diag[row_base + j] * w_row * d_atom;
                        }
                    }
                    None => {
                        // Dense layout genuinely contracts every free logit slot's
                        // `D_kk`, so the full-`K` `d` is intrinsic here; keep the
                        // single-source dense majorizer call.
                        let row_logits: Vec<f64> = (0..k_atoms)
                            .map(|k| self.assignment.logits[[row, k]])
                            .collect();
                        let d = penalty.psd_majorizer_abs_row_sums(&row_logits, scale);
                        let q = cache.row_dims[row];
                        let logit_dim = assignment_dim.min(q);
                        for atom in 0..logit_dim {
                            trace += inv_diag[row_base + atom] * w_row * d[atom];
                        }
                    }
                }
            }
            return Ok(0.5 * trace);
        }
        let mut hdiag = crate::assignment::assignment_prior_log_strength_hdiag_weighted(
            &self.assignment,
            rho,
            self.row_loss_weights.as_deref(),
        )?;
        if hdiag.is_empty() {
            return Ok(0.0);
        }
        // RAW selected-inverse diagonal: the per-row diagonal contraction uses the
        // DEFLATED inverse; the full kept-subspace + β-Schur/rotation deflation
        // correction `tr(inv_vv·(D − DΦ[D]))` is subtracted per row afterwards
        // (`deflation_block_correction`), exactly as the data trace does. The
        // cross-row off-diagonal pass below contracts only DISTINCT rows `i ≠ j`,
        // off any single-row `vᵢ`'s support, so it needs no deflation correction.
        let inv_diag = solver
            .latent_inverse_diagonal()
            .map_err(|err| format!("assignment_log_strength_hessian_trace: {err}"))?;
        let assignment_dim = self.assignment.assignment_coord_dim();
        let total_t = cache.delta_t_len();
        // #932 FRONT C: row-local Takahashi selected inverse on the plain arrow
        // for the per-row deflation correction below (the diagonal trace already
        // uses the cheap `latent_inverse_diagonal`); gauge / cross-row Woodbury
        // fall back to the per-row full-system `solve` loop.
        let fast_selected = solver.plain_selected_inverse_available();
        let selected_beta_inv = if fast_selected && cache.k > 0 {
            solver
                .beta_inv()
                .map_err(|err| format!("assignment_log_strength_hessian_trace: {err}"))?
        } else {
            Array2::<f64>::zeros((0, 0))
        };
        // #1416 cross-row IBP source: the per-row block that the deflation
        // factorizes is the NO-SELF base `H₀'` — the rank-one self curvature
        // `d_k·J_ik²` is DOWNDATED from each logit diagonal and re-applied through
        // the Woodbury carrier. The full-`H` diagonal contraction below still uses
        // the full `hdiag` (which carries that self term), but the per-row
        // DEFLATION correction must use `(∂H₀'/∂ρ)_tt`, i.e. `hdiag` MINUS the
        // downdated self term — otherwise the Daleckii–Krein correction
        // mis-attributes the (un-deflated) Woodbury self curvature's derivative to
        // the deflated subspace. For non-IBP modes there is no Woodbury source and
        // the self term is `0` (the deflated block IS the full block).
        // #1416 (compact-layout completion): the IBP cross-row Woodbury source is
        // installed for BOTH the dense and the compact (#1420 top-`k`) layouts (see
        // `set_ibp_cross_row_source`, which emits `(g_base + pos, atom, z'_ik)` for
        // the active set under a compact layout), so the deflated base `H₀'` is the
        // no-self block in BOTH layouts. The self-curvature downdate below must
        // therefore run regardless of layout — gating it to the dense path (the
        // pre-fix bug) left the compact deflation correction differentiating the
        // un-downdated full block. For non-IBP modes `ibp_assignment_third_channels`
        // returns `None`, there is no Woodbury source, and `self_curv` is
        // identically 0 (the deflated block IS the full block).
        // RAW channels: the `w·s·c` diagonal split needs the un-clamped `w·s'`, so
        // build raw and apply the gam#2144 majorization here.
        let mut cross_channels = ibp_assignment_third_channels_weighted(
            &self.assignment,
            rho,
            false,
            self.row_loss_weights.as_deref(),
        )?;
        let learnable_alpha = matches!(
            self.assignment.mode,
            AssignmentMode::IBPMap {
                learnable_alpha: true,
                ..
            }
        );
        // gam#2144/#1038: the assembled `H` carries the PSD-majorized IBP
        // curvature UNCONDITIONALLY (`ibp_psd_majorized_hdiag` + clamped Woodbury
        // `d` — the same doctrine as softmax's #1419 Gershgorin). Differentiate the
        // SAME operator: overwrite the per-slot diagonal with its majorizer and
        // clamp the rank-one coefficient (`cross_row_d`, and its learnable-α
        // derivative) to `max(·,0)`. `self_curv`, the diagonal trace, and the
        // cross-row off-diagonal pass all read these, so the whole ρ-trace stays on
        // the majorized operator the evidence log-det factors.
        if let Some(ch) = cross_channels.as_mut() {
            for row in 0..self.n_obs() {
                for atom in 0..k_atoms {
                    let slot = row * k_atoms + atom;
                    hdiag[slot] = super::construction_arrow_schur_assembly::ibp_psd_majorized_hdiag(
                        ch,
                        row,
                        k_atoms,
                        atom,
                        hdiag[slot],
                    );
                }
            }
            for k in 0..k_atoms {
                if ch.cross_row_d[k] < 0.0 {
                    ch.cross_row_d[k] = 0.0;
                    ch.cross_row_d_logalpha[k] = 0.0;
                }
            }
        }
        let self_curv = |row: usize, atom: usize| -> f64 {
            let Some(ch) = cross_channels.as_ref() else {
                return 0.0;
            };
            let d_k = if learnable_alpha {
                ch.cross_row_d_logalpha[atom]
            } else {
                ch.cross_row_d[atom]
            };
            let j = ch.z_jac[row * k_atoms + atom];
            d_k * j * j
        };
        let mut trace = 0.0_f64;
        // Hoisted RHS scratch for the gauge/Woodbury per-row solve fallback:
        // single-entry set/clear instead of a per-column total_t-sized zeroing.
        let mut rhs_t_scratch = Array1::<f64>::zeros(total_t);
        let rhs_beta_zero = Array1::<f64>::zeros(cache.k);
        for row in 0..self.n_obs() {
            let row_base = cache.row_offsets[row];
            let assignment_base = row * k_atoms;
            let q = cache.row_dims[row];
            // Per-row diagonal `(∂H₀'/∂ρ)_tt` for the deflation correction: the
            // assignment prior curves only the logit/assignment slots (coordinate
            // slots are 0 — ARD handles those), MINUS the downdated cross-row self
            // curvature. The full-`H` trace contraction keeps the full `hdiag`.
            let mut d_diag = Array1::<f64>::zeros(q);
            match self.last_row_layout {
                Some(ref layout) => {
                    for (pos, &atom) in layout.active_atoms[row].iter().enumerate() {
                        let d_slot = hdiag[assignment_base + atom];
                        trace += inv_diag[row_base + pos] * d_slot;
                        if pos < q {
                            d_diag[pos] = d_slot - self_curv(row, atom);
                        }
                    }
                }
                None => {
                    for free_idx in 0..assignment_dim {
                        let d_slot = hdiag[assignment_base + free_idx];
                        trace += inv_diag[row_base + free_idx] * d_slot;
                        if free_idx < q {
                            d_diag[free_idx] = d_slot - self_curv(row, free_idx);
                        }
                    }
                }
            }
            let dirs = cache
                .deflated_row_directions
                .get(row)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            if !dirs.is_empty() {
                let inv_vv = if fast_selected {
                    let (inv_vv, _inv_vbeta) = solver
                        .selected_inverse_row_blocks(row, &selected_beta_inv)
                        .map_err(|err| {
                            format!(
                                "assignment_log_strength_hessian_trace: selected inverse: {err}"
                            )
                        })?;
                    inv_vv
                } else {
                    let mut inv_vv = Array2::<f64>::zeros((q, q));
                    for col in 0..q {
                        rhs_t_scratch[row_base + col] = 1.0;
                        let solved = solver
                            .solve(rhs_t_scratch.view(), rhs_beta_zero.view())
                            .map_err(|err| {
                                format!(
                                    "assignment_log_strength_hessian_trace: selected inverse: {err}"
                                )
                            })?;
                        rhs_t_scratch[row_base + col] = 0.0;
                        for r in 0..q {
                            inv_vv[[r, col]] = solved.t[row_base + r];
                        }
                    }
                    inv_vv
                };
                let mut d_mat = Array2::<f64>::zeros((q, q));
                for s in 0..q {
                    d_mat[[s, s]] = d_diag[s];
                }
                let spectrum = cache
                    .deflation_row_spectra
                    .get(row)
                    .and_then(Option::as_ref);
                trace -= Self::deflation_block_correction(&inv_vv, &d_mat, dirs, spectrum);
            }
        }
        // #1416: the IBP prior Hessian is `H_p = d·J Jᵀ + diag(s, c)`, where the
        // rank-one `d·J Jᵀ` couples EVERY row pair `(i, j)` in a column `k`
        // through the shared empirical mass `M_k`. The assembled `H` carries the
        // full `H_full = H₀' + U D Uᵀ` (Woodbury, `set_ibp_cross_row_source`), and
        // for fixed alpha the entire IBP prior scales with `λ = eᵖ`, so
        // `∂H_p/∂ρ = H_p`. The diagonal loop above already captures the `i = j`
        // self terms (the `d·J_ik²` summand lives in `hdiag`); this pass adds the
        // omitted off-diagonal `½·d_k·Σ_{i≠j}(H⁻¹)_{ik,jk}·J_ik·J_jk`. Only IBP
        // has the cross-row rank-one source; for other diagonal modes
        // `ibp_assignment_third_channels` returns `None` and the trace stays the
        // pure diagonal contraction.
        //
        // #1416 (compact completion): this pass is LAYOUT-AGNOSTIC. Under the dense
        // layout atom `k`'s logit slot is local position `k`
        // (`row_offsets[i] + k`); under the compact (#1420 top-`k`) layout only the
        // row's active atoms carry coordinates and atom `k` lives at local position
        // `pos` of `active_atoms[row]` (`row_offsets[i] + pos`). The Woodbury source
        // and the θ-adjoint already use this active-slot mapping, so gating the
        // cross-row pass to the dense layout (the pre-fix bug) dropped the
        // off-diagonal term from `∂log|H|/∂ρ` whenever the budget/`top_k` engaged
        // the compact layout. We build per-column active sites `(row, t_index)` once
        // — exactly the θ-adjoint `col_sites` construction — then contract the
        // off-diagonal `i ≠ j` remainder with one solve per active site.
        if let Some(channels) = cross_channels.as_ref() {
            let n = self.n_obs();
            let total_t = cache.delta_t_len();
            // This trace is ½ ∂log|H|/∂ρ. For FIXED-α IBP the whole prior
            // scales with λ=eᵖ so ∂H_p/∂ρ = H_p and the rank-one coefficient
            // is the VALUE `cross_row_d[k] = w·s'_k`. For LEARNABLE-α this trace
            // is ½ ∂log|H|/∂logα, and the rank-one block's logα-derivative is
            // `∂d_k/∂logα = w·∂s'_k/∂logα` (`cross_row_d_logalpha[k]`) — the same
            // α-derivative the DIAGONAL channel (`hessian_diag_log_alpha_derivative`)
            // already uses. Using the value `s'_k` here (the pre-fix bug) made the
            // off-diagonal inconsistent with the diagonal and the α-gradient wrong.
            // (`learnable_alpha` is the same flag the self-curvature downdate uses.)
            // Per-column active sites `(row, global t-index)`. Layout-agnostic.
            let mut col_sites: Vec<Vec<(usize, usize)>> = vec![Vec::new(); k_atoms];
            match self.last_row_layout {
                Some(ref layout) => {
                    for row in 0..n {
                        let base = cache.row_offsets[row];
                        for (pos, &atom) in layout.active_atoms[row].iter().enumerate() {
                            col_sites[atom].push((row, base + pos));
                        }
                    }
                }
                None => {
                    for row in 0..n {
                        let base = cache.row_offsets[row];
                        for k in 0..k_atoms {
                            col_sites[k].push((row, base + k));
                        }
                    }
                }
            }
            let mut cross = 0.0_f64;
            // Hoisted RHS scratch: each active site sets exactly one t-slot, so
            // set-then-clear that single entry rather than allocating and zeroing
            // a total_t-sized vector per (column, site).
            let mut rhs_t_scratch = Array1::<f64>::zeros(total_t);
            let rhs_beta_zero = Array1::<f64>::zeros(cache.k);
            for k in 0..k_atoms {
                let d_k = if learnable_alpha {
                    channels.cross_row_d_logalpha[k]
                } else {
                    channels.cross_row_d[k]
                };
                if d_k == 0.0 || col_sites[k].len() < 2 {
                    continue;
                }
                for &(i, t_i) in &col_sites[k] {
                    let j_ik = channels.z_jac[i * k_atoms + k];
                    if j_ik == 0.0 {
                        continue;
                    }
                    // (H⁻¹) column at row `i`'s active logit-`k` slot.
                    rhs_t_scratch[t_i] = 1.0;
                    let solved = solver
                        .solve(rhs_t_scratch.view(), rhs_beta_zero.view())
                        .map_err(|err| format!("assignment_log_strength_hessian_trace: {err}"))?;
                    rhs_t_scratch[t_i] = 0.0;
                    for &(j, t_j) in &col_sites[k] {
                        if j == i {
                            continue;
                        }
                        let j_jk = channels.z_jac[j * k_atoms + k];
                        if j_jk == 0.0 {
                            continue;
                        }
                        cross += d_k * solved.t[t_j] * j_ik * j_jk;
                    }
                }
            }
            trace += cross;
        }
        Ok(0.5 * trace)
    }

    /// Derivative of the coordinate-block logdet
    /// `½ Σ_i log|H_tt^(i)|` with respect to the assignment-strength rho
    /// coordinate. The canonical criterion subtracts this term from the full
    /// joint logdet, so the outer gradient must subtract this trace too.
    pub(crate) fn coordinate_block_assignment_log_strength_hessian_trace(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
    ) -> Result<f64, String> {
        let k_atoms = self.k_atoms();
        let assignment_dim = self.assignment.assignment_coord_dim();
        let row_weights = self.row_loss_weights.as_deref();

        let softmax = match self.assignment.mode {
            AssignmentMode::Softmax {
                temperature,
                sparsity,
            } if k_atoms > 1 => {
                let inv_tau = temperature.recip();
                Some((
                    temperature,
                    rho.lambda_sparse() * sparsity * inv_tau * inv_tau,
                    gam_terms::analytic_penalties::SoftmaxAssignmentSparsityPenalty::new(
                        k_atoms,
                        temperature,
                    ),
                ))
            }
            AssignmentMode::Softmax { .. } => return Ok(0.0),
            _ => None,
        };
        let mut hdiag = if softmax.is_none() {
            crate::assignment::assignment_prior_log_strength_hdiag_weighted(
                &self.assignment,
                rho,
                row_weights,
            )?
        } else {
            Array1::<f64>::zeros(0)
        };
        if softmax.is_none() && hdiag.is_empty() {
            return Ok(0.0);
        }

        let mut ibp_channels =
            ibp_assignment_third_channels_weighted(&self.assignment, rho, false, row_weights)?;
        let learnable_alpha = matches!(
            self.assignment.mode,
            AssignmentMode::IBPMap {
                learnable_alpha: true,
                ..
            }
        );
        if let Some(channels) = ibp_channels.as_mut() {
            for row in 0..self.n_obs() {
                for atom in 0..k_atoms {
                    let index = row * k_atoms + atom;
                    hdiag[index] =
                        super::construction_arrow_schur_assembly::ibp_psd_majorized_hdiag(
                            channels,
                            row,
                            k_atoms,
                            atom,
                            hdiag[index],
                        );
                }
            }
            for atom in 0..k_atoms {
                if channels.cross_row_d[atom] < 0.0 {
                    channels.cross_row_d[atom] = 0.0;
                    channels.cross_row_d_logalpha[atom] = 0.0;
                }
            }
        }

        let mut total_trace = 0.0_f64;
        for row in 0..self.n_obs() {
            let q = cache.row_dims[row];
            let factor = cache.undamped_factor(row);
            let mut inverse = Array2::<f64>::zeros((q, q));
            let mut unit = Array1::<f64>::zeros(q);
            for col in 0..q {
                unit.fill(0.0);
                unit[col] = 1.0;
                let solved = cholesky_solve_vector(factor, unit.view());
                for inverse_row in 0..q {
                    inverse[[inverse_row, col]] = solved[inverse_row];
                }
            }
            let mut derivative = Array2::<f64>::zeros((q, q));
            if let Some((temperature, scale, penalty)) = softmax.as_ref() {
                let row_weight = row_weights.map_or(1.0, |weights| weights[row]);
                match self.last_row_layout {
                    Some(ref layout) => {
                        let assignments = crate::assignment::softmax_row(
                            self.assignment.logits.row(row),
                            *temperature,
                        );
                        let assignments = assignments
                            .as_slice()
                            .expect("softmax assignment row is contiguous");
                        let mean = softmax_majorizer_log_mean(assignments);
                        for (slot, &atom) in layout.logit_atoms[row].iter().enumerate() {
                            derivative[[slot, slot]] = row_weight
                                * active_softmax_gershgorin_majorizer_entry(
                                    assignments,
                                    atom,
                                    mean,
                                    *scale,
                                );
                        }
                    }
                    None => {
                        let logits = (0..k_atoms)
                            .map(|atom| self.assignment.logits[[row, atom]])
                            .collect::<Vec<_>>();
                        let curvature = penalty.psd_majorizer_abs_row_sums(&logits, *scale);
                        for atom in 0..assignment_dim.min(q) {
                            derivative[[atom, atom]] = row_weight * curvature[atom];
                        }
                    }
                }
            } else {
                let assignment_base = row * k_atoms;
                let self_curvature = |atom: usize| -> f64 {
                    let Some(channels) = ibp_channels.as_ref() else {
                        return 0.0;
                    };
                    let coefficient = if learnable_alpha {
                        channels.cross_row_d_logalpha[atom]
                    } else {
                        channels.cross_row_d[atom]
                    };
                    let jacobian = channels.z_jac[assignment_base + atom];
                    coefficient * jacobian * jacobian
                };
                match self.last_row_layout {
                    Some(ref layout) => {
                        for (slot, &atom) in layout.active_atoms[row].iter().enumerate() {
                            derivative[[slot, slot]] =
                                hdiag[assignment_base + atom] - self_curvature(atom);
                        }
                    }
                    None => {
                        for atom in 0..assignment_dim.min(q) {
                            derivative[[atom, atom]] =
                                hdiag[assignment_base + atom] - self_curvature(atom);
                        }
                    }
                }
            }
            let mut row_trace = 0.0_f64;
            for a in 0..q {
                for b in 0..q {
                    row_trace += inverse[[b, a]] * derivative[[a, b]];
                }
            }
            let directions = cache
                .deflated_row_directions
                .get(row)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            if !directions.is_empty() {
                let spectrum = cache
                    .deflation_row_spectra
                    .get(row)
                    .and_then(Option::as_ref);
                row_trace -=
                    Self::deflation_block_correction(&inverse, &derivative, directions, spectrum);
            }
            total_trace += row_trace;
        }
        Ok(0.5 * total_trace)
    }

    /// Matrix-free sibling of [`Self::assignment_log_strength_hessian_trace`]
    /// for assignment families whose prior curvature is row-local (softmax,
    /// threshold gate, and TopK).  Reconstructs each undeflated row's selected-
    /// inverse diagonal from the exact row-local inverse plus the shared
    /// `(z_j, S^-1 z_j)` reduced-Schur bundle:
    ///
    /// `diag(H^-1_tt) = diag(A_i^-1) + (1/m) sum_j
    ///   (A_i^-1 H_tbeta z_j) * (A_i^-1 H_tbeta S^-1 z_j)`.
    ///
    /// This is the missing assignment-strength trace in the matrix-free analytic
    /// rho-gradient cluster.  It deliberately refuses IBP cross-row curvature and
    /// per-row spectral/gauge deflation: the border-only bundle represents the
    /// plain row-block arrow inverse and cannot reconstruct either correction.
    pub(crate) fn assignment_log_strength_hessian_trace_from_probes(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        probes: &[Array1<f64>],
        sinv_probes: &[Array1<f64>],
    ) -> Result<f64, String> {
        if matches!(self.assignment.mode, AssignmentMode::IBPMap { .. })
            || cache.cross_row_woodbury.is_some()
        {
            return Err(
                "assignment_log_strength_hessian_trace_from_probes: IBP cross-row curvature \
                 is not represented by the border-only inverse-probe bundle"
                    .to_string(),
            );
        }
        let m = probes.len();
        if m == 0 || sinv_probes.len() != m {
            return Err(format!(
                "assignment_log_strength_hessian_trace_from_probes: need matching non-empty \
                 probe/solve bundles, got {m} probes and {} solves",
                sinv_probes.len()
            ));
        }
        let k_border = cache.k;
        for (label, set) in [("probe", probes), ("solve", sinv_probes)] {
            for (j, vector) in set.iter().enumerate() {
                if vector.len() != k_border {
                    return Err(format!(
                        "assignment_log_strength_hessian_trace_from_probes: {label} {j} has \
                         length {} != border dim {k_border}",
                        vector.len()
                    ));
                }
            }
        }

        let k_atoms = self.k_atoms();
        let softmax = match self.assignment.mode {
            AssignmentMode::Softmax {
                temperature,
                sparsity,
            } if k_atoms > 1 => {
                let inv_tau = 1.0 / temperature;
                Some((
                    temperature,
                    rho.lambda_sparse() * sparsity * inv_tau * inv_tau,
                    gam_terms::analytic_penalties::SoftmaxAssignmentSparsityPenalty::new(
                        k_atoms,
                        temperature,
                    ),
                ))
            }
            AssignmentMode::Softmax { .. } => return Ok(0.0),
            _ => None,
        };
        let hdiag = if softmax.is_none() {
            crate::assignment::assignment_prior_log_strength_hdiag_weighted(
                &self.assignment,
                rho,
                self.row_loss_weights.as_deref(),
            )?
        } else {
            Array1::zeros(0)
        };
        if softmax.is_none() && hdiag.is_empty() {
            return Ok(0.0);
        }
        let assignment_dim = self.assignment.assignment_coord_dim();
        let row_loss_weights = self.row_loss_weights.as_deref();
        let inv_m = 1.0 / m as f64;
        let mut trace = 0.0_f64;
        for row in 0..self.n_obs() {
            if cache
                .deflated_row_directions
                .get(row)
                .is_some_and(|directions| !directions.is_empty())
            {
                return Err(format!(
                    "assignment_log_strength_hessian_trace_from_probes: row {row} carries \
                     deflation directions; the plain-S^-1 bundle cannot reconstruct the \
                     Daleckii-Krein correction"
                ));
            }
            let q = cache.row_dims[row];
            let factor = cache.undamped_factor(row);
            let mut inverse_diagonal = Array1::<f64>::zeros(q);
            let mut unit = Array1::<f64>::zeros(q);
            for slot in 0..q {
                unit.fill(0.0);
                unit[slot] = 1.0;
                inverse_diagonal[slot] = cholesky_solve_vector(factor, unit.view())[slot];
            }

            let mut cross = Array1::<f64>::zeros(q);
            for j in 0..m {
                cross.fill(0.0);
                if !cache.apply_htbeta_row(row, probes[j].view(), &mut cross) {
                    return Err(format!(
                        "assignment_log_strength_hessian_trace_from_probes: H_tbeta^({row}) \
                         probe apply failed"
                    ));
                }
                let probe_row = cholesky_solve_vector(factor, cross.view());
                cross.fill(0.0);
                if !cache.apply_htbeta_row(row, sinv_probes[j].view(), &mut cross) {
                    return Err(format!(
                        "assignment_log_strength_hessian_trace_from_probes: H_tbeta^({row}) \
                         solve apply failed"
                    ));
                }
                let solve_row = cholesky_solve_vector(factor, cross.view());
                for slot in 0..q {
                    inverse_diagonal[slot] += inv_m * probe_row[slot] * solve_row[slot];
                }
            }

            if let Some((temperature, scale, penalty)) = softmax.as_ref() {
                let row_weight = row_loss_weights.map_or(1.0, |weights| weights[row]);
                match self.last_row_layout {
                    Some(ref layout) => {
                        let assignments = crate::assignment::softmax_row(
                            self.assignment.logits.row(row),
                            *temperature,
                        );
                        let assignments = assignments
                            .as_slice()
                            .expect("softmax row must be contiguous");
                        let mean = softmax_majorizer_log_mean(assignments);
                        for (slot, &atom) in layout.logit_atoms[row].iter().enumerate() {
                            let curvature = active_softmax_gershgorin_majorizer_entry(
                                assignments,
                                atom,
                                mean,
                                *scale,
                            );
                            trace += inverse_diagonal[slot] * row_weight * curvature;
                        }
                    }
                    None => {
                        let row_logits = (0..k_atoms)
                            .map(|atom| self.assignment.logits[[row, atom]])
                            .collect::<Vec<_>>();
                        let curvature = penalty.psd_majorizer_abs_row_sums(&row_logits, *scale);
                        let logit_dim = assignment_dim.min(inverse_diagonal.len());
                        for atom in 0..logit_dim {
                            trace += inverse_diagonal[atom] * row_weight * curvature[atom];
                        }
                    }
                }
            } else {
                let assignment_base = row * k_atoms;
                match self.last_row_layout {
                    Some(ref layout) => {
                        for (slot, &atom) in layout.active_atoms[row].iter().enumerate() {
                            trace += inverse_diagonal[slot] * hdiag[assignment_base + atom];
                        }
                    }
                    None => {
                        for slot in 0..assignment_dim.min(inverse_diagonal.len()) {
                            trace += inverse_diagonal[slot] * hdiag[assignment_base + slot];
                        }
                    }
                }
            }
        }
        Ok(0.5 * trace)
    }

    /// Per-row spectral-deflation correction `tr((H⁻¹)_tt · (D − DΦ[D]))` for one
    /// evidence ρ-component, to be SUBTRACTED from the raw-derivative trace
    /// `tr((H⁻¹)_tt · D)` the trace otherwise accumulates.
    ///
    /// The criterion VALUE re-deflates each per-row `H_tt` at every ρ, so the
    /// correct evidence gradient contracts `(H⁻¹)_tt` against the deflation-map
    /// derivative `DΦ[D]`, not the raw `D = (∂H_raw/∂ρ)_tt`. By Daleckii–Krein,
    /// in the row's RAW eigenbasis `U`,
    ///   `DΦ[D] = U (F ∘ (Uᵀ D U)) Uᵀ`,  `F_{ml} = (λ̃ₘ − λ̃ₗ)/(λₘ − λₗ)`
    /// (raw `λ` in the denominator, conditioned `λ̃` in the numerator; the
    /// diagonal / degenerate entry is `f'(λₘ) = 1` for an unclamped kept
    /// direction and `0` otherwise). Hence `D − DΦ[D] = U ((1−F) ∘ (Uᵀ D U)) Uᵀ`,
    /// whose kept×kept block is `0`, deflated×deflated block is the full `M`, and
    /// kept(m)×deflated(i) block carries the ROTATION coefficient
    /// `(1−λᵢ)/(λₘ−λᵢ)`. Contracting against the FULL deflated selected-inverse
    /// t-block `inv_vv` (which carries the β-Schur back-substitution) captures
    /// both the within-row kept-subspace term and the deferred β-Schur/rotation
    /// coupling in one pass, matching the re-deflating fixed-state FD oracle.
    ///
    /// `spectrum = Some` (spectral deflation): exact Daleckii–Krein. `None` with a
    /// non-empty `dirs` (gauge-only deflation, ρ-independent structural null):
    /// fall back to the within-row kept-subspace term `Σᵢ vᵢᵀ D vᵢ`.
    /// `inv_vv` is assumed symmetric (selected inverse of a symmetric PD system).
    // #1610 — `pub(crate)` so the ARD/latent-block helpers moved into
    // `construction_ard.rs` (pure code move to stay under the 10k-line ban gate)
    // can still call this from the sibling module.
    pub(crate) fn deflation_block_correction(
        inv_vv: &Array2<f64>,
        d_mat: &Array2<f64>,
        dirs: &[Array1<f64>],
        spectrum: Option<&RowDeflationSpectrum>,
    ) -> f64 {
        let q = inv_vv.nrows();
        let Some(spec) = spectrum else {
            // Gauge-only deflation: ρ-independent structural null → within-row term.
            let mut acc = 0.0_f64;
            for v in dirs {
                for a in 0..q {
                    let va = if a < v.len() { v[a] } else { 0.0 };
                    if va == 0.0 {
                        continue;
                    }
                    for b in 0..q {
                        let vb = if b < v.len() { v[b] } else { 0.0 };
                        acc += va * vb * d_mat[[a, b]];
                    }
                }
            }
            return acc;
        };
        let u = &spec.evecs;
        if u.nrows() != q || u.ncols() != q {
            return 0.0;
        }
        let raw = &spec.raw_evals;
        let cond = &spec.cond_evals;
        // M = Uᵀ D U, W = Uᵀ inv_vv U (both q×q, symmetric).
        let m = u.t().dot(d_mat).dot(u);
        let w = u.t().dot(inv_vv).dot(u);
        // correction = Σ_{m,l} W[m,l]·M[m,l]·(1 − F[m,l]).
        let mut acc = 0.0_f64;
        let eigen_scale = raw
            .iter()
            .chain(cond.iter())
            .copied()
            .fold(0.0_f64, |scale, value| scale.max(value.abs()));
        let gap_threshold = eigen_gap_threshold(eigen_scale, raw.len());
        for a in 0..q {
            for b in 0..q {
                let denom = raw[a] - raw[b];
                let f1 = if denom.abs() > gap_threshold {
                    (cond[a] - cond[b]) / denom
                } else if cond[a] == raw[a] {
                    1.0
                } else {
                    0.0
                };
                acc += w[[a, b]] * m[[a, b]] * (1.0 - f1);
            }
        }
        acc
    }

    /// β-tier selected inverse `(H⁻¹)_ββ`, shared across rows (#932 FRONT C). On
    /// the plain bordered arrow this is the cached dense `S⁻¹` formed once from the
    /// Schur factor; when a gauge / #1038 cross-row Woodbury is active the row-local
    /// Takahashi blocks are NOT valid, so it falls back to the per-β-coordinate
    /// `solve` loop (bit-identical, `O(n)` per column). `context` prefixes the
    /// caller's error text. Used by `logdet_theta_adjoint` to share one
    /// β selected-inverse across all row contractions.
    fn selected_inverse_beta_block(
        solver: &DeflatedArrowSolver<'_>,
        cache: &ArrowFactorCache,
        fast_selected: bool,
        context: &str,
    ) -> Result<Array2<f64>, String> {
        if cache.k == 0 {
            Ok(Array2::<f64>::zeros((0, 0)))
        } else if fast_selected {
            solver
                .beta_inv()
                .map_err(|err| format!("{context}: beta selected inverse: {err}"))
        } else {
            let mut beta_inv = Array2::<f64>::zeros((cache.k, cache.k));
            let rhs_t = Array1::<f64>::zeros(cache.delta_t_len());
            let mut rhs_beta = Array1::<f64>::zeros(cache.k);
            for col in 0..cache.k {
                rhs_beta[col] = 1.0;
                let solved = solver
                    .solve(rhs_t.view(), rhs_beta.view())
                    .map_err(|err| format!("{context}: beta selected inverse solve: {err}"))?;
                rhs_beta[col] = 0.0;
                for r in 0..cache.k {
                    beta_inv[[r, col]] = solved.beta[r];
                }
            }
            Ok(beta_inv)
        }
    }

    /// Per-row selected-inverse blocks `(inv_vv, inv_vbeta) = ((H⁻¹)_tt, (H⁻¹)_tβ)`
    /// for `row` (#932 FRONT C). Row-local Takahashi (`O(q·(q+K))`) on the plain
    /// arrow; a per-row full-system `solve` loop (`O(n·q)`) under gauge / cross-row
    /// Woodbury where the row-local blocks are not valid. `rhs_t_scratch` is a
    /// hoisted `delta_t_len()`-sized buffer, left zeroed on return; `rhs_beta_zero`
    /// is a zero β-RHS of length `cache.k`; `context` prefixes the error text.
    /// Used by `logdet_theta_adjoint`; the solve-invariant operands ride in
    /// [`SelectedInverseRowSolve`] (built once per outer solve), while only the
    /// per-row coordinates and reusable scratch vary per call.
    fn selected_inverse_row_blocks_or_solve(
        ctx: &SelectedInverseRowSolve<'_>,
        row: usize,
        base: usize,
        q: usize,
        rhs_t_scratch: &mut Array1<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>), String> {
        let solver = ctx.solver;
        let cache = ctx.cache;
        let beta_inv = ctx.beta_inv;
        let fast_selected = ctx.fast_selected;
        let rhs_beta_zero = ctx.rhs_beta_zero;
        let context = ctx.context;
        if fast_selected {
            solver
                .selected_inverse_row_blocks(row, beta_inv)
                .map_err(|err| format!("{context}: selected inverse: {err}"))
        } else {
            let mut inv_vv = Array2::<f64>::zeros((q, q));
            let mut inv_vbeta = Array2::<f64>::zeros((q, cache.k));
            for col in 0..q {
                rhs_t_scratch[base + col] = 1.0;
                let solved = solver
                    .solve(rhs_t_scratch.view(), rhs_beta_zero)
                    .map_err(|err| format!("{context}: selected inverse solve: {err}"))?;
                rhs_t_scratch[base + col] = 0.0;
                for r in 0..q {
                    inv_vv[[r, col]] = solved.t[base + r];
                }
                for b in 0..cache.k {
                    inv_vbeta[[col, b]] = solved.beta[b];
                }
            }
            Ok((inv_vv, inv_vbeta))
        }
    }

    pub(crate) fn border_channels_for_cache(
        &self,
        cache: &ArrowFactorCache,
    ) -> Result<Vec<SaeBorderChannel>, String> {
        let p = self.output_dim();
        let frames_active = self.last_frames_active && cache.k == self.factored_border_dim();
        let offsets = if frames_active {
            self.factored_beta_offsets()
        } else {
            self.beta_offsets()
        };
        let mut channels = Vec::with_capacity(cache.k);
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let m = atom.basis_size();
            let frame = if frames_active {
                self.frame_output_matrix(atom_idx)
            } else {
                Array2::<f64>::eye(p)
            };
            let r = frame.ncols();
            for basis_col in 0..m {
                for channel in 0..r {
                    let mut output = vec![0.0_f64; p];
                    for out_col in 0..p {
                        output[out_col] = frame[[out_col, channel]];
                    }
                    channels.push(SaeBorderChannel {
                        atom: atom_idx,
                        basis_col,
                        index: offsets[atom_idx] + basis_col * r + channel,
                        output,
                    });
                }
            }
        }
        if channels.len() != cache.k {
            return Err(format!(
                "border channel layout has {} entries but cache border has {}",
                channels.len(),
                cache.k
            ));
        }
        Ok(channels)
    }

    pub(crate) fn row_vars_for_cache_row(
        &self,
        row: usize,
        cache: &ArrowFactorCache,
    ) -> Result<Vec<SaeLocalRowVar>, String> {
        let q_row = cache.row_dims[row];
        let mut vars: Vec<Option<SaeLocalRowVar>> = vec![None; q_row];
        match self.last_row_layout {
            Some(ref layout) => {
                // #Bug1: logit vars go on the leading free-logit slots; the softmax
                // reference atom takes a coord block but no logit slot.
                for (j, &atom) in layout.logit_atoms[row].iter().enumerate() {
                    vars[j] = Some(SaeLocalRowVar::Logit { atom });
                }
                for (pos, &atom) in layout.active_atoms[row].iter().enumerate() {
                    let start = layout.coord_starts[row][pos];
                    let d = self.assignment.coords[atom].latent_dim();
                    for axis in 0..d {
                        vars[start + axis] = Some(SaeLocalRowVar::Coord { atom, axis });
                    }
                }
            }
            None => {
                let assignment_dim = self.assignment.assignment_coord_dim();
                let coord_offsets = self.assignment.coord_offsets();
                for atom in 0..assignment_dim {
                    vars[atom] = Some(SaeLocalRowVar::Logit { atom });
                }
                for atom in 0..self.k_atoms() {
                    let start = coord_offsets[atom];
                    let d = self.assignment.coords[atom].latent_dim();
                    for axis in 0..d {
                        vars[start + axis] = Some(SaeLocalRowVar::Coord { atom, axis });
                    }
                }
            }
        }
        vars.into_iter()
            .enumerate()
            .map(|(idx, v)| {
                v.ok_or_else(|| {
                    format!("row_vars_for_cache_row: row {row} position {idx} was not mapped")
                })
            })
            .collect()
    }

    pub(crate) fn atom_second_jets(&self) -> Result<Vec<Array4<f64>>, String> {
        let mut out = Vec::with_capacity(self.k_atoms());
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let coords = self.assignment.coords[atom_idx].as_matrix();
            let jet = if let Some(second) = atom.basis_second_jet.as_ref() {
                second.second_jet(coords.view())?
            } else {
                let evaluator = atom.basis_evaluator.as_ref().ok_or_else(|| {
                    format!(
                        "logdet_theta_adjoint: atom '{}' has no basis evaluator for second jets",
                        atom.name
                    )
                })?;
                evaluator
                    .second_jet_dyn(coords.view())
                    .ok_or_else(|| {
                        format!(
                            "logdet_theta_adjoint: atom '{}' basis does not expose analytic second jets",
                            atom.name
                        )
                    })??
            };
            let expected = (
                atom.n_obs(),
                atom.basis_size(),
                atom.latent_dim,
                atom.latent_dim,
            );
            if jet.dim() != expected {
                return Err(format!(
                    "logdet_theta_adjoint: atom '{}' second jet shape {:?}, expected {:?}",
                    atom.name,
                    jet.dim(),
                    expected
                ));
            }
            out.push(jet);
        }
        Ok(out)
    }

    // [#780 line-count gate] The per-row jet / reconstruction-channel cluster
    // (`reconstruction_row_program_for_logdet`, the const-generic
    // reconstruction / β-border channel fills and their dynamic dispatchers,
    // `row_jets_for_logdet`, `row_jets_for_logdet_batch4`, `batch4_assemble`,
    // and `refill_jet_window`) lives in the sibling
    // `construction_row_jet_logdet_channels.rs` file, inlined via `include!`
    // below at module scope as a second `impl SaeManifoldTerm` block. Splitting
    // it out keeps this tracked file under the 10k limit; `include!` preserves
    // the identical module scope and private-field access.

    pub(crate) fn assignment_prior_hdiag_derivative_entry(
        &self,
        rho: &SaeManifoldRho,
        row: usize,
        diag_atom: usize,
        wrt: SaeLocalRowVar,
        ibp_channels: Option<&IbpHessianDiagThirdChannels>,
    ) -> f64 {
        let SaeLocalRowVar::Logit { atom: wrt_atom } = wrt else {
            return 0.0;
        };
        // #Bug4: a FIXED logit (ungated atom, or every atom under frozen routing)
        // has its assembled `htt` diagonal entry ZEROED (see
        // `assignment_prior_grad_hdiag`), so the θ-adjoint third derivative of that
        // zeroed entry must also be zero. Mirror the IBP channel zeroing in
        // `ibp_assignment_third_channels`. The ThresholdGate/IBP branches below are
        // both diagonal (`diag_atom == wrt_atom`), so masking on `wrt_atom` suffices.
        if self.assignment.logit_is_fixed(wrt_atom) {
            return 0.0;
        }
        match self.assignment.mode {
            AssignmentMode::Softmax { .. } => {
                // #1038: the softmax entropy Hessian is now stored DENSE in
                // `block.htt` and its full θ-derivative `∂H_{k,j}/∂z_w` (diagonal
                // AND off-diagonal) is added inline in `logdet_theta_adjoint` from
                // the shared `row_dense_hessian_logit_derivative`. Returning the
                // diagonal contribution here too would double-count, so this
                // primitive is silent for softmax — the dense path is the single
                // source for value, logdet, and adjoint.
                0.0
            }
            AssignmentMode::ThresholdGate {
                temperature,
                threshold,
            } => {
                if diag_atom != wrt_atom {
                    return 0.0;
                }
                let logit = self.assignment.logits[[row, diag_atom]];
                if !crate::assignment::jumprelu_in_optimization_band(logit, threshold, temperature)
                {
                    return 0.0;
                }
                let inv_tau = 1.0 / temperature;
                let activation = gam_linalg::utils::stable_logistic((logit - threshold) * inv_tau);
                let slope = activation * (1.0 - activation);
                // #991 — this row's JumpReLU prior curvature in `htt` carries the
                // design weight `w_row`, so its θ-derivative carries the SAME
                // `w_row` (value/logdet/adjoint stay on one weighted branch).
                let w_row = self.row_loss_weights.as_deref().map_or(1.0, |w| w[row]);
                // #1415: P(ℓ)=λσ((ℓ−θ)/τ); P''(ℓ)=(λ/τ²)s(1−2a) so the third
                // derivative is P'''(ℓ)=(λ/τ³)·s·(1−6a+6a²), because
                // d/dℓ[s(1−2a)] = (1/τ)s[(1−2a)²−2s] = (1/τ)s(1−6a+6a²).
                w_row
                    * rho.lambda_sparse()
                    * slope
                    * (1.0 - 6.0 * activation + 6.0 * activation * activation)
                    * inv_tau
                    * inv_tau
                    * inv_tau
            }
            AssignmentMode::IBPMap { .. } => {
                // The assembled `htt` diagonal consumes
                // `IBPAssignmentPenalty::hessian_diag`, whose logit derivative
                // splits into a row-local direct-`z` channel and a global
                // empirical-`M_k` channel (π_k couples every row in column k).
                // This same-row primitive returns only the LOCAL direct-`z`
                // channel — and only on the matching logit (`diag_atom == w`),
                // since H_ik depends on no other row's z explicitly. The global
                // M_k channel is accumulated column-wise in
                // `logdet_theta_adjoint` (it needs the per-row selected-inverse
                // diagonals), so adding it here would double-count.
                if diag_atom != wrt_atom {
                    return 0.0;
                }
                match ibp_channels {
                    Some(ch) => ch.local_logit_third[row * ch.k_max + diag_atom],
                    None => 0.0,
                }
            }
            // Unreachable in practice: every TopK logit is `logit_is_fixed`, so
            // the mask above already returned 0.0 (no prior, no free logits).
            AssignmentMode::TopK { .. } => 0.0,
        }
    }

    pub(crate) fn ard_majorized_hessian_derivative(
        &self,
        rho: &SaeManifoldRho,
        row: usize,
        atom: usize,
        axis: usize,
    ) -> f64 {
        if rho.log_ard[atom].is_empty() {
            return 0.0;
        }
        let alpha = SaeManifoldRho::stable_exp_strength(rho.log_ard[atom][axis]);
        let periods = self.assignment.coords[atom].effective_axis_periods();
        let t = self.assignment.coords[atom].row(row)[axis];
        let prior = ArdAxisPrior::eval(alpha, t, periods[axis]);
        if prior.hess <= 0.0 {
            return 0.0;
        }
        match periods[axis] {
            None => 0.0,
            Some(period) => {
                let kappa = std::f64::consts::TAU / period;
                // HT row weighting: the assembled majorizer whose t-derivative this
                // feeds into the ½log|H| θ-adjoint is `w_row·max(V'',0)` (full
                // `w_row`, added directly to `htt` — NOT via the √w jet seam), so on
                // the positive branch its coordinate derivative is
                // `w_row·d/dt[α cos κt] = w_row·(−ακ sin κt)`. The data-fit `dH/dθ`
                // terms sharing this diagonal already carry full `w` (each is a
                // product of two √w-scaled jets, so √w·√w = w), so the correct single
                // factor for this prior term is likewise full `w_row`. `None`
                // weights ⇒ w_row = 1, bit-for-bit the historical derivative.
                let w_row = self.row_loss_weights.as_deref().map_or(1.0, |w| w[row]);
                -w_row * alpha * kappa * (kappa * t).sin()
            }
        }
    }

    pub fn outer_rho_gradient_ift_rhs(
        &self,
        rho: &SaeManifoldRho,
        j: usize,
        cache: &ArrowFactorCache,
    ) -> Result<SaeArrowVector, String> {
        let n_params = rho.to_flat().len();
        if j >= n_params {
            return Err(format!(
                "outer_rho_gradient_ift_rhs: coordinate {j} outside rho dim {n_params}"
            ));
        }
        let mut t = Array1::<f64>::zeros(cache.delta_t_len());
        let mut beta = Array1::<f64>::zeros(cache.k);
        if rho.sparse_flat_index() == Some(j) {
            let assignment_grad =
                crate::assignment::assignment_prior_log_strength_target_mixed_weighted(
                    &self.assignment,
                    rho,
                    self.row_loss_weights.as_deref(),
                )?;
            let k_atoms = self.k_atoms();
            let assignment_dim = self.assignment.assignment_coord_dim();
            for row in 0..self.n_obs() {
                let base = cache.row_offsets[row];
                let assignment_base = row * k_atoms;
                match self.last_row_layout {
                    Some(ref layout) => {
                        // #Bug1: assignment log-strength gradient lands on FREE logit
                        // slots only; softmax's reference atom has none (matching the
                        // dense `0..assignment_dim` = K−1 branch).
                        for (slot, &atom) in layout.logit_atoms[row].iter().enumerate() {
                            t[base + slot] = assignment_grad[assignment_base + atom];
                        }
                    }
                    None => {
                        for free_idx in 0..assignment_dim {
                            t[base + free_idx] = assignment_grad[assignment_base + free_idx];
                        }
                    }
                }
            }
        } else if (rho.smooth_flat_start()..rho.smooth_flat_start() + rho.log_lambda_smooth.len())
            .contains(&j)
        {
            // #1556: this layout-derived coordinate is one atom's smoothness
            // strength. `∂(penalty)/∂log λ_k = λ_k·S_k C_k` touches ONLY
            // atom `k`'s decoder block; every other atom's RHS is zero.
            let target_atom = j - rho.smooth_flat_start();
            let lambda = rho.lambda_smooth_for(target_atom);
            let frames_active = self.last_frames_active && cache.k == self.factored_border_dim();
            let offsets = if frames_active {
                self.factored_beta_offsets()
            } else {
                self.beta_offsets()
            };
            let atom = &self.atoms[target_atom];
            let m = atom.basis_size();
            let coeffs = if frames_active {
                match &atom.decoder_frame {
                    Some(frame) => frame.project_decoder(atom.decoder_coefficients.view())?,
                    None => atom.decoder_coefficients.clone(),
                }
            } else {
                atom.decoder_coefficients.clone()
            };
            let r = coeffs.ncols();
            let off = offsets[target_atom];
            for mu in 0..m {
                for channel in 0..r {
                    let mut acc = 0.0_f64;
                    for nu in 0..m {
                        let s_sym =
                            0.5 * (atom.smooth_penalty[[mu, nu]] + atom.smooth_penalty[[nu, mu]]);
                        acc += s_sym * coeffs[[nu, channel]];
                    }
                    beta[off + mu * r + channel] = lambda * acc;
                }
            }
        } else {
            // ARD coordinate `j`. `ard_flat_index` maps `(atom, axis)` onto the
            // flat coordinate for both parameterizations; a shared axis is owned
            // by SEVERAL atoms, and the RHS for that one outer coordinate is the
            // SUM of each owning atom's `∂g/∂log α_{atom,axis}` block (chain rule
            // through the broadcast). Those blocks land in disjoint per-atom row
            // slots of `t`, so accumulate every matching atom rather than
            // returning on the first. In `PerAtom` mode exactly one `(atom, axis)`
            // matches, reproducing the historical single-atom RHS.
            for atom in 0..rho.log_ard.len() {
                for axis in 0..rho.log_ard[atom].len() {
                    if rho.ard_flat_index(atom, axis) != j {
                        continue;
                    }
                    let alpha = SaeManifoldRho::stable_exp_strength(rho.log_ard[atom][axis]);
                    let periods = self.assignment.coords[atom].effective_axis_periods();
                    let row_w = self.row_loss_weights.as_deref();
                    for row in 0..self.n_obs() {
                        let row_t = self.assignment.coords[atom].row(row);
                        let prior = ArdAxisPrior::eval(alpha, row_t[axis], periods[axis]);
                        let Some(pos) = sae_coord_penalty_offset(
                            self.last_row_layout.as_ref(),
                            self.assignment.coord_offsets()[atom] + axis,
                            row,
                            atom,
                        ) else {
                            continue;
                        };
                        // HT row weighting: this RHS is `∂g/∂log α` of the inner-MAP
                        // stationarity gradient `g`, and the assembly writes that
                        // gradient as `w_row·V'` (full `w_row`, `construction_arrow_schur_assembly.rs`
                        // gt seam). The IFT operator `H` it feeds carries full `w_row`
                        // on this coordinate diagonal (`w·(D_data + prior'')`), so the
                        // RHS must carry the SAME full `w_row` to stay consistent — `V`
                        // is linear in α so `∂(w·V')/∂log α = w·V'`. `None` ⇒ w_row = 1,
                        // bit-for-bit the historical RHS.
                        let w_row = row_w.map_or(1.0, |w| w[row]);
                        t[cache.row_offsets[row] + pos] += w_row * prior.grad;
                    }
                }
            }
        }
        Ok(SaeArrowVector { t, beta })
    }

    /// #2231 — the crosscoder block coordinate's IFT RHS
    /// `∂g/∂log λ_ℓ = −½·Jᵀ_M Z̃^{(ℓ)}`, where `g` is the inner stationarity
    /// gradient, `Z̃^{(ℓ)}` is the CURRENTLY-SCALED stacked target masked to
    /// block `ℓ`'s columns, and `Jᵀ_M` is the same metric-whitened,
    /// `√w`-weighted data Jacobian the assembly's `gt = J̃ᵀẽ` uses (the target
    /// enters `g` only through the data residual `r̃ = f − Z̃`, and
    /// `∂Z̃_ℓ/∂log λ_ℓ = ½·Z̃_ℓ`). Feeding this RHS through
    /// `solve_exact_stationarity` gives the block coordinate the SAME
    /// `−½·Γᵀθ̂_ρ` Laplace adjoint every other ρ coordinate carries — without
    /// it the block gradient differentiates a fictitious criterion in which
    /// the fitted state is held fixed (#2087 desync class).
    pub(crate) fn crosscoder_block_ift_rhs(
        &self,
        cache: &ArrowFactorCache,
        target: ArrayView2<'_, f64>,
        col_range: std::ops::Range<usize>,
    ) -> Result<SaeArrowVector, String> {
        let n = self.n_obs();
        let p = self.output_dim();
        if target.nrows() != n || target.ncols() != p {
            return Err(format!(
                "crosscoder_block_ift_rhs: target shape ({}, {}) != ({n}, {p})",
                target.nrows(),
                target.ncols()
            ));
        }
        if col_range.end > p || col_range.start >= col_range.end {
            return Err(format!(
                "crosscoder_block_ift_rhs: block columns {col_range:?} outside output dim {p}"
            ));
        }
        let mut t = Array1::<f64>::zeros(cache.delta_t_len());
        let mut beta = Array1::<f64>::zeros(cache.k);
        let second_jets = self.atom_second_jets()?;
        let border = self.border_channels_for_cache(cache)?;
        let whiten = self.whiten_logdet_row_jets();
        let mut jet_window: std::collections::VecDeque<SaeRowJets> =
            std::collections::VecDeque::new();
        let mut jet_window_next = 0usize;
        for row in 0..n {
            let base = cache.row_offsets[row];
            if jet_window.is_empty() {
                jet_window_next = self.refill_jet_window(
                    jet_window_next,
                    cache,
                    &second_jets,
                    &border,
                    &mut jet_window,
                )?;
            }
            let mut jets = jet_window
                .pop_front()
                .ok_or_else(|| "crosscoder_block_ift_rhs: empty jet window".to_string())?;
            if whiten {
                self.apply_whiten_to_logdet_row_jets(row, &mut jets)?;
            }
            // `−½·√w·Z̃` on the block's columns, zero elsewhere; whitened by the
            // SAME row metric the jets carry so the product reconstructs
            // `−½·w·Jᵀ M Z̃^{(ℓ)}` (the jets already hold one `√w`/`Uᵀ` factor).
            let sqrt_w = self
                .row_loss_weights
                .as_deref()
                .map_or(1.0, |w| w[row].sqrt());
            let mut v: Vec<f64> = (0..p)
                .map(|col| {
                    if col_range.contains(&col) {
                        sqrt_w * target[[row, col]]
                    } else {
                        0.0
                    }
                })
                .collect();
            if whiten {
                let metric = self.row_metric.as_ref().ok_or_else(|| {
                    "crosscoder_block_ift_rhs: whitening metric absent".to_string()
                })?;
                Self::whiten_logdet_metric_vec(metric, row, p, &mut v)?;
            }
            for (var_idx, first) in jets.first.iter().enumerate() {
                t[base + var_idx] = -0.5 * sae_dot(first, &v);
            }
            for (channel_pos, channel) in border.iter().enumerate() {
                beta[channel.index] += -0.5 * sae_dot(&jets.beta[channel_pos], &v);
            }
        }
        Ok(SaeArrowVector { t, beta })
    }

    fn whiten_logdet_metric_vec(
        metric: &gam_problem::RowMetric,
        row: usize,
        p: usize,
        values: &mut Vec<f64>,
    ) -> Result<(), String> {
        if values.len() != p {
            return Err(format!(
                "logdet_theta_adjoint: row jet channel length {} != output dim {p}",
                values.len()
            ));
        }
        let rank = metric.metric_rank();
        let mut whitened = vec![0.0_f64; rank];
        for rank_col in 0..rank {
            let mut acc = 0.0_f64;
            for out_col in 0..p {
                acc += metric.factor_entry(row, out_col, rank_col) * values[out_col];
            }
            whitened[rank_col] = acc;
        }
        *values = whitened;
        Ok(())
    }

    /// Whiten every log-det row-jet channel by the row metric factor
    /// (`values ← Uᵀ values`), matching the assembly's whitened likelihood
    /// Hessian. Applies at any rank (full-rank ⇒ `rank == p`, length preserved;
    /// low-rank ⇒ `rank < p`, channels shrink to the whitened dim). Gated by
    /// [`whiten_logdet_row_jets`] at the call sites.
    fn apply_whiten_to_logdet_row_jets(
        &self,
        row: usize,
        jets: &mut SaeRowJets,
    ) -> Result<(), String> {
        let metric = self
            .row_metric
            .as_ref()
            .ok_or_else(|| "logdet_theta_adjoint: whitening metric absent".to_string())?;
        let p = self.output_dim();
        for first in jets.first.iter_mut() {
            Self::whiten_logdet_metric_vec(metric, row, p, first)?;
        }
        for second_row in jets.second.iter_mut() {
            for second in second_row.iter_mut() {
                Self::whiten_logdet_metric_vec(metric, row, p, second)?;
            }
        }
        for beta in jets.beta.iter_mut() {
            Self::whiten_logdet_metric_vec(metric, row, p, beta)?;
        }
        for beta_deriv_row in jets.beta_deriv.iter_mut() {
            for beta_deriv in beta_deriv_row.iter_mut() {
                Self::whiten_logdet_metric_vec(metric, row, p, beta_deriv)?;
            }
        }
        for beta_l_deriv_row in jets.beta_l_deriv.iter_mut() {
            for beta_l_deriv in beta_l_deriv_row.iter_mut() {
                Self::whiten_logdet_metric_vec(metric, row, p, beta_l_deriv)?;
            }
        }
        Ok(())
    }

    pub(crate) fn softmax_data_weight_product_logit_factor(
        assignments: &[f64],
        atom_a: usize,
        atom_b: usize,
        atom_w: usize,
        inv_tau: f64,
    ) -> f64 {
        let a_w = assignments[atom_w];
        let left = if atom_w == atom_a { 1.0 } else { 0.0 } - a_w;
        let right = if atom_w == atom_b { 1.0 } else { 0.0 } - a_w;
        (left + right) * inv_tau
    }

    pub(crate) fn logdet_theta_adjoint(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        solver: &DeflatedArrowSolver<'_>,
    ) -> Result<SaeArrowVector, String> {
        self.logdet_theta_adjoint_for_block(rho, cache, solver, true)
    }

    /// `Γ_tt = ∂_theta Σ_i log|H_tt^(i)|`, the state derivative of the
    /// coordinate-block logdet removed by the canonical rank-charge criterion.
    pub(crate) fn coordinate_block_logdet_theta_adjoint(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        solver: &DeflatedArrowSolver<'_>,
    ) -> Result<SaeArrowVector, String> {
        self.logdet_theta_adjoint_for_block(rho, cache, solver, false)
    }

    fn logdet_theta_adjoint_for_block(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        solver: &DeflatedArrowSolver<'_>,
        joint_block: bool,
    ) -> Result<SaeArrowVector, String> {
        // Γ_a = tr(H⁻¹ ∂H/∂θ_a) over the inner variables θ (#1006). `H` here is
        // the SAME object the evidence factor builds — Gauss-Newton data
        // curvature plus the prior majorizers / `hessian_diag` diagonals the
        // Newton/Schur Cholesky factorizes — so each block's θ-derivative channel
        // is differentiated on the criterion's own branch (no value/gradient
        // desync). The IBP-MAP assignment prior is the one block whose
        // `hessian_diag` couples every row in a column through the plug-in
        // empirical mass `M_k = Σ_i z_ik`; its logit derivative therefore has a
        // row-local channel (handled inline via
        // `assignment_prior_hdiag_derivative_entry`) and a cross-row channel
        // (accumulated column-wise after the row loop, below).
        if cache.arrow_log_det().is_none() {
            return Err(
                "logdet_theta_adjoint: cache lacks an authoritative joint-Hessian log-det \
                 for the selected-inverse operator"
                    .to_string(),
            );
        }
        let n = self.n_obs();
        let total_t = cache.delta_t_len();
        let mut gamma_t = Array1::<f64>::zeros(total_t);
        let mut gamma_beta = Array1::<f64>::zeros(cache.k);
        let second_jets = self.atom_second_jets()?;
        let border = self.border_channels_for_cache(cache)?;
        // #932 FRONT C: plain-arrow `(H⁻¹)_ββ = S⁻¹` formed once from the cached
        // Schur factor; gauge / #1038 cross-row Woodbury fall back to the per-β
        // `solve` loop where the row-local Takahashi blocks are not valid.
        let fast_selected = joint_block && solver.plain_selected_inverse_available();
        let beta_inv = if joint_block {
            Self::selected_inverse_beta_block(solver, cache, fast_selected, "logdet_theta_adjoint")?
        } else {
            Array2::<f64>::zeros((cache.k, cache.k))
        };
        // IBP `hessian_diag` logit third-derivative channels (#1006). The full
        // IBP Hessian also has per-column cross-row rank-one terms
        // `H_(i,k),(j,k) = d_k·J_ik·J_jk`; these ARE carried in `H` via the #1038
        // Woodbury source (`IbpCrossRowSource`, construction.rs:4710-4752), the
        // ρ-trace differentiates them (#1416,
        // `assignment_log_strength_hessian_trace`), AND this θ-adjoint now
        // differentiates them exactly too: the empirical-`M_k` channel below
        // contracts the shared-mass coupling of the DIAGONAL curvature, and the
        // cross-row Woodbury pass (further below, using `cross_row_dd` and
        // `logit_curvature`) contracts the `∂/∂ℓ_w (d_k·J_ik·J_jk)` rank-one
        // derivative — so value, logdet, ρ-trace, and θ-adjoint all differentiate
        // the one operator `H = H₀ + Σ_k d_k u_k u_kᵀ`.
        // gam#2144/#1038: the assembly PSD-majorizes the IBP curvature
        // UNCONDITIONALLY, so the θ-adjoint differentiates the MAJORIZED channels
        // (clamped `cross_row_d`, gated `cross_row_dd`/`m_channel`/
        // `local_logit_third`) — the exact derivative of the operator the evidence
        // log-det factors, on every IBP path. Anything else desyncs the outer-REML
        // gradient from the evidence (#2087).
        // gam#2144: whitening of the row jets tracks `whitens_likelihood()` at ANY
        // rank (the assembly whitens `JᵀU UᵀJ` for full- and low-rank alike) and is
        // independent of the PSD majorization.
        let whiten_row_jets = self.whiten_logdet_row_jets();
        let ibp_channels = ibp_assignment_third_channels_weighted(
            &self.assignment,
            rho,
            true,
            self.row_loss_weights.as_deref(),
        )?;
        let k_atoms = self.k_atoms();
        // #1038 softmax entropy: the dense per-row entropy Hessian written into
        // `block.htt` has off-diagonal logit terms whose θ-derivative the adjoint
        // must contract too (not just the diagonal). Build the SAME penalty +
        // `scale = λ/τ²` the assembly uses so value/logdet/adjoint differentiate
        // one operator. `None` for non-softmax modes (their diagonal/cross-row
        // channels are handled by `assignment_prior_hdiag_derivative_entry` and
        // the IBP column pass).
        let softmax_dense_adjoint: Option<(
            gam_terms::analytic_penalties::SoftmaxAssignmentSparsityPenalty,
            f64,
        )> = match self.assignment.mode {
            AssignmentMode::Softmax {
                temperature,
                sparsity,
            } if k_atoms > 1 => {
                let inv_tau = 1.0 / temperature;
                let scale = rho.lambda_sparse() * sparsity * inv_tau * inv_tau;
                Some((
                    gam_terms::analytic_penalties::SoftmaxAssignmentSparsityPenalty::new(
                        k_atoms,
                        temperature,
                    ),
                    scale,
                ))
            }
            _ => None,
        };
        // Per active logit site: row, atom, global t-index, raw selected-inverse
        // diagonal. The raw diagonal drives the empirical-M contraction and the
        // cross-row Woodbury self-subtraction. The cached unit-diagonal
        // Daleckii-Krein weight lets the later empirical-M pass correct only the
        // no-self row-base derivative, leaving the Woodbury self derivative raw.
        #[derive(Clone, Copy)]
        struct IbpLogitSite {
            row: usize,
            atom: usize,
            t_index: usize,
            raw_diag: f64,
            no_self_diag_deflation_weight: f64,
        }
        let mut ibp_logit_sites: Vec<IbpLogitSite> = Vec::new();

        // #1557 — reuse one K-sized scratch row across all N rows (alias-free).
        let mut assignments = Array1::<f64>::zeros(self.k_atoms());
        // #932 SIMD: jets are built in aligned 4-row SIMD batches through a
        // bounded (≤4-row) look-ahead window; unaligned / non-softmax / remainder
        // rows fall back to the scalar per-row path (bit-identical either way).
        let mut jet_window: std::collections::VecDeque<SaeRowJets> =
            std::collections::VecDeque::new();
        let mut jet_window_next = 0usize;
        // Hoisted RHS scratch for the gauge/Woodbury per-row solve fallback.
        let mut rhs_t_scratch = Array1::<f64>::zeros(total_t);
        let rhs_beta_zero = Array1::<f64>::zeros(cache.k);
        for row in 0..n {
            let q = cache.row_dims[row];
            let base = cache.row_offsets[row];
            let a_scratch = assignments.as_slice_mut().expect("contiguous scratch");
            self.assignment.try_assignments_row_into(row, a_scratch)?;
            if jet_window.is_empty() {
                jet_window_next = self.refill_jet_window(
                    jet_window_next,
                    cache,
                    &second_jets,
                    &border,
                    &mut jet_window,
                )?;
            }
            let mut jets = jet_window
                .pop_front()
                .expect("jet window must be non-empty");
            if whiten_row_jets {
                self.apply_whiten_to_logdet_row_jets(row, &mut jets)?;
            }

            // #932 FRONT C: row-local Takahashi on the plain arrow; per-row
            // full-system `solve` loop under gauge / cross-row Woodbury.
            let (inv_vv, inv_vbeta) = if joint_block {
                Self::selected_inverse_row_blocks_or_solve(
                    &SelectedInverseRowSolve {
                        solver,
                        cache,
                        beta_inv: &beta_inv,
                        fast_selected,
                        rhs_beta_zero: rhs_beta_zero.view(),
                        context: "logdet_theta_adjoint",
                    },
                    row,
                    base,
                    q,
                    &mut rhs_t_scratch,
                )?
            } else {
                let factor = cache.undamped_factor(row);
                let mut inverse = Array2::<f64>::zeros((q, q));
                let mut unit = Array1::<f64>::zeros(q);
                for col in 0..q {
                    unit[col] = 1.0;
                    let solved = cholesky_solve_vector(factor, unit.view());
                    unit[col] = 0.0;
                    for inverse_row in 0..q {
                        inverse[[inverse_row, col]] = solved[inverse_row];
                    }
                }
                (inverse, Array2::<f64>::zeros((q, cache.k)))
            };

            // Per-row UNIT-stiffness deflated directions: the selected inverse
            // `inv_vv` is the DEFLATED inverse (it assigns `1/λ̃ = 1` to each
            // `vᵢ`), so every `inv_vv`-weighted t–t contraction of `∂H/∂θ_w`
            // below spuriously contracts the RAW derivative where the re-deflating
            // criterion uses the deflation-map derivative `DΦ`. The kept-subspace Γ
            // subtracts `tr(inv_vv·(D − DΦ[D]))` over the t–t block via the same
            // Daleckii–Krein helper the ρ-traces use (the t–β / β–β blocks are not
            // deflated). IBP cross-row Woodbury caches factor the no-self base, so
            // the correction matrix below removes the local self derivative before
            // applying `DΦ`; the full self/off-row rank-one derivative stays in the
            // ordinary raw contractions.
            let defl_dirs = cache
                .deflated_row_directions
                .get(row)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let defl_spectrum = cache
                .deflation_row_spectra
                .get(row)
                .and_then(Option::as_ref);

            // Record each active logit's column, global t-index, and raw
            // selected-inverse diagonal for the IBP cross-row passes. Also cache
            // the per-slot Daleckii-Krein weight for a unit diagonal derivative:
            // the empirical-M `m_channel` later splits into a no-self row-base
            // derivative plus a rank-one self derivative, and only the no-self
            // piece belongs under the row deflation map.
            if ibp_channels.is_some() {
                for (pos, var) in jets.vars.iter().enumerate() {
                    if let SaeLocalRowVar::Logit { atom } = *var {
                        let raw_diag = inv_vv[[pos, pos]];
                        let no_self_diag_deflation_weight = if defl_dirs.is_empty() {
                            0.0
                        } else {
                            let mut unit_diag = Array2::<f64>::zeros((q, q));
                            unit_diag[[pos, pos]] = 1.0;
                            Self::deflation_block_correction(
                                &inv_vv,
                                &unit_diag,
                                defl_dirs,
                                defl_spectrum,
                            )
                        };
                        ibp_logit_sites.push(IbpLogitSite {
                            row,
                            atom,
                            t_index: base + pos,
                            raw_diag,
                            no_self_diag_deflation_weight,
                        });
                    }
                }
            }

            // #1419: when `w` is a logit and the assignment is softmax, the per-row
            // Gershgorin majorizer `D = diag(Σ_j|H_kj|)` is what the assembly wrote
            // into `htt` (the genuine Loewner majorizer that replaces the indefinite
            // exact entropy Hessian). Its full θ-derivative `∂D_{k,k}/∂z_w` (diagonal;
            // `∂D_kk/∂z_w = Σ_j sign(H_kj)·∂H_kj/∂z_w`) is the SAME operator the
            // assembly and logdet now differentiate, so value and adjoint stay on ONE
            // exact branch. Compute it once per logit `w` and add it at every logit
            // pair `(a,b)` below. The diagonal softmax case is therefore handled here,
            // NOT in `assignment_prior_hdiag_derivative_entry` (which returns 0 for
            // softmax to avoid double-counting).
            // #1410: the softmax majorizer θ-derivative `∂D_kk/∂z_w` is DIAGONAL
            // (`D` is diagonal), and the compact adjoint reads it only for this
            // row's `≤ top_k` active atoms. Compute the needed diagonal entry
            // directly from the softmax row `a` (= `assignments`, in hand) via
            // `active_softmax_majorizer_logit_derivative_entry`, instead of the old
            // per-(row, logit) full `K×K` `row_psd_majorizer_logit_derivative`
            // allocation. `m = Σ_j a_j l_j` is shared across all `(w, k)` pairs of
            // the row, so compute it once. `inv_tau` carries the softmax `∂a/∂z`
            // convention.
            let softmax_adjoint_row: Option<(&[f64], f64, f64, f64)> =
                match (softmax_dense_adjoint.as_ref(), self.assignment.mode) {
                    (Some((_penalty, scale)), AssignmentMode::Softmax { temperature, .. }) => {
                        let a = assignments
                            .as_slice()
                            .expect("softmax assignments row must be contiguous");
                        let m = softmax_majorizer_log_mean(a);
                        Some((a, m, *scale, 1.0 / temperature))
                    }
                    _ => None,
                };
            // #991 — the softmax majorizer written into `htt` carries this row's
            // design weight `w_row`, so its θ-derivative below carries the SAME
            // `w_row`; the data-curvature θ-derivative already carries `w` through
            // the √w-scaled jets, and the IBP prior derivative
            // (`assignment_prior_hdiag_derivative_entry`) is left unweighted.
            let w_row_prior = self.row_loss_weights.as_deref().map_or(1.0, |w| w[row]);
            for w in 0..q {
                let mut gamma = 0.0_f64;
                // The active logit `w` differentiates against; `None` unless this
                // slot is a softmax logit on the softmax path.
                let softmax_d_dw: Option<(&[f64], f64, f64, f64, usize)> =
                    match (softmax_adjoint_row, jets.vars[w]) {
                        (Some((a, m, scale, inv_tau)), SaeLocalRowVar::Logit { atom: atom_w }) => {
                            Some((a, m, scale, inv_tau, atom_w))
                        }
                        _ => None,
                    };
                let mut deflated_base_dh_mat = Array2::<f64>::zeros((q, q));
                for a in 0..q {
                    for b in 0..q {
                        let mut dh = match (softmax_d_dw, jets.vars[a], jets.vars[b]) {
                            (
                                Some((a_soft, _m, _scale, inv_tau, atom_w)),
                                SaeLocalRowVar::Coord { atom: atom_a, .. },
                                SaeLocalRowVar::Coord { atom: atom_b, .. },
                            ) => {
                                let h_ab = sae_dot(&jets.first[a], &jets.first[b]);
                                h_ab * Self::softmax_data_weight_product_logit_factor(
                                    a_soft, atom_a, atom_b, atom_w, inv_tau,
                                )
                            }
                            _ => {
                                sae_dot(&jets.second[a][w], &jets.first[b])
                                    + sae_dot(&jets.first[a], &jets.second[b][w])
                            }
                        };
                        // `∂D/∂z_w` is diagonal, so it contributes only when the two
                        // logit slots are the SAME atom (`atom_a == atom_b`).
                        if let (
                            Some((a_soft, m, scale, inv_tau, _atom_w)),
                            SaeLocalRowVar::Logit { atom: atom_a },
                            SaeLocalRowVar::Logit { atom: atom_b },
                        ) = (softmax_d_dw, jets.vars[a], jets.vars[b])
                        {
                            if atom_a == atom_b {
                                dh += w_row_prior
                                    * active_softmax_majorizer_logit_derivative_entry(
                                        a_soft, atom_a, _atom_w, m, scale, inv_tau,
                                    );
                            }
                        }
                        if a == b {
                            dh += match jets.vars[a] {
                                SaeLocalRowVar::Logit { atom } => self
                                    .assignment_prior_hdiag_derivative_entry(
                                        rho,
                                        row,
                                        atom,
                                        jets.vars[w],
                                        ibp_channels.as_ref(),
                                    ),
                                SaeLocalRowVar::Coord { atom, axis } if a == w => {
                                    self.ard_majorized_hessian_derivative(rho, row, atom, axis)
                                }
                                _ => 0.0,
                            };
                        }
                        let mut ibp_self_derivative = 0.0_f64;
                        // #2144: the row factor that spectral deflation conditions is
                        // the IBP no-self base `H0'`, because
                        // `solve_arrow_newton_step_with_options` downdates the
                        // row-local `d_k J_ik^2` self curvature before factoring and
                        // re-adds the full rank-one column through Woodbury. The trace
                        // above still contracts the derivative of the full diagonal
                        // against the full selected inverse; only the Daleckii-Krein
                        // deflation-map correction must see the derivative of the
                        // actually deflated row block. Therefore remove just the
                        // direct-local derivative of the downdated IBP self term from
                        // the matrix passed to `deflation_block_correction`. The
                        // empirical-M and off-row Woodbury channels remain in their
                        // existing passes.
                        if let (
                            Some(channels),
                            SaeLocalRowVar::Logit { atom: diag_atom },
                            SaeLocalRowVar::Logit { atom: wrt_atom },
                        ) = (ibp_channels.as_ref(), jets.vars[a], jets.vars[w])
                        {
                            if a == b && diag_atom == wrt_atom {
                                let idx = row * k_atoms + diag_atom;
                                ibp_self_derivative = 2.0
                                    * channels.cross_row_d[diag_atom]
                                    * channels.z_jac[idx]
                                    * channels.logit_curvature[idx];
                            }
                        }
                        let deflated_base_dh = dh - ibp_self_derivative;
                        if !joint_block {
                            // `htt_half` is formed from the factored IBP no-self
                            // row base. Its raw contraction therefore excludes
                            // the local diagonal piece reintroduced only by the
                            // joint cross-row Woodbury carrier.
                            dh = deflated_base_dh;
                        }
                        deflated_base_dh_mat[[a, b]] = deflated_base_dh;
                        gamma += inv_vv[[b, a]] * dh;
                    }
                }
                if !defl_dirs.is_empty() {
                    // The row factor/log-det operator is the spectrally
                    // conditioned `Φ(H_tt)`, while the local theta channels above
                    // assemble the raw row derivative `D`. Subtract
                    // `tr(inv_vv · (D - DΦ[D]))` for every deflated row, including
                    // the low-rank IBP majorizer path, so the theta adjoint
                    // differentiates the same operator as `arrow_log_det`,
                    // `apply_cached_arrow_hessian`, and the selected inverse.
                    gamma -= Self::deflation_block_correction(
                        &inv_vv,
                        &deflated_base_dh_mat,
                        defl_dirs,
                        defl_spectrum,
                    );
                }
                for a in 0..q {
                    for (beta_pos, channel) in border.iter().enumerate() {
                        let dh = sae_dot(&jets.second[a][w], &jets.beta[beta_pos])
                            + sae_dot(&jets.first[a], &jets.beta_deriv[w][beta_pos]);
                        gamma += 2.0 * inv_vbeta[[a, channel.index]] * dh;
                    }
                }
                for (beta_i, channel_i) in border.iter().enumerate() {
                    for (beta_j, channel_j) in border.iter().enumerate() {
                        let dh = sae_dot(&jets.beta_deriv[w][beta_i], &jets.beta[beta_j])
                            + sae_dot(&jets.beta[beta_i], &jets.beta_deriv[w][beta_j]);
                        gamma += beta_inv[[channel_i.index, channel_j.index]] * dh;
                    }
                }
                gamma_t[base + w] = gamma;
            }

            for (w_beta_pos, w_channel) in border.iter().enumerate() {
                let mut gamma = 0.0_f64;
                let mut dh_mat = Array2::<f64>::zeros((q, q));
                for a in 0..q {
                    for b in 0..q {
                        let dh = sae_dot(&jets.beta_l_deriv[a][w_beta_pos], &jets.first[b])
                            + sae_dot(&jets.first[a], &jets.beta_l_deriv[b][w_beta_pos]);
                        dh_mat[[a, b]] = dh;
                        gamma += inv_vv[[b, a]] * dh;
                    }
                }
                if !defl_dirs.is_empty() {
                    gamma -= Self::deflation_block_correction(
                        &inv_vv,
                        &dh_mat,
                        defl_dirs,
                        defl_spectrum,
                    );
                }
                for a in 0..q {
                    for (beta_pos, channel) in border.iter().enumerate() {
                        let dh = sae_dot(&jets.beta_l_deriv[a][w_beta_pos], &jets.beta[beta_pos]);
                        gamma += 2.0 * inv_vbeta[[a, channel.index]] * dh;
                    }
                }
                gamma_beta[w_channel.index] += gamma;
            }
        }

        // IBP cross-row empirical-`M_k` channel of Γ (#1006). The assembled
        // diagonal H_ik consumes `hessian_diag`, whose dependence on the column
        // mass M_k = Σ_i z_ik couples every row in a column. Differentiating
        // tr(H⁻¹ ∂H/∂ℓ_wk) on that shared branch:
        //   Γ_wk += [ Σ_i (H⁻¹)_ik,ik · ∂_M H_ik ] · J_wk = C_k · J_wk,
        // where ∂_M H_ik = `m_channel[i*K+k]` and J_wk = `z_jac[w*K+k]`. The
        // row-local direct-`z` channel was already added inline above; this pass
        // owns the empirical-mass branch. The no-self part of `m_channel` is a
        // derivative of the deflated row base `H₀'`, so it receives the same
        // Daleckii-Krein `DΦ` correction as the row-local channel; the rank-one
        // self part stays raw and is paired with the off-row Woodbury derivative.
        if let Some(channels) = ibp_channels.as_ref() {
            let mut col_coeff = vec![0.0_f64; k_atoms];
            for site in &ibp_logit_sites {
                let idx = site.row * k_atoms + site.atom;
                let j = channels.z_jac[idx];
                let self_mass = channels.cross_row_dd[site.atom] * j * j;
                let no_self_mass = channels.m_channel[idx] - self_mass;
                let raw_mass = if joint_block {
                    channels.m_channel[idx]
                } else {
                    no_self_mass
                };
                col_coeff[site.atom] +=
                    site.raw_diag * raw_mass - site.no_self_diag_deflation_weight * no_self_mass;
            }
            for site in &ibp_logit_sites {
                let idx = site.row * k_atoms + site.atom;
                gamma_t[site.t_index] += col_coeff[site.atom] * channels.z_jac[idx];
            }
            if !joint_block {
                return Ok(SaeArrowVector {
                    t: gamma_t,
                    beta: gamma_beta,
                });
            }

            // #1416 / #1641: the EXACT cross-row Woodbury derivative of Γ. The
            // assembled `H` carries the per-column rank-one block
            // `W_k = d_k·u_k u_kᵀ` with `u_k` the J-weighted column indicator
            // (`u_k[slot(i,k)] = J_ik`) and `d_k = w·s'_k` (`cross_row_d[k]`). Both
            // `d_k` (through `M_k`) and the `u_k` entries (through `ℓ_ik`) depend on
            // the logits, so
            //   ∂W_k/∂ℓ_wk = dd_k·J_wk·u_k u_kᵀ
            //               + d_k·c_wk·(e_w u_kᵀ + u_k e_wᵀ),
            // where `dd_k = ∂d_k/∂M_k = w·s''_k` (`cross_row_dd[k]`),
            // `c_wk = ∂J_wk/∂ℓ_wk` (`logit_curvature`), and `e_w` is the unit
            // vector at row `w`'s logit-`k` slot.
            //
            // The θ-adjoint contracts the FULL trace `Γ_wk = tr(H⁻¹ ∂H/∂ℓ_wk)`
            // (NOT the `½ tr` the ρ-trace uses — `fixed_state_logdet` differentiates
            // the full `log|H|`, and the per-row blocks above contract `inv_vv·dh`
            // with no ½). Critically, the `i=j` self curvature `w·s'_k·J_ik²` of the
            // rank-one block lives on the assembled `htt` DIAGONAL `H_ik`, so its
            // derivative is ALREADY differentiated by the row-local
            // `local_logit_third` channel (direct-z, `i=w`) and the `m_channel`
            // column pass (via `M_k`) above. This Woodbury pass must therefore add
            // ONLY the off-diagonal `i≠j` remainder — otherwise the self term is
            // double-counted (the #1641 defect: the pre-fix pass summed the full
            // `u_k u_kᵀ` including `i=j`, AND carried the ρ-trace ½, AND dropped the
            // factor 2 on the symmetric `e_w u_kᵀ + u_k e_wᵀ` term). Excluding `i=j`
            // is also why this pass needs no deflation correction: it contracts only
            // DISTINCT rows, off any single-row `vᵢ`'s support (matching the
            // #1416 ρ-trace cross-row pass).
            //
            // Contracting `tr(H⁻¹ ∂W_k/∂ℓ_wk)` over `i≠j` only:
            //   Γ_wk += dd_k·J_wk·( u_kᵀ H⁻¹ u_k − Σ_i P_ii·J_ik² )       (term A)
            //         + 2·d_k·c_wk·( (H⁻¹ u_k)_{slot(w,k)} − P_ww·J_wk )  (term B),
            // where `P_ii = (H⁻¹)_{slot(i,k),slot(i,k)}` is the selected-inverse
            // diagonal recorded in `ibp_logit_sites`. The subtracted self pieces are
            // exactly the `i=j` terms the diagonal channels own. Both `u_kᵀ H⁻¹ u_k`
            // and `(H⁻¹ u_k)` come from ONE solve per column, `x_k = H⁻¹ u_k` — so
            // the adjoint differentiates the SAME `H = H₀ + Σ_k W_k` the
            // value/logdet use, closing the one-operator contract on the rank-one
            // block too.
            //
            // Group the column sites once (the layout is mode-agnostic: dense or
            // compact, `ibp_logit_sites` already carries each active logit's
            // global t-index AND its selected-inverse diagonal `G_ii`), then per
            // column build `u_k`, solve, and distribute the OFF-DIAGONAL remainder.
            //
            // #1416 FIX: the diagonal (`i = w`) parts of term A and term B are
            // ALREADY supplied — `diag(term A) = dd_k·J_w·Σ_i G_ii·J_i²` by the
            // `m_channel` column pass above (whose `m_channel = w·(s''·J² + s'·c)`
            // carries the `s''·J²` self piece), and `diag(term B) = 2·d_k·c_w·G_ww·J_w`
            // by the inline `local_logit_third` self channel (whose
            // `s'·2J·∂_z J` piece is exactly that). So this pass must add ONLY the
            // cross-row off-diagonal remainder; double-counting the diagonal here
            // (the pre-fix `0.5·dd·J·uᵀGu + d·c·x_w` form, which is neither the
            // full nor the off-diagonal value) desynced the θ-adjoint from the FD
            // of `log|H|`. The exact `tr(H⁻¹ ∂W_k/∂ℓ_wk)` is
            //   Γ_wk += dd_k·J_wk·(uᵀ G u − Σ_i G_ii·J_ik²)   (term A, off-diagonal)
            //         + 2·d_k·c_wk·((G u)_w − G_ww·J_wk)        (term B, off-diagonal),
            // with `uᵀGu = Σ_i J_ik·(Gu)_i`, `(Gu) = x_k = H⁻¹ u_k` from one solve,
            // and `G_ii` the per-site selected-inverse diagonal.
            let total_t = cache.delta_t_len();
            // The Woodbury pass reconstructs the off-diagonal `(H⁻¹)_ij` from the
            // deflated solve `x_k = H⁻¹ u_k` and subtracts the `i=j` self term; the
            // self term must use the RAW deflated diagonal (matching `x_k`), NOT the
            // Daleckii–Krein-corrected diagonal the `M_k` pass uses.
            let mut col_sites: Vec<Vec<(usize, usize, f64)>> = vec![Vec::new(); k_atoms];
            for site in &ibp_logit_sites {
                col_sites[site.atom].push((site.row, site.t_index, site.raw_diag));
            }
            // Hoisted RHS scratch: fill only this column's active slots, solve,
            // then clear exactly those slots — no per-column total_t zeroing.
            let mut rhs_t_scratch = Array1::<f64>::zeros(total_t);
            let rhs_beta_zero = Array1::<f64>::zeros(cache.k);
            for atom in 0..k_atoms {
                let d_k = channels.cross_row_d[atom];
                let dd_k = channels.cross_row_dd[atom];
                if col_sites[atom].is_empty() || (d_k == 0.0 && dd_k == 0.0) {
                    continue;
                }
                // u_k as a full t-RHS: J at each active logit-k slot.
                for &(row, t_index, _g) in &col_sites[atom] {
                    rhs_t_scratch[t_index] = channels.z_jac[row * k_atoms + atom];
                }
                let x_k = solver
                    .solve(rhs_t_scratch.view(), rhs_beta_zero.view())
                    .map_err(|err| {
                        format!("logdet_theta_adjoint: IBP cross-row Woodbury solve: {err}")
                    })?;
                // Clear this column's active slots for the next atom's RHS.
                for &(_row, t_index, _g) in &col_sites[atom] {
                    rhs_t_scratch[t_index] = 0.0;
                }
                // (JᵀH⁻¹J)_k = u_kᵀ x_k, and the diagonal `Σ_i G_ii·J_ik²` that the
                // `m_channel` pass already counted (subtract it from term A so this
                // pass holds only the off-diagonal `i ≠ j` remainder).
                let mut jt_hinv_j = 0.0_f64;
                let mut diag_jt_g_j = 0.0_f64;
                for &(row, t_index, g_ii) in &col_sites[atom] {
                    let j = channels.z_jac[row * k_atoms + atom];
                    jt_hinv_j += j * x_k.t[t_index];
                    diag_jt_g_j += g_ii * j * j;
                }
                let off_diag_a = jt_hinv_j - diag_jt_g_j;
                for &(row, t_index, g_ii) in &col_sites[atom] {
                    let j_wk = channels.z_jac[row * k_atoms + atom];
                    let c_wk = channels.logit_curvature[row * k_atoms + atom];
                    // term A (off-diagonal) + term B (off-diagonal); the inline /
                    // `m_channel` passes already added the diagonal parts.
                    let off_diag_b = x_k.t[t_index] - g_ii * j_wk;
                    gamma_t[t_index] += dd_k * j_wk * off_diag_a + 2.0 * d_k * c_wk * off_diag_b;
                }
            }
        }

        Ok(SaeArrowVector {
            t: gamma_t,
            beta: gamma_beta,
        })
    }

    /// #2080 matrix-free θ-adjoint: the SAME `Γ = tr(H⁻¹ ∂H/∂θ)` the dense
    /// [`Self::logdet_theta_adjoint`] assembles, reconstructed from the shared
    /// selected-inverse probe bundle `(z_j, S⁻¹ z_j)` instead of the dense
    /// `DeflatedArrowSolver` selected inverse — the last new-math channel of the
    /// wide-p surrogate. It never materializes the `K×K` reduced-Schur `S⁻¹`
    /// (the one massive-K-infeasible object the dense β–β loop reads); everything
    /// folds onto the bundle:
    ///
    /// With `A_i = undamped_factor(i)`, `G_i = A_i⁻¹ H_tβ^(i)`, and the Rademacher
    /// probe identity `E[z zᵀ] = I` (EXACT at the full-basis probe set `√k·e_j`),
    /// the arrow inverse blocks the dense adjoint contracts are unbiased outer
    /// products of the row probe images `w_l = G_i z_l`, `s_l = G_i (S⁻¹ z_l)`:
    /// ```text
    ///   (H⁻¹)_tt[i]  = A_i⁻¹ + G_i S⁻¹ G_iᵀ ,  (G_i S⁻¹ G_iᵀ)[a,b] ≈ (1/m)Σ_l w_l[a] s_l[b]
    ///   (H⁻¹)_tβ[i]  = −G_i S⁻¹           ,  (G_i S⁻¹)[a,c]      ≈ (1/m)Σ_l w_l[a] (S⁻¹z_l)[c]
    /// ```
    /// so the t–t (`q×q`) and t–β (`q×K`) blocks are materialized per row (feasible:
    /// `q` small, `q×K` matches the dense t–β cost) and the dense contraction code is
    /// reused verbatim. Only the β–β term `Σ_ij S⁻¹[i,j]·∂H_βiβj` (dense: the `O(K²)`
    /// `beta_inv` double loop) is refolded as `tr(S⁻¹·M)`:
    /// `Σ_ij S⁻¹[i,j](⟨bd_i,b_j⟩+⟨b_i,bd_j⟩) = (1/m)Σ_l (⟨Rd_l,P_l⟩+⟨R_l,Q_l⟩)` with
    /// `P_l=Σ_j z_l[c_j] b_j`, `R_l=Σ_i (S⁻¹z_l)[c_i] b_i`, `Q_l=Σ_j z_l[c_j] bd_j`,
    /// `Rd_l=Σ_i (S⁻¹z_l)[c_i] bd_i` (`b`=`beta` jet, `bd`=`beta_deriv` jet).
    ///
    /// # Scope: softmax / euclidean / non-cross-row (hard refuse otherwise)
    ///
    /// The bundle spans ONLY the reduced-Schur border (`cache.k`), so the outer
    /// products reconstruct exactly the NO-SELF base inverse `(H₀')⁻¹ = A_i⁻¹ +
    /// G_i S⁻¹ G_iᵀ`. That equals the full `H⁻¹` only when the cache carries no
    /// T-space rank-R correction; two regimes are therefore hard-refused (routed to
    /// the dense channel, same discipline as invariant #2):
    ///
    /// * **Per-row deflation** (`deflated_row_directions`): the Daleckii–Krein
    ///   correction `−tr(inv_vv·(D − DΦ[D]))` needs the DEFLATED block the plain-S⁻¹
    ///   bundle does not carry.
    /// * **IBP cross-row Woodbury** (`ibp_channels` / `cache.cross_row_woodbury`):
    ///   `W_k = d_k u_k u_kᵀ` lives in the T-block, layered onto `H₀'` as a rank-R
    ///   correction (`full_inverse_apply`), NOT in the border the bundle spans —
    ///   folding it matrix-free needs the bundle to additionally carry `H₀'⁻¹U`, a
    ///   reduced-Schur/bundle design step tracked separately.
    ///
    /// On the accepted (softmax / euclidean / non-cross-row) regimes base = full
    /// inverse, both refuse conditions are absent, and the from-probes and dense
    /// θ-adjoints agree exactly at full-basis probes — the FD gate's acceptance.
    pub(crate) fn logdet_theta_adjoint_from_probes(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        probes: &[Array1<f64>],
        sinv_probes: &[Array1<f64>],
    ) -> Result<SaeArrowVector, String> {
        if cache.arrow_log_det().is_none() {
            return Err(
                "logdet_theta_adjoint_from_probes: cache lacks an authoritative joint-Hessian \
                 log-det for the selected-inverse operator"
                    .to_string(),
            );
        }
        let k_border = cache.k;
        let m = probes.len();
        if k_border > 0 {
            if m == 0 || sinv_probes.len() != m {
                return Err(format!(
                    "logdet_theta_adjoint_from_probes: need matching non-empty probe/solve \
                     bundles, got {m} probes and {} solves",
                    sinv_probes.len()
                ));
            }
            for (label, set) in [("probe", probes), ("solve", sinv_probes)] {
                for (j, v) in set.iter().enumerate() {
                    if v.len() != k_border {
                        return Err(format!(
                            "logdet_theta_adjoint_from_probes: {label} {j} has length {} != \
                             border dim {k_border}",
                            v.len()
                        ));
                    }
                }
            }
        }
        let inv_m = if m > 0 { 1.0 / m as f64 } else { 0.0 };
        let n = self.n_obs();
        let total_t = cache.delta_t_len();
        let mut gamma_t = Array1::<f64>::zeros(total_t);
        let mut gamma_beta = Array1::<f64>::zeros(k_border);

        // Deflation hard-refuse (see the docstring): the plain-S⁻¹ bundle cannot
        // reconstruct the Daleckii–Krein correction, so any deflated row routes the
        // whole fit to the dense channel.
        for row in 0..n {
            if cache
                .deflated_row_directions
                .get(row)
                .is_some_and(|d| !d.is_empty())
            {
                return Err(format!(
                    "logdet_theta_adjoint_from_probes: row {row} carries deflation directions; \
                     the plain-S⁻¹ bundle cannot reconstruct the Daleckii–Krein correction — \
                     route this fit through the dense channel"
                ));
            }
        }

        // Cross-row IBP Woodbury hard-refuse. The bundle spans only the reduced-Schur
        // BORDER (the decoder-β channels, `cache.k`); the #1038 cross-row rank-one
        // curvature `W_k = d_k u_k u_kᵀ` lives in the T-block, layered onto the NO-SELF
        // base `H₀'` as a rank-R correction (`full_inverse_apply` / the
        // `subtract_inverse_diagonal` step). The border-only outer products reconstruct
        // exactly `A_i⁻¹ + G_i S⁻¹ G_iᵀ = (H₀')⁻¹` per row — NOT the Woodbury-corrected
        // full inverse the dense adjoint contracts — so an IBP cross-row cache is routed
        // to the dense channel (same hard-refuse discipline as deflation; folding this
        // channel matrix-free needs the bundle to additionally carry `H₀'⁻¹U`, a
        // reduced-Schur/bundle design step). The from-probes lane therefore owns the
        // softmax / euclidean / non-cross-row regimes where base = full inverse.
        let ibp_channels = ibp_assignment_third_channels_weighted(
            &self.assignment,
            rho,
            true,
            self.row_loss_weights.as_deref(),
        )?;
        if ibp_channels.is_some() || cache.cross_row_woodbury.is_some() {
            return Err(
                "logdet_theta_adjoint_from_probes: cache carries an IBP cross-row Woodbury \
                 (T-space rank-R) curvature the border-only probe bundle cannot reconstruct — \
                 route this fit through the dense channel"
                    .to_string(),
            );
        }
        let second_jets = self.atom_second_jets()?;
        let border = self.border_channels_for_cache(cache)?;
        let whiten_row_jets = self.whiten_logdet_row_jets();
        let k_atoms = self.k_atoms();
        // Softmax entropy dense off-diagonal channel `scale = λ·sparsity/τ²` — the
        // SAME weight the dense adjoint (and the assembly) differentiate. The compact
        // per-active-atom majorizer derivative reads only this scale (not the full
        // penalty object), so we carry just the scalar.
        let softmax_dense_adjoint: Option<f64> = match self.assignment.mode {
            AssignmentMode::Softmax {
                temperature,
                sparsity,
            } if k_atoms > 1 => {
                let inv_tau = 1.0 / temperature;
                Some(rho.lambda_sparse() * sparsity * inv_tau * inv_tau)
            }
            _ => None,
        };

        let mut assignments = Array1::<f64>::zeros(self.k_atoms());
        let mut jet_window: std::collections::VecDeque<SaeRowJets> =
            std::collections::VecDeque::new();
        let mut jet_window_next = 0usize;

        for row in 0..n {
            let q = cache.row_dims[row];
            let base = cache.row_offsets[row];
            let a_scratch = assignments.as_slice_mut().expect("contiguous scratch");
            self.assignment.try_assignments_row_into(row, a_scratch)?;
            if jet_window.is_empty() {
                jet_window_next = self.refill_jet_window(
                    jet_window_next,
                    cache,
                    &second_jets,
                    &border,
                    &mut jet_window,
                )?;
            }
            let mut jets = jet_window
                .pop_front()
                .expect("jet window must be non-empty");
            if whiten_row_jets {
                self.apply_whiten_to_logdet_row_jets(row, &mut jets)?;
            }

            // A_i⁻¹ (q×q) via the row-local undamped Cholesky.
            let factor = cache.undamped_factor(row);
            let mut a_inv = Array2::<f64>::zeros((q, q));
            let mut e_j = Array1::<f64>::zeros(q);
            for j in 0..q {
                e_j.fill(0.0);
                e_j[j] = 1.0;
                let col = cholesky_solve_vector(factor, e_j.view());
                for r in 0..q {
                    a_inv[[r, j]] = col[r];
                }
            }

            // Row probe images w_l = G_i z_l, s_l = G_i (S⁻¹ z_l) — the bundle carriers
            // for the selected-inverse blocks (identical to the ARD from-probes path).
            let mut w_probes: Vec<Array1<f64>> = Vec::with_capacity(m);
            let mut s_probes: Vec<Array1<f64>> = Vec::with_capacity(m);
            if k_border > 0 {
                let mut b_tmp = Array1::<f64>::zeros(q);
                for l in 0..m {
                    b_tmp.fill(0.0);
                    if !cache.apply_htbeta_row(row, probes[l].view(), &mut b_tmp) {
                        return Err(format!(
                            "logdet_theta_adjoint_from_probes: H_tβ^({row}) probe apply failed"
                        ));
                    }
                    w_probes.push(cholesky_solve_vector(factor, b_tmp.view()));
                    b_tmp.fill(0.0);
                    if !cache.apply_htbeta_row(row, sinv_probes[l].view(), &mut b_tmp) {
                        return Err(format!(
                            "logdet_theta_adjoint_from_probes: H_tβ^({row}) solve apply failed"
                        ));
                    }
                    s_probes.push(cholesky_solve_vector(factor, b_tmp.view()));
                }
            }

            // (H⁻¹)_tt block: A_i⁻¹ + (1/m)Σ_l sym(w_l ⊗ s_l) (symmetrized outer product;
            // exact & symmetric at full-basis probes).
            // `w_probes`/`s_probes` are populated only when a border exists
            // (`k_border > 0`); their length is `m` there and `0` otherwise, so the
            // border-term loops below vanish cleanly on the borderless arrow.
            let mut inv_vv = a_inv.clone();
            for l in 0..w_probes.len() {
                for a in 0..q {
                    for b in 0..q {
                        inv_vv[[a, b]] += 0.5
                            * inv_m
                            * (w_probes[l][a] * s_probes[l][b] + s_probes[l][a] * w_probes[l][b]);
                    }
                }
            }
            // (H⁻¹)_tβ block (q×K): −(1/m)Σ_l w_l ⊗ (S⁻¹ z_l).
            let mut inv_vbeta = Array2::<f64>::zeros((q, k_border));
            for l in 0..w_probes.len() {
                for a in 0..q {
                    inv_vbeta
                        .row_mut(a)
                        .scaled_add(-inv_m * w_probes[l][a], &sinv_probes[l]);
                }
            }

            // Precompute the β–β fold carriers P_l, R_l (w-independent) per probe.
            let bjet_len = if k_border > 0 {
                jets.beta.first().map_or(0, Vec::len)
            } else {
                0
            };
            let mut p_probe: Vec<Vec<f64>> = Vec::with_capacity(m);
            let mut r_probe: Vec<Vec<f64>> = Vec::with_capacity(m);
            if k_border > 0 && bjet_len > 0 {
                for l in 0..m {
                    let mut p_l = vec![0.0_f64; bjet_len];
                    let mut r_l = vec![0.0_f64; bjet_len];
                    for (beta_pos, channel) in border.iter().enumerate() {
                        let zc = probes[l][channel.index];
                        let sc = sinv_probes[l][channel.index];
                        let bj = &jets.beta[beta_pos];
                        for c in 0..bjet_len {
                            p_l[c] += zc * bj[c];
                            r_l[c] += sc * bj[c];
                        }
                    }
                    p_probe.push(p_l);
                    r_probe.push(r_l);
                }
            }

            let softmax_adjoint_row: Option<(&[f64], f64, f64, f64)> =
                match (softmax_dense_adjoint, self.assignment.mode) {
                    (Some(scale), AssignmentMode::Softmax { temperature, .. }) => {
                        let a = assignments
                            .as_slice()
                            .expect("softmax assignments row must be contiguous");
                        let m_mean = softmax_majorizer_log_mean(a);
                        Some((a, m_mean, scale, 1.0 / temperature))
                    }
                    _ => None,
                };

            // #991 — same design weighting as the primary θ-adjoint path: the
            // softmax majorizer written into `htt` carries `w_row`, so its
            // θ-derivative does too.
            let w_row_prior = self.row_loss_weights.as_deref().map_or(1.0, |w| w[row]);
            for w in 0..q {
                let mut gamma = 0.0_f64;
                let softmax_d_dw: Option<(&[f64], f64, f64, f64, usize)> =
                    match (softmax_adjoint_row, jets.vars[w]) {
                        (Some((a, mm, scale, inv_tau)), SaeLocalRowVar::Logit { atom: atom_w }) => {
                            Some((a, mm, scale, inv_tau, atom_w))
                        }
                        _ => None,
                    };
                // t–t block: reuse the dense contraction (undeflated: no DΦ correction).
                for a in 0..q {
                    for b in 0..q {
                        let mut dh = match (softmax_d_dw, jets.vars[a], jets.vars[b]) {
                            (
                                Some((a_soft, _m, _scale, inv_tau, atom_w)),
                                SaeLocalRowVar::Coord { atom: atom_a, .. },
                                SaeLocalRowVar::Coord { atom: atom_b, .. },
                            ) => {
                                let h_ab = sae_dot(&jets.first[a], &jets.first[b]);
                                h_ab * Self::softmax_data_weight_product_logit_factor(
                                    a_soft, atom_a, atom_b, atom_w, inv_tau,
                                )
                            }
                            _ => {
                                sae_dot(&jets.second[a][w], &jets.first[b])
                                    + sae_dot(&jets.first[a], &jets.second[b][w])
                            }
                        };
                        if let (
                            Some((a_soft, mm, scale, inv_tau, _atom_w)),
                            SaeLocalRowVar::Logit { atom: atom_a },
                            SaeLocalRowVar::Logit { atom: atom_b },
                        ) = (softmax_d_dw, jets.vars[a], jets.vars[b])
                        {
                            if atom_a == atom_b {
                                dh += w_row_prior
                                    * active_softmax_majorizer_logit_derivative_entry(
                                        a_soft, atom_a, _atom_w, mm, scale, inv_tau,
                                    );
                            }
                        }
                        if a == b {
                            dh += match jets.vars[a] {
                                SaeLocalRowVar::Logit { atom } => self
                                    .assignment_prior_hdiag_derivative_entry(
                                        rho,
                                        row,
                                        atom,
                                        jets.vars[w],
                                        ibp_channels.as_ref(),
                                    ),
                                SaeLocalRowVar::Coord { atom, axis } if a == w => {
                                    self.ard_majorized_hessian_derivative(rho, row, atom, axis)
                                }
                                _ => 0.0,
                            };
                        }
                        gamma += inv_vv[[b, a]] * dh;
                    }
                }
                // t–β block: reuse the dense contraction with the reconstructed inv_vβ.
                for a in 0..q {
                    for (beta_pos, channel) in border.iter().enumerate() {
                        let dh = sae_dot(&jets.second[a][w], &jets.beta[beta_pos])
                            + sae_dot(&jets.first[a], &jets.beta_deriv[w][beta_pos]);
                        gamma += 2.0 * inv_vbeta[[a, channel.index]] * dh;
                    }
                }
                // β–β block: refolded as tr(S⁻¹·M) onto the probe bundle.
                if k_border > 0 && bjet_len > 0 {
                    for l in 0..m {
                        let mut q_l = vec![0.0_f64; bjet_len];
                        let mut rd_l = vec![0.0_f64; bjet_len];
                        for (beta_pos, channel) in border.iter().enumerate() {
                            let zc = probes[l][channel.index];
                            let sc = sinv_probes[l][channel.index];
                            let bd = &jets.beta_deriv[w][beta_pos];
                            for c in 0..bjet_len {
                                q_l[c] += zc * bd[c];
                                rd_l[c] += sc * bd[c];
                            }
                        }
                        gamma += inv_m * (sae_dot(&rd_l, &p_probe[l]) + sae_dot(&r_probe[l], &q_l));
                    }
                }
                gamma_t[base + w] = gamma;
            }

            for (w_beta_pos, w_channel) in border.iter().enumerate() {
                let mut gamma = 0.0_f64;
                for a in 0..q {
                    for b in 0..q {
                        let dh = sae_dot(&jets.beta_l_deriv[a][w_beta_pos], &jets.first[b])
                            + sae_dot(&jets.first[a], &jets.beta_l_deriv[b][w_beta_pos]);
                        gamma += inv_vv[[b, a]] * dh;
                    }
                }
                for a in 0..q {
                    for (beta_pos, channel) in border.iter().enumerate() {
                        let dh = sae_dot(&jets.beta_l_deriv[a][w_beta_pos], &jets.beta[beta_pos]);
                        gamma += 2.0 * inv_vbeta[[a, channel.index]] * dh;
                    }
                }
                gamma_beta[w_channel.index] += gamma;
            }
        }

        // No IBP empirical-M / cross-row Woodbury pass here: those channels are
        // hard-refused above (the border-only bundle cannot carry the T-space
        // rank-R Woodbury), so on every accepted cache `ibp_channels` is `None` and
        // the softmax/euclidean core folds above are the complete θ-adjoint.

        Ok(SaeArrowVector {
            t: gamma_t,
            beta: gamma_beta,
        })
    }

    /// Public analytic outer-ρ gradient at a converged inner state, constructing
    /// the deflated arrow solver from the supplied cache. Use this seam from
    /// integration tests and external consumers that have a converged
    /// `(loss, cache)` from [`Self::reml_criterion_with_cache`] but no access to
    /// the crate-private `DeflatedArrowSolver`.
    pub fn analytic_outer_rho_gradient_at_converged(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        loss: &SaeManifoldLoss,
        cache: &ArrowFactorCache,
    ) -> Result<SaeOuterRhoGradientComponents, String> {
        let solver = self.outer_gradient_arrow_solver(cache, &rho.lambda_smooth_vec())?;
        self.analytic_outer_rho_gradient_components(target, rho, loss, cache, &solver)
            .map_err(|e| e.to_string())
    }

    /// Compose the SAE LAML criterion as a sum of atoms (#931 SAE pilot).
    ///
    /// This is the single seam that establishes value↔gradient coherence for
    /// the SAE objective: it runs the inner solve once via
    /// [`Self::reml_criterion_with_cache`], reads the value decomposition
    /// (`loss.total() + extra_penalty_energy`, the rank-adjusted Laplace
    /// complexity, `occam`) and the
    /// matching gradient channels (`SaeOuterRhoGradientComponents`) from the
    /// SAME converged cache, and hands them to [`SaeCriterion::assemble`]. The
    /// returned criterion's [`SaeCriterion::value`] and
    /// [`SaeCriterion::gradient`] are then projections of one factorization —
    /// the outer optimizer can no longer evaluate a value path and a gradient
    /// path that disagree (the #752/#748/#901 desync class). The
    /// implicit-stationarity envelope correction (#1006's Γ term) is its own
    /// named atom, so the channel the desync class keeps dropping is visible
    /// rather than a silent zero.
    pub fn criterion_as_atoms(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
    ) -> Result<SaeCriterion, String> {
        let (production_value, loss, cache) = self.reml_criterion_with_cache(
            target,
            rho,
            registry,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
        )?;
        let log_det = arrow_log_det_from_cache(&cache).ok_or_else(|| {
            "criterion_as_atoms: arrow_log_det_from_cache returned None".to_string()
        })?;
        let residual = self.reconstruction_residual(target, rho)?;
        let dispersion = self.reconstruction_dispersion(
            &loss,
            &cache,
            rho,
            Some(residual.view()),
        )?;
        let d_eff = self.per_atom_realised_rank_dof(rho, dispersion)?;
        let n_eff = self.per_atom_effective_sample_size();
        let log_det_tt = coordinate_block_log_det(&cache)?;
        let laplace_complexity =
            rank_adjusted_laplace_complexity(log_det, log_det_tt, &d_eff, &n_eff)?;
        let occam = self.reml_occam_term(rho)?;
        let extra_penalty_energy = self
            .reml_extra_penalty_value_total(registry)
            .map_err(|err| format!("SaeManifoldTerm::criterion_as_atoms: {err}"))?;
        let data_fit_priors_value = loss.total() + extra_penalty_energy;

        let solver = self.outer_gradient_arrow_solver(&cache, &rho.lambda_smooth_vec())?;
        let components =
            self.analytic_outer_rho_gradient_components(target, rho, &loss, &cache, &solver)?;
        let criterion = SaeCriterion::assemble(
            data_fit_priors_value,
            laplace_complexity,
            occam,
            components.explicit,
            components.logdet_trace,
            components.occam,
            components.third_order_correction,
        );
        let assembled_value = criterion.value();
        let identity_roundoff = 64.0
            * f64::EPSILON
            * (1.0 + production_value.abs().max(assembled_value.abs()));
        if (assembled_value - production_value).abs() > identity_roundoff {
            return Err(format!(
                "criterion_as_atoms: assembled value {assembled_value:.17e} does not equal \
                 production value {production_value:.17e} from the same cache \
                 (roundoff={identity_roundoff:.3e})"
            ));
        }
        Ok(criterion)
    }

    // [#780 line-count gate] reconstruction_dispersion + assemble_shape_uncertainty
    // + complete_born_atom_shape_bands + shape_uncertainty_without_decoder_covariance
    // (the contiguous trailing methods of this impl block) were split into the
    // sibling construction_reconstruction.rs (declared in mod.rs); callers reach
    // them bare via use super::*.
}
