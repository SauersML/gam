// The fixed-ρ custom quasi-Laplace criterion and its complexity-pricing
// machinery (penalized_quasi_laplace_criterion* entries, rank-charge ledger,
// deflated-factor path) live in this sibling file as a second
// `impl SaeManifoldTerm` block, inlined via `include!` from construction.rs so
// it keeps the SAME module scope and private-field access. Keeps the tracked
// construction.rs under the 10k limit.

/// One coherent matrix-free outer sample. The value, factor cache, reduced
/// operator, and lossless rational derivative are all emitted by the same
/// frozen surrogate evaluation, so no consumer can accidentally differentiate
/// a reassembled or differently-randomized operator.
/// #2515 — what one gradient-bearing streaming evidence evaluation leaves
/// behind, as ONE object so the two operators cannot be taken from different
/// evaluations.
///
/// `majorizer_system` is `B`: the positive-definite Newton/IFT scale that
/// [`SaeManifoldTerm::solve_exact_stationarity_matrix_free`] reassembles
/// `A = B + ΔC` on top of. `exact_a_cache` is the factor cache of the exact
/// observed information — the operator whose reduced-Schur log-determinant the
/// criterion ranks and whose derivative representation the surrogate lane emits.
/// Handing the from-probes channels the majorizer's cache alongside this lane's
/// `A`-rooted bundle is the #2515 defect; keeping both in one struct, produced by
/// one call, is what stops them being paired across evaluations.
pub(crate) struct StreamingEvidenceArtifacts {
    pub(crate) majorizer_system: ArrowSchurSystem,
    pub(crate) exact_a_cache: ArrowFactorCache,
}

pub(crate) struct StreamingOuterEvaluation {
    pub(crate) cost: f64,
    pub(crate) loss: SaeManifoldLoss,
    pub(crate) cache: ArrowFactorCache,
    pub(crate) system: ArrowSchurSystem,
    /// The factor cache of the exact-`A` evidence operator this evaluation's
    /// `logdet_derivative_bundle` was produced from (#2515). The from-probes
    /// selected-inverse channels reconstruct `(H⁻¹)_tt = A_i⁻¹ + G_i S⁻¹ G_iᵀ`,
    /// so the row factors here and the `S⁻¹` in the bundle must be the same
    /// operator's; `cache` above stays `B`.
    pub(crate) exact_a_cache: ArrowFactorCache,
    /// Lossless low-rank derivative of the rational value (all shifts and the
    /// frozen deflation block). This, never the raw shift-zero inverse probes,
    /// owns the outer logdet trace and theta-adjoint channels.
    pub(crate) logdet_derivative_bundle: RationalLogdetDerivativeBundle,
    /// Optional raw `(z, S^-1 z)` bundle used only for EFS/MacKay proposal
    /// traces. Its root is not the rational surrogate derivative and it must
    /// never enter the authoritative outer gradient.
    pub(crate) efs_inverse_probe_bundle: Option<(Vec<Array1<f64>>, Vec<Array1<f64>>)>,
}

/// The two deliberately distinct scalar currencies of one stationarity
/// residual. The terminal polish is accepted in the posterior-null quotient,
/// while the ambient norm is a non-growth invariant; naming both prevents a
/// baseline from one space being paired with a model endpoint from the other
/// (#2762).
#[derive(Clone, Copy, Debug)]
struct ResidualMerits {
    quotient: f64,
    ambient: f64,
}

/// One accepted terminal-polish trial with the model prediction in the same
/// quotient currency that admitted it. The spectral step owns a model residual,
/// not a scalar merit; this caller owns the projection and therefore the price.
struct AcceptedTerminalResidualStep {
    damping: f64,
    trial_merits: ResidualMerits,
    predicted_quotient_decrease: f64,
    step: DampedResidualStep,
    system: Option<ArrowSchurSystem>,
}

impl SaeManifoldTerm {
    /// Custom penalized quasi-Laplace score for the SAE term at a fixed `ρ`.
    ///
    /// This is not a normalized LAML, REML, or evidence objective. The
    /// assignment priors (softmax entropy, ThresholdGate) have NO finite normalizer:
    /// for softmax the reference-logit chart sends `P(ℓ)→0` as a free logit →±∞
    /// so `∫ e^{−λP} dℓ = ∞`, and ThresholdGate's bounded penalty `0<P<λ` keeps
    /// `e^{−λP}` bounded below over an unbounded domain, also divergent. There is
    /// therefore no ρ-independent assignment-prior normalizer that can be dropped
    /// as a constant. The smoothing-penalty `−½log|λS|_+` term IS a genuine
    /// (proper-Gaussian) REML normalizer and is kept exactly; the rest is a
    /// penalized quasi-Laplace score (custom curvature term `½log|B|` around the
    /// inner optimum), which the engine minimizes over ρ.
    ///
    /// Runs the inner `(t, β)` arrow-Schur Newton solve to convergence at the
    /// supplied ρ (with NO in-loop ARD update — ρ is owned by the engine),
    /// then forms the custom penalized quasi-Laplace cost
    ///
    /// ```text
    /// V(ρ) = ℓ_pen(t̂, β̂; ρ) + E_extra
    ///        + ½ log|A| − ½ log|A_tt| + Σ_k ½ · dof_k · log(max(N_eff_k, 1))
    ///        − ½ · p · (Σ_k rank S_k) · log λ_smooth
    /// ```
    ///
    /// where `ℓ_pen = loss.total()` is the penalised objective at the inner
    /// optimum and the middle line is the #2a occupancy-aware BIC/Laplace
    /// complexity assembled by `rank_adjusted_quasi_laplace_complexity` from
    /// the EXACT observed information: `log|A|` joint, `log|A_tt|` on the
    /// coordinate block, plus the per-atom realised-DOF rank charge.
    ///
    /// #2a SUPERSEDED the majorizer form. This doc previously described the
    /// charge as `½ log|B|` over the PSD / Gauss--Newton arrow-Schur factor, and
    /// argued that because `B_tt` carries `α = exp(log_ard)` on its diagonal,
    /// `½ log|B|` rises as α grows and balances the `−½·n·log α` already inside
    /// `loss.ard` — concluding the criterion therefore needs no clamp to stay
    /// finite on a collapsing axis. **That balance no longer exists**, because
    /// `− ½ log|A_tt|` subtracts the coordinate block, which is the only place α
    /// enters, and the rank charge `Σ ½·dof·log(max(N_eff,1))` is α-free.
    ///
    /// gam#2627 measured the consequence on a collapsed axis (`‖t₁‖² ≈ 4e-10`,
    /// `n = 4`): `dV/d log α = −2.0000000000` to ~1e-12 with zero curvature,
    /// i.e. exactly `−n/2` — the `loss.ard` term standing alone. Whether
    /// `−½·n·log α` should have been dropped from `V` when #2a retired the
    /// majorizer coordinate term, or whether #2a owes `V` a replacement
    /// α-charge, is an open question on that issue. Until it is settled, do not
    /// rely on the retired balance above as a reason this criterion is
    /// clamp-free on a collapsing axis.
    ///
    /// The final `−½·p·rank(S)·log λ_smooth` term is the smoothing-penalty
    /// normaliser `−½ log|λ S|_+` restricted to its ρ-dependent part: `S_k` is
    /// shared across all `p` decoder output channels (the `⊗ I_p` Kronecker
    /// structure), so `log|λ S|_+ = p·rank(S)·log λ + p·log|S|_+`, and the
    /// `½ p·log|S|_+` piece is ρ-independent. The ρ-independent additive
    /// constants that ARE dropped here (they shift `V` by a constant and do not
    /// affect the ρ-argmin) are the formal `2π` Gaussian constant and the base
    /// `½ p·log|S|_+` penalty logdet. #1421: NO assignment-prior normalizer is
    /// dropped, because none exists (softmax/ThresholdGate priors are improper — see
    /// the doc on this function): the quasi-Laplace score simply omits a
    /// normalizer that is not a finite constant.
    ///
    /// Returns `(V, loss)` so the engine can both rank ρ and surface the inner
    /// loss breakdown.
    pub fn penalized_quasi_laplace_criterion(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
    ) -> Result<(f64, SaeManifoldLoss), SaeCriterionError> {
        self.penalized_quasi_laplace_criterion_with_refine_policy(
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

    pub(crate) fn penalized_quasi_laplace_criterion_with_refine_policy(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
        refine_progress_extension: bool,
    ) -> Result<(f64, SaeManifoldLoss), SaeCriterionError> {
        self.penalized_quasi_laplace_criterion_with_refine_policy_and_lane(
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

    /// [`Self::penalized_quasi_laplace_criterion_with_refine_policy`] with the #2080 surrogate lane
    /// threaded to the streaming `log|S|` evidence term. `lane = None` is the
    /// bit-identical SLQ path; on the dense (non-streaming) branch the lane is
    /// unused (the dense evidence has its own factor-cache log-det).
    pub(crate) fn penalized_quasi_laplace_criterion_with_refine_policy_and_lane(
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
    ) -> Result<(f64, SaeManifoldLoss), SaeCriterionError> {
        self.assignment.validate_rho_domain(rho)?;
        // #976 evidence-ledger scope: one criterion evaluation = one per-atom
        // reseed budget. The joint-fit driver no longer clears the ledger on
        // evidence re-entries (each refine round used to get a fresh budget and
        // could fire an unguarded reseed once per round — the ‖g‖-spike /
        // progress-budget-collapse pathology), so the criterion entry owns the
        // clear.
        self.collapse_events.clear();
        let plan = self.streaming_plan()?.admitted_or_error(
            self.n_obs(),
            self.output_dim(),
            self.k_atoms(),
        )?;
        if plan.streaming {
            // #1225: streaming and dense MUST optimize the SAME mathematical
            // objective — the full penalized quasi-Laplace criterion `loss.total() + extra_penalty +
            // ½ log|H| − Occam`. The streaming branch previously returned only
            // `loss.total() + extra_penalty_energy`, dropping the Laplace
            // normalizer `½ log|H|` and the Occam term, so large shapes (exactly
            // where streaming is needed) were ranked by penalized loss rather than
            // penalized quasi-Laplace — and dense vs streaming disagreed on the objective. Route
            // through the streaming exact-logdet path, which assembles a
            // chunk-by-chunk `½ log|H|_stream` and the same `−Occam`/extra-penalty
            // terms as the dense `penalized_quasi_laplace_criterion_with_cache`.
            //
            // ⚠ #2509 — THE TWO `log|H|` ARE NOT THE SAME OPERATOR TODAY. #2330
            // Phase-2 migrated the DENSE lane to the exact observed information
            // `A = ∇²_θθ L = B + ΔC` and left this one on the Arrow–Schur
            // majorizer `B`, so the objectives split by exactly
            // `½·[(log|A| − log|A_tt|) − (log|B| − log|B_tt|)]` whenever `ΔC ≠ 0`
            // (residual curvature, softmax entropy-minus-majorizer, the periodic
            // ARD concave clamp, ordered Beta–Bernoulli). The #1225 statement
            // above is the CONTRACT, and it is currently unmet on this branch;
            // `criterion_lane_gap_is_exactly_the_evidence_logdet_gap_2509` pins
            // that the split is confined to that pair and nothing else.
            self.penalized_quasi_laplace_criterion_streaming_exact_with_lane(
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
            let (v, loss, _cache) = self
                .penalized_quasi_laplace_criterion_with_cache_refine_policy(
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

    /// As [`Self::penalized_quasi_laplace_criterion`], but also returns the converged undamped
    /// `ArrowFactorCache` so callers (the EFS fixed-point step) can read the
    /// selected-inverse traces `(H⁻¹)_tt` / `(H⁻¹)_ββ` without re-factoring.
    /// The cache is the single shared O(K³) Direct factor; both the
    /// log-determinant criterion and the Fellner-Schall ρ-step consume it.
    pub fn penalized_quasi_laplace_criterion_with_cache(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
    ) -> Result<(f64, SaeManifoldLoss, ArrowFactorCache), SaeCriterionError> {
        self.penalized_quasi_laplace_criterion_with_cache_refine_policy(
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

    pub(crate) fn penalized_quasi_laplace_criterion_with_cache_refine_policy(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
        refine_progress_extension: bool,
    ) -> Result<(f64, SaeManifoldLoss, ArrowFactorCache), SaeCriterionError> {
        let criterion_entered = std::time::Instant::now();
        self.assignment.validate_rho_domain(rho)?;
        // #976 evidence-ledger scope (see `penalized_quasi_laplace_criterion_with_refine_policy_
        // and_lane`): direct cache-lane callers also get a fresh per-evaluation
        // reseed budget here; the double clear when routed through the value
        // entry is an idempotent no-op.
        self.collapse_events.clear();
        let admission_plan = self.streaming_plan()?.admitted_or_error(
            self.n_obs(),
            self.output_dim(),
            self.k_atoms(),
        )?;
        if !admission_plan.direct_logdet_admitted() {
            // The cache-returning penalized quasi-Laplace entry is used by the EFS/outer lanes that
            // need selected-inverse traces in addition to the scalar evidence.
            // Large SAE fits cannot form the dense `N · q · border_dim`
            // evidence slab (`q = K(1+d)`, `border_dim = Σ_k M_k · p`), so the
            // correct implementation is not to reject here and force callers
            // onto a value-only path.  Route through the streaming evidence
            // implementation instead: it reuses the converged per-row factor
            // cache for traces and recomputes the reduced-Schur logdet by
            // chunks / matrix-free matvecs, keeping peak memory at the admitted
            // streaming working set rather than the dense n·k·p floor.
            return self.penalized_quasi_laplace_criterion_streaming_exact_with_cache(
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
        log::info!(
            "[SAE-ENTRY] initial joint fit starts {:.2}s after criterion entry",
            criterion_entered.elapsed().as_secs_f64(),
        );
        let initial_fit = self.run_joint_fit_arrow_schur_for_quasi_laplace(
            target,
            &mut rho_fixed,
            registry,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
        )?;
        log::info!(
            "[SAE-ENTRY] initial joint fit done {:.2}s after criterion entry",
            criterion_entered.elapsed().as_secs_f64(),
        );
        let mut loss = initial_fit.loss;
        let mut criterion_fixed_point = initial_fit.fixed_point;

        // 2. Drive the inner (t, β) solve to the KKT/step-converged optimum and
        //    take one final UNDAMPED factor there to obtain the joint Hessian
        //    log-determinant. We force ridge = 0 and the dense `Direct` Schur
        //    mode so `arrow_log_det_from_cache` returns the exact
        //    `log|H| = Σ_i log|H_tt^(i)| + log|Schur_β|` (it rejects damped
        //    factors and InexactPCG caches, which have no dense Schur factor).
        //    This is the same evidence convention the main GAM penalized quasi-Laplace path uses.
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
            .with_gpu_policy(self.gpu_policy)
            .with_newton_schur_tikhonov(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR)
            .with_evidence_unit_deflation(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR);
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
            &mut criterion_fixed_point,
            &options,
            refine_progress_extension,
        )?;
        self.record_criterion_gauge_deflation_count(
            cache.gauge_deflated_directions,
            refine_progress_extension,
        )?;
        loss.criterion_gauge_deflated_directions = cache.gauge_deflated_directions;
        // #2330 Phase-2: rank the EXACT observed-information Laplace term ½log|A|
        // (A = B + ΔC = ∇²_θθ L), not the majorizer surrogate ½log|B|. One
        // eigendecomposition yields BOTH the joint log|A| and the coordinate-block
        // log|A_tt|, applying the shared PD floor; an indefinite A (a majorizer
        // saddle) returns the typed IndefiniteObservedInformation refusal, which
        // makes saddle-ρ probe-infeasible (+inf) and steers the outer away until
        // the #2336 accepted-lane saddle-escape lands.
        let (log_det, log_det_tt) =
            self.exact_observed_information_log_dets(rho, target, &cache)?;

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
            .map_err(|err| format!("SaeManifoldTerm::penalized_quasi_laplace_criterion: {err}"))?;

        let v = {
            // #5/(B): replace the COORDINATE-block ½log|H_tt| in the Laplace
            // complexity with the honest BIC ½·d_eff·log n on each atom's realised
            // decoder rank. The decoder-scale mispricing (`½log(a²‖B‖²)` scale,
            // over-charging real atoms + rewarding a²‖B‖²→0) lives ENTIRELY in the
            // coordinate block (`H_tt ∝ ‖B‖²`); the β/Schur block is
            // ‖B‖-independent (ρ⁰ coupling) and stays. `d_eff` is rotation-
            // invariant, so it accepts a real rank-2 circle but does not
            // distinguish clean-vs-blend (producer's job). A certified vanished
            // atom is a typed boundary before rank pricing.
            // Decoder disappearance is certified first from the raw output-frame
            // residual and gated decoder Grams. It has no tuned noise multiple:
            // the boundary is derived from the residual reduction's floating-point
            // backward error, and proof-unavailable is surfaced loudly.
            let residual = self.reconstruction_residual(target, rho)?;
            let mut grams = self.empty_decoder_gram_accumulator();
            self.accumulate_decoder_gram(&mut grams)?;
            let n_eff = self.per_atom_effective_sample_size();
            let residual_energy =
                self.residual_energy_for_vanishing(residual.view())?;
            match self.vanished_atoms_from_signal_upper_bound(
                &grams,
                &n_eff,
                residual_energy.mean_square(),
            )? {
                VanishedAtomsProof::Certified {
                    atoms: Some(atoms),
                    ..
                } => return Err(SaeCriterionError::VanishedAtoms(atoms)),
                VanishedAtomsProof::Certified { atoms: None, .. } => {}
                VanishedAtomsProof::Unavailable { reason } => {
                    return Err(SaeCriterionError::Numerical(format!(
                        "decoder-vanishing proof unavailable: {reason}"
                    )));
                }
            }
            let disp = self
                .reconstruction_dispersion(&loss, &cache, rho, Some(residual.view()))
                .map_err(|e| {
                    format!(
                        "SaeManifoldTerm::penalized_quasi_laplace_criterion: rank-charge dispersion is required: {e}"
                    )
                })?;
            let d_eff = self.rank_dof_from_grams(&grams, &n_eff, rho, disp)?;
            // Occupancy-aware effective sample size N_eff,k = Σ_i a_{ik}², the #2a
            // per-atom BIC log-scale (same quantity `per_atom_realised_rank_dof` uses
            // internally for the MP edge; recomputed here — a cheap Σa² — to price the
            // charge in the same currency).
            // #5/#2498 — the same-state gated-signal certificate above owns the
            // categorical Laplace-validity boundary. Do not manufacture a second
            // disappearance verdict from `d_eff == 0`: DOF also contains the
            // smooth-basis charge and is not a physical reconstruction signal.
            // #2a — occupancy-aware BIC/Laplace scale. The shared scalar helper
            // owns the exact replacement
            // `0.5 log|H| - 0.5 log|H_tt| + rank_charge`; dense, streaming, and
            // criterion-as-atoms assembly therefore cannot drift apart.
            // log_det (= log|A|) and log_det_tt (= log|A_tt|) are produced together
            // above from the exact observed information; `coordinate_block_log_det`
            // (the majorizer ½log|B_tt|) is no longer the ranked coordinate term.
            let quasi_laplace_complexity =
                rank_adjusted_quasi_laplace_complexity(log_det, log_det_tt, &d_eff, &n_eff)?;
            loss.total() + extra_penalty_energy + quasi_laplace_complexity - occam
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
    /// `re_anchor == false` (value-probe / line-search lanes): the transient
    /// count is READ-ONLY — the anchor, the drift-direction memory, and the
    /// reversal budget are all left untouched. #2253/#1037: the criterion's
    /// quotient dimension may only move at ACCEPTED iterates; a probe that
    /// re-anchored mid-line-search let the bookkeeping dimension flicker inside
    /// a Wolfe bracket (a live discontinuity generator between two probes of
    /// the same search), and a bracket of probes could burn the reversal budget
    /// that exists to catch a genuinely oscillating ACCEPTED trajectory. Each
    /// deflated direction contributes the ρ-independent `log 1 = 0` to
    /// `½log|H|`, so skipping the probe-lane anchor move never changes any
    /// probe's value.
    pub(crate) fn record_criterion_gauge_deflation_count(
        &mut self,
        count: usize,
        re_anchor: bool,
    ) -> Result<(), String> {
        if !re_anchor {
            return Ok(());
        }
        match self.expected_criterion_gauge_deflated_directions {
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
                let is_reversal = self.criterion_gauge_deflation_last_delta_sign != 0
                    && delta_sign != self.criterion_gauge_deflation_last_delta_sign;
                self.criterion_gauge_deflation_last_delta_sign = delta_sign;
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
                    self.criterion_gauge_deflation_reanchors += 1;
                }
                let reversal_budget = self
                    .k_atoms()
                    .saturating_mul(
                        SAE_ATOM_COLLAPSE_RESEED_BUDGET
                            + SAE_DICTIONARY_COCOLLAPSE_RESEED_BUDGET
                            + 1,
                    )
                    .saturating_add(1);
                if self.criterion_gauge_deflation_reanchors > reversal_budget {
                    return Err(format!(
                        "SaeManifoldTerm::penalized_quasi_laplace_criterion: row-gauge criterion deflation count \
                         oscillated (reversed direction {} times, last {expected}->{count}) within \
                         one optimization, exceeding the {reversal_budget}-reversal budget for {} \
                         atoms; the quotient dimension is not stabilizing, refusing to compare \
                         Laplace normalizers",
                        self.criterion_gauge_deflation_reanchors,
                        self.k_atoms()
                    ));
                }
                log::debug!(
                    "SaeManifoldTerm::penalized_quasi_laplace_criterion: per-row criterion deflation count changed \
                     {expected}->{count} (a benign per-row conditioning drift across the ρ-walk; \
                     reversal {}/{reversal_budget}); re-anchoring the Laplace normalizer comparison \
                     to the new dimension",
                    self.criterion_gauge_deflation_reanchors
                );
                self.expected_criterion_gauge_deflated_directions = Some(count);
                Ok(())
            }
            None => {
                self.expected_criterion_gauge_deflated_directions = Some(count);
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
    /// (`penalized_quasi_laplace_criterion_streaming_exact`) criterion paths route through this same
    /// driver, so they converge to the identical inner state (#847).
    ///
    /// ⚠ #2509 — a shared inner state is NOT a shared log-determinant. #2330
    /// Phase-2 changed which OPERATOR each lane factors at that shared state:
    /// dense prices the exact observed information `A = B + ΔC`
    /// (`exact_observed_information_log_dets`), streaming prices the Arrow–Schur
    /// majorizer `B` (`streaming_exact_arrow_log_det`). The #847 bit-identity
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
            // the decoded-derivative gauge floor (`tangent·tangent > 1e-24`) can
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
        let mut saw_refine_progress = false;
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
        // #2653 — repeated STALL-branch polish is governed by what the former
        // ordinal cap was trying to approximate: contraction of either
        // accepted KKT currency. A raw count of eight retired the production
        // K=1 circle tail while both raw and quotient residuals were still
        // falling. The certificate advances only when one Pareto frontier
        // moves beyond floating-point resolution, so it admits every useful
        // ninth-or-later rescue while a repeated plateau still terminates.
        let mut stall_polish_progress =
            super::stall_polish_progress::StallPolishProgressCertificate::new(
                f64::EPSILON.sqrt(),
            );
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
            // `arrow_log_det_from_cache` remains exact.
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
                log::debug!(
                    "SAE criterion factor accepted at KKT stationarity: ‖g‖={grad_norm:.6e} \
                     ‖Π⊥null g‖={quotient_grad_norm:.6e} tol={grad_tolerance:.6e} \
                     ‖Δ‖={:.6e} ‖Π⊥null Δ‖={:.6e} after {total_inner_iter} inner iterations",
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
                            saw_refine_progress,
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
                        saw_refine_progress |=
                            Self::refine_round_made_progress(previous_refine_grad_norm, grad_norm);
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
                saw_refine_progress,
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
                // cost-lane disagreement at one ρ). A solve that is MEASURABLY
                // DESCENDING (`saw_refine_progress`) is an unfinished
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
                    let decrement_sq = sae_manifold_newton_directional_decrease(
                        &sys,
                        limit_factor.delta_t.view(),
                        limit_factor.delta_beta.view(),
                    )
                    .max(0.0);
                    let limit_scale = self
                        .penalized_objective_total(target, rho_fixed, registry, 1.0)
                        .map(|obj| obj.abs() + 1.0)
                        .unwrap_or(f64::INFINITY);
                    let predicted_relative_decrease = 0.5 * decrement_sq / limit_scale;
                    if predicted_relative_decrease <= SAE_MANIFOLD_INNER_OBJECTIVE_STALL_REL_TOL {
                        log::debug!(
                            "SAE inner limit-boundary decrement acceptance: ‖g‖={grad_norm:.6e} \
                             (tol {grad_tolerance:.6e}) ½λ²/scale={predicted_relative_decrease:.6e} \
                             after {total_inner_iter} inner iterations"
                        );
                        return Ok(limit_factor.cache);
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
                if (saw_refine_progress || gradient_stationary) && budget_escalation_extra == 0 {
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
                    // phase ever running. Try it before refusing: a committed
                    // step strictly contracts ‖g‖ OR the exact Newton decrement
                    // (the accept test is `obj_ok && (grad_ok || decrement_ok)`;
                    // its no-contraction bail exits cheaply on genuinely
                    // hopeless states), and on progress the loop resumes with
                    // fresh accounting — the
                    // loop-top KKT gate and the idempotence certificate remain
                    // the sole acceptance authority, exactly as at the stall
                    // branch.
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
                            saw_refine_progress = true;
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
                        let newton_decrement_sq = sae_manifold_newton_directional_decrease(
                            &sys,
                            final_dt.view(),
                            final_db.view(),
                        )
                        .max(0.0);
                        let excursion_cert =
                            0.5 * newton_decrement_sq / final_objective_scale;
                        // #2228 — the acceptance verdict keys on the BEST-SEEN
                        // certificate, not the excursion the polish left. The
                        // band is UNCHANGED; a best-seen plateau ABOVE it is a
                        // solver stall, refused honestly below with the best-
                        // seen ‖g‖. When best-seen clears the band we certify
                        // THERE (restore + re-factor) — the continuation is
                        // over, so nothing consumes the restore.
                        let best_clears = best_seen
                            .as_ref()
                            .is_some_and(|(c, _, _)| {
                                *c < excursion_cert
                                    && *c <= SAE_MANIFOLD_INNER_OBJECTIVE_STALL_REL_TOL
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
                                log::debug!(
                                    "SAE #2228 certify-at-best-seen: ‖g‖ {grad_norm:.6e} \
                                     \u{2192} {best_g:.6e}, ½λ²/scale {excursion_cert:.6e} \
                                     \u{2192} {best_cert:.6e} after {total_inner_iter} iters"
                                );
                                return Ok(best_factor.cache);
                            }
                            // Re-factor at best-seen failed: restore the
                            // excursion so state + final_cache stay consistent,
                            // then fall through to the honest refusal below.
                            self.restore_mutable_state(&excursion)?;
                        } else if excursion_cert
                            <= SAE_MANIFOLD_INNER_OBJECTIVE_STALL_REL_TOL
                        {
                            log::debug!(
                                "SAE inner final-gate decrement acceptance: ‖g‖={grad_norm:.6e} \
                                 (tol {grad_tolerance:.6e}) λ²={newton_decrement_sq:.6e} \
                                 ½λ²/scale={excursion_cert:.6e} after \
                                 {total_inner_iter} inner iterations"
                            );
                            return Ok(final_cache);
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
            saw_refine_progress |=
                Self::refine_round_made_progress(previous_refine_grad_norm, grad_norm);
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
            log::info!(
                "[SAE-REFINE] round={refine_rounds} penalized_objective={new_loss_total:.10e} \
                 ‖g‖={grad_norm:.6e}",
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
                    if predicted_relative_decrease <= SAE_MANIFOLD_INNER_OBJECTIVE_STALL_REL_TOL {
                        return Ok(stationary_cache);
                    }
                    let progress_verdict = stall_polish_progress.observe(
                        stationary_grad_norm,
                        stationary_quotient_grad_norm,
                        grad_tolerance,
                    );
                    stall_polish_permitted = progress_verdict.permits_continuation();
                    log::debug!(
                        "SAE inner stall-polish continuation certificate: {progress_verdict:?}"
                    );
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
                // the iterate to the exact-Hessian terminal Newton phase; a
                // committed step strictly contracts ‖g‖ or the exact Newton
                // decrement, so the refine loop resumes with fresh progress
                // instead of refusing. The phase
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
                        // Anti-runaway cap ONLY — the polish's acceptance gate
                        // requires strict contraction of ‖g‖ OR of the exact
                        // Newton decrement λ² per step, and its bail fires the
                        // first step that contracts NEITHER, so the loop
                        // terminates numerically on its own. The termination
                        // therefore rests on the decrement arm wherever raw ‖g‖
                        // is non-monotone, which is exactly the indefinite /
                        // stiff regime this phase exists for; reading this
                        // sentence as a single-currency ‖g‖ contract is wrong,
                        // and gam#2715 read the resulting trace as a defect in
                        // the gate rather than as the design it is. Measured
                        // (tier-0 fixtures, host lane): at 12 the polish
                        // silently expired at ‖g‖ = 6.48e-5 against a 6.11e-5
                        // band — refused 1.07× from convergence purely by cap.
                        // Near the marginally-indefinite root the quotient
                        // GMRES steps contract slower than pure quadratic, so
                        // the cap must not impersonate a convergence bound.
                        64,
                        &mut best_seen,
                    )? {
                        *criterion_fixed_point = false;
                        consecutive_objective_stalls = 0;
                        saw_refine_progress = true;
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
                    let orbit = self.descend_gauge_orbit_at_terminal_candidate(
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
                        saw_refine_progress = true;
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

    /// Run the likelihood-flat-block mover on the state this terminal path
    /// would actually retain (#2762).
    ///
    /// The terminal Newton polish records the smallest `½λ²/scale` state in
    /// `best_seen`, while the refine loop may subsequently leave a different
    /// live excursion in `self`.  A terminal descent on that excursion followed
    /// by the refusal's restore of `best_seen` discards the descent and reports
    /// the untouched state as the fixed point.  Restore first, so the mover and
    /// the terminal diagnostic have one state authority.
    ///
    /// A committed move invalidates the saved decrement certificate: it belongs
    /// to the pre-move state and must not restore over the move later.  An inert
    /// call leaves the certificate in place because `self` is then still exactly
    /// that saved state.
    pub(crate) fn descend_gauge_orbit_at_terminal_candidate(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        penalized_gram_scale: &[f64],
        best_seen: &mut Option<(f64, f64, SaeManifoldMutableState)>,
        max_rounds: usize,
    ) -> Result<GaugeOrbitDescent, String> {
        if let Some((_, _, best_state)) = best_seen.as_ref() {
            self.restore_mutable_state(best_state)?;
        }
        let outcome =
            self.descend_gauge_orbit(target, rho, registry, penalized_gram_scale, max_rounds)?;
        if outcome.moved() {
            *best_seen = None;
        }
        Ok(outcome)
    }

    /// The empty per-row `ArrowRowGaugeDeflation` that opts a system into per-row
    /// spectral discovery (the #974 low-rank-whiten seam). An intrinsic flat /
    /// indefinite `H_tt` direction is then deflated to UNIT stiffness
    /// (`log 1 = 0`, ρ-independent, the quotient pseudo-determinant convention),
    /// so the ridge-0 factor is PD-by-deflation and the criterion log-det finite;
    /// a full-rank block has no sub-floor eigenvalue and is untouched.
    ///
    /// Shared by the acceptance-site installer
    /// [`Self::ensure_row_gauge_deflation_for_quasi_laplace`] and by the two
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
    /// be FACTORED for an accepted criterion log-det, never on the loop `sys` fed
    /// to `probe_undamped_evidence_row_factors` — the #2080 infeasible-ρ probe is
    /// contractually the UNDAMPED (non-deflated) per-row verdict (#2080/#2228).
    pub(crate) fn ensure_row_gauge_deflation_for_quasi_laplace(sys: &mut ArrowSchurSystem) {
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

    /// Largest componentwise Jacobi-scaled KKT gradient in parameter units.
    ///
    /// Each gradient component is divided by the diagonal curvature of its own
    /// block before the blocks are aggregated. The ordering is load-bearing:
    /// decoder gradients and curvatures are both extensive in the rows assigned
    /// to an atom, while coordinate components are row-local. Normalizing an
    /// already-aggregated L2 norm would retain a spurious `sqrt(K)` dependence;
    /// the max of individually scaled components is intensive in both `n` and
    /// `K`.
    pub fn system_scaled_grad_max(
        sys: &ArrowSchurSystem,
    ) -> Result<f64, SaeInnerKktScaleError> {
        let mut scaled_max = 0.0_f64;
        for (row_index, row) in sys.rows.iter().enumerate() {
            let gradient_len = row.gt.len();
            let (curvature_rows, curvature_cols) = row.htt.dim();
            let block = SaeInnerKktScaleBlock::CoordinateRow { row: row_index };
            if (curvature_rows, curvature_cols) != (gradient_len, gradient_len) {
                return Err(
                    SaeInnerKktScaleError::GradientCurvatureShapeMismatch {
                        block,
                        gradient_len,
                        curvature_rows,
                        curvature_cols,
                    },
                );
            }
            for component in 0..gradient_len {
                let gradient = row.gt[component];
                if !gradient.is_finite() {
                    return Err(SaeInnerKktScaleError::NonFiniteGradient {
                        block,
                        component,
                        value: gradient,
                    });
                }
                let curvature = row.htt[[component, component]];
                if !curvature.is_finite()
                    || curvature < 0.0
                    || (curvature == 0.0 && gradient != 0.0)
                {
                    return Err(SaeInnerKktScaleError::InvalidCurvature {
                        block,
                        component,
                        gradient,
                        curvature,
                    });
                }
                if curvature > 0.0 {
                    let scaled = gradient.abs() / curvature;
                    if !scaled.is_finite() {
                        return Err(SaeInnerKktScaleError::NonFiniteScaledGradient {
                            block,
                            component,
                            gradient,
                            curvature,
                        });
                    }
                    scaled_max = scaled_max.max(scaled);
                }
            }
        }

        let block = SaeInnerKktScaleBlock::SharedDecoder;
        let diagonal = sys.shared_block_diagonal();
        if sys.gb.len() != sys.k || diagonal.len() != sys.k {
            return Err(
                SaeInnerKktScaleError::GradientCurvatureShapeMismatch {
                    block,
                    gradient_len: sys.gb.len(),
                    curvature_rows: diagonal.len(),
                    curvature_cols: diagonal.len(),
                },
            );
        }
        for component in 0..sys.k {
            let gradient = sys.gb[component];
            if !gradient.is_finite() {
                return Err(SaeInnerKktScaleError::NonFiniteGradient {
                    block,
                    component,
                    value: gradient,
                });
            }
            let curvature = diagonal[component];
            if !curvature.is_finite()
                || curvature < 0.0
                || (curvature == 0.0 && gradient != 0.0)
            {
                return Err(SaeInnerKktScaleError::InvalidCurvature {
                    block,
                    component,
                    gradient,
                    curvature,
                });
            }
            if curvature > 0.0 {
                let scaled = gradient.abs() / curvature;
                if !scaled.is_finite() {
                    return Err(SaeInnerKktScaleError::NonFiniteScaledGradient {
                        block,
                        component,
                        gradient,
                        curvature,
                    });
                }
                scaled_max = scaled_max.max(scaled);
            }
        }
        Ok(scaled_max)
    }

    /// The sole acceptance gate for a differentiable inner-envelope root.
    /// Objective stagnation, a finite deflated factor, or a small Newton
    /// decrement may diagnose conditioning but cannot substitute for raw or
    /// quotient KKT stationarity.
    /// #2228 DIAGNOSTIC — the INTENSIVE companion of the bar this loop enforces.
    ///
    /// `quasi_laplace_kkt_stationary` compares an EXTENSIVE L2 gradient norm
    /// (`Σ_i ‖g_t^(i)‖² + ‖g_β‖²`, a sum over rows AND atoms) against
    /// `1e-5 · (1 + ‖x‖₂)`, whose right side grows only like `sqrt(#params)`.
    /// `system_scaled_grad_max` / `inner_iterate_max` are the componentwise,
    /// Jacobi-curvature-scaled pair whose own doc says they "remove row-count,
    /// atom-count, and basis-scale extensivity" — and
    /// `SaeInstalledInnerKktAudit::certifies` already accepts that pair as
    /// SUFFICIENT. Its only production consumer is `installed_inner_kkt_audit`
    /// (the external-state certification entry), so no refusal raised by this
    /// loop has ever reported the intensive ratio and nobody can tell a units
    /// artefact from a genuinely non-stationary iterate. Reporting it costs one
    /// assembly on a path that is already returning `Err`.
    ///
    /// Diagnostic only: it does not decide, accept, or relax anything.
    fn intensive_kkt_diagnostic(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
    ) -> String {
        let system = match self.assemble_arrow_schur(target, rho, registry) {
            Ok(system) => system,
            Err(reason) => return format!("intensive=unresolved(assembly: {reason})"),
        };
        let scaled_max = match Self::system_scaled_grad_max(&system) {
            Ok(value) => value,
            Err(reason) => return format!("intensive=unresolved(scaled-grad: {reason})"),
        };
        let iterate_max = match self.inner_iterate_max() {
            Ok(value) => value,
            Err(reason) => return format!("intensive=unresolved(iterate-max: {reason})"),
        };
        let bound = SAE_MANIFOLD_INNER_GRAD_REL_TOL * iterate_max;
        let ratio = if bound > 0.0 {
            scaled_max / bound
        } else {
            f64::INFINITY
        };
        format!(
            "intensive_scaled_max={scaled_max:.6e}, intensive_bound={bound:.6e}, \
             intensive_over_bound={ratio:.3e}"
        )
    }

    /// #2762 PROBE — what the quotient removes, priced as objective motion.
    ///
    /// `quotient_residual_norm_sq` projects the KKT residual onto the complement
    /// of the chart-gauge orbit + decoder nulls before the gate reads it, on the
    /// premise that the penalized objective is flat along the removed span.
    /// `quotient_gradient_norm_sq`'s own doc records that the premise is FALSE
    /// (gam#2715/#2720: the orbit is a symmetry of the likelihood, not of the
    /// posterior) and that the precondition `maxᵢ |gᵀvᵢ| ≤ tolerance` is
    /// available at the projection site and never checked.
    ///
    /// This diagnostic checks it, and goes one step further: a nonzero `gᵀv` is
    /// a first-order statement, and a first-order statement about a direction
    /// with near-zero curvature does not say whether any FINITE motion along it
    /// actually lowers the objective. So each removed direction is also walked
    /// with a two-sided geometric line search on `penalized_objective_total` —
    /// the exact scalar the inner solve descends — and the best realized
    /// decrease is reported next to the first-order derivative.
    ///
    /// Diagnostic only: nothing here gates, accepts, or relaxes anything. It is
    /// paid on the refusal path, which is already returning `Err`.
    fn gauge_orbit_descent_diagnostic(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        lambda_smooth: &[f64],
    ) -> String {
        let system = match self.assemble_arrow_schur(target, rho, registry) {
            Ok(system) => system,
            Err(reason) => return format!("orbit=unresolved(assembly: {reason})"),
        };
        let n = self.n_obs();
        let q = self.assignment.row_block_dim();
        let dense_len = n.saturating_mul(q);
        let border_dim = self.factored_border_dim();
        if system.rows.len() != n || system.row_offsets.len() != n + 1 || system.gb.len() != border_dim
        {
            return "orbit=unresolved(non-dense layout)".to_string();
        }
        let mut gradient = Array1::<f64>::zeros(dense_len + border_dim);
        for (row_index, row) in system.rows.iter().enumerate() {
            let base = system.row_offsets[row_index];
            let dim = system.row_dims[row_index];
            if base + dim > dense_len || row.gt.len() < dim {
                return "orbit=unresolved(row layout)".to_string();
            }
            for axis in 0..dim {
                gradient[base + axis] = row.gt[axis];
            }
        }
        for (index, &value) in system.gb.iter().enumerate() {
            gradient[dense_len + index] = value;
        }

        // Same span, same order, same Gram--Schmidt as the projection the gate
        // reads, so what is measured here is exactly what is removed there.
        let gauges = match (
            self.dense_step_gauge_vectors(),
            self.joint_decoder_beta_null_directions(lambda_smooth),
            self.decoder_channel_null_directions(),
        ) {
            (Ok(chart), Ok(beta_null), Ok(channel_null)) => chart
                .into_iter()
                .chain(beta_null)
                .chain(channel_null)
                .collect::<Vec<_>>(),
            (Err(reason), _, _) | (_, Err(reason), _) | (_, _, Err(reason)) => {
                return format!("orbit=unresolved(gauge basis: {reason})");
            }
        };
        let mut orthonormal: Vec<Array1<f64>> = Vec::new();
        for mut gauge in gauges {
            if gauge.len() != gradient.len() {
                continue;
            }
            for basis in &orthonormal {
                let coeff = gauge.dot(basis);
                for index in 0..gauge.len() {
                    gauge[index] -= coeff * basis[index];
                }
            }
            let norm_sq = gauge.iter().map(|value| value * value).sum::<f64>();
            if norm_sq <= 1.0e-24 || !norm_sq.is_finite() {
                continue;
            }
            let inv_norm = norm_sq.sqrt().recip();
            for value in gauge.iter_mut() {
                *value *= inv_norm;
            }
            orthonormal.push(gauge);
        }
        if orthonormal.is_empty() {
            return "orbit=empty(no direction removed)".to_string();
        }
        let mut max_derivative = 0.0_f64;
        for basis in &orthonormal {
            max_derivative = max_derivative.max(gradient.dot(basis).abs());
        }

        // The steepest removed direction is the projection of `−g` onto the
        // removed span: one line search on it bounds what the whole span offers.
        let mut descent = Array1::<f64>::zeros(gradient.len());
        for basis in &orthonormal {
            let coeff = gradient.dot(basis);
            for index in 0..descent.len() {
                descent[index] -= coeff * basis[index];
            }
        }
        let descent_norm = descent.dot(&descent).sqrt();
        if !(descent_norm.is_finite() && descent_norm > 0.0) {
            return format!(
                "orbit_dim={}, orbit_max_dderiv={max_derivative:.6e}, orbit_descent=degenerate",
                orthonormal.len(),
            );
        }
        for value in descent.iter_mut() {
            *value /= descent_norm;
        }
        let base_objective = match self.penalized_objective_total(target, rho, registry, 1.0) {
            Ok(value) => value,
            Err(reason) => return format!("orbit=unresolved(objective: {reason})"),
        };
        let snapshot = self.snapshot_mutable_state();
        let mut best_decrease = 0.0_f64;
        let mut best_alpha = 0.0_f64;
        let mut alpha = 1.0e-8_f64;
        while alpha <= 1.0e3 {
            let applied = self
                .apply_newton_step(
                    descent.slice(s![..dense_len]),
                    descent.slice(s![dense_len..]),
                    alpha,
                )
                .is_ok();
            if applied
                && let Ok(trial) = self.penalized_objective_total(target, rho, registry, 1.0)
                && trial.is_finite()
                && base_objective - trial > best_decrease
            {
                best_decrease = base_objective - trial;
                best_alpha = alpha;
            }
            if self.restore_mutable_state(&snapshot).is_err() {
                return format!(
                    "orbit_dim={}, orbit_max_dderiv={max_derivative:.6e}, \
                     orbit_descent=unresolved(restore failed at α={alpha:.3e})",
                    orthonormal.len(),
                );
            }
            alpha *= 10.0;
        }
        let relative = if base_objective.abs() > 0.0 {
            best_decrease / base_objective.abs()
        } else {
            f64::INFINITY
        };

        // THE AMBIENT CONTROL, and it is the one that decides what this refusal
        // means. `g ≠ 0` on a differentiable objective makes `−g/‖g‖` a descent
        // direction, so if NO step along it lowers `penalized_objective_total`,
        // the assembled gradient is not the gradient of the scalar the line
        // search descends — an objective↔gradient desync — and no amount of
        // solver work can close a gap the two functions disagree about. The
        // one-sided finite difference is reported beside the analytic slope so
        // the two are compared rather than asserted.
        let mut steepest = gradient.clone();
        let steepest_norm = steepest.dot(&steepest).sqrt();
        let ambient = if steepest_norm.is_finite() && steepest_norm > 0.0 {
            for value in steepest.iter_mut() {
                *value /= -steepest_norm;
            }
            let analytic_slope = gradient.dot(&steepest);
            let mut ambient_best_drop = 0.0_f64;
            let mut ambient_best_alpha = 0.0_f64;
            let mut finite_difference = f64::NAN;
            let mut alpha = 1.0e-8_f64;
            while alpha <= 1.0e3 {
                let applied = self
                    .apply_newton_step(
                        steepest.slice(s![..dense_len]),
                        steepest.slice(s![dense_len..]),
                        alpha,
                    )
                    .is_ok();
                if applied && let Ok(trial) = self.penalized_objective_total(target, rho, registry, 1.0)
                {
                    if (alpha - 1.0e-6).abs() < 1.0e-18 && trial.is_finite() {
                        finite_difference = (trial - base_objective) / alpha;
                    }
                    if trial.is_finite() && base_objective - trial > ambient_best_drop {
                        ambient_best_drop = base_objective - trial;
                        ambient_best_alpha = alpha;
                    }
                }
                if self.restore_mutable_state(&snapshot).is_err() {
                    break;
                }
                alpha *= 10.0;
            }
            let ratio = if analytic_slope != 0.0 {
                finite_difference / analytic_slope
            } else {
                f64::NAN
            };
            format!(
                "ambient_slope={analytic_slope:.6e}, ambient_fd_slope={finite_difference:.6e} \
                 (fd/analytic {ratio:.6e}), ambient_best_objective_drop={ambient_best_drop:.6e} \
                 at α={ambient_best_alpha:.3e}"
            )
        } else {
            "ambient=degenerate".to_string()
        };

        format!(
            "orbit_dim={}, orbit_max_dderiv={max_derivative:.6e}, \
             orbit_best_objective_drop={best_decrease:.6e} at α={best_alpha:.3e} \
             (objective {base_objective:.6e}, relative {relative:.6e}), {ambient}",
            orthonormal.len(),
        )
    }

    pub(crate) fn quasi_laplace_kkt_stationary(
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
    /// undamped (ridge-0) criterion factorization, and read back both KKT residual
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
        Self::ensure_row_gauge_deflation_for_quasi_laplace(sys);
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
    /// Gauss–Newton inner loop is missing, globalized as a Levenberg–Marquardt
    /// trust region on the stationarity residual (#2762).
    ///
    /// The MM/GN inner solver is guaranteed descent but converges LINEARLY with
    /// contraction rate → 1 exactly where real data puts it: high residual (the
    /// GN data block drops first-order residual curvature) and huge near-flat
    /// bands (t-reparameterization, penalty-flat frame orientation). Measured
    /// stable-tail contraction on the production repro is 0.9965–0.9979 per
    /// iteration, i.e. ~1,800–3,000 uninterrupted iterations to close the gap
    /// from the objective-stall plateau (‖g‖ ≈ 1.4) to the KKT band — against a
    /// ~1e3 refine budget. The stall detector fires precisely when the MM phase
    /// has entered that crawl: from there, Newton on the EXACT Hessian is
    /// locally quadratic and closes the same gap in O(10) steps, making the
    /// strict KKT contract REACHABLE instead of loosened.
    ///
    /// # One merit, and it is the one the gate reads
    ///
    /// This phase solves `g(θ) = 0`, and the gate that judges it is a bound on
    /// `‖g‖` (raw or gauge-quotient). So the merit here is `½‖g‖²` and nothing
    /// else. It is a function of the STATE, not of any operator evaluated at the
    /// state, which is what makes a comparison across two states mean something.
    ///
    /// The `#2762` defect was that the acceptance test compared
    /// `gᵀB(θ₊)⁻¹g(θ₊)` — the trial state's decrement in the MAJORIZER metric —
    /// against `gᵀA⁺g` at the pre-state, in the EXACT-Hessian metric. Same
    /// bilinear form, two different operators, measured 67x apart on the
    /// witness; every step passed it, `‖g‖` rose 15x–107x per accepted step, and
    /// 482 consecutive steps were accepted after the baseline was made
    /// self-consistent, because `gᵀB(θ)⁻¹g(θ)` can fall while `‖g‖` rises
    /// whenever `B` stiffens.
    ///
    /// # Why the step is damped, and why that is the actual root cause
    ///
    /// Fixing the merit alone does not converge this phase, and the measurement
    /// says why. At the `#2015` witness — `‖g‖ = 1.23e-4`, the WHOLE residual
    /// inside the retained range — the undamped step is `‖Δ‖ = 0.44`, its full
    /// application drives the merit `7.5e-9 → 6.1e0`, and an Armijo test on
    /// `½‖g‖²` first passes at `α = 4.9e-4`, buying 0.03%. The step's LENGTH is
    /// set entirely by the near-null eigendirections of `A`; the residual is
    /// carried by the well-conditioned ones. No scalar step length separates
    /// them — shrinking the step to keep the flat direction inside the model
    /// shrinks the useful directions by the same factor.
    ///
    /// Damping does separate them. `A` is already materialized and
    /// diagonalized here, so the whole Levenberg–Marquardt path
    /// [`ExactHessianSpectralBlock::damped_residual_step`] is available in
    /// closed form at one diagonal pass per point — including the modeled
    /// residual this caller prices in its quotient merit. On the same witness
    /// `ν = 5.7e-7` gives `‖Δ‖ = 4.6e-4`, drives
    /// the merit `7.5e-9 → 6.2e-11` (`‖g‖ 1.23e-4 → 1.11e-5`, past a `7.1e-5`
    /// tolerance in ONE step) at a measured/predicted ratio of `0.9992`.
    ///
    /// # The ladder, and why every number in it is derived
    ///
    /// * The first trial is `ν = 0` — the undamped step this phase has always
    ///   taken — so the quadratic tail near a well-conditioned root is
    ///   unchanged, and a state that never needed damping never pays for it.
    /// * The ladder then runs from `λ_min²` to `λ_max²` over the RETAINED
    ///   spectrum by [`opt::constants::RIDGE_GROWTH`]: below `λ_min²` a damping
    ///   cannot move the flattest resolved direction, above `λ_max²` it has
    ///   already flattened every direction there is.
    /// * The accepted damping is CARRIED to the next step (divided by the same
    ///   growth, and snapped back to `0` once it falls under `λ_min²`), so a
    ///   converging tail walks back to the pure Newton step by itself.
    /// * A trial is accepted when its MEASURED merit reduction is at least the
    ///   shared Armijo fraction [`SAE_MANIFOLD_ARMIJO_C1`] of the reduction its
    ///   own closed-form model predicted, with the shared round-off cushion.
    ///   Model-predicted reduction is monotonically decreasing along the ladder,
    ///   so a ladder that falls under the round-off floor
    ///   [`SAE_MANIFOLD_DIRECTIONAL_DECREASE_REL_FLOOR`] × merit is exhausted —
    ///   that is a proof of termination, not a cap.
    ///
    /// Consequences worth stating as properties:
    ///
    /// * every accepted step STRICTLY decreases the quantity the refusal is
    ///   denominated in, so this phase can no longer leave the state worse than
    ///   it found it — which is what it measurably did on both `#2762` witnesses;
    /// * the merit is monotone across steps by construction, so no
    ///   cross-iteration contraction bail is needed and none is kept;
    /// * a trial costs ONE assembly. It used to cost an assembly plus a full
    ///   arrow factorization, because the merit it evaluated needed one.
    ///
    /// Indefiniteness of `A` needs no special handling: `Δ(ν)` solves
    /// `(A² + ν)Δ = −Ag`, whose operator is positive semidefinite for every
    /// symmetric `A`, so a resolved negative mode is descended, not reflected.
    /// Every internal failure degrades to `Ok(false)` (fall through to the
    /// historical stall accounting), never to a new error class, and a rejected
    /// trial restores the snapshot bit-for-bit.
    ///
    /// Returns `Ok(true)` when at least one step was committed (the caller
    /// re-enters the refine loop, whose existing raw/quotient KKT gate +
    /// idempotence certificate remain the SOLE acceptance authority — this
    /// phase mints nothing).
    /// A residual measured in the two currencies the inner gate speaks: the
    /// gauge-QUOTIENT merit `½‖Π⊥null r‖²` — which is what
    /// [`Self::quasi_laplace_kkt_stationary`] is a bound on, since the quotient
    /// norm is clamped at or below the raw one — and the AMBIENT merit `½‖r‖²`.
    ///
    /// Returned together because the polish accepts on the first and holds the
    /// second as an invariant: a step may not buy quotient progress by pumping
    /// residual into the gauge orbit, which is the only way a projected norm can
    /// fall without the residual falling.
    fn residual_merits(
        &self,
        residual: &SaeArrowVector,
        penalized_gram_scale: &[f64],
    ) -> ResidualMerits {
        let ambient_norm_sq =
            residual.t.dot(&residual.t) + residual.beta.dot(&residual.beta);
        let quotient_norm_sq = self
            .quotient_gradient_norm_sq(
                residual.t.view(),
                residual.beta.view(),
                ambient_norm_sq,
                penalized_gram_scale,
            )
            .unwrap_or(ambient_norm_sq);
        ResidualMerits {
            quotient: 0.5 * quotient_norm_sq,
            ambient: 0.5 * ambient_norm_sq,
        }
    }

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
        // #2228 — caller's cross-round best-seen accumulator, keyed on the
        // ½λ²/scale certificate. Captured HERE because the polish is where the
        // decrement is evaluated per step. #2762: the excursion it was
        // introduced to undo can no longer be produced by this phase, whose
        // merit is monotone; it remains the CALLER's accumulator across rounds
        // and across the other movers, and is left keyed on the caller's own
        // acceptance currency.
        best_seen: &mut Option<(f64, f64, SaeManifoldMutableState)>,
    ) -> Result<bool, String> {
        let mut made_progress = false;
        // Warm-carried Levenberg--Marquardt damping. `0` is the undamped exact
        // Newton step, so the very first trial of the very first step is
        // byte-identical to the step this phase has always proposed.
        let mut damping = 0.0_f64;
        // #2267 — the polish's own elapsed clock, one candidate denominator for the
        // size predicate this route still lacks.
        let polish_started = std::time::Instant::now();
        for step in 0..max_steps {
            let step_started = std::time::Instant::now();
            let mut sys = self
                .assemble_arrow_schur(target, rho_fixed, registry)
                .map_err(|err| format!("SaeManifoldTerm::terminal_exact_newton_polish: {err}"))?;
            let assemble_seconds = step_started.elapsed().as_secs_f64();
            let grad_norm_sq = Self::system_grad_norm_sq(&sys);
            if !grad_norm_sq.is_finite() {
                log::debug!("terminal Newton bail: non-finite ‖g‖² at entry");
                break;
            }
            let grad_norm = grad_norm_sq.sqrt();
            let quotient_grad_norm =
                self.quotient_gradient_norm_from_system(&sys, grad_norm_sq, lambda_smooth);
            if Self::quasi_laplace_kkt_stationary(grad_norm, quotient_grad_norm, grad_tolerance) {
                // In the band: hand back to the refine loop, whose gate +
                // idempotence certificate decide acceptance.
                return Ok(true);
            }
            // Ridge-0 deflated criterion factor = the B-preconditioner for the
            // exact-pencil GMRES (identical to the outer IFT's preconditioner).
            // It is NOT this phase's merit (#2762): it supplies the factor cache
            // the exact-A materialization is built on, and the ½λ²/scale
            // certificate the CALLER accumulates in `best_seen`.
            let factor = match self.factor_deflated_evidence_with_grad_norms(
                &mut sys,
                lambda_smooth,
                options,
            ) {
                Ok(factor) => factor,
                Err(err) => {
                    log::debug!(
                        "terminal Newton bail: deflated criterion factor at ‖g‖={grad_norm:.6e}: {err}"
                    );
                    break;
                }
            };
            let decrement_sq = sae_manifold_newton_directional_decrease(
                &sys,
                factor.delta_t.view(),
                factor.delta_beta.view(),
            )
            .max(0.0);
            let cert = if objective_scale.is_finite() && objective_scale > 0.0 {
                0.5 * decrement_sq / objective_scale
            } else {
                f64::INFINITY
            };
            if cert.is_finite() && best_seen.as_ref().is_none_or(|(c, _, _)| cert < *c) {
                *best_seen = Some((cert, grad_norm, self.snapshot_mutable_state()));
            }
            // #2472 — one line per Newton step, so a criterion evaluation that
            // has not returned can be read as "still contracting" or "grinding
            // at a fixed ‖g‖" from the log alone. Bounded by `max_steps`.
            log::info!(
                "[SAE-NEWTON] polish step {}/{max_steps}: ‖g‖={grad_norm:.6e} \
                 (quotient {quotient_grad_norm:.6e}, tol {grad_tolerance:.3e}) \
                 λ²={decrement_sq:.6e} cert={cert:.6e}",
                step + 1,
            );
            let cache = factor.cache;
            // #2267 — FORECAST the dense exact-stationarity step before entering it,
            // and state it next to the two quantities any bar would be denominated
            // against: the assemble this step already paid, and the time this polish
            // call has burned so far. Measured on the shipped ladder's K=8 rung, the
            // materialization's column loop alone is 406.5 s of a step the loop is
            // permitted to repeat 64 times; measured on the shipped 160-row demo the
            // whole step is ~1.6 s. A predicate that refuses the first and admits the
            // second has to be read off BOTH, which is what this line is for. No bar
            // is applied here: choosing one from a single fixture is how a literal
            // gets laundered into a threshold.
            match self.exact_stationarity_materialization_forecast(rho_fixed, target, &cache) {
                Ok((forecast_dim, forecast)) => log::info!(
                    "[SAE-NEWTON] step {}/{max_steps} FORECAST: dim={forecast_dim}, \
                     materialization >= {:.3} s (column loop only, eigendecomposition \
                     NOT included), assemble={:.3} s, polish elapsed={:.3} s",
                    step + 1,
                    forecast.as_secs_f64(),
                    assemble_seconds,
                    polish_started.elapsed().as_secs_f64(),
                ),
                Err(err) => log::info!(
                    "[SAE-NEWTON] step {}/{max_steps} FORECAST unavailable: {err}",
                    step + 1,
                ),
            }
            // The stationarity residual `g` as one ambient vector. The damped
            // path below solves against `−g`; reporting the model residual for
            // `g` itself keeps the predicted and the measured merit in the same
            // units.
            let mut residual_t = Array1::<f64>::zeros(cache.delta_t_len());
            let mut offset = 0usize;
            for row in &sys.rows {
                for (axis, &g) in row.gt.iter().enumerate() {
                    residual_t[offset + axis] = g;
                }
                offset += row.gt.len();
            }
            let residual = SaeArrowVector {
                t: residual_t,
                beta: sys.gb.clone(),
            };
            // ONE eigendecomposition of `A` per step; every damping below reads
            // it. This is the same materialization the undamped solve paid for.
            let geometry =
                match self.materialize_exact_stationarity_geometry(rho_fixed, target, &cache) {
                    Ok(geometry) => geometry,
                    Err(err) => {
                        log::debug!(
                            "terminal Newton bail: dense exact-stationarity geometry at \
                             ‖g‖={grad_norm:.6e}: {err}"
                        );
                        break;
                    }
                };
            let Some((curvature_min, curvature_max)) = geometry.retained_curvature_extremes()
            else {
                log::debug!(
                    "terminal Newton bail: every direction of A is inside its own null band at \
                     ‖g‖={grad_norm:.6e} — no step of this operator can move the residual",
                );
                break;
            };
            let smallest_damping = curvature_min * curvature_min;
            let largest_damping = curvature_max * curvature_max;
            // The gate is `raw ≤ tol OR quotient ≤ tol`, and the quotient norm
            // is clamped at or below the raw one, so the gate IS the quotient
            // bound and the quotient merit is the phase's currency. The ambient
            // merit is carried alongside as the invariant, not as a second
            // acceptance test (see `residual_merits`).
            let pre_merits = ResidualMerits {
                quotient: 0.5 * quotient_grad_norm * quotient_grad_norm,
                ambient: 0.5 * grad_norm_sq,
            };
            // Round-off floor on the MODEL's predicted reduction, in the merit's
            // own units — the same relative floor the majorized Armijo lane
            // applies to its directional decrease. A prediction below it is
            // f64 noise in the quadratic model, not a step worth measuring.
            let predicted_floor =
                SAE_MANIFOLD_DIRECTIONAL_DECREASE_REL_FLOOR * pre_merits.quotient;
            let snapshot = self.snapshot_mutable_state();
            let backtrack_started = std::time::Instant::now();
            let mut trials = 0usize;
            let mut accepted: Option<AcceptedTerminalResidualStep> = None;
            let mut nu = damping;
            loop {
                let damped = match geometry.damped_residual_step(&residual, nu) {
                    Ok(damped) => damped,
                    Err(err) => {
                        log::debug!(
                            "terminal Newton bail: damped residual step at ν={nu:.6e}: {err}"
                        );
                        break;
                    }
                };
                let model_merits = self.residual_merits(&damped.model_residual, lambda_smooth);
                let predicted_quotient_decrease =
                    pre_merits.quotient - model_merits.quotient;
                if !(predicted_quotient_decrease.is_finite()
                    && predicted_quotient_decrease > predicted_floor)
                {
                    // The model's predicted reduction decreases monotonically in
                    // ν, so no larger damping on this ladder can clear the floor
                    // either: the ladder is exhausted, and it is exhausted for a
                    // stated reason rather than at a trial count.
                    log::debug!(
                        "terminal Newton: damping ladder exhausted at ν={nu:.6e} — predicted \
                         quotient-merit reduction {predicted_quotient_decrease:.6e} is under the \
                         round-off floor {predicted_floor:.6e} (quotient merit \
                         {:.6e})",
                        pre_merits.quotient,
                    );
                    break;
                }
                trials += 1;
                let (trial_merits, trial_system) = if self
                    .apply_newton_step(damped.step.t.view(), damped.step.beta.view(), 1.0)
                    .is_ok()
                {
                    match self.assemble_arrow_schur(target, rho_fixed, registry) {
                        Ok(trial_sys) => {
                            let ambient_sq = Self::system_grad_norm_sq(&trial_sys);
                            let quotient = self.quotient_gradient_norm_from_system(
                                &trial_sys,
                                ambient_sq,
                                lambda_smooth,
                            );
                            (
                                ResidualMerits {
                                    quotient: 0.5 * quotient * quotient,
                                    ambient: 0.5 * ambient_sq,
                                },
                                Some(trial_sys),
                            )
                        }
                        Err(_) => (
                            ResidualMerits {
                                quotient: f64::INFINITY,
                                ambient: f64::INFINITY,
                            },
                            None,
                        ),
                    }
                } else {
                    (
                        ResidualMerits {
                            quotient: f64::INFINITY,
                            ambient: f64::INFINITY,
                        },
                        None,
                    )
                };
                let sufficient = SAE_MANIFOLD_ARMIJO_C1 * predicted_quotient_decrease;
                // Acceptance: a measured reduction of the GATE's currency worth
                // at least the shared Armijo fraction of what this step's own
                // model predicted. Invariant, not a second currency: the ambient
                // residual may not GROW, so quotient progress can never be
                // bought by pumping residual into the gauge orbit — which is the
                // only way a projected norm falls while the residual does not.
                if trial_merits.quotient.is_finite()
                    && trial_merits.ambient.is_finite()
                    && pre_merits.quotient - trial_merits.quotient
                        >= sufficient - opt::armijo_roundoff_cushion(pre_merits.quotient)
                    && trial_merits.ambient
                        <= pre_merits.ambient + opt::armijo_roundoff_cushion(pre_merits.ambient)
                {
                    accepted = Some(AcceptedTerminalResidualStep {
                        damping: nu,
                        trial_merits,
                        predicted_quotient_decrease,
                        step: damped,
                        system: trial_system,
                    });
                    break;
                }
                self.restore_mutable_state(&snapshot)?;
                let next = if nu > 0.0 {
                    nu * opt::constants::RIDGE_GROWTH
                } else {
                    smallest_damping
                };
                if next > largest_damping {
                    log::debug!(
                        "terminal Newton: damping ladder exhausted at ν={next:.6e} — past \
                         λ_max²={largest_damping:.6e}, where every direction is already damped"
                    );
                    break;
                }
                // A rung that does not strictly advance is not a rung. `λ_min²`
                // underflows to zero once the retained spectrum is below
                // `1e-154`, and without this the ν = 0 trial would be retried
                // forever; the exhaustion floor above cannot catch it, because
                // the undamped model can predict a large reduction while
                // delivering none. Termination is a property of the ladder, so
                // it is enforced on the ladder.
                if !(next > nu) {
                    log::debug!(
                        "terminal Newton: damping ladder cannot advance past ν={nu:.6e} \
                         (λ_min²={smallest_damping:.6e} is not representable above it)"
                    );
                    break;
                }
                nu = next;
            }
            let Some(accepted) = accepted else {
                log::debug!(
                    "terminal Newton bail: no damping on [{smallest_damping:.6e}, \
                     {largest_damping:.6e}] bought a sufficient measured decrease of the \
                     residual merit at ‖g‖={grad_norm:.6e} ({trials} trial(s))"
                );
                break;
            };
            made_progress = true;
            let gate_norm = quotient_grad_norm;
            // #2762 — SPEND THE BUDGET ONLY WHILE THE PHASE IS ON TRACK TO FINISH.
            //
            // A step here costs one dense eigendecomposition of `A` — measured
            // 13.4 s at `dim = 519` on the `zz2015` witness, against 0.14 s of
            // assembly and 0.28 s for the whole damping ladder. The step COUNT
            // is therefore the entire cost of this phase, and a fixed cap prices
            // every entry at the worst case.
            //
            // The gate this phase is trying to reach bounds
            // `min(‖g‖, ‖Π⊥null g‖)`, so that is the quantity to extrapolate,
            // and it is read off the system the accepted trial already
            // assembled — the test costs nothing and fires one whole
            // eigendecomposition earlier than it could at the next loop top. At
            // the contraction the step actually delivered, the band is
            // `ln(tol/gate)/ln(contraction)` steps away; if that exceeds the
            // steps left, this phase cannot finish on its current trajectory and
            // every further step is one it will not be paid for. Measured on
            // `zz2015`: steps 1-4 take `‖g‖ 14.54 → 0.724`, and steps 5-64 buy
            // 15% for 60 x 13.4 s.
            //
            // Stopping here is neither a refusal nor final. The merit is
            // monotone, so everything gained is kept; `made_progress` is already
            // true, so the refine loop takes another window and may re-arm this
            // phase, and the trajectory is re-measured from scratch when it does.
            if let Some(system) = accepted.system.as_ref() {
                let after_sq = Self::system_grad_norm_sq(system);
                let after_gate =
                    self.quotient_gradient_norm_from_system(system, after_sq, lambda_smooth);
                if Self::quasi_laplace_kkt_stationary(
                    after_sq.sqrt(),
                    after_gate,
                    grad_tolerance,
                ) {
                    // The step landed in the band. Say so without paying for the
                    // next loop top's assembly to rediscover it.
                    log::debug!(
                        "SAE terminal Newton reached the KKT band at step {}: gate norm \
                         {gate_norm:.6e} → {after_gate:.6e} against tol {grad_tolerance:.6e}",
                        step + 1,
                    );
                    return Ok(true);
                }
                let contraction = if gate_norm > 0.0 {
                    after_gate / gate_norm
                } else {
                    f64::INFINITY
                };
                let remaining = (max_steps - step - 1) as f64;
                let projected_steps = if contraction > 0.0 && contraction < 1.0 {
                    (grad_tolerance / after_gate).ln() / contraction.ln()
                } else {
                    f64::INFINITY
                };
                if !(projected_steps <= remaining) {
                    log::debug!(
                        "terminal Newton: stopping on trajectory, not on budget — step {} \
                         contracted the gate norm by {contraction:.6e} ({gate_norm:.6e} → \
                         {after_gate:.6e}), which puts the {grad_tolerance:.6e} band \
                         {projected_steps:.1} steps away against {remaining:.0} remaining",
                        step + 1,
                    );
                    break;
                }
            }
            // Walk back toward the undamped Newton step: a damping under
            // `λ_min²` cannot move the flattest resolved direction, so it IS the
            // undamped step and is carried as exactly that.
            damping = accepted.damping / opt::constants::RIDGE_GROWTH;
            if damping < smallest_damping {
                damping = 0.0;
            }
            // The penalized objective at the committed state, next to the merit
            // the step was accepted on: this phase descends ‖g‖², the refine
            // window descends the objective, and whether they agree on the
            // direction of progress is readable only with both on one line.
            let committed_objective = self
                .penalized_objective_total(target, rho_fixed, registry, 1.0)
                .unwrap_or(f64::NAN);
            log::info!(
                "[SAE-NEWTON] step {} phases: assemble={assemble_seconds:.2}s \
                 trials={trials} in {:.2}s (ν={:.6e}, ‖Δ‖={:.6e}, damped rank {}/{}) \
                 total={:.2}s penalized_objective={committed_objective:.10e}",
                step + 1,
                backtrack_started.elapsed().as_secs_f64(),
                accepted.damping,
                accepted.step.step_norm_sq.sqrt(),
                accepted.step.retained_rank,
                geometry.eigenvalues.len(),
                step_started.elapsed().as_secs_f64(),
            );
            log::debug!(
                "SAE terminal Newton step committed: quotient merit {:.6e} → {:.6e} \
                 (predicted quotient reduction {:.6e}, measured {:.6e}, ratio {:.4e}); \
                 ambient merit {:.6e} → {:.6e}; ‖g‖ {grad_norm:.6e} → {:.6e}, tol \
                 {grad_tolerance:.6e}",
                pre_merits.quotient,
                accepted.trial_merits.quotient,
                accepted.predicted_quotient_decrease,
                pre_merits.quotient - accepted.trial_merits.quotient,
                if accepted.predicted_quotient_decrease > 0.0 {
                    (pre_merits.quotient - accepted.trial_merits.quotient)
                        / accepted.predicted_quotient_decrease
                } else {
                    f64::NAN
                },
                pre_merits.ambient,
                accepted.trial_merits.ambient,
                (2.0 * accepted.trial_merits.ambient).max(0.0).sqrt(),
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
        let mut raw_gauges = self
            .joint_chart_gauge_basis_for_arrow_layout(
                &cache.row_offsets,
                cache.k,
                "outer_gradient_arrow_solver chart gauges",
            )
            .map_err(OuterGradientError::internal)?;
        // #2253: everything pushed above comes from `dense_step_gauge_vectors`
        // — the closed-form CHART gauge orbit (circle/torus phase, and the
        // translation/scale orbits of the linear/euclidean/duchon/poincaré
        // patches).
        //
        // #2720 — READ THE SCOPE OF "EXACT" HERE CAREFULLY. This comment used
        // to call them "EXACT criterion symmetries … flat by construction",
        // and that sentence is what put the same orbit into the inner
        // CONVERGENCE quotient, where it certified non-stationary points at up
        // to 76 170x the KKT tolerance. They are exact symmetries of the
        // RECONSTRUCTION (measured `1e-16` relative) and NOT of the criterion:
        // the ARD prior on `t` and the smoothness prior on `β` are written on
        // the chart coordinates and move along the orbit — the dilation field
        // by `−7.82` on an objective of `165`
        // (`tests_gauge_posterior_flatness_2720`).
        //
        // What justifies deflating them HERE is a different property and a
        // weaker one: this block runs only after the conditioning gate has
        // already flagged a near-singular joint Hessian, and the orbit carries
        // NO data-fit CURVATURE (only the priors'), so it is a genuine
        // near-null direction OF THE OPERATOR BEING INVERTED. Deflation is then
        // a pseudo-inverse choice on an ill-conditioned solve, not a claim that
        // the criterion is flat. That distinction is the whole of #2720, and it
        // is written here because this is the other site the claim reached.
        //
        // Remember the boundary so the exact-gauge subspace can be deflated
        // UNCONDITIONALLY, keeping the deflation COUNT stable across the ρ-walk
        // (a borderline eigenvalue flickering across the Rayleigh floor
        // re-anchors ½log|H| and desyncs the fixed-ρ criterion gradient from
        // its value).
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
            let mapped = self
                .dense_joint_vector_in_arrow_layout(
                    dir.view(),
                    &cache.row_offsets,
                    cache.k,
                    "outer_gradient_arrow_solver decoder-beta null",
                )
                .map_err(OuterGradientError::internal)?;
            raw_gauges.push(mapped);
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
            let mapped = self
                .dense_joint_vector_in_arrow_layout(
                    dir.view(),
                    &cache.row_offsets,
                    cache.k,
                    "outer_gradient_arrow_solver decoder-channel null",
                )
                .map_err(OuterGradientError::internal)?;
            raw_gauges.push(mapped);
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
        // is `gauge_span[0..exact_basis_count]` (reconstruction-flat by
        // construction, hence data-fit-curvature-free — NOT criterion-flat, see
        // the scope note at the candidate site above); add any
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
        self.assignment.validate_rho_domain(rho)?;
        let mut acc = 0.0_f64;
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let rank_s = Self::symmetric_rank(atom.smooth_penalty())?;
            // Penalized decoder dimension: `r_k` coordinate channels carry the
            // `S_k` roughness penalty (full-`B` path ⇒ `r_k == p`).
            let penalized_channel_dim = atom.border_frame_rank() * rank_s;
            let log_lambda = rho.log_lambda_smooth[atom_idx];
            acc += 0.5 * (penalized_channel_dim as f64) * log_lambda;
        }
        // `V = … − occam`, so the net occam SUBTRACTS the penalty normalizer.
        Ok(acc)
    }

    /// Per-atom derivative `∂(occam)/∂log λ_smooth[k]` (#1556): atom `k`'s entry
    /// is `½·r_k·rank(S_k)` throughout the validated log-strength domain,
    /// matching the per-atom Occam term exactly. The
    /// unpenalized-frame `frame_dim` term carries no `log λ` dependence and is
    /// absent from both. Returns one entry per atom in atom order.
    pub(crate) fn reml_occam_log_lambda_smooth_derivative(
        &self,
        rho: &SaeManifoldRho,
    ) -> Result<Vec<f64>, String> {
        self.assignment.validate_rho_domain(rho)?;
        let mut out = Vec::with_capacity(self.atoms.len());
        for atom in self.atoms.iter() {
            let rank_s = Self::symmetric_rank(atom.smooth_penalty())?;
            let penalized_channel_dim = atom.border_frame_rank() * rank_s;
            out.push(0.5 * (penalized_channel_dim as f64));
        }
        Ok(out)
    }

    /// Streaming criterion that RETURNS the converged arrow-factor cache — the
    /// per-row factored Hessian (matrix-free, feasible at massive K; the dense
    /// `border_dim²` Schur is NEVER formed here), so the EFS hyperparameter lane
    /// can take its matrix-free ARD / smoothness traces off this cache in the
    /// streaming regime instead of hard-erroring on the dense criterion path. The
    /// log-determinant is the chunked matrix-free `streaming_exact_arrow_log_det`.
    /// Convenience over [`Self::penalized_quasi_laplace_criterion_streaming_exact_with_cache_and_lane`]
    /// with no #2080 surrogate lane (bit-identical SLQ evidence).
    pub fn penalized_quasi_laplace_criterion_streaming_exact_with_cache(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
    ) -> Result<(f64, SaeManifoldLoss, ArrowFactorCache), SaeCriterionError> {
        self.penalized_quasi_laplace_criterion_streaming_exact_with_cache_and_lane(
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

    /// [`Self::penalized_quasi_laplace_criterion_streaming_exact_with_cache`] with the #2080 surrogate
    /// lane threaded to the streaming `log|S|` term (`None` = bit-identical SLQ).
    pub fn penalized_quasi_laplace_criterion_streaming_exact_with_cache_and_lane(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
        lane: Option<&mut SurrogateLaneState>,
    ) -> Result<(f64, SaeManifoldLoss, ArrowFactorCache), SaeCriterionError> {
        let (cost, loss, cache, _artifacts) = self
            .penalized_quasi_laplace_criterion_streaming_exact_with_cache_lane_and_system(
                target,
                rho,
                registry,
                inner_max_iter,
                learning_rate,
                ridge_ext_coord,
                ridge_beta,
                lane,
            )?;
        Ok((cost, loss, cache))
    }

    /// Matrix-free outer value/gradient artifact. Unlike the scalar/cache
    /// convenience entries, this requires the rational surrogate to retain its
    /// complete weighted shifted-solve derivative and the exact
    /// `ArrowSchurSystem` used to produce it. Optional shift-zero inverse probes
    /// are requested separately and are scoped to EFS proposals.
    ///
    /// Per-row spectral deflation is ADMITTED (#2515). It was refused here for as
    /// long as the arrow route and the dense route priced a deflated direction
    /// differently; since #2673 unified the classification metric they do not, and
    /// the two complete gradients agree to `1.6e-9` relative on #2712's certified
    /// deflated anchor. The body carries the four eras of that refusal's stated
    /// reason and the measurement that ended it, because the recurring defect at
    /// this seam is a refusal outliving the disagreement it was written for.
    pub(crate) fn penalized_quasi_laplace_streaming_outer_evaluation(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
        lane: &mut SurrogateLaneState,
        need_efs_inverse_probes: bool,
    ) -> Result<StreamingOuterEvaluation, SaeCriterionError> {
        lane.request_logdet_derivative_bundle();
        if need_efs_inverse_probes {
            lane.request_inverse_probes();
        }
        let evaluated = self
            .penalized_quasi_laplace_criterion_streaming_exact_with_cache_lane_and_system(
                target,
                rho,
                registry,
                inner_max_iter,
                learning_rate,
                ridge_ext_coord,
                ridge_beta,
                Some(&mut *lane),
            );
        let (cost, loss, cache, artifacts) = match evaluated {
            Ok(evaluated) => evaluated,
            Err(error) => {
                drop(lane.take_logdet_derivative_bundle());
                drop(lane.take_inverse_probes());
                return Err(error);
            }
        };
        let logdet_derivative_bundle = lane.take_logdet_derivative_bundle().ok_or_else(|| {
            SaeCriterionError::Numerical(
                "streaming outer evaluation did not emit the rational value's derivative bundle"
                    .to_string(),
            )
        })?;
        let efs_inverse_probe_bundle = lane.take_inverse_probes();
        if need_efs_inverse_probes && efs_inverse_probe_bundle.is_none() {
            return Err(SaeCriterionError::Numerical(
                "streaming EFS evaluation did not emit its requested shift-zero inverse probes"
                    .to_string(),
            ));
        }
        let StreamingEvidenceArtifacts {
            majorizer_system: system,
            exact_a_cache,
        } = artifacts.ok_or_else(|| {
            SaeCriterionError::Numerical(
                "streaming outer evaluation did not retain its matrix-free evidence system \
                 and exact-A factor cache"
                    .to_string(),
            )
        })?;
        // #2515 — THE SPECTRAL-DEFLATION REFUSAL THAT STOOD HERE IS GONE, AND THE
        // NUMBER IT WAS RETAINED ON IS WHY. Four eras, each disproved by a
        // measurement rather than by an argument; keeping all four because the
        // failure mode this seam keeps producing is a refusal whose justification
        // outlives the defect it was written for.
        //
        // Era 1 said the from-probes cluster could not price per-row deflation.
        // #2712 disproved that: `A_i` is the conditioned row Cholesky, so the
        // reconstructed block IS the deflated `(H⁻¹)_tt`, and every channel applies
        // the same Daleckii–Krein correction its dense sibling applies.
        //
        // Era 2 named the #2499/#2515 β-Schur smoothness-EDF desync — the dense
        // route contracting a β-Schur deflated pseudo-inverse while the bundle
        // contracted "whatever `S⁻¹` it carries". Fixed by the typed
        // `BundleEvidenceGeometry`: the bundle now carries the exact observed
        // information's own reduced Schur AND its row factors, and the two routes
        // agree to `1.57e-14` on the complete gradient at a non-deflating state
        // (`laplace_value_and_gradient_are_route_invariant_2515`).
        //
        // Era 3 (`ac66e624d`) measured, on #2712's certified deflated anchor, a
        // complete-gradient gap of `9.131537e0` against `‖g‖∞ = 5.004339e0` and
        // attributed it to two floors in two metrics — the dense route flooring the
        // spectrum of the materialized `A` against an ABSOLUTE band while the arrow
        // route conditions per row. It said, in as many words, that the way to lift
        // this gate was to reconcile those two prices.
        //
        // Era 4: #2673 reconciled them (`00c1fe139`, `758c9d336`) — the absolute
        // floor is deleted and BOTH sites now classify a direction by its curvature
        // in the majorizer metric, `max(dim·ε·‖A‖₂, √ε·vᵀBv)`. That was not done for
        // this gate and this gate was not re-measured against it. Re-measured now,
        // same anchor, same comparison, `zz_attribute_deflated_route_classification_2515`
        // and `exact_a_route_parity_holds_on_a_deflated_cache_2515`:
        //
        //     majorizer deflated rows 10, exact-A deflated rows 10
        //     complete gradient max|Δ| = 2.798722e-8 against ‖g‖∞ = 1.726754e1
        //                              = 1.62e-9 RELATIVE, from 1.8 relative
        //
        // and END TO END through this very function, forced onto the streaming route
        // at the same state (`forced_streaming_admits_a_deflating_state_and_matches_-
        // dense_2515`):
        //
        //     cost      dense 1.7469252484e1   streaming 1.7469252476e1
        //     gradient  max|Δ| = 3.301233e-8 against ‖g‖∞ = 1.726754e1
        //
        // and across a ρ ladder of deflating states rather than one anchor
        // (`exact_a_route_parity_holds_across_a_deflating_rho_ladder_2515`), where the
        // gap stays ABSOLUTE at ~3e-8 while ‖g‖∞ moves over a decade — which is the
        // signature of the attributed cause below and not of a route-dependent
        // criterion.
        //
        // and the classification is now agreed direction for direction: on that
        // anchor the dense route pins nothing, prices no clamp-attributable negative,
        // and reads `log|A_tt| = 2.2623032065e1` against the arrow route's
        // `2.2623032490e1` over the same thirty directions.
        //
        // The residual `1.6e-9` is NOT machine precision and is not noise; it is
        // attributed, and the attribution is under test. The dense route materializes
        // `A` through `apply_cached_arrow_hessian`, which applies the CONDITIONED row
        // factor, so a `B`-deflated direction enters the dense `A` as `1 + ΔC_vv`;
        // the arrow route assembles `B_raw + ΔC` and unit-pins the result, so the
        // same direction is exactly `1`. Both honour "a deflated direction is unit
        // stiffness"; they disagree about whether `ΔC` is added before or after the
        // pinning, and `ΔC_vv ~ 1e-8` there. See
        // `dense_exact_a_prices_a_b_deflated_direction_as_one_plus_delta_c_2515`.
        //
        // So the criterion is one criterion on both routes, and a deflating state is
        // ADMITTED rather than refused. `cache` is still the `B` stationarity
        // geometry the IFT solve rides and `exact_a_cache` is still what the
        // from-probes channels reconstruct their inverse blocks from — the two are
        // different factorizations of different operators by design (#2515), which
        // is exactly why neither one deflating is a reason to withhold the gradient.
        Ok(StreamingOuterEvaluation {
            cost,
            loss,
            cache,
            system,
            exact_a_cache,
            logdet_derivative_bundle,
            efs_inverse_probe_bundle,
        })
    }

    /// #2515 — ONE CRITERION EVALUATION = ONE OBJECTIVE, and the evidence
    /// assembly is part of the evaluation.
    ///
    /// `converge_inner_for_undamped_logdet` freezes the collapse-prevention gates,
    /// converges, and then RESTORES the flag. The evidence assembly that prices
    /// the criterion runs after that restore, so `assemble_arrow_schur_scaled`
    /// re-refreshed all three gates from the MOVED state: the factor cache held
    /// the entry-state gates and the system it is paired with held the
    /// post-convergence ones. `validate_matrix_free_arrow_pair` then refused the
    /// pair, and the streaming outer gradient did not exist at all on a state the
    /// dense route ranks and differentiates without complaint:
    ///
    /// ```text
    /// smooth=-1.10  dense     cost=1.8195496423e1  ||g||inf=1.580471e1
    ///               streaming cost=1.8195496415e1  GRADIENT REFUSED: … refuses a
    ///                         stale matrix-free system/cache pair (row fingerprint
    ///                         4241518385832902043 vs 17638973738998200310,
    ///                         manifold fingerprint EQUAL)
    /// ```
    ///
    /// The manifold fingerprints match and the row ones do not, which is the
    /// signature `evidence_assembly_row_fingerprint_sources_2515` attributes: with
    /// gates held frozen the two assemblers agree bit for bit, and with one
    /// assembler the gate state alone moves the row fingerprint. That test was
    /// landed by `b5506eeaa` naming this as "Cause 2 (real, but not sufficient)";
    /// its Cause 1 was retracted in `60feddc2e`, which fixed the fingerprint's
    /// IDENTITY but not the state the fingerprint correctly reports as different.
    ///
    /// So the freeze belongs at the EVALUATION scope, which is here. The body is
    /// the same function; this wrapper only decides when the gates move.
    fn penalized_quasi_laplace_criterion_streaming_exact_with_cache_lane_and_system(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
        lane: Option<&mut SurrogateLaneState>,
    ) -> Result<
        (
            f64,
            SaeManifoldLoss,
            ArrowFactorCache,
            Option<StreamingEvidenceArtifacts>,
        ),
        SaeCriterionError,
    > {
        self.assignment.validate_rho_domain(rho)?;
        let mut rho_fixed = rho.clone();
        // The initial fit stays OUTSIDE the freeze, deliberately. The dense sibling
        // `penalized_quasi_laplace_criterion_with_cache` runs the identical driver
        // outside its own freeze, and the two routes have to put the inner solve at
        // the SAME state or the criterion they each price is a different criterion —
        // which is the defect this issue is, in the one place it would be easiest to
        // reintroduce while fixing it.
        let initial_fit = self.run_joint_fit_arrow_schur_for_quasi_laplace(
            target,
            &mut rho_fixed,
            registry,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
        )?;
        let gates_were_frozen = self.freeze_collapse_prevention_gates();
        let out = self.penalized_quasi_laplace_criterion_streaming_exact_gate_frozen(
            target,
            rho,
            &mut rho_fixed,
            registry,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
            initial_fit,
            lane,
        );
        self.streaming_gates_frozen = gates_were_frozen;
        out
    }

    fn penalized_quasi_laplace_criterion_streaming_exact_gate_frozen(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        rho_fixed: &mut SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
        initial_fit: crate::manifold::fit_drivers::EvidenceJointFitOutcome,
        lane: Option<&mut SurrogateLaneState>,
    ) -> Result<
        (
            f64,
            SaeManifoldLoss,
            ArrowFactorCache,
            Option<StreamingEvidenceArtifacts>,
        ),
        SaeCriterionError,
    > {
        let mut loss = initial_fit.loss;
        let mut criterion_fixed_point = initial_fit.fixed_point;
        // Drive the inner (t, β) state to the SAME KKT/step-converged optimum the
        // dense `penalized_quasi_laplace_criterion_with_cache` reaches before factoring. At that
        // optimum the per-row `H_tt^(i)` blocks are PD, so the undamped
        // (`ridge_t = 0`) streaming factorization in `streaming_exact_arrow_log_det`
        // succeeds — without this, a state stopped after only `inner_max_iter`
        // steps can leave a rank-deficient / indefinite row block (`p_out = 1` →
        // rank-1 `JᵀJ`, softmax negative-logit curvature) that surfaces
        // `PerRowFactorFailed` at base ridge 0. Sharing the driver puts both lanes
        // at the SAME inner state — but not, since #2330 Phase-2, on the same
        // evidence operator; see the #2509 note below.
        let options = ArrowSolveOptions::direct()
            .with_gpu_policy(self.gpu_policy)
            .with_newton_schur_tikhonov(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR)
            .with_evidence_unit_deflation(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR);
        // The converged arrow-factor cache is the per-row factored Hessian
        // (matrix-free, feasible at massive K — the dense border_dim² Schur is
        // never materialised here); it is RETURNED so the EFS lane can take its
        // matrix-free ARD/smoothness traces off it. The log-determinant itself is
        // recomputed chunk-by-chunk in `streaming_exact_arrow_log_det` to bound
        // peak memory.
        //
        // #2509 Phase-2b — that recomputation now prices the exact observed
        // information `A = B + ΔC` on BOTH of its branches, via
        // [`Self::exact_a_evidence_system`]: the four
        // `apply_exact_hessian_minus_b` channels are assembled into per-row arrow
        // blocks and folded into a second system whose log-determinant the
        // criterion takes. `OrderedBetaBernoulli` — the one channel that couples
        // rows within an atom column and therefore has no per-row arrow block —
        // REFUSES by name rather than falling back to `B`.
        //
        // The cache returned here is still `B`: it is the Newton/IFT scale and
        // the positive-definite preconditioner, and `apply_exact_hessian_minus_b`
        // adds `ΔC` on top of it, so promoting it to `A` would double-count.
        // Before this, the criteria differed by exactly
        // `½·[(log|A| − log|A_tt|) − (log|B| − log|B_tt|)]` — 22.32 units on the
        // `reml_retries_refinement_after_non_pd_undamped_evidence_factor` witness.
        let mut converged_cache = self.converge_inner_for_undamped_logdet(
            target,
            rho,
            rho_fixed,
            registry,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
            &mut loss,
            &mut criterion_fixed_point,
            &options,
            true,
        )?;
        // #9: accumulate the per-atom Grams + N_eff + log_det_tt in the same
        // log-det pass. These are required by the canonical rank-charge criterion.
        let mut rank_inputs = StreamingRankInputs::default();
        // #2515 — an INDEFINITE exact-A verdict from the arrow evidence route must
        // arrive here as the SAME typed error the dense route raises, not as a
        // generic `Numerical`. Both routes are saying "this state is a saddle, so
        // `½log|A|` is not a Laplace normalizer"; the outer solver reads the typed
        // one as an infeasible ρ (`+inf`, steer away) and the untyped one as a
        // defect that aborts the fit. Same verdict, two behaviours, chosen by which
        // route the memory planner picked — this issue's genus one level up.
        let (log_det, evidence_artifacts) = self
            .streaming_exact_arrow_log_det_with_lane_and_system(
                target,
                rho,
                registry,
                Some(&mut rank_inputs),
                lane,
            )
            .map_err(SaeCriterionError::from_arrow_refusal)?;
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
        // matching the full-batch `penalized_quasi_laplace_criterion_with_cache` path so streaming
        // and dense criteria rank the identical penalized objective.
        let extra_penalty_energy =
            self.reml_extra_penalty_value_total(registry)
                .map_err(|err| {
                    format!(
                        "SaeManifoldTerm::penalized_quasi_laplace_criterion_streaming_exact: {err}"
                    )
                })?;
        let v = {
            let ri = rank_inputs;
            // #9/#5 streaming rank charge: replace the coordinate-block ½log|H_tt|
            // (= log_det_tt/2, exposed by the log-det pass) with Σ ½·d_eff·log n on
            // each atom's realised decoder rank, priced through the SAME
            // `rank_dof_from_grams` MP hard count as the dense path off the
            // chunk-accumulated Grams. The β/Schur block (the ‖B‖-independent part
            // of log_det) is untouched by the rank charge — but it is where #2509
            // lives. The shared seam is `0.5*(log_det − log_det_tt) + rank_charge`,
            // and on THIS lane `log_det = log_det_tt + log|S_B|` by construction,
            // so the criterion's whole exposure to the A-vs-B operator split is
            // `log|S_A|` against `log|S_B|`: the per-row t-block log-dets cancel.
            // (On the dense lane the two log-dets come from two independent
            // spectral classifications of `A` and `A_tt`, so their difference is
            // `log|S_A|` only where neither PD floor deflates a direction.)
            let residual = self.reconstruction_residual(target, rho)?;
            let residual_energy =
                self.residual_energy_for_vanishing(residual.view())?;
            match self.vanished_atoms_from_signal_upper_bound(
                &ri.grams,
                &ri.n_eff,
                residual_energy.mean_square(),
            )? {
                VanishedAtomsProof::Certified {
                    atoms: Some(atoms),
                    ..
                } => return Err(SaeCriterionError::VanishedAtoms(atoms)),
                VanishedAtomsProof::Certified { atoms: None, .. } => {}
                VanishedAtomsProof::Unavailable { reason } => {
                    return Err(SaeCriterionError::Numerical(format!(
                        "streaming decoder-vanishing proof unavailable: {reason}"
                    )));
                }
            }
            let disp = self
                .reconstruction_dispersion(
                    &loss,
                    &converged_cache,
                    rho,
                    Some(residual.view()),
                )
                .map_err(|e| {
                    format!(
                        "SaeManifoldTerm::penalized_quasi_laplace_criterion_streaming_exact: rank-charge dispersion is required: {e}"
                    )
                })?;
            let d_eff = self.rank_dof_from_grams(&ri.grams, &ri.n_eff, rho, disp)?;
            // #5/#2498: the typed gated-signal proof above is the sole
            // disappearance verdict. The scalar rank-charge seam only prices the
            // already-certified live state.
            let quasi_laplace_complexity =
                rank_adjusted_quasi_laplace_complexity(log_det, ri.log_det_tt, &d_eff, &ri.n_eff)?;
            loss.total() + extra_penalty_energy + quasi_laplace_complexity - occam
        };
        Ok((v, loss, converged_cache, evidence_artifacts))
    }

    /// `Self::penalized_quasi_laplace_criterion_streaming_exact` with the #2080 surrogate lane
    /// threaded to the streaming `log|S|` term (`None` = bit-identical SLQ).
    pub fn penalized_quasi_laplace_criterion_streaming_exact_with_lane(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
        lane: Option<&mut SurrogateLaneState>,
    ) -> Result<(f64, SaeManifoldLoss), SaeCriterionError> {
        let (cost, loss, _cache) = self
            .penalized_quasi_laplace_criterion_streaming_exact_with_cache_and_lane(
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

    /// #2509/#2515 Phase-2b — the arrow evidence operator carrying the EXACT
    /// observed information `A = ∇²_θθ L`, derived from an already-assembled
    /// Arrow–Schur majorizer `B` by folding in the per-row `ΔC = A − B` blocks.
    ///
    /// The Laplace criterion is `½log|∇²_θθ(objective)|`, and `A` IS that
    /// Hessian by construction. `B` is the positive-definite scale /
    /// preconditioner for `A` — and, since #2673, the METRIC every direction of
    /// `A` is classified in at both the value and the gradient site (see
    /// `sae_exact_a_identifiability_floor`); a
    /// preconditioner is not the operator it preconditions. Pricing `log|B|`
    /// here while the dense lane prices `log|A|` is exactly the defect: the same
    /// statistical state was ranked ~22 criterion units apart because a host
    /// memory predicate, not the model, chose the operator.
    ///
    /// **`B` is returned untouched.** This is a SECOND operator, not a mutation:
    /// the Newton/IFT solves keep `B` as their (positive-definite, factorable)
    /// scale, and `apply_exact_hessian_minus_b` — which adds `ΔC` on top of `B`
    /// itself — cannot double-count `ΔC`.
    ///
    /// **Ordering.** The `ΔC` assembler needs only the arrow LAYOUT — per-row
    /// dimensions and the border dimension — never a factorization, so it reads
    /// `row_dims` / `k` off the UNFACTORED system. That removes the apparent
    /// factor-then-assemble two-pass: `ArrowFactorCache` was only ever being
    /// used as a carrier for those two layout facts (see
    /// [`Self::border_channels_for_border_dim`],
    /// [`Self::row_vars_for_row_dim`], `refill_jet_window_with_row_dims`).
    ///
    /// `ΔC_ββ ≡ 0` (the decoder is linear in β), so `hbb` / `penalty_op` are
    /// untouched and the whole correction lands in the row blocks and the
    /// eliminated Schur sum.
    ///
    /// `ΔC_tβ` is carried by COMPOSING the installed matrix-free row operator
    /// rather than by a dense supplement, because
    /// `StreamingArrowSchur::from_system` drops the dense `row.htbeta` slabs
    /// whenever a row operator is installed — writing `ΔC` there would be
    /// silently discarded, i.e. would price `B` while claiming `A`. Systems with
    /// no row operator carry `ΔC` in the dense slab, which for them IS the
    /// operator.
    pub(crate) fn exact_a_evidence_system(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        majorizer: &ArrowSchurSystem,
    ) -> Result<ArrowSchurSystem, String> {
        let border_dim = majorizer.k;
        let row_dims: Vec<usize> = majorizer.row_dims.to_vec();
        // The assembled blocks are `O(Σ q_i·(q_i + K))`. Refuse above the
        // in-core budget rather than OOM: an admitted-then-killed run reads as a
        // green route. (Follow-up: an active-atom-sparse `ΔC_tβ` operator, which
        // removes the `K` factor entirely.)
        let delta_bytes: u128 = row_dims
            .iter()
            .map(|&q| (q as u128) * ((q as u128) + (border_dim as u128)) * 8)
            .sum();
        let budget = crate::manifold::sae_host_in_core_budget_bytes().0 as u128;
        if delta_bytes > budget {
            return Err(format!(
                "SaeManifoldTerm::exact_a_evidence_system: the assembled exact-A correction needs \
                 {delta_bytes} bytes over {} rows at border {border_dim}, above the {budget}-byte \
                 in-core budget; this route must refuse rather than price the Arrow-Schur \
                 majorizer B and call it the exact observed information A (#2509)",
                row_dims.len()
            ));
        }
        let delta =
            self.assemble_exact_hessian_minus_b_rows(rho, target, &row_dims, border_dim)?;
        if delta.len() != majorizer.rows.len() {
            return Err(format!(
                "SaeManifoldTerm::exact_a_evidence_system: assembled {} exact-A correction rows \
                 for a {}-row arrow system",
                delta.len(),
                majorizer.rows.len()
            ));
        }
        // #2515 — retain the operands of the ONE raw exact-A classification.
        // The factorization must measure every direction against B_raw and must
        // restore the exactly-known clamp basin before calling it a saddle.  The
        // delta rows recover B_raw from the A system without retaining a second
        // full arrow system; the clamp diagonal is assembled before any
        // conditioning, from the same row layout as ΔC.
        let clamp = self.materialize_ard_concave_clamp_diagonal_for_rows(rho, &row_dims)?;
        let mut clamp_base = 0usize;
        let classification_rows: std::sync::Arc<
            [gam_solve::arrow_schur::ExactAClassificationRow],
        > = delta
            .into_iter()
            .zip(row_dims.iter().copied())
            .map(|(block, q)| {
                let clamp_diag = clamp.slice(s![clamp_base..clamp_base + q]).to_owned();
                clamp_base += q;
                gam_solve::arrow_schur::ExactAClassificationRow {
                    delta_tt: block.tt,
                    delta_tbeta: block.tbeta,
                    clamp_diag,
                }
            })
            .collect::<Vec<_>>()
            .into();
        let border = self.border_channels_for_border_dim(border_dim)?;
        let classification_indices: std::sync::Arc<[usize]> = border
            .iter()
            .map(|channel| channel.index)
            .collect::<Vec<_>>()
            .into();
        let mut system = majorizer.clone();
        // The CUDA descriptor describes `B`'s cross-block sparsity, so it cannot
        // stand in for `A`; the generic closures are the authoritative path.
        system.device_sae_pcg = None;
        for (row_idx, (row, block)) in system
            .rows
            .iter_mut()
            .zip(classification_rows.iter())
            .enumerate()
        {
            let q = block.delta_tt.nrows();
            if row.htt.dim() != (q, q) {
                return Err(format!(
                    "SaeManifoldTerm::exact_a_evidence_system: row {row_idx} exact-A correction is \
                     {q}x{q} but the arrow row block is {:?}",
                    row.htt.dim()
                ));
            }
            for a in 0..q {
                for b in 0..q {
                    row.htt[[a, b]] += block.delta_tt[[a, b]];
                }
            }
        }
        match (
            majorizer.htbeta_matvec.clone(),
            majorizer.htbeta_transpose_matvec.clone(),
        ) {
            (None, _) => {
                for (row_idx, (row, block)) in
                    system
                        .rows
                        .iter_mut()
                        .zip(classification_rows.iter())
                        .enumerate()
                {
                    let q = block.delta_tt.nrows();
                    if row.htbeta.dim() != (q, border_dim) {
                        return Err(format!(
                            "SaeManifoldTerm::exact_a_evidence_system: row {row_idx} has no \
                             matrix-free cross-block operator and its dense slab is {:?}, not \
                             ({q}, {border_dim}); the exact-A correction has nowhere to land",
                            row.htbeta.dim()
                        ));
                    }
                    for a in 0..q {
                        for (beta_pos, channel) in border.iter().enumerate() {
                            row.htbeta[[a, channel.index]] +=
                                block.delta_tbeta[[a, beta_pos]];
                        }
                    }
                }
            }
            (Some(base_forward), Some(base_transpose)) => {
                let forward_blocks = std::sync::Arc::clone(&classification_rows);
                let forward_indices = std::sync::Arc::clone(&classification_indices);
                let transpose_blocks = std::sync::Arc::clone(&classification_rows);
                let transpose_indices = std::sync::Arc::clone(&classification_indices);
                // #2515 — the COMPOSED operator's content identity: the base
                // operator's own identity (which the majorizer published when it
                // was assembled) combined with the exact-A correction blocks this
                // wraps it in. Falling back to the base's `Arc` address would
                // reintroduce the allocation-identity defect one layer up, since
                // this closure is freshly allocated on every call.
                let composed_fingerprint = {
                    let mut hasher = gam_runtime::warm_start::Fingerprinter::new();
                    hasher.write_str("sae-exact-a-htbeta-composed-v1");
                    match majorizer.htbeta_operator_fingerprint {
                        Some(base_fp) => {
                            hasher.write_bool(true);
                            hasher.write_u64(base_fp);
                        }
                        None => hasher.write_bool(false),
                    }
                    hasher.write_usize(classification_rows.len());
                    for block in classification_rows.iter() {
                        hasher.write_f64_array2(&block.delta_tbeta);
                    }
                    hasher.write_usize(classification_indices.len());
                    for &index in classification_indices.iter() {
                        hasher.write_usize(index);
                    }
                    hasher.finish_u64()
                };
                system.set_row_htbeta_operator_with_fingerprint(
                    move |row, x, out| {
                        base_forward(row, x, out);
                        let block = &forward_blocks[row].delta_tbeta;
                        for a in 0..block.nrows() {
                            let mut acc = 0.0_f64;
                            for (beta_pos, &index) in forward_indices.iter().enumerate() {
                                acc += block[[a, beta_pos]] * x[index];
                            }
                            out[a] += acc;
                        }
                    },
                    move |row, v, out| {
                        base_transpose(row, v, out);
                        let block = &transpose_blocks[row].delta_tbeta;
                        for a in 0..block.nrows() {
                            let va = v[a];
                            if va == 0.0 {
                                continue;
                            }
                            for (beta_pos, &index) in transpose_indices.iter().enumerate() {
                                out[index] += block[[a, beta_pos]] * va;
                            }
                        }
                    },
                    composed_fingerprint,
                );
            }
            (Some(_), None) => {
                return Err(
                    "SaeManifoldTerm::exact_a_evidence_system: the majorizer installed a \
                     matrix-free cross-block operator without its declared sparse transpose, so \
                     the exact-A correction cannot be composed without changing which operator \
                     the reduced Schur applies (#2509)"
                        .to_string(),
                );
            }
        }
        system.exact_a_classification = Some(
            gam_solve::arrow_schur::ExactAClassificationGeometry {
                rows: classification_rows,
                border_indices: classification_indices,
            },
        );
        system.refresh_row_hessian_fingerprint();
        Ok(system)
    }

    /// Assemble the one whole-row matrix-free evidence system at the current
    /// fitted state. The dense reduced Schur is never formed: the returned
    /// system retains only the structured shared-block and row-cross operators.
    ///
    /// This single source of truth is consumed both by the rational
    /// log-determinant and by #2230's exact-stationarity IFT solve, ensuring the
    /// value and assignment-strength residual cannot reassemble different
    /// operators. Optional rank inputs are accumulated from the same full chunk.
    /// The returned pair is the system and the whole-row CHUNK TERM it was
    /// assembled from.
    ///
    /// The chunk term — not `self` — is the term whose `last_row_layout` /
    /// `last_frames_active` describe the returned system's arrow layout, because
    /// it is the receiver `assemble_arrow_schur_scaled` was called on. Anything
    /// that reads that layout back (the #2509 exact-`A` row assembly) must use
    /// this term, or it can silently index a DIFFERENT active-set layout than the
    /// system it is correcting.
    pub(crate) fn assemble_full_matrix_free_evidence_system(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        mut rank_inputs: Option<&mut StreamingRankInputs>,
    ) -> Result<(ArrowSchurSystem, SaeManifoldTerm), String> {
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
            full_chunk.accumulate_decoder_gram(&mut inputs.grams)?;
            let assignments = full_chunk.assignment.assignments();
            for atom in 0..inputs.n_eff.len() {
                let support = SupportMeasure::from_assignment_matrix(assignments.view(), atom)
                    .expect("streaming full-rank chunk assignment shape must match atoms");
                inputs.n_eff[atom] += support.fisher_n();
            }
        }
        let mut system = full_chunk
            .assemble_arrow_schur_scaled(target, rho, registry, 1.0)
            .map_err(|error| format!("SaeManifoldTerm::streaming_exact_arrow_log_det: {error}"))?;
        // The exact-stationarity inverse consumes this system with the factor
        // cache emitted from it. Persist the completed row/registry fingerprint
        // now so the stale-pair guard compares two identities from the same
        // assembled operator instead of the constructor sentinel `0`.
        system.refresh_row_hessian_fingerprint();
        Ok((system, full_chunk))
    }

    fn streaming_exact_arrow_log_det_with_lane_and_system(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        mut rank_inputs: Option<&mut StreamingRankInputs>,
        mut lane: Option<&mut SurrogateLaneState>,
    ) -> Result<(f64, Option<StreamingEvidenceArtifacts>), String> {
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
        let plan = self.streaming_plan()?.admitted_or_error(
            self.n_obs(),
            self.output_dim(),
            self.k_atoms(),
        )?;
        // A gradient-bearing streaming evaluation always uses the rational
        // matrix-free value, even when a chunked dense Schur would barely fit:
        // only the rational lane emits the frozen selected-inverse bundle whose
        // contractions are the exact derivative of that value. Value-only SLQ
        // callers retain the historical memory-derived split.
        if plan.estimated_dense_schur_bytes > plan.in_core_budget_bytes || lane.is_some() {
            // #988 memory-matrix-free evidence route. The dense k×k reduced Schur
            // (≈8 GB at the K=32k manifold border) does NOT fit the in-core
            // budget, so estimate log|S| via Stochastic Lanczos Quadrature on the
            // matrix-free `schur_matvec` apply (`gam_solve::arrow_schur::
            // matrix_free_arrow_evidence_log_det`) instead of assembling +
            // Cholesky-factoring the dense Schur. Peak memory is the per-row block
            // storage the inner PCG already holds, not the extra O(k²) dense S.
            //
            // #2515 — the operator this factors is `a_sys`, the EXACT OBSERVED
            // INFORMATION, whose sign is a modelling verdict and not a rounding
            // artefact. `with_evidence_unit_deflation` deflates on `λ < floor` —
            // one-sided — so it swallowed every negative direction of `A` however
            // large and priced it as the ρ-independent null `log 1 = 0` with a `1/λ
            // → 1` inverse, while the dense route classified the same direction as
            // #2336 clamp-attributable curvature (priced at its basin) or as a
            // genuine saddle (the typed `IndefiniteObservedInformation` refusal
            // that makes the ρ infeasible). Measured on #2712's deflated anchor at
            // `log λ_smooth = −1.05`: reduced-Schur eigenvalues `−7.997610e-3` and
            // `−2.033493e-3`, five decades outside the `1e-8` band, both pinned to
            // `+1`, and the two complete outer gradients `1.009` RELATIVE apart.
            let options = ArrowSolveOptions::direct()
                .with_gpu_policy(self.gpu_policy)
                .with_newton_schur_tikhonov(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR)
                .with_indefinite_refusing_evidence_unit_deflation(
                    gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR,
                );
            // Assemble the WHOLE system once (a single "chunk" over all rows) so the
            // matrix-free reduced-Schur apply `v ↦ S·v` can iterate every row; the
            // per-row block storage is exactly what the inner solve already holds.
            let (sys, chunk_term) = self.assemble_full_matrix_free_evidence_system(
                target,
                rho,
                registry,
                rank_inputs.as_deref_mut(),
            )?;
            // #2509/#2515 Phase-2b: the log-determinant is the LAPLACE
            // normalizer, so it must be taken off the exact observed information
            // `A = B + ΔC`, not off the Arrow-Schur majorizer `B`. `B` itself is
            // returned unchanged below as the solve/IFT scale.
            let a_sys = chunk_term.exact_a_evidence_system(target, rho, &sys)?;
            // #2080: the reduced-Schur `log|S|` term. `lane = None` runs the
            // bit-identical SLQ estimate; `lane = Some(state)` swaps in the frozen
            // derived-rank rational surrogate (matrix-free, value+ρ-gradient one
            // functional). `log_det_tt` (the Σ log|H_tt| coordinate block) is exact
            // on the shared factorization either way.
            //
            // #2515 — a lane-bearing evaluation takes the GRADIENT-BEARING entry
            // point, which emits the exact-`A` row factorization alongside the
            // value and the derivative bundle. All three then come from ONE
            // factorization of ONE operator, which is what lets the outer gradient
            // reconstruct `(H⁻¹)_tt = A_i⁻¹ + G_i S_A⁻¹ G_iᵀ` instead of splicing
            // `A`'s reduced Schur onto `B`'s row blocks. The value-only entry is
            // retained verbatim for `lane = None` (bit-identical SLQ).
            let (log_det_tt, log_det_schur, exact_a_cache) = match lane.as_deref_mut() {
                Some(lane) => {
                    let evaluated = gam_solve::arrow_schur::matrix_free_arrow_evidence_evaluation(
                        &a_sys,
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
                            "SaeManifoldTerm::streaming_exact_arrow_log_det: matrix-free criterion log-det: {err:?}"
                        )
                    })?;
                    (
                        evaluated.log_det_tt,
                        evaluated.log_det_schur,
                        Some(evaluated.factor_cache),
                    )
                }
                None => {
                    let (log_det_tt, log_det_schur) = matrix_free_arrow_evidence_log_det_surrogate(
                        &a_sys,
                        0.0,
                        0.0,
                        &options,
                        SCHUR_SLQ_LOGDET_PROBES,
                        SCHUR_SLQ_LOGDET_LANCZOS_STEPS,
                        SCHUR_SLQ_LOGDET_SEED,
                        None,
                    )
                    .map_err(|err| {
                        format!(
                            "SaeManifoldTerm::streaming_exact_arrow_log_det: matrix-free criterion log-det: {err:?}"
                        )
                    })?;
                    (log_det_tt, log_det_schur, None)
                }
            };
            if !log_det_schur.is_finite() {
                return Err(format!(
                    "SaeManifoldTerm::streaming_exact_arrow_log_det: matrix-free reduced-Schur \
                     log|S| non-finite ({log_det_schur})"
                ));
            }
            if let Some(ri) = rank_inputs.as_deref_mut() {
                ri.log_det_tt = log_det_tt;
            }
            return Ok((
                log_det_tt + log_det_schur,
                exact_a_cache.map(|exact_a_cache| StreamingEvidenceArtifacts {
                    majorizer_system: sys,
                    exact_a_cache,
                }),
            ));
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
        // #2515 — same substitution as the matrix-free branch above, and for the
        // same reason: every factorization below is of `exact_a_evidence_system`'s
        // output, so a resolved negative direction is a saddle verdict rather than
        // a numerical null, and unit-pinning it would price a saddle as `log 1 = 0`.
        let options = ArrowSolveOptions::direct()
            .with_gpu_policy(self.gpu_policy)
            .with_newton_schur_tikhonov(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR)
            .with_indefinite_refusing_evidence_unit_deflation(
                gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR,
            );
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
                chunk.accumulate_decoder_gram(&mut ri.grams)?;
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
            // #2509/#2515 Phase-2b — same substitution as the matrix-free branch:
            // the Laplace normalizer is `log|A|`, and every `ΔC` channel except
            // ordered Beta-Bernoulli is row-local, so the correction is
            // chunk-additive exactly as the majorizer is. (`penalty_scale` scales
            // only β-side penalties, and `ΔC_ββ ≡ 0`.)
            let sys = chunk.exact_a_evidence_system(z_chunk, rho, &sys)?;
            let mut streaming = StreamingArrowSchur::from_system(&sys, sys.rows.len().max(1));
            let (chunk_log_det_tt, chunk_schur) = streaming
                .reduced_schur_and_log_det_tt(0.0, 0.0, &options)
                .map_err(|err| format!("SaeManifoldTerm::streaming_exact_arrow_log_det: {err}"))?;
            log_det_tt += chunk_log_det_tt;
            for row in 0..border_dim {
                for col in 0..border_dim {
                    schur_acc[[row, col]] += chunk_schur[[row, col]];
                }
            }
            start = end;
        }
        let log_det_schur = StreamingArrowSchur::reduced_schur_log_det(&schur_acc, &options)
            .map_err(|err| format!("SaeManifoldTerm::streaming_exact_arrow_log_det: {err}"))?;
        if let Some(ri) = rank_inputs.as_deref_mut() {
            ri.log_det_tt = log_det_tt;
        }
        Ok((log_det_tt + log_det_schur, None))
    }

    /// Per-atom decoder-smoothness penalty quadratic form (#1556): entry `k` is
    /// the λ-free `<B_k, ½(S_k+S_kᵀ)·B_k> = Σ_oc B_k[:,oc]ᵀ S_k B_k[:,oc]`, the
    /// per-atom denominator of atom `k`'s λ_smooth Fellner-Schall update. The sum
    /// over atoms is `βᵀ(⊕_k S_k ⊗ I_p)β`, the un-scaled total penalty energy.
    /// `S_k` is symmetrised defensively (as the assembler does); the per-atom
    /// `½(S+Sᵀ)·B_k` GEMMs ride the multi-GPU batched smoothness GEMM. Device-free
    /// and sub-threshold groups use exact CPU products; admitted failures propagate.
    pub(crate) fn decoder_smoothness_quadratic_form_per_atom(&self) -> Result<Vec<f64>, String> {
        let sb_inputs: Vec<(ArrayView2<'_, f64>, ArrayView2<'_, f64>)> = self
            .atoms
            .iter()
            .map(|atom| {
                (
                    atom.smooth_penalty().view(),
                    atom.decoder_coefficients().view(),
                )
            })
            .collect();
        let sb_all = batched_smooth_sb(&sb_inputs, true, self.gpu_policy)?;
        let mut per_atom = vec![0.0_f64; self.atoms.len()];
        for (atom_idx, (atom, sb)) in self.atoms.iter().zip(sb_all.iter()).enumerate() {
            per_atom[atom_idx] = (atom.decoder_coefficients() * sb).sum();
        }
        Ok(per_atom)
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
            (self.beta_offsets(), Box::new(move |_: usize| p))
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
        // #2253/#2228 λ→0 boundary: the plain per-column back-substitution
        // divides by the doubly-null (data-null ∧ penalty-null) β-Schur pivots
        // at the ρ lower face and returns `Inf`/`NaN` — the EDF value is the
        // ONLY outer-gradient piece that contracts `(H⁻¹)_ββ`, so it is the
        // piece that diverges while the criterion value stays finite. Route
        // every column through the deflated spectral pseudo-inverse instead:
        // the eigendecomposition happens ONCE (`schur_deflated_applier`), a
        // doubly-null direction contributes exactly 0 dof (it is
        // unidentifiable, not a real degree of freedom), and in the interior
        // no direction deflates so the trace matches the plain path to
        // round-off.
        let apply = cache.schur_deflated_applier()?;
        let mut per_atom = vec![0.0_f64; self.atoms.len()];
        let mut m_col = Array1::<f64>::zeros(k);
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let s = atom.smooth_penalty();
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
                    let z = apply(m_col.view());
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
            (self.beta_offsets(), Box::new(move |_: usize| p))
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
        // #2253/#2228 λ→0 boundary: route the β-only columns through the ONE
        // deflated spectral pseudo-inverse (see
        // `decoder_smoothness_effective_dof_per_atom`) so a doubly-null decoder
        // direction contributes 0 dof instead of `Inf`/`NaN`. With a zero
        // t-RHS the full arrow solve's β component IS the β-Schur selected
        // inverse (`solve(0, m).beta = S_β⁻¹ m`), so the deflated applier is
        // the exact drop-in — but ONLY on the plain bordered arrow. When a
        // gauge Woodbury deflation is installed (`!plain_selected_inverse_
        // available`) the solve carries a rank-R gauge correction the β-Schur
        // applier omits; there the known nulls are already stiffened by
        // `κQQᵀ`, so the plain per-column solve stays (finite by
        // construction of the gauge stiffness).
        let mut per_atom = vec![0.0_f64; self.atoms.len()];
        let mut m_col = Array1::<f64>::zeros(k);
        let deflated_apply = if solver.plain_selected_inverse_available() {
            Some(cache.schur_deflated_applier().map_err(|e| {
                format!("decoder_smoothness_effective_dof_with_solver_per_atom: {e:?}")
            })?)
        } else {
            None
        };
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let s = atom.smooth_penalty();
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
                    let z = match deflated_apply.as_ref() {
                        Some(apply) => apply(m_col.view()),
                        None => solver.solve(zero_t.view(), m_col.view())?.beta,
                    };
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
        self.assignment.validate_rho_domain(rho)?;
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
            let scale = rho.lambda_sparse()? * sparsity * inv_tau * inv_tau;
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
                    Some(_) => {}
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
        let inv_diag = solver
            .latent_inverse_diagonal()
            .map_err(|err| format!("assignment_log_strength_hessian_trace: {err}"))?;
        let assignment_dim = self.assignment.assignment_coord_dim();
        let total_t = cache.delta_t_len();
        // #932 FRONT C: row-local Takahashi selected inverse on the plain arrow
        // for the per-row deflation correction below (the diagonal trace already
        // uses the cheap `latent_inverse_diagonal`); gauge-deflated systems fall
        // back to the per-row full-system `solve` loop.
        let fast_selected = solver.plain_selected_inverse_available();
        let selected_beta_inv = if fast_selected && cache.k > 0 {
            solver
                .beta_inv()
                .map_err(|err| format!("assignment_log_strength_hessian_trace: {err}"))?
        } else {
            Array2::<f64>::zeros((0, 0))
        };
        let learnable_alpha = matches!(
            self.assignment.mode,
            AssignmentMode::OrderedBetaBernoulli {
                learnable_alpha: true,
                ..
            }
        );
        let ordered_channels = ordered_beta_bernoulli_psd_majorizer_third_channels_weighted(
            &self.assignment,
            rho,
            self.row_loss_weights.as_deref(),
        )?;
        // The integrated marginal's mass-Hessian coefficient is strictly
        // negative, so its cross-row rank-one block has the zero PSD Loewner
        // majorizer. Retain only the positive part of the row-local
        // concrete-Jacobian term, matching assembly exactly.
        if let Some(ch) = ordered_channels.as_ref() {
            for row in 0..self.n_obs() {
                for atom in 0..k_atoms {
                    let slot = row * k_atoms + atom;
                    hdiag[slot] = if learnable_alpha {
                        super::construction_arrow_schur_assembly::ordered_beta_bernoulli_psd_majorized_log_alpha_hdiag(
                            ch, row, k_atoms, atom, hdiag[slot],
                        )
                    } else {
                        super::construction_arrow_schur_assembly::ordered_beta_bernoulli_psd_majorized_hdiag(
                            ch, row, k_atoms, atom, hdiag[slot],
                        )
                    };
                }
            }
        }
        let mut trace = 0.0_f64;
        // Hoisted RHS scratch for the gauge-deflated per-row solve fallback:
        // single-entry set/clear instead of a per-column total_t-sized zeroing.
        let mut rhs_t_scratch = Array1::<f64>::zeros(total_t);
        let rhs_beta_zero = Array1::<f64>::zeros(cache.k);
        for row in 0..self.n_obs() {
            let row_base = cache.row_offsets[row];
            let assignment_base = row * k_atoms;
            let q = cache.row_dims[row];
            // Per-row diagonal `(∂H/∂ρ)_tt` for the deflation correction: the
            // assignment prior curves only the logit/assignment slots (coordinate
            // slots are zero; ARD handles those).
            let mut d_diag = Array1::<f64>::zeros(q);
            match self.last_row_layout {
                Some(ref layout) => {
                    for (pos, &atom) in layout.active_atoms[row].iter().enumerate() {
                        let d_slot = hdiag[assignment_base + atom];
                        trace += inv_diag[row_base + pos] * d_slot;
                        if pos < q {
                            d_diag[pos] = d_slot;
                        }
                    }
                }
                None => {
                    for free_idx in 0..assignment_dim {
                        let d_slot = hdiag[assignment_base + free_idx];
                        trace += inv_diag[row_base + free_idx] * d_slot;
                        if free_idx < q {
                            d_diag[free_idx] = d_slot;
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
        Ok(0.5 * trace)
    }

    /// Derivative of the coordinate-block logdet
    /// `½ Σ_i log|H_tt^(i)|` with respect to the assignment-strength rho
    /// coordinate. The canonical criterion subtracts this term from the full
    /// joint logdet, so the outer gradient must subtract this trace too.
    /// `operator` (#2515) selects the `∂H/∂ρ_sparse` operand exactly as the joint
    /// leg's does: `B`'s diagonal Gershgorin majorizer, or `A`'s dense entropy
    /// Hessian. Both legs of `½log|H| − ½log|H_tt|` must name the same operator.
    pub(crate) fn coordinate_block_assignment_log_strength_hessian_trace(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        operator: EvidenceOperator,
    ) -> Result<f64, String> {
        self.assignment.validate_rho_domain(rho)?;
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
                    rho.lambda_sparse()? * sparsity * inv_tau * inv_tau,
                ))
            }
            AssignmentMode::Softmax { .. } => return Ok(0.0),
            _ => None,
        };
        // Per-row softmax assignment scratch, reused across rows (the softmax arm
        // reads `a` rather than raw logits so both operator arms come off ONE
        // vector — see `softmax_sparse_curvature_rho_derivative_block`).
        let mut softmax_assignments = Array1::<f64>::zeros(k_atoms);
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

        let ordered_beta_bernoulli_channels =
            ordered_beta_bernoulli_psd_majorizer_third_channels_weighted(
                &self.assignment,
                rho,
                row_weights,
            )?;
        let learnable_alpha = matches!(
            self.assignment.mode,
            AssignmentMode::OrderedBetaBernoulli {
                learnable_alpha: true,
                ..
            }
        );
        if let Some(channels) = ordered_beta_bernoulli_channels.as_ref() {
            for row in 0..self.n_obs() {
                for atom in 0..k_atoms {
                    let index = row * k_atoms + atom;
                    hdiag[index] = if learnable_alpha {
                        super::construction_arrow_schur_assembly::ordered_beta_bernoulli_psd_majorized_log_alpha_hdiag(
                            channels, row, k_atoms, atom, hdiag[index],
                        )
                    } else {
                        super::construction_arrow_schur_assembly::ordered_beta_bernoulli_psd_majorized_hdiag(
                            channels, row, k_atoms, atom, hdiag[index],
                        )
                    };
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
            if let Some((_temperature, scale)) = softmax.as_ref() {
                let row_weight = row_weights.map_or(1.0, |weights| weights[row]);
                match self.last_row_layout {
                    Some(_) => {}
                    None => {
                        self.assignment.try_assignments_row_into(
                            row,
                            softmax_assignments
                                .as_slice_mut()
                                .expect("softmax assignment scratch is contiguous"),
                        )?;
                        let a_soft = softmax_assignments
                            .as_slice()
                            .expect("softmax assignment scratch is contiguous");
                        let m = softmax_majorizer_log_mean(a_soft);
                        let slot_atoms: Vec<usize> = (0..assignment_dim.min(q)).collect();
                        let block = softmax_sparse_curvature_rho_derivative_block(
                            a_soft,
                            &slot_atoms,
                            m,
                            *scale,
                            row_weight,
                            operator,
                        );
                        for (a, _) in slot_atoms.iter().enumerate() {
                            for (b, _) in slot_atoms.iter().enumerate() {
                                derivative[[a, b]] = block[[a, b]];
                            }
                        }
                    }
                }
            } else {
                let assignment_base = row * k_atoms;
                match self.last_row_layout {
                    Some(ref layout) => {
                        for (slot, &atom) in layout.active_atoms[row].iter().enumerate() {
                            derivative[[slot, slot]] = hdiag[assignment_base + atom];
                        }
                    }
                    None => {
                        for atom in 0..assignment_dim.min(q) {
                            derivative[[atom, atom]] = hdiag[assignment_base + atom];
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
            let spectrum = cache
                .deflation_row_spectra
                .get(row)
                .and_then(Option::as_ref);
            // A clamp-basin classification changes the priced spectrum without
            // unit-deflating a direction.  Its direction list is therefore empty,
            // but the stored raw/conditioned spectrum still owns a non-identity
            // Daleckii--Krein map and must differentiate it (#2515/#2336).
            if spectrum.is_some() || !directions.is_empty() {
                row_trace -=
                    Self::deflation_block_correction(&inverse, &derivative, directions, spectrum);
            }
            total_trace += row_trace;
        }
        Ok(0.5 * total_trace)
    }

    /// Matrix-free sibling of [`Self::assignment_log_strength_hessian_trace`]
    /// for assignment families whose majorized prior curvature is row-local.
    /// Reconstructs each row's selected-inverse block from the exact row-local
    /// inverse plus the shared `(z_j, S^-1 z_j)` reduced-Schur bundle
    /// ([`row_selected_inverse_from_probes`]):
    ///
    /// `H^-1_tt = A_i^-1 + G_i S^-1 G_i^T`,
    /// `diag` = `diag(A_i^-1) + (1/m) sum_j (G_i z_j) * (G_i S^-1 z_j)`.
    ///
    /// This is the missing assignment-strength trace in the matrix-free analytic
    /// rho-gradient cluster. Per-row deflation is PRICED, not refused (#2712):
    /// the reconstructed block is the DEFLATED one (`A_i` is the conditioned row
    /// block), so each branch applies the same deflation treatment as its dense
    /// counterpart — the within-row kept-subspace diagonal on the softmax branch,
    /// the full Daleckii–Krein `tr(inv_vv·(D − DΦ[D]))` elsewhere.
    pub(crate) fn assignment_log_strength_hessian_trace_from_probes(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        probes: &[Array1<f64>],
        sinv_probes: &[Array1<f64>],
        operator: EvidenceOperator,
    ) -> Result<f64, String> {
        self.assignment.validate_rho_domain(rho)?;
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
                    rho.lambda_sparse()? * sparsity * inv_tau * inv_tau,
                ))
            }
            AssignmentMode::Softmax { .. } => return Ok(0.0),
            _ => None,
        };
        let mut softmax_assignments = Array1::<f64>::zeros(k_atoms);
        let mut hdiag = if softmax.is_none() {
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
        let ordered_channels = ordered_beta_bernoulli_psd_majorizer_third_channels_weighted(
            &self.assignment,
            rho,
            self.row_loss_weights.as_deref(),
        )?;
        let learnable_alpha = matches!(
            self.assignment.mode,
            AssignmentMode::OrderedBetaBernoulli {
                learnable_alpha: true,
                ..
            }
        );
        if let Some(channels) = ordered_channels.as_ref() {
            for row in 0..self.n_obs() {
                for atom in 0..k_atoms {
                    let index = row * k_atoms + atom;
                    hdiag[index] = if learnable_alpha {
                        super::construction_arrow_schur_assembly::ordered_beta_bernoulli_psd_majorized_log_alpha_hdiag(
                            channels, row, k_atoms, atom, hdiag[index],
                        )
                    } else {
                        super::construction_arrow_schur_assembly::ordered_beta_bernoulli_psd_majorized_hdiag(
                            channels, row, k_atoms, atom, hdiag[index],
                        )
                    };
                }
            }
        }
        let assignment_dim = self.assignment.assignment_coord_dim();
        let row_loss_weights = self.row_loss_weights.as_deref();
        let mut trace = 0.0_f64;
        for row in 0..self.n_obs() {
            let q = cache.row_dims[row];
            // The DEFLATED row-block selected inverse from the shared bundle
            // (#2712). The `t–β` block is not contracted here, so it is not built.
            let (inv_vv, _) = row_selected_inverse_from_probes(
                cache,
                row,
                probes,
                sinv_probes,
                false,
                "assignment_log_strength_hessian_trace_from_probes",
            )?;
            let inverse_diagonal = inv_vv.diag().to_owned();
            let dirs = cache
                .deflated_row_directions
                .get(row)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let spectrum = cache
                .deflation_row_spectra
                .get(row)
                .and_then(Option::as_ref);

            if let Some((_temperature, scale)) = softmax.as_ref() {
                let row_weight = row_loss_weights.map_or(1.0, |weights| weights[row]);
                match self.last_row_layout {
                    Some(_) => {}
                    None => {
                        self.assignment.try_assignments_row_into(
                            row,
                            softmax_assignments
                                .as_slice_mut()
                                .expect("softmax assignment scratch is contiguous"),
                        )?;
                        let a_soft = softmax_assignments
                            .as_slice()
                            .expect("softmax assignment scratch is contiguous");
                        let m = softmax_majorizer_log_mean(a_soft);
                        let logit_dim = assignment_dim.min(inverse_diagonal.len());
                        let slot_atoms: Vec<usize> = (0..logit_dim).collect();
                        let block = softmax_sparse_curvature_rho_derivative_block(
                            a_soft,
                            &slot_atoms,
                            m,
                            *scale,
                            row_weight,
                            operator,
                        );
                        match operator {
                            EvidenceOperator::Majorizer => {
                                // `∂B/∂ρ_sparse` is DIAGONAL, so the contraction is
                                // the kept-subspace diagonal — matching the dense
                                // softmax branch's `latent_inverse_diagonal_kept`:
                                // the deflated inverse assigns `1/λ̃ = 1` to each
                                // `vᵢ`, and a ρ-independent direction must
                                // contribute 0.
                                for atom in 0..logit_dim {
                                    let kept = inverse_diagonal[atom]
                                        - dirs
                                            .iter()
                                            .map(|v| v.get(atom).copied().unwrap_or(0.0).powi(2))
                                            .sum::<f64>();
                                    trace += kept * block[[atom, atom]];
                                }
                            }
                            EvidenceOperator::ExactObservedInformation => {
                                // `∂A/∂ρ_sparse` is the DENSE entropy Hessian, so the
                                // diagonal shortcut does not apply: contract the full
                                // block and take the general Daleckii–Krein deflation
                                // correction, exactly as the non-softmax arm below.
                                let mut d_mat = Array2::<f64>::zeros((q, q));
                                for a in 0..logit_dim {
                                    for b in 0..logit_dim {
                                        d_mat[[a, b]] = block[[a, b]];
                                        trace += inv_vv[[b, a]] * block[[a, b]];
                                    }
                                }
                                // Basin pricing changes `cond_evals` without
                                // creating a unit-deflated direction.  The
                                // spectrum, not the null-direction list, is the
                                // certificate that this derivative map is live.
                                if spectrum.is_some() || !dirs.is_empty() {
                                    trace -= Self::deflation_block_correction(
                                        &inv_vv, &d_mat, dirs, spectrum,
                                    );
                                }
                            }
                        }
                    }
                }
            } else {
                let assignment_base = row * k_atoms;
                // Per-row diagonal `(∂H/∂ρ)_tt` for the deflation correction: the
                // assignment prior curves only the logit/assignment slots.
                let mut d_diag = Array1::<f64>::zeros(q);
                match self.last_row_layout {
                    Some(ref layout) => {
                        for (slot, &atom) in layout.active_atoms[row].iter().enumerate() {
                            let d_slot = hdiag[assignment_base + atom];
                            trace += inverse_diagonal[slot] * d_slot;
                            if slot < q {
                                d_diag[slot] = d_slot;
                            }
                        }
                    }
                    None => {
                        for slot in 0..assignment_dim.min(inverse_diagonal.len()) {
                            let d_slot = hdiag[assignment_base + slot];
                            trace += inverse_diagonal[slot] * d_slot;
                            if slot < q {
                                d_diag[slot] = d_slot;
                            }
                        }
                    }
                }
                if spectrum.is_some() || !dirs.is_empty() {
                    // Same Daleckii–Krein correction the dense sibling subtracts,
                    // against the same deflated `inv_vv` (#2712).
                    let mut d_mat = Array2::<f64>::zeros((q, q));
                    for slot in 0..q {
                        d_mat[[slot, slot]] = d_diag[slot];
                    }
                    trace -= Self::deflation_block_correction(&inv_vv, &d_mat, dirs, spectrum);
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
        // M = Uᵀ D U, W = Uᵀ inv_vv U (both q×q, symmetric).
        let m = u.t().dot(d_mat).dot(u);
        let w = u.t().dot(inv_vv).dot(u);
        // correction = Σ_{m,l} W[m,l]·M[m,l]·(1 − F[m,l]).
        let f = Self::row_deflation_frechet_coefficients(spec, q);
        let mut acc = 0.0_f64;
        for a in 0..q {
            for b in 0..q {
                acc += w[[a, b]] * m[[a, b]] * (1.0 - f[[a, b]]);
            }
        }
        acc
    }

    /// Fold the row's Daleckii–Krein deflation differential into the single
    /// t–t weight consumed by `SaeRowJetContraction::Trace` (#2333).
    ///
    /// For every symmetric derivative block `D`, the returned `E` satisfies
    /// `sum(E⊙D) = tr(inv_vv·D) - deflation_block_correction(inv_vv,D)`. In
    /// the spectral case this is `U ((Uᵀ inv_vv U) ⊙ F) Uᵀ`, using the
    /// exact same `F` and gap convention as the correction. Gauge-only rows fold
    /// the structural-null subtraction directly; undeflated or malformed
    /// spectral rows preserve the raw selected inverse, matching the correction's
    /// zero branch.
    fn deflation_folded_trace_weight(
        inv_vv: &Array2<f64>,
        dirs: &[Array1<f64>],
        spectrum: Option<&RowDeflationSpectrum>,
    ) -> Array2<f64> {
        let q = inv_vv.nrows();
        let Some(spec) = spectrum else {
            let mut e = inv_vv.clone();
            for v in dirs {
                for a in 0..q {
                    let va = v.get(a).copied().unwrap_or(0.0);
                    if va == 0.0 {
                        continue;
                    }
                    for b in 0..q {
                        e[[a, b]] -= va * v.get(b).copied().unwrap_or(0.0);
                    }
                }
            }
            return e;
        };
        let u = &spec.evecs;
        if u.nrows() != q || u.ncols() != q {
            return inv_vv.clone();
        }
        let mut folded = u.t().dot(inv_vv).dot(u);
        let f = Self::row_deflation_frechet_coefficients(spec, q);
        for a in 0..q {
            for b in 0..q {
                folded[[a, b]] *= f[[a, b]];
            }
        }
        u.dot(&folded).dot(&u.t())
    }

    /// The Daleckii–Krein coefficient matrix `F` of the per-row spectral
    /// deflation map `Φ`, in the row's RAW eigenbasis:
    ///
    /// ```text
    ///   F[a,b] = (λ̃_a − λ̃_b) / (λ_a − λ_b)      (raw λ below, conditioned λ̃ above)
    /// ```
    ///
    /// with the degenerate/near-degenerate entry taken as the diagonal limit
    /// `f'(λ_a)`, which is `1` for an unclamped KEPT direction and `0` for a
    /// deflated one. Single source for the two consumers that need it: the trace
    /// form [`Self::deflation_block_correction`] (`tr(inv·(D − DΦ[D]))`) and the
    /// operator form [`Self::row_deflation_map_derivative`] (`DΦ[D]` itself).
    /// `spec.evecs` is assumed `q×q` (checked by both callers).
    fn row_deflation_frechet_coefficients(spec: &RowDeflationSpectrum, q: usize) -> Array2<f64> {
        let raw = &spec.raw_evals;
        let cond = &spec.cond_evals;
        let conditioning = &spec.conditioning;
        let eigen_scale = raw
            .iter()
            .chain(cond.iter())
            .copied()
            .fold(0.0_f64, |scale, value| scale.max(value.abs()));
        let gap_threshold = eigen_gap_threshold(eigen_scale, raw.len());
        let mut f = Array2::<f64>::zeros((q, q));
        for a in 0..q {
            for b in 0..q {
                let denom = raw[a] - raw[b];
                f[[a, b]] = if denom.abs() > gap_threshold {
                    (cond[a] - cond[b]) / denom
                } else if conditioning[a] == RowSpectralConditioning::Raw {
                    1.0
                } else {
                    0.0
                };
            }
        }
        f
    }

    /// #2500 — the per-row spectral-deflation map's Daleckii–Krein differential
    /// as an OPERATOR: `DΦ[D] = U (F ∘ (Uᵀ D U)) Uᵀ`, the ρ-derivative of the
    /// CONDITIONED block `Φ(H_raw)` given the ρ-derivative `D` of the RAW block.
    ///
    /// This is the operator whose trace against a selected inverse
    /// [`Self::deflation_block_correction`] already reports as
    /// `tr(inv·(D − DΦ[D]))`; the channels that materialize `∂H/∂ρ` as a MATRIX
    /// (the PATH C operator map) need the operator itself, because the block they
    /// contract against — the arrow factors, and hence `apply_cached_arrow_hessian`
    /// and every `A = B + ΔC` built on it — carries the CONDITIONED spectrum. A
    /// raw `D` there over-claims curvature on exactly the deflated directions,
    /// where the installed operator is the ρ-INDEPENDENT unit stiffness.
    ///
    /// `spectrum = None` with non-empty `dirs` is gauge-only deflation: a
    /// ρ-independent structural null, so the map is the two-sided projection
    /// `DΦ[D] = P D P`, `P = I − Σᵢ vᵢvᵢᵀ`. That is the operator form of the same
    /// `Σᵢ vᵢᵀD vᵢ` fallback the trace uses (`inv` is `P inv P` on a gauge-deflated
    /// row, so the two agree under the trace).
    pub(crate) fn row_deflation_map_derivative(
        d_mat: &Array2<f64>,
        dirs: &[Array1<f64>],
        spectrum: Option<&RowDeflationSpectrum>,
    ) -> Option<Array2<f64>> {
        let q = d_mat.nrows();
        let Some(spec) = spectrum else {
            if dirs.is_empty() {
                return None;
            }
            let mut p = Array2::<f64>::eye(q);
            for v in dirs {
                for a in 0..q {
                    let va = if a < v.len() { v[a] } else { 0.0 };
                    if va == 0.0 {
                        continue;
                    }
                    for b in 0..q {
                        let vb = if b < v.len() { v[b] } else { 0.0 };
                        p[[a, b]] -= va * vb;
                    }
                }
            }
            return Some(p.dot(d_mat).dot(&p));
        };
        let u = &spec.evecs;
        if u.nrows() != q || u.ncols() != q {
            return None;
        }
        let f = Self::row_deflation_frechet_coefficients(spec, q);
        let mut m = u.t().dot(d_mat).dot(u);
        for a in 0..q {
            for b in 0..q {
                m[[a, b]] *= f[[a, b]];
            }
        }
        Some(u.dot(&m).dot(&u.t()))
    }

    /// β-tier selected inverse `(H⁻¹)_ββ`, shared across rows (#932 FRONT C). On
    /// the plain bordered arrow this is the cached dense `S⁻¹` formed once from the
    /// Schur factor; when gauge deflation is active the row-local
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
    /// arrow; a per-row full-system `solve` loop (`O(n·q)`) under gauge
    /// deflation, where the row-local blocks are not valid. `rhs_t_scratch` is a
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
        self.border_channels_for_border_dim(cache.k)
    }

    /// [`Self::border_channels_for_cache`] against a border dimension read
    /// directly off an ArrowSchurSystem instead of a factor cache.
    ///
    /// #2509 Phase-2b: the exact-`A` row assembly must run BEFORE anything is
    /// factored (its blocks are what gets factored), so it cannot take its
    /// layout from an `ArrowFactorCache`. `cache.k` and `sys.k` are the same
    /// border dimension by construction — the cache is built from the system —
    /// so this is the same layout with the factorization ordering removed.
    pub(crate) fn border_channels_for_border_dim(
        &self,
        border_dim: usize,
    ) -> Result<Vec<SaeBorderChannel>, String> {
        let p = self.output_dim();
        let frames_active = self.last_frames_active && border_dim == self.factored_border_dim();
        let offsets = if frames_active {
            self.factored_beta_offsets()
        } else {
            self.beta_offsets()
        };
        let mut channels = Vec::with_capacity(border_dim);
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
        if channels.len() != border_dim {
            return Err(format!(
                "border channel layout has {} entries but cache border has {}",
                channels.len(),
                border_dim
            ));
        }
        Ok(channels)
    }

    pub(crate) fn row_vars_for_cache_row(
        &self,
        row: usize,
        cache: &ArrowFactorCache,
    ) -> Result<Vec<SaeLocalRowVar>, String> {
        self.row_vars_for_row_dim(row, cache.row_dims[row])
    }

    /// [`Self::row_vars_for_cache_row`] against a row dimension read directly
    /// off an ArrowSchurSystem (`sys.row_dims[row]`) instead of a factor cache.
    /// Same layout, no factorization prerequisite (#2509 Phase-2b).
    pub(crate) fn row_vars_for_row_dim(
        &self,
        row: usize,
        q_row: usize,
    ) -> Result<Vec<SaeLocalRowVar>, String> {
        let mut vars: Vec<Option<SaeLocalRowVar>> = vec![None; q_row];
        match self.last_row_layout {
            Some(ref layout) => {
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

    /// #2330 Patch D — raw basis THIRD jets `∂³φ` per atom, `Some(Array5)` shaped
    /// `(n_obs, basis, d, d, d)` when the atom's base evaluator exposes an
    /// analytic third jet (`SaeBasisThirdJet::third_jet_dyn`), else `None`. Used
    /// only by the exact-A θ-adjoint's residual-curvature leg `⟨error_metric,
    /// ∂³f⟩` on the dense route; an atom without a third jet contributes no such
    /// leg (skipped, not errored) so mixed-basis terms degrade to the
    /// second-order-only exact-A gradient rather than refusing.
    pub(crate) fn atom_third_jets(&self) -> Result<Vec<Option<ndarray::Array5<f64>>>, String> {
        let mut out = Vec::with_capacity(self.k_atoms());
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let coords = self.assignment.coords[atom_idx].as_matrix();
            let jet = match atom.basis_evaluator.as_ref() {
                Some(ev) => match ev.third_jet_dyn(coords.view()) {
                    Some(Ok(jet)) => {
                        let expected = (
                            atom.n_obs(),
                            atom.basis_size(),
                            atom.latent_dim(),
                            atom.latent_dim(),
                            atom.latent_dim(),
                        );
                        if jet.dim() != expected {
                            return Err(format!(
                                "atom_third_jets: atom '{}' third jet shape {:?}, expected {:?}",
                                atom.name,
                                jet.dim(),
                                expected
                            ));
                        }
                        Some(jet)
                    }
                    Some(Err(e)) => return Err(e),
                    None => None,
                },
                None => None,
            };
            out.push(jet);
        }
        Ok(out)
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
                atom.latent_dim(),
                atom.latent_dim(),
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
    // `row_jets_for_logdet`, and `refill_jet_window`) lives in the sibling
    // `construction_row_jet_logdet_channels.rs` file, inlined via `include!`
    // below at module scope as a second `impl SaeManifoldTerm` block. Splitting
    // it out keeps this tracked file under the 10k limit; `include!` preserves
    // the identical module scope and private-field access.

    pub(crate) fn assignment_prior_hdiag_derivative_entry(
        &self,
        threshold_strength: f64,
        row: usize,
        diag_atom: usize,
        wrt: SaeLocalRowVar,
        ordered_beta_bernoulli_channels: Option<&OrderedBetaBernoulliHessianDiagThirdChannels>,
    ) -> f64 {
        let SaeLocalRowVar::Logit { atom: wrt_atom } = wrt else {
            return 0.0;
        };
        // #Bug4: a FIXED logit (ungated atom, or every atom under frozen routing)
        // has its assembled `htt` diagonal entry ZEROED (see
        // `assignment_prior_grad_hdiag`), so the θ-adjoint third derivative of that
        // zeroed entry must also be zero. Mirror the ordered Beta--Bernoulli channel zeroing in
        // `ordered_beta_bernoulli_psd_majorizer_third_channels`. The ThresholdGate/ordered Beta--Bernoulli branches below are
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
                let inv_tau = 1.0 / temperature;
                let activation = gam_linalg::utils::stable_logistic((logit - threshold) * inv_tau);
                let slope = activation * (1.0 - activation);
                // #991 — this row's ThresholdGate prior curvature in `htt` carries the
                // design weight `w_row`, so its θ-derivative carries the SAME
                // `w_row` (value/logdet/adjoint stay on one weighted branch).
                let w_row = self.row_loss_weights.as_deref().map_or(1.0, |w| w[row]);
                // #1415: P(ℓ)=λσ((ℓ−θ)/τ); P''(ℓ)=(λ/τ²)s(1−2a) so the third
                // derivative is P'''(ℓ)=(λ/τ³)·s·(1−6a+6a²), because
                // d/dℓ[s(1−2a)] = (1/τ)s[(1−2a)²−2s] = (1/τ)s(1−6a+6a²).
                w_row
                    * threshold_strength
                    * slope
                    * (1.0 - 6.0 * activation + 6.0 * activation * activation)
                    * inv_tau
                    * inv_tau
                    * inv_tau
            }
            AssignmentMode::OrderedBetaBernoulli { .. } => {
                // The assembled `htt` diagonal consumes
                // `OrderedBetaBernoulliPenalty::hessian_diag`, whose logit derivative
                // splits into a row-local direct-`z` channel and a global
                // empirical-`M_k` channel (the integrated marginal couples every
                // row in column `k`).
                // This same-row primitive returns only the LOCAL direct-`z`
                // channel — and only on the matching logit (`diag_atom == w`),
                // since H_ik depends on no other row's z explicitly. The global
                // M_k channel is accumulated column-wise in
                // `logdet_theta_adjoint` (it needs the per-row selected-inverse
                // diagonals), so adding it here would double-count.
                if diag_atom != wrt_atom {
                    return 0.0;
                }
                match ordered_beta_bernoulli_channels {
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
        alpha: f64,
        row: usize,
        atom: usize,
        axis: usize,
    ) -> f64 {
        let periods = self.assignment.coords[atom].effective_axis_periods();
        let t = self.assignment.coords[atom].row(row)[axis];
        match periods[axis] {
            None => 0.0,
            Some(period) => {
                let kappa = std::f64::consts::TAU / period;
                let phase = kappa * t;
                let (sin, cos) = phase.sin_cos();
                // Exact `∂/∂t` of the assembled majorizer entry
                // `w_row·psd_majorizer_hess = w_row·α·s_{τ₀}(cos κt)` (#2339):
                //   d/dt = w_row·α·s'_{τ₀}(cos κt)·(−κ sin κt)
                //        = −w_row·α·κ·sin(κt)·logistic(cos κt / τ₀).
                // The logistic factor `clamp_slope` is the smooth replacement for
                // the old hard `1{cos κt > 0}` branch indicator (`τ₀→0` recovers
                // it), so both the convex and concave halves now flow through one
                // analytic expression — C¹ across the clamp seam.
                //
                // HT row weighting: the assembled majorizer is `w_row·V''_clamped`
                // (full `w_row`, added directly to `htt` — NOT via the √w jet
                // seam), so its coordinate derivative carries the same full
                // `w_row`. The data-fit `dH/dθ` terms sharing this diagonal already
                // carry full `w` (a product of two √w-scaled jets), so the correct
                // single factor for this prior term is likewise full `w_row`.
                // `None` weights ⇒ w_row = 1.
                let slope = ArdAxisPrior::clamp_slope(cos);
                let w_row = self.row_loss_weights.as_deref().map_or(1.0, |w| w[row]);
                -w_row * alpha * kappa * sin * slope
            }
        }
    }

    /// #2330 Phase-2 — the EXACT (un-clamped) periodic-ARD curvature θ-derivative
    /// for `A = B + ΔC`. `ard_majorized_hessian_derivative` differentiates the
    /// PSD majorizer `w·max(α cos κt, 0)` (zero on the clamped half); the exact
    /// prior Hessian `w·α cos κt` is signed, so its θ-derivative is
    /// `∂/∂t[w·α cos κt] = −w·α κ sin κt` on BOTH branches. That is exactly
    /// `∂B/∂θ_ard + ∂ΔC/∂θ_ard` (the majorizer half + the restored negative half),
    /// i.e. the ARD leg of `∂A/∂θ`. Euclidean axes have constant curvature ⇒ 0.
    pub(crate) fn ard_exact_hessian_derivative(
        &self,
        alpha: f64,
        row: usize,
        atom: usize,
        axis: usize,
    ) -> f64 {
        let periods = self.assignment.coords[atom].effective_axis_periods();
        match periods[axis] {
            None => 0.0,
            Some(period) => {
                let kappa = std::f64::consts::TAU / period;
                let t = self.assignment.coords[atom].row(row)[axis];
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
        self.assignment.validate_rho_domain(rho)?;
        let ard_precisions = self.validated_ard_precisions(rho)?;
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
                    Some(_) => {}
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
            let lambda = rho.lambda_smooth_for(target_atom)?;
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
                    Some(frame) => frame.project_decoder(atom.decoder_coefficients().view())?,
                    None => atom.decoder_coefficients().clone(),
                }
            } else {
                atom.decoder_coefficients().clone()
            };
            let r = coeffs.ncols();
            let off = offsets[target_atom];
            for mu in 0..m {
                for channel in 0..r {
                    let mut acc = 0.0_f64;
                    for nu in 0..m {
                        let s_sym = 0.5
                            * (atom.smooth_penalty()[[mu, nu]] + atom.smooth_penalty()[[nu, mu]]);
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
                    let alpha = ard_precisions[atom][axis];
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
        if matches!(self.assignment.mode, AssignmentMode::Softmax { .. }) {
            // #2304 resident path: the packed channel tensors are reduced in
            // place (on device when the plan admits it) and only the per-row
            // t/β coefficients return.
            //
            // The probe is `−½·√w·Z̃` on the block's columns, zero elsewhere
            // (the −½ applied at emit time). With a whitening metric, the
            // historical consumer whitened BOTH the jets and this vector to
            // rank space and dotted there; `⟨Uᵀa, Uᵀv⟩ = ⟨a, U(Uᵀv)⟩`
            // exactly, so the metric folds into the probe as `M_n v` and the
            // raw jets are contracted directly.
            let probe_for_row = |row: usize| -> Result<Vec<f64>, String> {
                let sqrt_w = self
                    .row_loss_weights
                    .as_deref()
                    .map_or(1.0, |w| w[row].sqrt());
                let v: Vec<f64> = (0..p)
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
                    Ok(metric.apply_metric_row(row, ndarray::aview1(&v)))
                } else {
                    Ok(v)
                }
            };
            self.contracted_softmax_linear_rhs(
                cache,
                &second_jets,
                &border,
                probe_for_row,
                |row, q, t_row, beta_row| {
                    let base = cache.row_offsets[row];
                    for (var_idx, &value) in t_row.iter().enumerate().take(q) {
                        t[base + var_idx] = -0.5 * value;
                    }
                    for (channel, &value) in border.iter().zip(beta_row) {
                        beta[channel.index] += -0.5 * value;
                    }
                    Ok(())
                },
            )?;
            return Ok(SaeArrowVector { t, beta });
        }
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
            // The non-softmax rank-space dot: jets are whitened to `Uᵀ·`
            // channels, so the vector is whitened the same way (never
            // `M_n v` here — that fold belongs to the contracted path above).
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
            for var_idx in 0..jets.vars.len() {
                t[base + var_idx] = -0.5 * sae_dot(jets.first(var_idx), &v);
            }
            for (channel_pos, channel) in border.iter().enumerate() {
                beta[channel.index] += -0.5 * sae_dot(jets.beta(channel_pos), &v);
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
        if jets.channels.p() != p {
            return Err(format!(
                "logdet_theta_adjoint: packed row jet width {} != output dim {p}",
                jets.channels.p()
            ));
        }
        let rank = metric.metric_rank();
        let q = jets.channels.q();
        let n_beta = jets.channels.n_beta();
        let mut whitened = crate::row_jet_program::SaeScheduledRowJets::zeros(q, rank, n_beta);
        let apply = |input: &[f64], output: &mut [f64]| {
            for rank_col in 0..rank {
                let mut acc = 0.0_f64;
                for out_col in 0..p {
                    acc += metric.factor_entry(row, out_col, rank_col) * input[out_col];
                }
                output[rank_col] = acc;
            }
        };
        for a in 0..q {
            apply(jets.first(a), whitened.first_mut(a));
            for b in 0..q {
                apply(jets.second(a, b), whitened.second_mut(a, b));
            }
            for beta_pos in 0..n_beta {
                apply(
                    jets.beta_deriv(a, beta_pos),
                    whitened.beta_deriv_mut(a, beta_pos),
                );
                apply(
                    jets.beta_l_deriv(a, beta_pos),
                    whitened.beta_l_deriv_mut(a, beta_pos),
                );
            }
        }
        for beta_pos in 0..n_beta {
            apply(jets.beta(beta_pos), whitened.beta_mut(beta_pos));
        }
        jets.channels = whitened;
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
        // The joint leg of this entry point is the `B`-majorizer Γ by definition:
        // the exact-A joint adjoint is owned by `logdet_theta_adjoint_dense` (with
        // the priced pseudo-inverse) and by `logdet_theta_adjoint_from_probes`.
        self.logdet_theta_adjoint_for_block(
            rho,
            cache,
            solver,
            true,
            EvidenceOperator::Majorizer,
            None,
        )
    }

    /// `Γ_tt = ∂_theta Σ_i log|H_tt^(i)|`, the state derivative of the
    /// coordinate-block logdet removed by the canonical rank-charge criterion.
    /// #2515 — the coordinate-block leg takes NO solver. Its `joint_block = false`
    /// arm never reaches one: `fast_selected` short-circuits on `joint_block`,
    /// `beta_inv` is the zero block, and every row's `(H⁻¹)_tt` is the row-local
    /// Cholesky inverse. Accepting a solver here only created a way for a caller
    /// to pair this leg with a different operator's inverse than the joint leg it
    /// is subtracted from, which is the class of defect #2515 is about.
    pub(crate) fn coordinate_block_logdet_theta_adjoint(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        operator: EvidenceOperator,
        residual_target: Option<ArrayView2<'_, f64>>,
    ) -> Result<SaeArrowVector, String> {
        let solver = DeflatedArrowSolver::plain(cache);
        self.logdet_theta_adjoint_for_block(rho, cache, &solver, false, operator, residual_target)
    }

    fn logdet_theta_adjoint_for_block(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        solver: &DeflatedArrowSolver<'_>,
        joint_block: bool,
        // #2515 — which operator's `∂H/∂θ` this differentiates. `Γ_joint − Γ_tt`
        // is ONE difference; the dense exact-A route builds both legs with
        // `exact_a = true` off the SAME eigensystem, so a coordinate leg left on
        // the majorizer would desync the difference even when the joint leg is
        // right.
        operator: EvidenceOperator,
        residual_target: Option<ArrayView2<'_, f64>>,
    ) -> Result<SaeArrowVector, String> {
        let exact_a = operator.is_exact_a();
        self.assignment.validate_rho_domain(rho)?;
        let ard_precisions = self.validated_ard_precisions(rho)?;
        let threshold_strength = match self.assignment.mode {
            AssignmentMode::ThresholdGate { .. } => rho.lambda_sparse()?,
            _ => 0.0,
        };
        // Γ_a = tr(H⁻¹ ∂H/∂θ_a) over the inner variables θ (#1006). `H` here is
        // the SAME object the criterion factor builds — Gauss-Newton data
        // curvature plus the prior majorizers / `hessian_diag` diagonals the
        // Newton/Schur Cholesky factorizes — so each block's θ-derivative channel
        // is differentiated on the criterion's own branch (no value/gradient
        // desync). The integrated ordered Beta--Bernoulli prior is the one block
        // whose row-local majorizer depends on the shared active mass
        // `M_k = Σ_i z_ik`; its logit derivative therefore has a
        // row-local channel and a shared-mass channel accumulated column-wise
        // after the row loop.
        if cache.arrow_log_det().is_none() {
            return Err(
                "logdet_theta_adjoint: cache lacks an authoritative joint-Hessian log-det \
                 for the selected-inverse operator"
                    .to_string(),
            );
        }
        if matches!(self.assignment.mode, AssignmentMode::Softmax { .. }) {
            return self.contracted_softmax_trace_adjoint(
                rho,
                cache,
                solver,
                joint_block,
                operator,
                residual_target,
            );
        }
        let n = self.n_obs();
        let total_t = cache.delta_t_len();
        let mut gamma_t = Array1::<f64>::zeros(total_t);
        let mut gamma_beta = Array1::<f64>::zeros(cache.k);
        let second_jets = self.atom_second_jets()?;
        // #2330 Patch D residual-curvature legs on the exact-A arm (#2515). Same
        // gating as `logdet_theta_adjoint_dense`: exact-A AND a target, else the
        // pre-Patch-D behaviour bit-for-bit.
        let patchd_residual = exact_a.then_some(residual_target).flatten();
        let patchd_third_jets = if patchd_residual.is_some() {
            Some(self.atom_third_jets()?)
        } else {
            None
        };
        let patchd_is_obb = matches!(
            self.assignment.mode,
            AssignmentMode::OrderedBetaBernoulli { .. }
        );
        let patchd_obb_inv_tau = match self.assignment.mode {
            AssignmentMode::OrderedBetaBernoulli { temperature, .. } => 1.0 / temperature,
            _ => 0.0,
        };
        let border = self.border_channels_for_cache(cache)?;
        // #932 FRONT C: plain-arrow `(H⁻¹)_ββ = S⁻¹` formed once from the cached
        // Schur factor; gauge-deflated systems fall back to the per-β `solve`
        // loop where the row-local Takahashi blocks are not valid.
        let fast_selected = joint_block && solver.plain_selected_inverse_available();
        let beta_inv = if joint_block {
            Self::selected_inverse_beta_block(solver, cache, fast_selected, "logdet_theta_adjoint")?
        } else {
            Array2::<f64>::zeros((cache.k, cache.k))
        };
        // Exact derivatives of the ordered Beta--Bernoulli PSD majorizer. The
        // negative-semidefinite mass rank-one block has zero majorizer; the
        // retained row-local diagonal depends on `M_k`, so its derivative splits
        // into a same-row term and a columnwise empirical-mass term below.
        // gam#2144: whitening of the row jets tracks `whitens_likelihood()` at ANY
        // rank (the assembly whitens `JᵀU UᵀJ` for full- and low-rank alike) and is
        // independent of the PSD majorization.
        let whiten_row_jets = self.whiten_logdet_row_jets();
        let ordered_beta_bernoulli_channels =
            ordered_beta_bernoulli_psd_majorizer_third_channels_weighted(
                &self.assignment,
                rho,
                self.row_loss_weights.as_deref(),
            )?;
        // Per active logit site: row, atom, global t-index, selected-inverse
        // diagonal, and the unit-diagonal Daleckii--Krein correction weight.
        #[derive(Clone, Copy)]
        struct OrderedBetaBernoulliLogitSite {
            row: usize,
            atom: usize,
            t_index: usize,
            raw_diag: f64,
            diag_deflation_weight: f64,
        }
        let mut ordered_beta_bernoulli_logit_sites: Vec<OrderedBetaBernoulliLogitSite> = Vec::new();

        // #1557 — reuse one K-sized scratch row across all N rows (alias-free).
        let k_atoms = self.k_atoms();
        let mut assignments = Array1::<f64>::zeros(k_atoms);
        // The resident softmax program returned through the Trace seam above;
        // this hand window is exclusively the distinct non-softmax program.
        let mut jet_window: std::collections::VecDeque<SaeRowJets> =
            std::collections::VecDeque::new();
        let mut jet_window_next = 0usize;
        // Hoisted RHS scratch for the gauge-deflated per-row solve fallback.
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
            // full-system `solve` loop under gauge deflation.
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
            // deflated).
            let defl_dirs = cache
                .deflated_row_directions
                .get(row)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let defl_spectrum = cache
                .deflation_row_spectra
                .get(row)
                .and_then(Option::as_ref);

            // Record each active logit's column, global t-index, selected-inverse
            // diagonal, and per-slot Daleckii--Krein weight for a unit diagonal
            // derivative. The empirical-mass pass uses these to differentiate
            // the same spectrally conditioned row-local majorizer.
            if ordered_beta_bernoulli_channels.is_some() {
                for (pos, var) in jets.vars.iter().enumerate() {
                    if let SaeLocalRowVar::Logit { atom } = *var {
                        let raw_diag = inv_vv[[pos, pos]];
                        let diag_deflation_weight = if defl_dirs.is_empty() {
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
                        ordered_beta_bernoulli_logit_sites.push(OrderedBetaBernoulliLogitSite {
                            row,
                            atom,
                            t_index: base + pos,
                            raw_diag,
                            diag_deflation_weight,
                        });
                    }
                }
            }

            let w_row_prior = self.row_loss_weights.as_deref().map_or(1.0, |w| w[row]);
            // #2330 Patch D per-row residual context on the exact-A arm (#2515).
            let patchd_error_metric: Option<Vec<f64>> = patchd_residual.map(|tgt| {
                self.patchd_row_error_metric(row, w_row_prior, tgt, &assignments, whiten_row_jets)
            });
            let patchd_sqrt_w = w_row_prior.sqrt();
            let patchd_ctx: Option<PatchDResidualCtx<'_>> =
                patchd_error_metric.as_deref().map(|em| PatchDResidualCtx {
                    row,
                    error_metric: em,
                    sqrt_w: patchd_sqrt_w,
                    assignments: &assignments,
                    second_jets: &second_jets,
                    third_jets: patchd_third_jets.as_deref(),
                    is_obb: patchd_is_obb,
                    inv_tau: patchd_obb_inv_tau,
                });
            for w in 0..q {
                let mut gamma = 0.0_f64;
                let mut deflated_base_dh_mat = Array2::<f64>::zeros((q, q));
                for a in 0..q {
                    for b in 0..q {
                        let mut dh = sae_dot(jets.second(a, w), jets.first(b))
                            + sae_dot(jets.first(a), jets.second(b, w));
                        if exact_a {
                            // #2330 Patch D (1a) on the coordinate block — the same
                            // residual-curvature leg the joint routes carry.
                            dh += sae_dot(jets.first(w), jets.second(a, b));
                        }
                        if let Some(ctx) = patchd_ctx.as_ref() {
                            dh += self.patchd_residual_third_leg(
                                ctx,
                                jets.vars[a],
                                jets.vars[b],
                                jets.vars[w],
                            );
                        }
                        if a == b {
                            dh += match jets.vars[a] {
                                SaeLocalRowVar::Logit { atom } => self
                                    .assignment_prior_hdiag_derivative_entry(
                                        threshold_strength,
                                        row,
                                        atom,
                                        jets.vars[w],
                                        ordered_beta_bernoulli_channels.as_ref(),
                                    ),
                                SaeLocalRowVar::Coord { atom, axis }
                                    if a == w && !ard_precisions[atom].is_empty() =>
                                {
                                    // The majorizer writes `α·softplus_{τ₀}(cos κt)`
                                    // into `H_tt`; `A` carries the unclamped
                                    // `α·cos κt`.
                                    if exact_a {
                                        self.ard_exact_hessian_derivative(
                                            ard_precisions[atom][axis],
                                            row,
                                            atom,
                                            axis,
                                        )
                                    } else {
                                        self.ard_majorized_hessian_derivative(
                                            ard_precisions[atom][axis],
                                            row,
                                            atom,
                                            axis,
                                        )
                                    }
                                }
                                _ => 0.0,
                            };
                        }
                        deflated_base_dh_mat[[a, b]] = dh;
                        gamma += inv_vv[[b, a]] * dh;
                    }
                }
                if !defl_dirs.is_empty() {
                    // The row factor/log-det operator is the spectrally
                    // conditioned `Φ(H_tt)`, while the local theta channels above
                    // assemble the raw row derivative `D`. Subtract
                    // `tr(inv_vv · (D - DΦ[D]))` for every deflated row, including
                    // the low-rank ordered Beta--Bernoulli majorizer path, so the theta adjoint
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
                        let mut dh = sae_dot(jets.second(a, w), jets.beta(beta_pos))
                            + sae_dot(jets.first(a), jets.beta_deriv(w, beta_pos));
                        if exact_a {
                            dh += sae_dot(jets.first(w), jets.beta_deriv(a, beta_pos));
                        }
                        if let Some(ctx) = patchd_ctx.as_ref() {
                            dh += self.patchd_residual_third_leg_beta(
                                ctx,
                                jets.vars[a],
                                jets.vars[w],
                                channel,
                            );
                        }
                        gamma += 2.0 * inv_vbeta[[a, channel.index]] * dh;
                    }
                }
                for (beta_i, channel_i) in border.iter().enumerate() {
                    for (beta_j, channel_j) in border.iter().enumerate() {
                        let dh = sae_dot(jets.beta_deriv(w, beta_i), jets.beta(beta_j))
                            + sae_dot(jets.beta(beta_i), jets.beta_deriv(w, beta_j));
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
                        let dh = sae_dot(jets.beta_l_deriv(a, w_beta_pos), jets.first(b))
                            + sae_dot(jets.first(a), jets.beta_l_deriv(b, w_beta_pos));
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
                        let dh = sae_dot(jets.beta_l_deriv(a, w_beta_pos), jets.beta(beta_pos));
                        gamma += 2.0 * inv_vbeta[[a, channel.index]] * dh;
                    }
                }
                gamma_beta[w_channel.index] += gamma;
            }
        }

        // Empirical-mass channel of the row-local ordered Beta--Bernoulli
        // majorizer. Its diagonal depends on `M_k = Σ_i z_ik`, so a logit in
        // row `w` differentiates every retained row-local diagonal in column
        // `k`. The Daleckii--Krein weight applies to that same diagonal.
        if let Some(channels) = ordered_beta_bernoulli_channels.as_ref() {
            let mut column_coefficient = vec![0.0_f64; k_atoms];
            for site in &ordered_beta_bernoulli_logit_sites {
                let index = site.row * k_atoms + site.atom;
                column_coefficient[site.atom] +=
                    (site.raw_diag - site.diag_deflation_weight) * channels.m_channel[index];
            }
            for site in &ordered_beta_bernoulli_logit_sites {
                let index = site.row * k_atoms + site.atom;
                gamma_t[site.t_index] += column_coefficient[site.atom] * channels.z_jac[index];
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
    /// # Per-row deflation (#2712)
    ///
    /// Deflated rows are priced here, not refused. `cache.undamped_factor(i)`
    /// factorizes the spectrally CONDITIONED `Φ(H_tt^(i))` and the reduced Schur
    /// behind the bundle is that same conditioned arrow's, so the reconstructed
    /// `A_i⁻¹ + G_i S⁻¹ G_iᵀ` IS the deflated `(H⁻¹)_tt` the dense route contracts
    /// (see [`row_selected_inverse_from_probes`]). The Daleckii–Krein correction
    /// `−tr(inv_vv·(D − DΦ[D]))` is then applied through the same
    /// [`Self::deflation_block_correction`] helper the dense route uses, on the
    /// t-slot channels, the border channels, and the ordered Beta–Bernoulli
    /// shared-mass diagonal alike — none of whose operands involves `S⁻¹`. The
    /// from-probes and dense θ-adjoints therefore agree exactly at full-basis
    /// probes on the deflated regime too — the FD gate's acceptance.
    /// `exact_a` (#2515 B-full) selects WHICH operator's θ-adjoint this returns.
    ///
    /// `false` — the historical `½log|B|` adjoint `Γ = tr(B⁻¹ ∂B/∂θ)`.
    /// `true`  — `Γ = tr(A⁻¹ ∂A/∂θ)` for `A = B + ΔC`, matching
    /// [`SaeManifoldTerm::logdet_theta_adjoint_dense`] called with
    /// `exact_a = true` and `residual_target = None`.
    ///
    /// The port is possible because the dense contraction never needs a dense
    /// inverse. Enumerated over the whole of `logdet_theta_adjoint_dense`, every
    /// subscript of its `inv` is one of exactly three shapes — the row-local `t–t`
    /// block, the `t–β` border, and the `β–β` block — with NO cross-row
    /// off-diagonal entry `inv[[base_i + a, base_j + b]]` for `i ≠ j`. All three
    /// are what this function already reconstructs from the border bundle
    /// (`inv_vv`, `inv_vbeta`, and the refolded `S⁻¹` trace), so `exact_a` changes
    /// only the `dh` OPERANDS and never the contraction structure.
    ///
    /// SCOPE — the #2330 Patch-D residual THIRD-derivative legs
    /// (`⟨error_metric, ∂³f⟩`) are not carried here, which is why the reference
    /// above pins `residual_target = None`: the dense route skips those legs under
    /// that argument too, so the two are comparable term for term. They are
    /// additive later and need `atom_third_jets()`, a per-row quantity already
    /// available matrix-free. `OrderedBetaBernoulli`'s cross-row adjoint is
    /// excluded because the streaming evidence lane refuses that family by name
    /// (#2509 Phase-2b) rather than pricing `B` and calling it `A`.
    pub(crate) fn logdet_theta_adjoint_from_probes(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        probes: &[Array1<f64>],
        sinv_probes: &[Array1<f64>],
        operator: EvidenceOperator,
        // #2330 Patch D — the data target, required ONLY for the exact-A
        // residual-curvature legs. `None` reproduces the pre-Patch-D behaviour
        // exactly (the third-derivative leg is skipped), matching
        // `logdet_theta_adjoint_dense`'s own contract argument for argument.
        residual_target: Option<ArrayView2<'_, f64>>,
    ) -> Result<SaeArrowVector, String> {
        let exact_a = operator.is_exact_a();
        self.assignment.validate_rho_domain(rho)?;
        let ard_precisions = self.validated_ard_precisions(rho)?;
        // Threshold-gate sparsity strength for the assignment-prior H-diagonal
        // derivative (#1006/#1556): the ThresholdGate penalty differentiates
        // `λ_sparse`, every other assignment mode contributes zero. Same binding
        // the dense adjoint path builds; the probe path consumes it identically.
        let threshold_strength = match self.assignment.mode {
            AssignmentMode::ThresholdGate { .. } => rho.lambda_sparse()?,
            _ => 0.0,
        };
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

        let ordered_beta_bernoulli_channels =
            ordered_beta_bernoulli_psd_majorizer_third_channels_weighted(
                &self.assignment,
                rho,
                self.row_loss_weights.as_deref(),
            )?;
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
                Some(rho.lambda_sparse()? * sparsity * inv_tau * inv_tau)
            }
            _ => None,
        };

        // #2330 Patch D residual-curvature legs, ported from
        // `logdet_theta_adjoint_dense` (#2515). Active only on the exact-A route
        // WITH a target, exactly as there. The third jets are a per-row quantity
        // and cost nothing this lane cannot pay.
        let patchd_residual = exact_a.then_some(residual_target).flatten();
        let patchd_third_jets = if patchd_residual.is_some() {
            Some(self.atom_third_jets()?)
        } else {
            None
        };
        // The ordered-Beta–Bernoulli Patch-D channel is a CROSS-ROW adjoint and
        // has no per-row arrow block; the streaming evidence lane refuses that
        // family by name (#2509 Phase-2b) rather than pricing `B` and calling it
        // `A`, so reaching here with it would mean the refusal was bypassed.
        let patchd_is_obb = matches!(
            self.assignment.mode,
            AssignmentMode::OrderedBetaBernoulli { .. }
        );
        if patchd_residual.is_some() && patchd_is_obb {
            return Err(
                "logdet_theta_adjoint_from_probes: the exact-A Patch-D residual legs are not \
                 modelled for ordered Beta--Bernoulli here — its prior-curvature adjoint couples \
                 rows within an atom column and has no per-row arrow block. The streaming \
                 evidence lane refuses this family upstream; refusing rather than emitting a \
                 gradient short by that channel"
                    .to_string(),
            );
        }
        let patchd_obb_inv_tau = 0.0_f64;
        let mut assignments = Array1::<f64>::zeros(self.k_atoms());
        let mut jet_window: std::collections::VecDeque<SaeRowJets> =
            std::collections::VecDeque::new();
        let mut jet_window_next = 0usize;
        let mut ordered_beta_bernoulli_logit_sites: Vec<(usize, usize, usize, f64)> = Vec::new();

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

            // The DEFLATED row-block selected inverse `(H⁻¹)_tt = A_i⁻¹ + G_i S⁻¹ G_iᵀ`
            // and border block `(H⁻¹)_tβ = −G_i S⁻¹`, from the shared bundle. `A_i` is
            // the SPECTRALLY CONDITIONED row block, so these are the same objects the
            // dense `selected_inverse_row_blocks` returns — including on a deflated row
            // (#2712; see `row_selected_inverse_from_probes`).
            let (inv_vv, inv_vbeta) = row_selected_inverse_from_probes(
                cache,
                row,
                probes,
                sinv_probes,
                true,
                "logdet_theta_adjoint_from_probes",
            )?;

            // Per-row UNIT-stiffness deflated directions. `inv_vv` above is the
            // DEFLATED inverse (it assigns `1/λ̃ = 1` to each `vᵢ`), so every
            // `inv_vv`-weighted t–t contraction of the RAW `∂H/∂θ_w` below over-claims
            // curvature exactly where the re-deflating criterion uses the deflation-map
            // derivative `DΦ`. The kept-subspace Γ subtracts `tr(inv_vv·(D − DΦ[D]))`
            // through the SAME Daleckii–Krein helper the dense route uses — every
            // operand of which (`inv_vv`, `dirs`, `spectrum`, and the locally
            // assembled raw `D`) is in hand here.
            let defl_dirs = cache
                .deflated_row_directions
                .get(row)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let defl_spectrum = cache
                .deflation_row_spectra
                .get(row)
                .and_then(Option::as_ref);

            // #2330 Patch D per-row residual context (#2515 port). `w_row_prior`
            // is bound below for the majorizer legs; the residual weighting is the
            // row's own design weight, read here through the one authority both
            // routes share.
            let patchd_w_row = self.row_loss_weights.as_deref().map_or(1.0, |w| w[row]);
            let patchd_error_metric: Option<Vec<f64>> = patchd_residual.map(|tgt| {
                self.patchd_row_error_metric(row, patchd_w_row, tgt, &assignments, whiten_row_jets)
            });
            let patchd_sqrt_w = patchd_w_row.sqrt();
            let patchd_ctx: Option<PatchDResidualCtx<'_>> =
                patchd_error_metric.as_deref().map(|em| PatchDResidualCtx {
                    row,
                    error_metric: em,
                    sqrt_w: patchd_sqrt_w,
                    assignments: &assignments,
                    second_jets: &second_jets,
                    third_jets: patchd_third_jets.as_deref(),
                    is_obb: patchd_is_obb,
                    inv_tau: patchd_obb_inv_tau,
                });

            if ordered_beta_bernoulli_channels.is_some() {
                for (position, variable) in jets.vars.iter().enumerate() {
                    if let SaeLocalRowVar::Logit { atom } = *variable {
                        // Same per-slot Daleckii–Krein weight for a unit diagonal
                        // derivative the dense route records, so the shared-mass column
                        // pass below differentiates the same conditioned majorizer.
                        let diag_deflation_weight = if defl_dirs.is_empty() {
                            0.0
                        } else {
                            let mut unit_diag = Array2::<f64>::zeros((q, q));
                            unit_diag[[position, position]] = 1.0;
                            Self::deflation_block_correction(
                                &inv_vv,
                                &unit_diag,
                                defl_dirs,
                                defl_spectrum,
                            )
                        };
                        ordered_beta_bernoulli_logit_sites.push((
                            row,
                            atom,
                            base + position,
                            inv_vv[[position, position]] - diag_deflation_weight,
                        ));
                    }
                }
            }

            // Precompute the β–β fold carriers P_l, R_l (w-independent) per probe.
            let bjet_len = if k_border > 0 {
                if jets.channels.n_beta() == 0 {
                    0
                } else {
                    jets.channels.p()
                }
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
                        let bj = jets.beta(beta_pos);
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
                // t–t block: reuse the dense contraction. On a deflated row the raw
                // per-slot derivative is retained as a matrix so the Daleckii–Krein
                // correction can be applied to it after the loop; on a PD row the
                // matrix stays `0×0` and nothing is allocated.
                let mut deflated_base_dh_mat = if defl_dirs.is_empty() {
                    Array2::<f64>::zeros((0, 0))
                } else {
                    Array2::<f64>::zeros((q, q))
                };
                for a in 0..q {
                    for b in 0..q {
                        let mut dh = match (softmax_d_dw, jets.vars[a], jets.vars[b]) {
                            (
                                Some((a_soft, _m, _scale, inv_tau, atom_w)),
                                SaeLocalRowVar::Coord { atom: atom_a, .. },
                                SaeLocalRowVar::Coord { atom: atom_b, .. },
                            ) => {
                                let h_ab = sae_dot(jets.first(a), jets.first(b));
                                h_ab * Self::softmax_data_weight_product_logit_factor(
                                    a_soft, atom_a, atom_b, atom_w, inv_tau,
                                )
                            }
                            _ => {
                                sae_dot(jets.second(a, w), jets.first(b))
                                    + sae_dot(jets.first(a), jets.second(b, w))
                            }
                        };
                        if exact_a {
                            // #2330 Patch D (1a) — `A = B + ΔC` carries the residual
                            // curvature `ΔC_tt[a,b] = ⟨error_metric, ∂²f_ab⟩` the
                            // Gauss-Newton assembly drops, and that block moves with
                            // `θ_w` too:
                            //   `∂ΔC_tt[a,b]/∂θ_w = ⟨∂error_metric/∂θ_w, ∂²f_ab⟩`
                            //                      `+ ⟨error_metric, ∂³f_abw⟩`.
                            // `∂error_metric/∂θ_w` is `jets.first(w)` in this jet
                            // convention, so the first leg is a plain jet dot.
                            dh += sae_dot(jets.first(w), jets.second(a, b));
                        }
                        if let Some(ctx) = patchd_ctx.as_ref() {
                            // #2515 — the SECOND Patch-D leg `⟨error_metric, ∂³f_abw⟩`,
                            // through the same helper the dense route calls. It was
                            // previously scoped out of this port for want of
                            // `atom_third_jets()`; that is a per-row quantity and is
                            // built once above.
                            dh += self.patchd_residual_third_leg(
                                ctx,
                                jets.vars[a],
                                jets.vars[b],
                                jets.vars[w],
                            );
                        }
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
                                        threshold_strength,
                                        row,
                                        atom,
                                        jets.vars[w],
                                        ordered_beta_bernoulli_channels.as_ref(),
                                    ),
                                SaeLocalRowVar::Coord { atom, axis }
                                    if a == w && !ard_precisions[atom].is_empty() =>
                                {
                                    // The majorizer writes `α·softplus_{τ₀}(cos κt)`
                                    // into `H_tt`; `A` carries the unclamped
                                    // `α·cos κt`. Their difference is exactly the
                                    // `negative_hessian_remainder` the dense exact-A
                                    // route adds, pinned by
                                    // `exact_a_ard_operator_derivative_is_the_unmajorized_hessian_2515`.
                                    if exact_a {
                                        self.ard_exact_hessian_derivative(
                                            ard_precisions[atom][axis],
                                            row,
                                            atom,
                                            axis,
                                        )
                                    } else {
                                        self.ard_majorized_hessian_derivative(
                                            ard_precisions[atom][axis],
                                            row,
                                            atom,
                                            axis,
                                        )
                                    }
                                }
                                _ => 0.0,
                            };
                        }
                        if !defl_dirs.is_empty() {
                            deflated_base_dh_mat[[a, b]] = dh;
                        }
                        gamma += inv_vv[[b, a]] * dh;
                    }
                }
                if !defl_dirs.is_empty() {
                    // The row factor / log-det operator is the spectrally conditioned
                    // `Φ(H_tt)`, while the channels above assemble the RAW row
                    // derivative `D`. Subtract `tr(inv_vv·(D − DΦ[D]))` so the
                    // from-probes θ-adjoint differentiates the same operator as
                    // `arrow_log_det`, `apply_cached_arrow_hessian`, the selected
                    // inverse, and the dense θ-adjoint (#2712).
                    gamma -= Self::deflation_block_correction(
                        &inv_vv,
                        &deflated_base_dh_mat,
                        defl_dirs,
                        defl_spectrum,
                    );
                }
                // t–β block: reuse the dense contraction with the reconstructed inv_vβ.
                for a in 0..q {
                    for (beta_pos, channel) in border.iter().enumerate() {
                        // #2330 Patch D (1a), t–β leg: `ΔC_tβ[a,β]` moves with
                        // `θ_w` through the residual exactly as the t–t block does.
                        let mut dh = sae_dot(jets.second(a, w), jets.beta(beta_pos))
                            + sae_dot(jets.first(a), jets.beta_deriv(w, beta_pos))
                            + if exact_a {
                                sae_dot(jets.first(w), jets.beta_deriv(a, beta_pos))
                            } else {
                                0.0
                            };
                        if let Some(ctx) = patchd_ctx.as_ref() {
                            dh += self.patchd_residual_third_leg_beta(
                                ctx,
                                jets.vars[a],
                                jets.vars[w],
                                channel,
                            );
                        }
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
                            let bd = jets.beta_deriv(w, beta_pos);
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
                let mut dh_mat = if defl_dirs.is_empty() {
                    Array2::<f64>::zeros((0, 0))
                } else {
                    Array2::<f64>::zeros((q, q))
                };
                for a in 0..q {
                    for b in 0..q {
                        let dh = sae_dot(jets.beta_l_deriv(a, w_beta_pos), jets.first(b))
                            + sae_dot(jets.first(a), jets.beta_l_deriv(b, w_beta_pos));
                        if !defl_dirs.is_empty() {
                            dh_mat[[a, b]] = dh;
                        }
                        gamma += inv_vv[[b, a]] * dh;
                    }
                }
                if !defl_dirs.is_empty() {
                    // The border channels differentiate the same conditioned t–t block,
                    // so they carry the same Daleckii–Krein correction (#2712).
                    gamma -= Self::deflation_block_correction(
                        &inv_vv,
                        &dh_mat,
                        defl_dirs,
                        defl_spectrum,
                    );
                }
                for a in 0..q {
                    for (beta_pos, channel) in border.iter().enumerate() {
                        let dh = sae_dot(jets.beta_l_deriv(a, w_beta_pos), jets.beta(beta_pos));
                        gamma += 2.0 * inv_vbeta[[a, channel.index]] * dh;
                    }
                }
                gamma_beta[w_channel.index] += gamma;
            }
        }

        if let Some(channels) = ordered_beta_bernoulli_channels.as_ref() {
            let mut column_coefficient = vec![0.0_f64; k_atoms];
            for &(row, atom, _t_index, inverse_diagonal) in &ordered_beta_bernoulli_logit_sites {
                let index = row * k_atoms + atom;
                column_coefficient[atom] += inverse_diagonal * channels.m_channel[index];
            }
            for &(row, atom, t_index, _inverse_diagonal) in &ordered_beta_bernoulli_logit_sites {
                let index = row * k_atoms + atom;
                gamma_t[t_index] += column_coefficient[atom] * channels.z_jac[index];
            }
        }

        Ok(SaeArrowVector {
            t: gamma_t,
            beta: gamma_beta,
        })
    }

    // [#780 line-count gate] reconstruction_dispersion + assemble_shape_uncertainty
    // + recompute_joint_shape_uncertainty + unavailable_shape_uncertainty
    // (the contiguous trailing methods of this impl block) were split into the
    // sibling construction_reconstruction.rs (declared in mod.rs); callers reach
    // them bare via use super::*.
}
