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

/// Which lane priced one streaming evaluation's evidence, with what its derivative reads.
pub(crate) enum StreamingEvidence {
    Bundle(StreamingEvidenceArtifacts),
    /// #2234 — a closure-certified circle orbit, priced by the arrow orbit lane, whose geometry is
    /// the whole of what its derivative reads.
    ArrowOrbit(ArrowOrbitGeometry),
}

pub(crate) struct StreamingOuterEvaluation {
    pub(crate) cost: f64,
    pub(crate) loss: SaeManifoldLoss,
    pub(crate) cache: ArrowFactorCache,
    pub(crate) evidence: StreamingOuterEvidence,
}

/// What one streaming outer evaluation's derivative reads beside its `B` cache, by lane.
pub(crate) enum StreamingOuterEvidence {
    Bundle(StreamingBundleEvidence),
    /// #2234 step 1a — the arrow orbit lane's bordered elimination the value was priced off.
    ArrowOrbit(ArrowOrbitGeometry),
}

pub(crate) struct StreamingBundleEvidence {
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
    predicted_objective_decrease: f64,
    step: DampedResidualStep,
    system: Option<ArrowSchurSystem>,
}

/// #2283 — one committed trial of the arrow exact-A polish step, carrying what the
/// caller's band check, shift carry and phase log read.
struct ShiftedTerminalStep {
    shift: f64,
    ridge_escalations: usize,
    pre_objective: f64,
    committed_objective: f64,
    predicted_objective_decrease: f64,
    /// `Δᵀ(A + σI)Δ/‖Δ‖² = −gᵀΔ/‖Δ‖²`: the shifted operator's curvature along the
    /// committed step, the ladder's first rung above `σ = 0`.
    curvature_along_step: f64,
    step_norm_sq: f64,
    trials: usize,
    system: Option<ArrowSchurSystem>,
}

/// #2731/#2228 — what one terminal-polish call learned from its committed steps
/// that bought less than their own quadratic model: the largest rung (damping or
/// shift) such a step was taken at, and the shortest such step. While the memory
/// holds, a carry walks back only as far as its growth bound keeps the next step
/// inside that radius, instead of staying where it is.
#[derive(Clone, Copy)]
struct RefutedRung {
    rung: f64,
    radius: f64,
}

impl RefutedRung {
    fn record(previous: Option<Self>, rung: f64, step_norm: f64) -> Self {
        previous.map_or(
            Self {
                rung,
                radius: step_norm,
            },
            |refuted| Self {
                rung: refuted.rung.max(rung),
                radius: refuted.radius.min(step_norm),
            },
        )
    }

    /// Whether walking back to `walked_back` stays refuted: it lands at or below the
    /// refuted rung, and the walked-back step can still be as long as a refuted step.
    /// `growth` bounds how much longer than the committed step, `step_norm` long,
    /// the walked-back step can be.
    fn holds(&self, walked_back: f64, growth: f64, step_norm: f64) -> bool {
        walked_back <= self.rung && growth * step_norm >= self.radius
    }
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
    ///        + ½ log|A| + Σ_k ½ · dof_k · log(max(N_eff_k, 1))
    ///        − ½ · p · (Σ_k rank S_k) · log λ_smooth
    /// ```
    ///
    /// where `ℓ_pen = loss.total()` is the penalised objective at the inner
    /// optimum and the middle line is the #2a occupancy-aware BIC/Laplace
    /// complexity assembled by `rank_adjusted_quasi_laplace_complexity` from
    /// the EXACT observed information: the joint `log|A|`, coordinate block
    /// included, plus the per-atom realised-DOF rank charge.
    ///
    /// #2a superseded the majorizer form (`½ log|B|` over the PSD /
    /// Gauss--Newton arrow-Schur factor), and this seam then also subtracted the
    /// coordinate block `½ log|A_tt|`. That subtraction removed the only place
    /// `α = exp(log_ard)` enters the complexity while `loss.ard` kept its
    /// `−½·n·log α` normalizer; the rank charge `Σ ½·dof·log(max(N_eff,1))` is
    /// α-free. gam#2627 measured the consequence on a collapsed axis
    /// (`‖t₁‖² ≈ 4e-10`, `n = 4`): `dV/d log α = −2.0000000000` to ~1e-12 with
    /// zero curvature, exactly `−n/2`, the `loss.ard` term standing alone.
    /// #2668 keeps the coordinate block inside `log|A|`. On a Euclidean ARD axis
    /// `A_tt` carries α on its diagonal, so `∂(½ log|A|)/∂ log α = ½·α·tr[A⁻¹]_tt`
    /// and V's log-α stationarity condition is the MacKay fixed point
    /// `α = n/(‖t‖² + tr[A⁻¹]_tt)` that the EFS step iterates.
    ///
    /// The final `−½·p·rank(S)·log λ_smooth` term is the smoothing-penalty
    /// normaliser `−½ log|λ S|_+` restricted to its ρ-dependent part: `S_k` is
    /// shared across all `p` decoder output channels (the `⊗ I_p` Kronecker
    /// structure), so `log|λ S|_+ = p·rank(S)·log λ + p·log|S|_+`, and the
    /// `½ p·log|S|_+` piece is ρ-independent. The ρ-independent additive
    /// constants that ARE dropped here (they shift `V` by a constant and do not
    /// affect the ρ-argmin) are the formal `2π` Gaussian constant and the base
    /// `½ p·log|S|_+` penalty logdet. #2933 F45: the assignment-prior normalizers are not
    /// constants. The ThresholdGate partition `log[(1 − e^{−λ})/λ]` per free gate and the
    /// learnable-concentration ordered Beta--Bernoulli partition `Σ_k log C(a_k, N)` are
    /// carried in `loss.assignment_sparsity` with their ρ-derivatives. The softmax entropy
    /// and fixed-concentration ordered Beta--Bernoulli energies have finite normalizers
    /// that depend on `λ_sparse` but are not computed, so for those families `V` omits a
    /// ρ-dependent prior term and scores an unnormalized regularization energy.
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
        self.penalized_quasi_laplace_criterion_priced_with_lane(
            target,
            rho,
            registry,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
            refine_progress_extension,
            lane,
        )
        .map(|(value, loss, _priced)| (value, loss))
    }

    /// [`Self::penalized_quasi_laplace_criterion_with_refine_policy_and_lane`] with what the
    /// dense branch priced its value on: the converged factor cache and the exact-`A` spectral
    /// block (#2267). A value probe hands them to the gradient lane's evaluation at the same
    /// ρ, which otherwise re-converges and decomposes again the state it was handed. `None`
    /// on the streaming routes, which materialize no `A`.
    pub(crate) fn penalized_quasi_laplace_criterion_priced_with_lane(
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
    ) -> Result<
        (
            f64,
            SaeManifoldLoss,
            Option<(ArrowFactorCache, DenseExactAGeometry)>,
        ),
        SaeCriterionError,
    > {
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
            // `½·(log|A| − log|B|)` whenever `ΔC ≠ 0`
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
            .map(|(value, loss)| (value, loss, None))
        } else {
            let (v, loss, cache, geometry) = self.penalized_quasi_laplace_criterion_with_geometry(
                target,
                rho,
                registry,
                inner_max_iter,
                learning_rate,
                ridge_ext_coord,
                ridge_beta,
                refine_progress_extension,
            )?;
            Ok((v, loss, geometry.map(|geometry| (cache, geometry))))
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
        self.penalized_quasi_laplace_criterion_with_geometry(
            target,
            rho,
            registry,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
            refine_progress_extension,
        )
        .map(|(value, loss, cache, ..)| (value, loss, cache))
    }

    /// [`Self::penalized_quasi_laplace_criterion_with_cache_refine_policy`] with the dense
    /// exact-`A` spectral block the value was priced on (#2267), which the dense outer
    /// evaluation hands to its derivative. `None` on the streaming route, which materializes
    /// no `A`.
    pub(crate) fn penalized_quasi_laplace_criterion_with_geometry(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
        refine_progress_extension: bool,
    ) -> Result<
        (
            f64,
            SaeManifoldLoss,
            ArrowFactorCache,
            Option<DenseExactAGeometry>,
        ),
        SaeCriterionError,
    > {
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
            return self
                .penalized_quasi_laplace_criterion_streaming_exact_with_cache(
                    target,
                    rho,
                    registry,
                    inner_max_iter,
                    learning_rate,
                    ridge_ext_coord,
                    ridge_beta,
                )
                .map(|(value, loss, cache)| (value, loss, cache, None));
        }
        // 1. Run the inner (t, β) Newton solve to its numerical fixed point at
        //    FIXED ρ. Evidence uses the idempotence polish rather than stopping
        //    at the first coarse-KKT-band hit: the value and its implicit
        //    derivative must describe the same differentiable root (#2253).
        let mut rho_fixed = rho.clone();
        log::debug!(
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
        log::debug!(
            "[SAE-ENTRY] initial joint fit done {:.2}s after criterion entry",
            criterion_entered.elapsed().as_secs_f64(),
        );
        let mut loss = initial_fit.loss;
        let mut criterion_fixed_point = initial_fit.fixed_point;

        // 2. Drive the inner (t, β) solve to the KKT/step-converged optimum and
        //    take one final UNDAMPED factor there to obtain the joint Hessian
        //    log-determinant. We force ridge = 0 and the dense `Direct` Schur
        //    mode so `ArrowFactorCache::arrow_log_det` returns the exact
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
        let options = self.evidence_factor_options();
        // #2080 — the evidence root is converged, priced and, where `A` refuses on
        // resolved negative basin curvature, descended and converged again, all in
        // ONE gate-frozen scope, so every objective value compared below belongs to
        // the same objective (#2228 Zeno ratchet).
        let gates_were_frozen = self.freeze_collapse_prevention_gates();
        let evidence_root = loop {
            let cache = match self.converge_inner_for_undamped_logdet_gate_frozen(
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
            ) {
                Ok(cache) => cache,
                Err(err) => break Err(SaeCriterionError::from(err)),
            };
            loss.criterion_gauge_deflated_directions = cache.gauge_deflated_directions;
            // #2933 F05 — the root is priced under the gates frozen where the initial
            // joint fit ended, and they are not re-derived at the root. Re-deriving
            // `w ← W(θ̂)` and converging again until the root reproduced its own gates
            // (38c121f19) does not terminate in general: on a co-firing, near-collinear
            // K=2 fixture the repulsion gate switches off at the root it switched on and
            // back on at the next, a period-2 cycle that re-converged 189 times in 297 s
            // without pricing (job 1129738). Its fixed point `w = W(θ̂(ρ))` would also
            // move with ρ, which the analytic outer gradient does not differentiate.
            // `SaeManifoldOuterObjective` declares the gates its first priced root read
            // and holds them for the whole hyperparameter solve (see
            // `CollapsePreventionGates`), so re-evaluating a ρ from a root prices the
            // same objective.
            // #2330 Phase-2: rank the EXACT observed-information Laplace term ½log|A|
            // (A = B + ΔC = ∇²_θθ L), not the majorizer surrogate ½log|B|. One
            // eigendecomposition yields the joint log|A|, applying the shared PD
            // floor. An indefinite A the concave clamps do not explain is a saddle,
            // not a mode, and ½log|A| is not a Laplace normaliser there.
            //
            // A saddle is descended before it is refused. Pool job 507123 read the
            // refused directions on the #2080 wide-p fixture as gate logits (logit
            // share ≥ 0.9999999 on the 28 modes read, of 121 logged) with vᵀBv within
            // 5e-4 of 1 on all but one: the unit stiffness the evidence factor
            // substitutes where the majorizer has no curvature. At a saturated gate,
            // per-coordinate stationarity leaves the exact curvature along the logit
            // at about g_ℓ/τ (derived, not measured): a sub-tolerance slope toward
            // switching the gate on, which needs a long step that no Newton, MM or
            // damped-residual mover takes. #2336's refuted escape stepped to the line
            // minimum and re-converged under RE-frozen gates, climbing above the
            // saddle. Here the objective stays frozen and a descent commits only above
            // the material floor, so the walk strictly descends one objective and
            // ends. A saddle no refused direction can descend keeps the typed refusal,
            // which the outer search reads as an infeasible ρ.
            let mut saddle_directions = Vec::new();
            match self.exact_observed_information_log_dets_with_saddle_directions(
                rho,
                target,
                &cache,
                &mut saddle_directions,
            ) {
                Ok((log_det, geometry)) => break Ok((cache, log_det, geometry)),
                Err(err @ SaeCriterionError::IndefiniteObservedInformation { .. })
                    if inner_max_iter > 0 =>
                {
                    match self.descend_exact_a_saddle(
                        target,
                        rho,
                        registry,
                        &cache,
                        &saddle_directions,
                    ) {
                        Ok(true) => criterion_fixed_point = false,
                        Ok(false) => break Err(err),
                        Err(descent_err) => break Err(SaeCriterionError::from(descent_err)),
                    }
                }
                Err(err) => break Err(err),
            }
        };
        // #2933 F05 — a priced root leaves its gates declared, so re-pricing this
        // state (the shape-uncertainty recompute, a fitted term's next evaluation)
        // prices the objective this value belongs to rather than one re-frozen at
        // the re-pricing's own entry state. A refused evaluation hands back the
        // gate state it was given.
        let priced = (|| -> Result<
            (
                f64,
                SaeManifoldLoss,
                ArrowFactorCache,
                Option<DenseExactAGeometry>,
            ),
            SaeCriterionError,
        > {
            let (cache, log_det, mut geometry) = evidence_root?;

            // 3. Smoothing-prior normalizer `−½·Σ_k log|λ_k S_k ⊗ I_{r_k}|_+`
            //    (issue #972, #2933 F26): the `r_k·rank(S_k)·log λ_smooth` Occam term plus
            //    the base pseudo-determinant `r_k·log|S_k|_+`, so equivalent splits of one
            //    precision `λ_k S_k` price one value. The single seam is `reml_occam_term`,
            //    shared with the streaming path so both rank the identical normalizer.
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
                // #5/(B): the Laplace complexity is ½log|A| plus the honest BIC
                // ½·d_eff·log n on each atom's realised decoder rank. The coordinate
                // block stays inside log|A| (#2668): every row of `A_tt` carries the ARD
                // precision α that balances the `−½·n·log α` normalizer in `loss.ard`,
                // and subtracting the block, as this seam once did to remove the
                // decoder-scale term (`H_tt ∝ ‖B‖²`), left V falling linearly in log α
                // on a collapsing axis. `d_eff` is rotation-invariant, so it accepts a
                // real rank-2 circle but does not distinguish clean-vs-blend (producer's
                // job). A certified vanished atom is a typed boundary before rank
                // pricing.
                // Decoder disappearance is certified first from the raw output-frame
                // residual and gated decoder Grams. It has no tuned noise multiple:
                // the boundary is derived from the residual reduction's floating-point
                // backward error, and proof-unavailable is surfaced loudly.
                let residual = self.reconstruction_residual(target, rho)?;
                let mut grams = self.empty_decoder_gram_accumulator();
                self.accumulate_decoder_gram(&mut grams)?;
                let n_eff = self.per_atom_effective_sample_size();
                let residual_energy = self.residual_energy_for_vanishing(residual.view())?;
                match self.vanished_atoms_from_signal_upper_bound(
                    &grams,
                    &n_eff,
                    residual_energy.mean_square(),
                )? {
                    VanishedAtomsProof::Certified {
                        atoms: Some(atoms), ..
                    } => return Err(SaeCriterionError::VanishedAtoms(atoms)),
                    VanishedAtomsProof::Certified { atoms: None, .. } => {}
                    VanishedAtomsProof::Unavailable { reason } => {
                        return Err(SaeCriterionError::Numerical(format!(
                            "decoder-vanishing proof unavailable: {reason}"
                        )));
                    }
                }
                // #2933 F36 — the divergence reads the eigensystem ½log|A| was priced on:
                // the same materialization at the same cache and target, so the dense
                // criterion decomposes `A` once per evaluation, not twice.
                let dispersion = self
                    .reconstruction_dispersion_with_geometry(
                        &loss,
                        &cache,
                        rho,
                        residual.view(),
                        Some(HeldResponseGeometry::FixedFrame(&geometry.block)),
                    )
                    .map_err(|e| {
                        format!(
                            "SaeManifoldTerm::penalized_quasi_laplace_criterion: rank-charge dispersion is required: {e}"
                        )
                    })?;
                // #2933 F39 — the gradient's rank-charge derivative at this state reads the
                // dispersion off the geometry it is handed, so the fitted-response divergence
                // is formed once per evaluated state, not once for the value and again for the
                // gradient.
                geometry.rank_charge_dispersion = Some(dispersion);
                let disp = dispersion.raw_output_noise_variance;
                let d_eff = self.rank_dof_from_grams(&grams, &n_eff, rho, disp)?;
                // Occupancy-aware effective sample size N_eff,k = Σ_i a_{ik}², the #2a
                // per-atom BIC log-scale (same quantity `rank_dof_from_grams` uses
                // internally for the MP edge; recomputed here — a cheap Σa² — to price the
                // charge in the same currency).
                // #5/#2498 — the same-state gated-signal certificate above owns the
                // categorical Laplace-validity boundary. Do not manufacture a second
                // disappearance verdict from `d_eff == 0`: DOF also contains the
                // smooth-basis charge and is not a physical reconstruction signal.
                // #2a — occupancy-aware BIC/Laplace scale. The shared scalar helper
                // owns `0.5 log|A| + rank_charge`; dense, streaming, and
                // criterion-as-atoms assembly therefore cannot drift apart.
                // log_det (= log|A|, coordinate block included) comes from the exact
                // observed information above (#2668).
                let quasi_laplace_complexity =
                    rank_adjusted_quasi_laplace_complexity(log_det, &d_eff, &n_eff)?;
                let value = loss.total() + extra_penalty_energy + quasi_laplace_complexity - occam;
                // #2228 — the criterion's terms at the cache the `[SAE-ACCEPT]` line named, so
                // a split between two lanes at one ρ says which term moved.
                log::debug!(
                    "[SAE-CRITERION] V={value:.10e}: loss={:.10e} \
                     extra_penalty={extra_penalty_energy:.6e} ½log|A|={:.6e} rank_charge={:.6e} \
                     occam={occam:.6e}",
                    loss.total(),
                    0.5 * log_det,
                    quasi_laplace_complexity - 0.5 * log_det,
                );
                value
            };
            Ok((v, loss, cache, Some(geometry)))
        })();
        if priced.is_err() {
            self.streaming_gates_frozen = gates_were_frozen;
        }
        priced
    }

    /// Run the likelihood-flat-block mover on the LIVE state at the refine loop's
    /// objective-stall fixed-point claim, and consume the saved decrement
    /// certificate when the mover commits (#2762, #2228).
    ///
    /// `best_seen` is keyed on the `½λ²/scale` certificate, not on the penalized
    /// objective, so the state it names can sit materially above the live
    /// excursion. Every mover of the refine loop commits only objective
    /// decreases, so the live state is the objective incumbent, and this visit is
    /// a continuation path: the loop refuses only after
    /// `SAE_MANIFOLD_INNER_OBJECTIVE_STALL_MIN_ROUNDS` stalled rounds. Restoring
    /// `best_seen` before descending traded that incumbent for a worse state.
    /// Measured on the p=2048, charts=8 curved-tier cell (#2283, perf2731 job
    /// 448182): the polish and the next refine round took the objective from
    /// 9.4299854865e4 to 9.4299574102e4, this visit restored the 9.4299854865e4
    /// state, and the loop replayed the identical polish before refusing. Only
    /// the terminal exits restore `best_seen`: the refusal's report and the
    /// certify-at-best-seen gate.
    ///
    /// A committed move invalidates the saved certificate, because it belongs to
    /// a state the live one has left; a later refusal then reports the live state
    /// and cannot restore over the descent. An inert call leaves it in place.
    pub(crate) fn descend_gauge_orbit_consuming_best_seen(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        penalized_gram_scale: &[f64],
        best_seen: &mut Option<(f64, f64, SaeManifoldMutableState)>,
        max_rounds: usize,
    ) -> Result<GaugeOrbitDescent, String> {
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

    /// Largest componentwise Jacobi-scaled KKT gradient: a diagonal-preconditioned
    /// first-order residual in parameter units.
    ///
    /// It is not the remaining Newton displacement `H⁻¹g` (#2933 F08). A diagonal
    /// cannot see coupled weakly curved directions, so it is reported and certifies
    /// nothing.
    ///
    /// Each gradient component is divided by the diagonal curvature of its own
    /// block before the blocks are aggregated. The ordering is load-bearing:
    /// decoder gradients and curvatures are both extensive in the rows assigned
    /// to an atom, while coordinate components are row-local. Normalizing an
    /// already-aggregated L2 norm would retain a spurious `sqrt(K)` dependence;
    /// the max of individually scaled components is intensive in both `n` and
    /// `K`.
    pub fn system_scaled_grad_max(sys: &ArrowSchurSystem) -> Result<f64, SaeInnerKktScaleError> {
        let mut scaled_max = 0.0_f64;
        for (row_index, row) in sys.rows.iter().enumerate() {
            let gradient_len = row.gt.len();
            let (curvature_rows, curvature_cols) = row.htt.dim();
            let block = SaeInnerKktScaleBlock::CoordinateRow { row: row_index };
            if (curvature_rows, curvature_cols) != (gradient_len, gradient_len) {
                return Err(SaeInnerKktScaleError::GradientCurvatureShapeMismatch {
                    block,
                    gradient_len,
                    curvature_rows,
                    curvature_cols,
                });
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
            return Err(SaeInnerKktScaleError::GradientCurvatureShapeMismatch {
                block,
                gradient_len: sys.gb.len(),
                curvature_rows: diagonal.len(),
                curvature_cols: diagonal.len(),
            });
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
            if !curvature.is_finite() || curvature < 0.0 || (curvature == 0.0 && gradient != 0.0) {
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

    /// #2228 DIAGNOSTIC — the INTENSIVE companion of the bar this loop enforces.
    ///
    /// `quasi_laplace_kkt_stationary` compares an EXTENSIVE L2 gradient norm
    /// (`Σ_i ‖g_t^(i)‖² + ‖g_β‖²`, a sum over rows AND atoms) against
    /// `1e-5 · (1 + ‖x‖₂)`, whose right side grows only like `sqrt(#params)`.
    /// `system_scaled_grad_max` / `inner_iterate_max` are the componentwise,
    /// Jacobi-curvature-scaled pair whose own doc says they "remove row-count,
    /// atom-count, and basis-scale extensivity". The installed-state audit reports
    /// that pair and, since #2933 F08, does not certify on it: a diagonal-scaled
    /// residual is not a displacement. Its only production consumer is
    /// `installed_inner_kkt_audit` (the external-state certification entry), so no refusal raised by this
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
        if system.rows.len() != n
            || system.row_offsets.len() != n + 1
            || system.gb.len() != border_dim
        {
            return "orbit=unresolved(non-dense layout)".to_string();
        }
        let mut gradient = Array1::<f64>::zeros(dense_len + border_dim);
        // Dense product chart, as in `descend_gauge_orbit`: a hard-TopK row is
        // expanded into its dense slot, never copied at its compact offset (#2228).
        let compact_layout = self.last_row_layout.clone();
        let mut full_row = vec![0.0_f64; q];
        for (row_index, row) in system.rows.iter().enumerate() {
            let base = system.row_offsets[row_index];
            let dim = system.row_dims[row_index];
            if row.gt.len() < dim {
                return "orbit=unresolved(row layout)".to_string();
            }
            match compact_layout.as_ref() {
                Some(layout) => {
                    if dim != layout.row_q_active(row_index) {
                        return "orbit=unresolved(row layout)".to_string();
                    }
                    let compact_row: Vec<f64> = row.gt.iter().take(dim).copied().collect();
                    layout.expand_row(row_index, &compact_row, &mut full_row);
                    for axis in 0..q {
                        gradient[row_index * q + axis] = full_row[axis];
                    }
                }
                None => {
                    if base + dim > dense_len {
                        return "orbit=unresolved(row layout)".to_string();
                    }
                    for axis in 0..dim {
                        gradient[base + axis] = row.gt[axis];
                    }
                }
            }
        }
        let arrow_row_offsets = system.row_offsets.clone();
        for (index, &value) in system.gb.iter().enumerate() {
            gradient[dense_len + index] = value;
        }

        // The block the mover descends (chart orbit, decoder β-null, decoder
        // channel-null), from the SAME constructor and Gram--Schmidt the mover
        // uses, so what is measured here is what the mover could have descended.
        let orthonormal: Vec<Array1<f64>> = match self.likelihood_flat_block_basis(lambda_smooth)
        {
            Ok(basis) => basis
                .into_iter()
                .filter(|vector| vector.len() == gradient.len())
                .collect(),
            Err(reason) => return format!("orbit=unresolved(gauge basis: {reason})"),
        };
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
        // Both probes run the mover's own line minimization, with its endpoints and
        // commit floor, so a zero drop here means what it means there.
        let material_floor =
            SAE_MANIFOLD_INNER_OBJECTIVE_STALL_REL_TOL * (1.0 + base_objective.abs());
        let snapshot = self.snapshot_mutable_state();
        // Both walks step through the assembled arrow chart; under hard TopK a
        // dense-length direction fails every trial (#2228).
        let descent_step = match self.dense_joint_vector_in_arrow_layout(
            descent.view(),
            &arrow_row_offsets,
            border_dim,
            "SaeManifoldTerm::gauge_orbit_descent_diagnostic",
        ) {
            Ok(step) => step,
            Err(reason) => return format!("orbit=unresolved(layout: {reason})"),
        };
        let step_coord_len = arrow_row_offsets[n];
        let orbit = match self.minimize_objective_along(
            target,
            rho,
            registry,
            descent_step.view(),
            step_coord_len,
            base_objective,
            descent_norm,
            0.0,
            material_floor,
            &snapshot,
        ) {
            Ok(line) => line,
            Err(reason) => {
                return format!(
                    "orbit_dim={}, orbit_max_dderiv={max_derivative:.6e}, \
                     orbit_descent=unresolved({reason})",
                    orthonormal.len(),
                );
            }
        };
        let best_decrease = base_objective - orbit.value;
        let relative = if base_objective.abs() > 0.0 {
            best_decrease / base_objective.abs()
        } else {
            f64::INFINITY
        };

        // THE AMBIENT CONTROL. `g ≠ 0` on a differentiable objective makes `−g/‖g‖`
        // a descent direction, so if trials along it were evaluated and none met
        // sufficient decrease of `penalized_objective_total`, the assembled gradient
        // is not the gradient of the scalar the line search descends: an
        // objective↔gradient desync. A zero drop reads that way only beside
        // `finite_trials > 0`. A trial whose step failed to apply, or whose
        // objective was not finite, carries no information about the objective, so
        // those are counted and the first reason is named instead of being reported
        // as the same zero (#2228).
        let mut steepest = gradient.clone();
        let steepest_norm = steepest.dot(&steepest).sqrt();
        let ambient = if steepest_norm.is_finite() && steepest_norm > 0.0 {
            for value in steepest.iter_mut() {
                *value /= -steepest_norm;
            }
            let analytic_slope = gradient.dot(&steepest);
            let steepest_step = match self.dense_joint_vector_in_arrow_layout(
                steepest.view(),
                &arrow_row_offsets,
                border_dim,
                "SaeManifoldTerm::gauge_orbit_descent_diagnostic",
            ) {
                Ok(step) => step,
                Err(reason) => return format!("ambient=unresolved(layout: {reason})"),
            };
            match self.minimize_objective_along(
                target,
                rho,
                registry,
                steepest_step.view(),
                step_coord_len,
                base_objective,
                steepest_norm,
                0.0,
                material_floor,
                &snapshot,
            ) {
                Ok(line) => format!(
                    "ambient_slope={analytic_slope:.6e}, ambient_best_objective_drop={:.6e} \
                     at α={:.3e} (finite_trials={}, failed_trials={}, first_failure={:?})",
                    base_objective - line.value,
                    line.alpha,
                    line.finite_trials,
                    line.failed_trials,
                    line.first_failure,
                ),
                Err(reason) => format!("ambient=unresolved({reason})"),
            }
        } else {
            "ambient=degenerate".to_string()
        };

        format!(
            "orbit_dim={}, orbit_max_dderiv={max_derivative:.6e}, \
             orbit_best_objective_drop={best_decrease:.6e} at α={:.3e} \
             (objective {base_objective:.6e}, relative {relative:.6e}, finite_trials={}, \
             failed_trials={}, first_failure={:?}), {ambient}",
            orthonormal.len(),
            orbit.alpha,
            orbit.finite_trials,
            orbit.failed_trials,
            orbit.first_failure,
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

    /// The undamped evidence factorization options every inner acceptance lane
    /// factors with: Direct, Newton–Schur Tikhonov and unit-stiffness evidence
    /// deflation, both at the shared spectral floor.
    pub(crate) fn evidence_factor_options(&self) -> ArrowSolveOptions {
        ArrowSolveOptions::direct()
            .with_gpu_policy(self.gpu_policy)
            .with_newton_schur_tikhonov(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR)
            .with_evidence_unit_deflation(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR)
    }

    /// The affine-invariant inner acceptance certificate (#2226/#2228/#2253).
    /// `relative_decrease` is `½λ²/scale`, with `λ² = −gᵀΔ` the Newton decrement
    /// on the deflated exact factor and `scale` the objective scale. At or below
    /// the stall detector's no-meaningful-change band, no step lowers the
    /// penalized objective by a resolvable amount, however large the ambient ‖g‖
    /// is along a stiff direction. The objective-stall, budget-limit, best-seen
    /// and final-gate acceptances and the installed-state audit all read this
    /// one predicate, so a state the native inner solve accepts is a state the
    /// zero-step audit accepts (#2263).
    pub(crate) fn inner_decrement_certifies(relative_decrease: f64) -> bool {
        relative_decrease.is_finite()
            && relative_decrease <= SAE_MANIFOLD_INNER_OBJECTIVE_STALL_REL_TOL
    }

    /// `λ² = −gᵀΔ` of an inner acceptance factor's step, as the decrement
    /// certificate reads it (#2228, SPEC rule 22). `f64::max` returns its non-NaN
    /// operand, so the former `.max(0.0)` priced a NaN decrement as 0, and a
    /// materially negative one as 0 too, and 0 certifies. A negative decrement
    /// means the factor's step is not a descent direction, where no quadratic-model
    /// certificate exists. A negative value within the contraction's own rounding
    /// band, `accumulation_band(terms, Σ|gᵢΔᵢ|)` (the band the Armijo lane reads),
    /// reads as 0; any other value that is not a finite non-negative number reads
    /// as NaN, which [`Self::inner_decrement_certifies`] refuses.
    pub(crate) fn inner_certificate_decrement_sq(
        sys: &ArrowSchurSystem,
        delta_t: ndarray::ArrayView1<'_, f64>,
        delta_beta: ndarray::ArrayView1<'_, f64>,
    ) -> f64 {
        let decrease = sae_manifold_newton_directional_decrease(sys, delta_t, delta_beta);
        let raw = decrease.value;
        if !raw.is_finite() {
            return f64::NAN;
        }
        if raw >= 0.0 {
            return raw;
        }
        if decrease.rounding_band.is_finite() && -raw <= decrease.rounding_band {
            0.0
        } else {
            f64::NAN
        }
    }

    /// `½λ²/(|f| + 1)` at the installed state, priced exactly as the decrement
    /// acceptances price it: assemble at `(term, rho)`, take the deflated evidence
    /// factor with [`Self::evidence_factor_options`], and read the majorizer Newton
    /// decrement off that factor's discarded step. Where that decrement admits, the
    /// exact verdict decides, as it does for every native decrement acceptance
    /// (#2933 F08, [`Self::certified_decrement_acceptance`]):
    ///
    /// - A classified state reports its exact `½λ²/scale`.
    /// - A clamp basin, or a state above the dense admission, reports the majorizer's.
    /// - A saddle, or dense geometry that cannot be formed, is `Err`.
    ///
    /// The state is not moved.
    pub(crate) fn installed_newton_decrement_relative(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
    ) -> Result<f64, String> {
        let options = self.evidence_factor_options();
        let lambda_smooth = rho.lambda_smooth_vec().map_err(|err| err.to_string())?;
        let scale = self
            .penalized_objective_total(target, rho, registry, 1.0)
            .map_err(|err| err.to_string())?
            .abs()
            + 1.0;
        let mut sys = self
            .assemble_arrow_schur(target, rho, registry)
            .map_err(|err| err.to_string())?;
        let factor =
            self.factor_deflated_evidence_with_grad_norms(&mut sys, &lambda_smooth, &options)?;
        let decrement_sq = Self::inner_certificate_decrement_sq(
            &sys,
            factor.delta_t.view(),
            factor.delta_beta.view(),
        );
        // The audit refuses exactly the decrements the native lanes refuse.
        if !decrement_sq.is_finite() {
            return Err(format!(
                "installed-state Newton decrement is not a certificate: λ²={decrement_sq:e}"
            ));
        }
        let majorizer_relative = 0.5 * decrement_sq / scale;
        if !Self::inner_decrement_certifies(majorizer_relative) {
            return Ok(majorizer_relative);
        }
        match self.refined_root_verdict(target, rho, registry, &mut sys, &factor.cache)? {
            RefinedRootVerdict::Certified { relative, .. }
            | RefinedRootVerdict::Refused(RefinedRootRefusal::DecrementAboveTolerance {
                relative,
                ..
            }) => Ok(relative),
            RefinedRootVerdict::ClampBasin { .. } | RefinedRootVerdict::Unclassified => {
                Ok(majorizer_relative)
            }
            refused @ RefinedRootVerdict::Refused(_) => Err(format!(
                "the exact information refuses the installed state [{}]: {refused}",
                refused.tag()
            )),
        }
    }

    /// Install the per-row spectral deflation on an ACCEPTANCE system, take its
    /// undamped (ridge-0) criterion factorization, and read back both KKT residual
    /// norms (raw and quotient) off the SAME assembled system. This is the
    /// objective-stall diagnostic factorization (#1095/#2228/#1094): the returned
    /// [`DeflatedEvidenceFactor`] carries the finite deflated cache plus the
    /// discarded Newton step retained for the affine Newton-decrement diagnostic
    /// (#2226) that [`Self::inner_decrement_certifies`] accepts on. A solve failure surfaces as `Err`,
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
    ) -> usize {
        // Flat affine-gauge valleys can keep crawling productively after the
        // historical base budget. Extend only when the measured KKT residual has
        // shown a real finite round-to-round drop; true stalls end at the base
        // work budget (#968/#1029). Value-order probes pass the base budget as
        // their progress budget, so this branch cannot make probes expensive.
        //
        // #2230 COST-PROPORTIONAL EXTENSION: the latest pair of KKT residuals is
        // the progress verdict. A historical `|=` latch meant ONE gradient drop
        // anywhere granted the 16×/64× extended budget for the rest of the evaluation — an
        // oscillating or stalled tail then ground the full extended budget on
        // every criterion eval (the #1094 “kept extending via progress from
        // earlier rounds” pathology, and the
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
            Self::refine_round_made_progress(previous_grad_norm, grad_norm);
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
    /// # Objective-globalized spectral step
    ///
    /// The KKT residual remains the convergence certificate, but it cannot be
    /// the globalization currency: on a negative eigenmode of the exact Hessian,
    /// objective descent increases `||g||`.  The old `(A^2 + nu) delta = -A g`
    /// path therefore rejected the very move that left the saddle and made an
    /// entire outer-rho neighborhood look infeasible (#2080/#2228).
    ///
    /// The step now uses the absolute spectral Hessian,
    /// `delta_i = -g_i / (|lambda_i| + sqrt(nu))`.  Thus `g^T delta < 0` on every
    /// retained mode, irrespective of curvature sign, while it is ordinary
    /// Newton on a positive-definite basin.  The existing derived damping ladder
    /// controls its radius, and Armijo acceptance is measured against the actual
    /// penalized objective.  Rejected trials restore the snapshot bit-for-bit.
    /// No KKT tolerance changes: only the converged refine-loop gate can mint a
    /// fit.

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
        // #2283 — the warm-carried shift of the arrow exact-A step, the damping's
        // counterpart on states whose dense geometry the ledger does not admit.
        let mut shift = 0.0_f64;
        // #2731 — what this call learned from committed steps that bought less than
        // their own model predicted: the largest damping (and shift) such a step was
        // taken at, and the shortest such step. The carry does not walk back to a rung
        // at or below it while the walked-back step could be as long as a refuted
        // step, because returning there from a comparable state is a 2-cycle, not a
        // descent.
        //
        // #2228 — the rung alone is not a nearby iterate. Pool job 585774 at
        // `9544f0703` (`/scratch.global/sauer354/pool/sae2228/g4.seed2132.585774.txt`,
        // C=4 K=4 softmax, second polish call) committed step 6 at σ = 5.915e-1 with
        // ‖Δ‖ = 4.709 and agreement 0.158. The memory then held σ = 5.915 for steps
        // 7–64 at agreement 1.93–2.06 while ‖Δ‖ fell from 3.85e-2 to 1.07e-2 and ‖g‖
        // from 2.23e-1 to 6.37e-2 against tol 2.347e-3. At step 64 the measured
        // decrease, 6.811e-4, matched the quadratic model with `A` itself,
        // `½(−gᵀΔ) + ½σ‖Δ‖² = 6.812e-4`, to the printed digits, and the shift was
        // 99.4% of the curvature along the step. Those steps were 120–440× shorter
        // than the refuted one, and the memory never let the shift down.
        let mut refuted_damping: Option<RefutedRung> = None;
        let mut refuted_shift: Option<RefutedRung> = None;
        for step in 0..max_steps {
            let step_started = std::time::Instant::now();
            // #2267 — name each step to the process monitor; the guard ends with the
            // loop body on every exit.
            let polish_step_scope = gam_runtime::process_monitor::track_scope(format!(
                "sae terminal Newton polish step {}/{max_steps}",
                step + 1
            ));
            let mut sys = self
                .assemble_arrow_schur(target, rho_fixed, registry)
                .map_err(|err| format!("SaeManifoldTerm::terminal_exact_newton_polish: {err}"))?;
            let assemble_seconds = step_started.elapsed().as_secs_f64();
            let grad_norm_sq = Self::system_grad_norm_sq(&sys);
            if !grad_norm_sq.is_finite() {
                log::trace!("terminal Newton bail: non-finite ‖g‖² at entry");
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
                    log::trace!(
                        "terminal Newton bail: deflated criterion factor at ‖g‖={grad_norm:.6e}: {err}"
                    );
                    break;
                }
            };
            let decrement_sq = Self::inner_certificate_decrement_sq(
                &sys,
                factor.delta_t.view(),
                factor.delta_beta.view(),
            );
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
            log::debug!(
                "[SAE-NEWTON] polish step {}/{max_steps}: ‖g‖={grad_norm:.6e} \
                 (quotient {quotient_grad_norm:.6e}, tol {grad_tolerance:.3e}) \
                 λ²={decrement_sq:.6e} cert={cert:.6e}",
                step + 1,
            );
            let cache = factor.cache;
            // #2228 — which form of `A` this step is taken on is decided by the work of
            // the two forms, not only by whether the dense one fits. The dense geometry
            // (`materialize_exact_stationarity_geometry` below) pays one `O(dim³)`
            // symmetric eigendecomposition per step, and `dim = coords + border` grows
            // with ROWS. The arrow form (`shifted_exact_newton_polish_trials`) factors
            // each row block and the `k × k` border Schur complement,
            // `Σᵢ qᵢ³/3 + Σᵢ qᵢ²·k + k³/3`, linear in rows: at the #2132 anchor's C=3
            // shape (508 rows, `q = 5`, `k = 72`, dim 2612) about `1.1e6` flops against
            // `1.8e10·c`, and pool job 578251 read the dense step at ~4.8 s there, 40+
            // times in one call. Every assignment mode whose observed information is
            // arrow-structured therefore steps on the arrow form. Ordered Beta–Bernoulli
            // is the exception: its prior couples every row, so
            // `exact_a_evidence_system` cannot express `A`, and the dense geometry stays
            // its route under the #2724/#2283 memory admission at the EXACT dimension it
            // would build (the #2283 cell's 96 000 rows × 2 charts is 192 000
            // coordinates). A declined ordered Beta–Bernoulli geometry still tries the
            // arrow form, which refuses it and ends the phase.
            let exact_dim = sae_exact_stationarity_dim(cache.delta_t_len(), cache.k);
            let dense_admitted =
                sae_exact_stationarity_admitted(exact_dim, self.host_available_bytes);
            let dense_geometry_route = matches!(
                self.assignment.mode,
                AssignmentMode::OrderedBetaBernoulli { .. }
            ) && dense_admitted;
            if !dense_geometry_route {
                log::debug!(
                    "[SAE-NEWTON] step {}/{max_steps} steps on the arrow exact-A system (the \
                     dense geometry at dim={exact_dim} would hold {} resident bytes; admitted \
                     by the carried host reading of {} bytes: {dense_admitted})",
                    step + 1,
                    sae_exact_stationarity_resident_bytes(exact_dim),
                    self.host_available_bytes,
                );
                let backtrack_started = std::time::Instant::now();
                let Some(committed) = self.shifted_exact_newton_polish_trials(
                    target, rho_fixed, registry, options, &sys, shift,
                )?
                else {
                    log::trace!(
                        "terminal Newton bail: no shift on the arrow exact-A ladder bought \
                         sufficient Armijo decrease of the penalized objective at \
                         ‖g‖={grad_norm:.6e}"
                    );
                    break;
                };
                made_progress = true;
                if let Some(system) = committed.system.as_ref() {
                    let after_sq = Self::system_grad_norm_sq(system);
                    let after_gate =
                        self.quotient_gradient_norm_from_system(system, after_sq, lambda_smooth);
                    if Self::quasi_laplace_kkt_stationary(
                        after_sq.sqrt(),
                        after_gate,
                        grad_tolerance,
                    ) {
                        log::trace!(
                            "SAE terminal Newton reached the KKT band at arrow exact-A step {}: \
                             gate norm {quotient_grad_norm:.6e} → {after_gate:.6e} against tol \
                             {grad_tolerance:.6e}",
                            step + 1,
                        );
                        return Ok(true);
                    }
                }
                // The dense phase's carry, on the shift. The step's quadratic model buys
                // exactly half of `−gᵀΔ`, so a measured decrease at or above that walks
                // the shift back toward the undamped Newton step, and a smaller one
                // starts the next step one rung more damped. A shift at or under `ε` of
                // the operator's curvature along the step changes no digit of `A + σI`
                // there, so it IS the undamped step and is carried as exactly that.
                let model_agreement = (committed.pre_objective - committed.committed_objective)
                    / (0.5 * committed.predicted_objective_decrease);
                // The same refuted-rung memory as the dense carry below (#2731, #2228).
                // Walking σ back to σ′ lengthens each positive-curvature component of the
                // step, `uᵀg/(λ + σ)`, by `(λ + σ)/(λ + σ′) ≤ σ/σ′`.
                let step_norm = committed.step_norm_sq.sqrt();
                shift = if model_agreement >= 1.0 {
                    let walked_back = committed.shift / opt::constants::RIDGE_GROWTH;
                    let walked_back =
                        if walked_back <= f64::EPSILON * committed.curvature_along_step {
                            0.0
                        } else {
                            walked_back
                        };
                    let growth = committed.shift / walked_back;
                    if let Some(refuted) = refuted_shift
                        .filter(|refuted| refuted.holds(walked_back, growth, step_norm))
                    {
                        // #2731 — the rung below could make the step as long as a refuted
                        // one, so σ walks back only to `σ′ = σ·‖Δ‖/r`, which lengthens
                        // every positive-curvature component by at most `σ/σ′ = r/‖Δ‖`.
                        // Job 605564 (`b7945fab1`) held σ = 8.956239e-3 over polish steps
                        // 3–44 at agreement 1.85–1.990 while ‖Δ‖ went 0.168 → 0.109 under
                        // step 1's refuted radius 1.094; `σ′` was 1.375e-3 at step 3.
                        let bound = committed.shift * step_norm / refuted.radius;
                        if bound <= f64::EPSILON * committed.curvature_along_step {
                            0.0
                        } else {
                            bound.min(committed.shift)
                        }
                    } else {
                        walked_back
                    }
                } else {
                    refuted_shift = Some(RefutedRung::record(
                        refuted_shift,
                        committed.shift,
                        step_norm,
                    ));
                    if committed.shift > 0.0 {
                        committed.shift * opt::constants::RIDGE_GROWTH
                    } else {
                        committed.curvature_along_step
                    }
                };
                log::debug!(
                    "[SAE-NEWTON] step {} arrow exact-A phases: assemble={assemble_seconds:.2}s \
                     trials={} in {:.2}s (σ={:.6e}, ridge escalations {}, ‖Δ‖={:.6e}, model \
                     agreement {model_agreement:.3e}) total={:.2}s \
                     penalized_objective={:.10e}",
                    step + 1,
                    committed.trials,
                    backtrack_started.elapsed().as_secs_f64(),
                    committed.shift,
                    committed.ridge_escalations,
                    committed.step_norm_sq.sqrt(),
                    step_started.elapsed().as_secs_f64(),
                    committed.committed_objective,
                );
                continue;
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
                        log::trace!(
                            "terminal Newton bail: dense exact-stationarity geometry at \
                             ‖g‖={grad_norm:.6e}: {err}"
                        );
                        break;
                    }
                };
            let Some((curvature_min, curvature_max)) = geometry.retained_curvature_extremes()
            else {
                log::trace!(
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
            // The smallest predicted reduction the acceptance test below can verify.
            // That test admits `pre − trial ≥ c1·pred − cushion`, with the round-off
            // cushion `opt::armijo_roundoff_cushion(pre)`. At or under
            // `pred = cushion / c1` its right-hand side is not positive, so a trial
            // that did not lower the objective, or raised it within round-off, would
            // commit and report progress. Above this floor every committed step is a
            // strict Armijo decrease (#2861), and a ladder whose prediction falls
            // under it has no verifiable decrease left to buy.
            let pre_objective = self.penalized_objective_total(target, rho_fixed, registry, 1.0)?;
            let predicted_floor =
                opt::armijo_roundoff_cushion(pre_objective) / SAE_MANIFOLD_ARMIJO_C1;
            let snapshot = self.snapshot_mutable_state();
            let backtrack_started = std::time::Instant::now();
            let mut trials = 0usize;
            let mut accepted: Option<AcceptedTerminalResidualStep> = None;
            let mut nu = damping;
            loop {
                let damped = match geometry.damped_objective_step(&residual, nu) {
                    Ok(damped) => damped,
                    Err(err) => {
                        log::trace!(
                            "terminal Newton bail: damped residual step at ν={nu:.6e}: {err}"
                        );
                        break;
                    }
                };
                let predicted_objective_decrease =
                    -(residual.t.dot(&damped.step.t) + residual.beta.dot(&damped.step.beta));
                if !(predicted_objective_decrease.is_finite()
                    && predicted_objective_decrease > predicted_floor)
                {
                    // The model's predicted reduction decreases monotonically in
                    // ν, so no larger damping on this ladder can clear the floor
                    // either: the ladder is exhausted, and it is exhausted for a
                    // stated reason rather than at a trial count.
                    log::trace!(
                        "terminal Newton: damping ladder exhausted at ν={nu:.6e} — predicted \
                         objective decrease {predicted_objective_decrease:.6e} is under the \
                         round-off floor {predicted_floor:.6e}",
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
                let trial_objective = self
                    .penalized_objective_total(target, rho_fixed, registry, 1.0)
                    .unwrap_or(f64::INFINITY);
                let sufficient = SAE_MANIFOLD_ARMIJO_C1 * predicted_objective_decrease;
                // Acceptance is Armijo descent in the scalar objective. The residual is
                // deliberately not constrained here: at negative curvature, genuine
                // objective descent can and generally does increase its norm.
                if trial_objective.is_finite()
                    && pre_objective - trial_objective
                        >= sufficient - opt::armijo_roundoff_cushion(pre_objective)
                {
                    accepted = Some(AcceptedTerminalResidualStep {
                        damping: nu,
                        trial_merits,
                        predicted_objective_decrease,
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
                    log::trace!(
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
                    log::trace!(
                        "terminal Newton: damping ladder cannot advance past ν={nu:.6e} \
                         (λ_min²={smallest_damping:.6e} is not representable above it)"
                    );
                    break;
                }
                nu = next;
            }
            let Some(accepted) = accepted else {
                log::trace!(
                    "terminal Newton bail: no damping on [{smallest_damping:.6e}, \
                     {largest_damping:.6e}] bought sufficient Armijo decrease of the \
                     penalized objective at ‖g‖={grad_norm:.6e} ({trials} trial(s))"
                );
                break;
            };
            made_progress = true;
            let gate_norm = quotient_grad_norm;
            // The committed step bought an Armijo decrease of the penalized
            // objective. Nothing below extrapolates the gate norm to decide whether
            // to take the next one (#2267/#2283): the phase's currency is the
            // objective, and along a resolved negative-curvature mode the gate is not
            // monotone across accepted steps. On the #2283 documented cell one call's
            // gate went 4.42e-1 → 3.16 → 13.6 → 7.44 → 13.0 → 10.9 → 6.88 → 23.3 →
            // 6.01 → 6.48 → 5.76 while the objective fell by 2.4, and reading step
            // 10's one-step contraction (0.889) as a rate put the band 80.7 steps
            // away and ended the phase. A one-step ratio of a non-monotone sequence
            // is not a rate. The phase ends at the band, when no damping buys an
            // Armijo decrease, at a bail, or at `max_steps`.
            if let Some(system) = accepted.system.as_ref() {
                let after_sq = Self::system_grad_norm_sq(system);
                let after_gate =
                    self.quotient_gradient_norm_from_system(system, after_sq, lambda_smooth);
                if Self::quasi_laplace_kkt_stationary(after_sq.sqrt(), after_gate, grad_tolerance) {
                    // The step landed in the band. Say so without paying for the
                    // next loop top's assembly to rediscover it.
                    log::trace!(
                        "SAE terminal Newton reached the KKT band at step {}: gate norm \
                         {gate_norm:.6e} → {after_gate:.6e} against tol {grad_tolerance:.6e}",
                        step + 1,
                    );
                    return Ok(true);
                }
            }
            // The penalized objective at the committed state, next to the gate
            // norm it left behind. Both the phase and the refine window descend
            // the objective (#2861), so a step that lowers the objective while
            // raising the gate norm is the negative-curvature case above, and it
            // is readable as such only with both on one line.
            let committed_objective = self
                .penalized_objective_total(target, rho_fixed, registry, 1.0)
                .unwrap_or(f64::NAN);
            // #2731 — carry the damping by how well the step's own model predicted
            // what the step bought. The step minimizes `gᵀΔ + ½Δᵀ(|A| + √ν)Δ` over
            // the retained modes, so that model decreases by exactly half of
            // `predicted_objective_decrease`. Were the objective the quadratic with
            // Hessian `A`, the measured decrease would be at least that on every
            // retained mode: `c²/M − ½λc²/M² ≥ ½c²/M` for `M = |λ| + √ν ≥ |λ|`, for
            // either sign of `λ`. A measured decrease below the model's says the
            // step outran the second-order model, so the next step starts one rung
            // more damped; at or above it the damping walks back toward the
            // undamped Newton step. The Armijo test above admits any decrease of at
            // least `ARMIJO_C1 = 1e-4` of the linear prediction, so acceptance alone
            // says nothing about agreement. Job 539190 (`p = 2048, charts = 32`) read
            // the gate norm contract 3.21× and 4.08× on the two steps taken at
            // ν = 4.4e-5, and go from 2.964212e0 to 1.187672e1 across the eleven
            // taken at ν ≤ 4.4e-6, the rungs the unconditional walk-back returned to.
            // Job 578261 (`d40a922e0`, same cell) read the plain walk-back settle into
            // a 2-cycle over polish steps 59–64: ν = 7.68e-7 (agreement 1.09–1.14)
            // took ‖g‖ 1.18 → 0.044, and the walked-back ν = 7.68e-8 (agreement
            // 0.44–0.81) took it back to 1.18, until `max_steps` ended the call at a
            // refine entry ‖g‖ of 3.49e-2 against tol 2.501e-3. So a rung that bought
            // less than its model is remembered. While the walked-back step could be as
            // long as the shortest refuted step, the walk-back goes only as far as the
            // growth bound keeps the step inside that radius. The two steps of such a
            // 2-cycle undo each other, so they are about equally long, and the memory
            // holds there. It lets go once the committed steps are shorter than a
            // refuted one by more than a rung's growth (#2228).
            let model_agreement = (pre_objective - committed_objective)
                / (0.5 * accepted.predicted_objective_decrease);
            let step_norm = accepted.step.step_norm_sq.sqrt();
            damping = if model_agreement >= 1.0 {
                // A damping under `λ_min²` cannot move the flattest resolved
                // direction, so it IS the undamped step and is carried as exactly
                // that.
                let walked_back = accepted.damping / opt::constants::RIDGE_GROWTH;
                let walked_back = if walked_back < smallest_damping { 0.0 } else { walked_back };
                // Each retained component of the step is `uᵢᵀg/(|λᵢ| + √ν)`, so walking
                // ν back to ν′ lengthens it by at most `(λ_min + √ν)/(λ_min + √ν′)`.
                let growth = (curvature_min + accepted.damping.sqrt())
                    / (curvature_min + walked_back.sqrt());
                if let Some(refuted) = refuted_damping
                    .filter(|refuted| refuted.holds(walked_back, growth, step_norm))
                {
                    // The same bound on ν: `√ν′ = (λ_min + √ν)·‖Δ‖/r − λ_min` keeps every
                    // retained component inside the refuted radius (#2731).
                    let root = (curvature_min + accepted.damping.sqrt()) * step_norm
                        / refuted.radius
                        - curvature_min;
                    let bound = if root > 0.0 { root * root } else { 0.0 };
                    let bound = if bound < smallest_damping { 0.0 } else { bound };
                    bound.min(accepted.damping)
                } else {
                    walked_back
                }
            } else {
                refuted_damping = Some(RefutedRung::record(
                    refuted_damping,
                    accepted.damping,
                    step_norm,
                ));
                if accepted.damping > 0.0 {
                    accepted.damping * opt::constants::RIDGE_GROWTH
                } else {
                    smallest_damping
                }
            };
            log::debug!(
                "[SAE-NEWTON] step {} phases: assemble={assemble_seconds:.2}s \
                 trials={trials} in {:.2}s (ν={:.6e}, ‖Δ‖={:.6e}, damped rank {}/{}, \
                 ‖g_null‖={:.6e} of ‖g‖={grad_norm:.6e}, model agreement \
                 {model_agreement:.3e}) total={:.2}s \
                 penalized_objective={committed_objective:.10e}",
                step + 1,
                backtrack_started.elapsed().as_secs_f64(),
                accepted.damping,
                accepted.step.step_norm_sq.sqrt(),
                accepted.step.retained_rank,
                geometry.eigenvalues.len(),
                accepted.step.excluded_gradient_norm_sq.sqrt(),
                step_started.elapsed().as_secs_f64(),
            );
            log::trace!(
                "SAE terminal Newton step committed: quotient merit {:.6e} → {:.6e} \
                 (predicted quotient reduction {:.6e}, measured {:.6e}, ratio {:.4e}); \
                 ambient merit {:.6e} → {:.6e}; ‖g‖ {grad_norm:.6e} → {:.6e}, tol \
                 {grad_tolerance:.6e}",
                pre_merits.quotient,
                accepted.trial_merits.quotient,
                accepted.predicted_objective_decrease,
                pre_merits.quotient - accepted.trial_merits.quotient,
                if accepted.predicted_objective_decrease > 0.0 {
                    (pre_merits.quotient - accepted.trial_merits.quotient)
                        / accepted.predicted_objective_decrease
                } else {
                    f64::NAN
                },
                pre_merits.ambient,
                accepted.trial_merits.ambient,
                (2.0 * accepted.trial_merits.ambient).max(0.0).sqrt(),
            );
            drop(polish_step_scope);
        }
        Ok(made_progress)
    }

    /// #2283 — the objective-globalized terminal step on the ARROW form of the exact
    /// observed information, for a state whose dense exact-stationarity geometry the
    /// #2724 ledger does not admit.
    ///
    /// The dense phase reads `Δ = −Σᵢ uᵢ(uᵢᵀg)/(|λᵢ| + √ν)` off one eigendecomposition
    /// of `A`, whose `dim × dim` blocks grow with rows (dim 192,288 at the #2283
    /// cell). Declining that geometry used to end the phase, so a plateau at any row
    /// count the ledger refuses fell through to the stall refusal. The shifted Newton
    /// step `Δ(σ) = −(A + σI)⁻¹g` meets the same descent contract with no `dim²`
    /// object: [`Self::exact_a_evidence_system`] is `A` in arrow form, and the
    /// escalating solve factors it row by row plus the border Schur complement at the
    /// smallest ridge that is positive definite. For any such factorization
    ///
    /// ```text
    ///   gᵀΔ = −gᵀ(A + σI)⁻¹g < 0,     gᵀΔ + ½Δᵀ(A + σI)Δ = ½·gᵀΔ,
    /// ```
    ///
    /// so the step descends at either sign of curvature, its quadratic model buys
    /// exactly half of `−gᵀΔ` (the identity the caller's model-agreement carry reads),
    /// and `−gᵀΔ(σ)` falls as `σ` grows, which is what lets the verifiability floor end
    /// the ladder for a stated reason. The first rung above `σ = 0` is the shifted
    /// operator's own curvature along the rejected step, `Δᵀ(A + σI)Δ/‖Δ‖² =
    /// −gᵀΔ/‖Δ‖²`; later rungs grow by `RIDGE_GROWTH`. Acceptance, and the floor under
    /// the prediction, are the dense ladder's own, so a committed step is a strict
    /// Armijo decrease of the penalized objective (#2861), and a rejected trial
    /// restores the snapshot. `None` means no rung bought that decrease, or a solve
    /// failed.
    fn shifted_exact_newton_polish_trials(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho_fixed: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        options: &ArrowSolveOptions,
        majorizer: &ArrowSchurSystem,
        carried_shift: f64,
    ) -> Result<Option<ShiftedTerminalStep>, String> {
        let exact = match self.exact_a_evidence_system(target, rho_fixed, majorizer, 1.0) {
            Ok(exact) => exact,
            Err(err) => {
                log::trace!("terminal Newton bail: arrow exact-A system: {err}");
                return Ok(None);
            }
        };
        // The escalating solve reuses a supplied resident device frame for every
        // trial, and a frame describes the system it was prepared for. This system
        // is `A`, so it prepares its own.
        let mut exact_options = options.clone();
        exact_options.sae_resident_frame = None;
        let total_t: usize = exact.rows.iter().map(|row| row.gt.len()).sum();
        let pre_objective = self.penalized_objective_total(target, rho_fixed, registry, 1.0)?;
        // The dense ladder's floor: the smallest prediction whose Armijo threshold
        // `c1·pred − cushion` is positive, so every trial the test below admits
        // lowered the objective.
        let predicted_floor = opt::armijo_roundoff_cushion(pre_objective) / SAE_MANIFOLD_ARMIJO_C1;
        let snapshot = self.snapshot_mutable_state();
        let mut shift = carried_shift;
        let mut trials = 0usize;
        loop {
            let (delta_t, delta_beta, diagnostics) =
                match gam_solve::arrow_schur::solve_with_lm_escalation_inner(
                    &exact,
                    shift,
                    shift,
                    &exact_options,
                ) {
                    Ok(solution) => solution,
                    Err(err) => {
                        log::trace!(
                            "terminal Newton bail: arrow exact-A solve at σ={shift:.6e}: {err}"
                        );
                        return Ok(None);
                    }
                };
            if delta_t.len() != total_t {
                return Err(format!(
                    "SaeManifoldTerm::terminal_exact_newton_polish: the arrow exact-A step has \
                     {} coordinates for a system whose rows hold {total_t}",
                    delta_t.len()
                ));
            }
            let mut directional = exact.gb.dot(&delta_beta);
            let mut offset = 0usize;
            for row in &exact.rows {
                for (axis, &g) in row.gt.iter().enumerate() {
                    directional += g * delta_t[offset + axis];
                }
                offset += row.gt.len();
            }
            let predicted_objective_decrease = -directional;
            if !(predicted_objective_decrease.is_finite()
                && predicted_objective_decrease > predicted_floor)
            {
                log::trace!(
                    "terminal Newton: arrow exact-A ladder exhausted at σ={shift:.6e} — \
                     predicted objective decrease {predicted_objective_decrease:.6e} is under \
                     the verifiable floor {predicted_floor:.6e}",
                );
                return Ok(None);
            }
            trials += 1;
            let step_norm_sq = delta_t.dot(&delta_t) + delta_beta.dot(&delta_beta);
            let trial_system =
                match self.apply_newton_step(delta_t.view(), delta_beta.view(), 1.0) {
                    Ok(()) => match self.assemble_arrow_schur(target, rho_fixed, registry) {
                        Ok(trial_sys) => Some(trial_sys),
                        Err(err) => {
                            log::trace!(
                                "terminal Newton: arrow exact-A trial assembly at σ={shift:.6e}: \
                                 {err}"
                            );
                            None
                        }
                    },
                    Err(err) => {
                        log::trace!(
                            "terminal Newton: arrow exact-A trial step at σ={shift:.6e}: {err}"
                        );
                        None
                    }
                };
            let trial_objective = if trial_system.is_some() {
                self.penalized_objective_total(target, rho_fixed, registry, 1.0)
                    .unwrap_or(f64::INFINITY)
            } else {
                f64::INFINITY
            };
            let sufficient = SAE_MANIFOLD_ARMIJO_C1 * predicted_objective_decrease;
            if trial_objective.is_finite()
                && pre_objective - trial_objective
                    >= sufficient - opt::armijo_roundoff_cushion(pre_objective)
            {
                return Ok(Some(ShiftedTerminalStep {
                    shift,
                    ridge_escalations: diagnostics.ridge_escalations,
                    pre_objective,
                    committed_objective: trial_objective,
                    predicted_objective_decrease,
                    curvature_along_step: predicted_objective_decrease / step_norm_sq,
                    step_norm_sq,
                    trials,
                    system: trial_system,
                }));
            }
            self.restore_mutable_state(&snapshot)?;
            let next = if shift > 0.0 {
                shift * opt::constants::RIDGE_GROWTH
            } else {
                predicted_objective_decrease / step_norm_sq
            };
            if !(next.is_finite() && next > shift) {
                log::trace!(
                    "terminal Newton: arrow exact-A ladder cannot advance past σ={shift:.6e} \
                     (next rung {next:.6e})"
                );
                return Ok(None);
            }
            shift = next;
        }
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
        // A candidate that lies in the span of the stored bases must come out of
        // modified Gram–Schmidt as rounding, and nothing more. Each projection forms
        // one length-`full_len` inner product and updates every entry with a product
        // and a subtraction, leaking at most `γ_{full_len+4}·‖g₀‖`; it also leaks what
        // the stored bases' own loss of orthogonality leaves behind, at most
        // `Σ_j ω_j·‖g₀‖`. After `k` projections a dependent residual therefore stays
        // inside `k·(γ_{full_len+4} + Σ_j ω_j)·‖g₀‖`, and a basis stored from a
        // residual `r` of a candidate `g₀` carries `ω = band·‖g₀‖/‖r‖`.
        let projection_growth = gam_linalg::roundoff::accumulation_growth(full_len + 4);
        let mut orthogonality_defect = 0.0_f64;
        for (raw_idx, mut gauge) in raw_gauges.into_iter().enumerate() {
            let initial_norm_sq = gauge.iter().map(|v| v * v).sum::<f64>();
            for basis in &gauge_span {
                let coeff = gauge.dot(basis);
                for i in 0..gauge.len() {
                    gauge[i] -= coeff * basis[i];
                }
            }
            let norm_sq = gauge.iter().map(|v| v * v).sum::<f64>();
            let band = gauge_span.len() as f64 * (projection_growth + orthogonality_defect);
            if !(norm_sq.is_finite() && norm_sq > band * band * initial_norm_sq) {
                continue;
            }
            orthogonality_defect += band * (initial_norm_sq / norm_sq).sqrt();
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
            // An orthonormal combination with a unit eigenvector column has unit norm,
            // so only an exactly zero or non-finite direction is refused.
            let norm_sq = direction.iter().map(|v| v * v).sum::<f64>();
            if !(norm_sq.is_finite() && norm_sq > 0.0) {
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
        // floor already kept a gauge, its residual here lies inside the band below
        // and it is not double-counted.
        //
        // The band is the span construction's modified Gram–Schmidt band, taken
        // against the kept directions. Each kept direction is a normalized
        // combination `G·v` of the span bases with a computed eigenvector column `v`,
        // so against another kept direction it is off orthogonality by at most
        // `‖GᵀG − I‖₂ ≤ span_rank·Σ_j ω_j` (the span's own defect), plus the
        // eigenvector columns' orthogonality (`O(span_rank·u)` for a Householder-based
        // symmetric eigensolver, counted as `γ_{span_rank}`), plus the combination's
        // formation, `γ_{span_rank}·√span_rank` per vector.
        let kept_defect = span_rank as f64 * orthogonality_defect
            + gam_linalg::roundoff::accumulation_growth(span_rank)
                * (1.0 + 2.0 * (span_rank as f64).sqrt());
        let mut kept_orthogonality_defect = orthonormal.len() as f64 * kept_defect;
        for exact_idx in 0..exact_basis_count {
            let mut direction = gauge_span[exact_idx].clone();
            let initial_norm_sq = direction.iter().map(|v| v * v).sum::<f64>();
            let band = orthonormal.len() as f64 * (projection_growth + kept_orthogonality_defect);
            for kept in &orthonormal {
                let coeff = direction.dot(kept);
                for row in 0..direction.len() {
                    direction[row] -= coeff * kept[row];
                }
            }
            let norm_sq = direction.iter().map(|v| v * v).sum::<f64>();
            if !(norm_sq.is_finite() && norm_sq > band * band * initial_norm_sq) {
                continue;
            }
            kept_orthogonality_defect += band * (initial_norm_sq / norm_sq).sqrt();
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

    /// Smoothing-prior normalizer
    /// `½·Σ_k log|λ_k S_k ⊗ I_{r_k}|_+ = ½·Σ_k r_k·(rank(S_k)·log λ_k + log|S_k|_+)`
    /// (issue #972; #1556 per-atom λ; #2933 F26).
    ///
    /// Atom `k`'s decoder prior on its penalized subspace is the Gaussian with
    /// precision `λ_k S_k ⊗ I_{r_k}`: the `S_k` roughness penalty acts on `r_k`
    /// coordinate channels (`r_k == p` on the full-`B` path, the smaller frame rank
    /// when a Grassmann frame is active), each contributing `rank(S_k)` penalized
    /// directions. Its negative-log normalizer is `−½ log|λ_k S_k ⊗ I_{r_k}|_+`, which
    /// `V = … − occam` carries. Directions in `null(S_k)` carry a flat unit-density
    /// prior and no normalizer.
    ///
    /// #2933 F26 — this used to keep only `½ r_k·rank(S_k)·log λ_k`, which depends on
    /// how the one precision `λ_k S_k` is split between its factors: `S_k → c·S_k`,
    /// `λ_k → λ_k/c` leaves the prior, the fitted state and `log|A|` unchanged but
    /// moved `V` by `½ r_k·rank(S_k)·log c` (4.605 nats at `r_k·rank(S_k) = 2`,
    /// `c = 100`), and it left `V` blind to a curvature-parameterised `S_k(κ)`.
    /// `log|S_k|_+` is taken on the eigenspace [`Self::symmetric_rank`] counts. It is
    /// constant in `log λ_k`, so [`Self::reml_occam_log_lambda_smooth_derivative`] is
    /// unchanged; its curvature channel is [`Self::reml_occam_kappa_derivative`].
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
            let (rank_s, log_pdet_s) =
                Self::symmetric_rank_and_log_pseudodeterminant(atom.smooth_penalty())?;
            // Penalized decoder channels: `r_k` coordinate channels carry the
            // `S_k` roughness penalty (full-`B` path ⇒ `r_k == p`).
            let channels = atom.border_frame_rank() as f64;
            let log_lambda = rho.log_lambda_smooth[atom_idx];
            acc += 0.5 * channels * (rank_s as f64 * log_lambda + log_pdet_s);
        }
        // `V = … − occam`, so the net occam SUBTRACTS the penalty normalizer.
        Ok(acc)
    }

    /// Per-curvature-coordinate derivative `∂(occam)/∂κ_k = ½·r_k·tr(S_k⁺ ∂S_k/∂κ_k)`
    /// (#2933 F26). A raw sectional curvature moves `occam` only through
    /// `log|S_k(κ_k)|_+`. Returns `(flat index, derivative)` for every atom in
    /// `rho.kappa_atoms` whose penalty is curvature-parameterised, on the stratum
    /// where `rank(S_k)` is locally constant (the constant-curvature Dirichlet Gram
    /// keeps the constants null at every κ).
    pub(crate) fn reml_occam_kappa_derivative(
        &self,
        rho: &SaeManifoldRho,
    ) -> Result<Vec<(usize, f64)>, String> {
        self.assignment.validate_rho_domain(rho)?;
        let mut out = Vec::with_capacity(rho.kappa_atoms.len());
        for (flat, atom_idx, ds) in self.kappa_penalty_derivatives(rho)? {
            let atom = &self.atoms[atom_idx];
            let differential =
                Self::symmetric_log_pseudodeterminant_differential(atom.smooth_penalty(), ds)?;
            out.push((flat, 0.5 * atom.border_frame_rank() as f64 * differential));
        }
        Ok(out)
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
    /// log-determinant is the chunked matrix-free `streaming_exact_arrow_log_det_with_lane_and_system`.
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
        let artifacts = artifacts.ok_or_else(|| {
            SaeCriterionError::Numerical(
                "streaming outer evaluation did not retain its matrix-free evidence system \
                 and exact-A factor cache"
                    .to_string(),
            )
        });
        let (system, exact_a_cache) = match artifacts {
            Ok(StreamingEvidence::Bundle(StreamingEvidenceArtifacts {
                majorizer_system,
                exact_a_cache,
            })) => (majorizer_system, exact_a_cache),
            Ok(StreamingEvidence::ArrowOrbit(geometry)) => {
                // #2234 — the orbit lane prices the value off its own elimination and asks the
                // rational lane for nothing.
                drop(lane.take_logdet_derivative_bundle());
                drop(lane.take_inverse_probes());
                return Ok(StreamingOuterEvaluation {
                    cost,
                    loss,
                    cache,
                    evidence: StreamingOuterEvidence::ArrowOrbit(geometry),
                });
            }
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
            evidence: StreamingOuterEvidence::Bundle(StreamingBundleEvidence {
                system,
                exact_a_cache,
                logdet_derivative_bundle,
                efs_inverse_probe_bundle,
            }),
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
            Option<StreamingEvidence>,
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
        // #2933 F05 — as on the dense route: a priced root leaves its gates declared.
        if out.is_err() {
            self.streaming_gates_frozen = gates_were_frozen;
        }
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
            Option<StreamingEvidence>,
        ),
        SaeCriterionError,
    > {
        let mut loss = initial_fit.loss;
        let mut criterion_fixed_point = initial_fit.fixed_point;
        // Drive the inner (t, β) state to the SAME KKT/step-converged optimum the
        // dense `penalized_quasi_laplace_criterion_with_cache` reaches before factoring. At that
        // optimum the per-row `H_tt^(i)` blocks are PD, so the undamped
        // (`ridge_t = 0`) streaming factorization in `streaming_exact_arrow_log_det_with_lane_and_system`
        // succeeds — without this, a state stopped after only `inner_max_iter`
        // steps can leave a rank-deficient / indefinite row block (`p_out = 1` →
        // rank-1 `JᵀJ`, softmax negative-logit curvature) that surfaces
        // `PerRowFactorFailed` at base ridge 0. Sharing the driver puts both lanes
        // at the SAME inner state — but not, since #2330 Phase-2, on the same
        // evidence operator; see the #2509 note below.
        let options = self.evidence_factor_options();
        // The converged arrow-factor cache is the per-row factored Hessian
        // (matrix-free, feasible at massive K — the dense border_dim² Schur is
        // never materialised here); it is RETURNED so the EFS lane can take its
        // matrix-free ARD/smoothness traces off it. The log-determinant itself is
        // recomputed chunk-by-chunk in `streaming_exact_arrow_log_det_with_lane_and_system` to bound
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
        // #2234 — a closure-certified circle orbit is integrated exactly on this route too, off the
        // same predicate the dense route stiffens on: the arrow orbit lane prices
        // `log|A_s| − log det N − 2·Σ log I_k + K·log 2π` off one elimination of the bordered
        // operator where the stiffened pencil is certified free of band and negative directions,
        // and every other orbit state refuses by the lane's name, so the two routes never price
        // one state differently without saying so.
        let orbit_generators: Vec<CircleOrbitGenerator> = self
            .separated_compact_orbit_pricing(rho, target, &converged_cache)?
            .into_iter()
            .filter_map(|pricing| match pricing {
                CompactOrbitPricing::ExactCircle(generator) => Some(generator),
                CompactOrbitPricing::Laplace { .. } => None,
            })
            .collect();
        // #9: accumulate the per-atom Grams + N_eff in the same log-det pass.
        // These are required by the canonical rank-charge criterion.
        let mut rank_inputs = StreamingRankInputs::default();
        // #2515 — an INDEFINITE exact-A verdict from the arrow evidence route must
        // arrive here as the SAME typed error the dense route raises, not as a
        // generic `Numerical`. Both routes are saying "this state is a saddle, so
        // `½log|A|` is not a Laplace normalizer"; the outer solver reads the typed
        // one as an infeasible ρ (`+inf`, steer away) and the untyped one as a
        // defect that aborts the fit. Same verdict, two behaviours, chosen by which
        // route the memory planner picked — this issue's genus one level up.
        let (log_det, evidence_artifacts) = if orbit_generators.is_empty() {
            self.streaming_exact_arrow_log_det_with_lane_and_system(
                target,
                rho,
                registry,
                Some(&mut rank_inputs),
                lane,
            )
            .map_err(SaeCriterionError::from_arrow_refusal)?
        } else {
            // The rank charge's Grams and effective sizes come off the same full-row
            // assembly the streaming log-determinant accumulates them from.
            rank_inputs.grams = self.empty_decoder_gram_accumulator();
            rank_inputs.n_eff = vec![0.0; self.k_atoms()];
            self.assemble_full_matrix_free_evidence_system(
                target,
                rho,
                registry,
                Some(&mut rank_inputs),
            )?;
            let geometry =
                self.arrow_orbit_geometry(rho, target, &converged_cache, orbit_generators)?;
            (
                geometry.log_det(),
                Some(StreamingEvidence::ArrowOrbit(geometry)),
            )
        };
        // The returned row-factor cache and the external matrix-free log|S|
        // estimate are one evidence operator. Stamp the authoritative joint
        // value onto the cache so from-probes theta-adjoint consumers can verify
        // that their selected-inverse bundle differentiates a live log-det,
        // exactly as dense caches do through their Schur-factor path.
        converged_cache.joint_hessian_log_det = Some(log_det);
        converged_cache.schur_factor_is_undamped = true;
        // #3439 — the periodic phases' circle volume, priced off this `B` cache as the dense
        // lane prices it off its own (`periodic_phase_marginal`). The orbit lane integrates its
        // atoms' collective shift already, so only the phases outside its orbits are priced.
        // The stamp above stays the operator's `log|A|`: the correction is not a determinant.
        let orbit_atoms: &[CircleOrbitGenerator] = match evidence_artifacts.as_ref() {
            Some(StreamingEvidence::ArrowOrbit(geometry)) => &geometry.orbit_generators,
            _ => &[],
        };
        let phase_correction = match self.periodic_phase_marginal(&converged_cache, orbit_atoms)? {
            Some((correction, _)) => {
                log::debug!(
                    "[SAE-CRITERION streaming] periodic phase circle volume: ½Δlog|A|={:.6e}",
                    0.5 * correction
                );
                correction
            }
            None => 0.0,
        };
        let log_det = log_det + phase_correction;
        let occam = self.reml_occam_term(rho)?;
        // Extra penalized-objective energy (#671/#737 + full-objective
        // completion: registry penalties + repulsion + separation barrier),
        // matching the full-batch `penalized_quasi_laplace_criterion_with_cache` path so streaming
        // and dense criteria rank the identical penalized objective.
        let extra_penalty_energy =
            self.reml_extra_penalty_value_total(registry)
                .map_err(|err| {
                    format!(
                        "SaeManifoldTerm::penalized_quasi_laplace_criterion_streaming_exact_with_lane: {err}"
                    )
                })?;
        let v = {
            let ri = rank_inputs;
            // #9/#5 streaming rank charge: the criterion charges ½·log_det, the
            // whole joint arrow log-det with the coordinate blocks included, as the
            // dense lane charges ½log|A|, and adds Σ ½·d_eff·log n on each atom's
            // realised decoder rank, priced through the SAME `rank_dof_from_grams`
            // MP hard count as the dense path off the chunk-accumulated Grams. The
            // shared seam is `0.5*log_det + rank_charge`. On THIS lane
            // `log_det = Σ log|H_tt| + log|S_B|` by construction (plus the periodic
            // phases' circle volume, which both lanes price alike), so the per-row
            // t-block log-dets enter both lanes identically and the criterion's
            // exposure to the A-vs-B operator split (#2509) is `log|S_A|` against
            // `log|S_B|`.
            let residual = self.reconstruction_residual(target, rho)?;
            let residual_energy = self.residual_energy_for_vanishing(residual.view())?;
            match self.vanished_atoms_from_signal_upper_bound(
                &ri.grams,
                &ri.n_eff,
                residual_energy.mean_square(),
            )? {
                VanishedAtomsProof::Certified {
                    atoms: Some(atoms), ..
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
                    residual.view(),
                )
                .map_err(|e| {
                    format!(
                        "SaeManifoldTerm::penalized_quasi_laplace_criterion_streaming_exact_with_lane: rank-charge dispersion is required: {e}"
                    )
                })?
                .raw_output_noise_variance;
            let d_eff = self.rank_dof_from_grams(&ri.grams, &ri.n_eff, rho, disp)?;
            // #5/#2498: the typed gated-signal proof above is the sole
            // disappearance verdict. The scalar rank-charge seam only prices the
            // already-certified live state.
            let quasi_laplace_complexity =
                rank_adjusted_quasi_laplace_complexity(log_det, &d_eff, &ri.n_eff)?;
            let value = loss.total() + extra_penalty_energy + quasi_laplace_complexity - occam;
            // #2515 — the dense lane's `[SAE-CRITERION]` terms, on this lane, so a split
            // between the two routes at one ρ says which term moved.
            log::debug!(
                "[SAE-CRITERION streaming] V={value:.10e}: loss={:.10e} \
                 extra_penalty={extra_penalty_energy:.6e} ½log|A|={:.6e} rank_charge={:.6e} \
                 occam={occam:.6e}",
                loss.total(),
                0.5 * log_det,
                quasi_laplace_complexity - 0.5 * log_det,
            );
            value
        };
        Ok((v, loss, converged_cache, evidence_artifacts))
    }

    /// The streaming exact criterion with the #2080 surrogate lane
    /// threaded to the streaming `log|S|` term (`None` = bit-identical SLQ).
    pub(crate) fn penalized_quasi_laplace_criterion_streaming_exact_with_lane(
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
    /// `sae_exact_a_pencil_floor`); a
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
    /// `ΔC_ββ` is leg (5) of `ΔC` (#2828): the β-tier decoder priors install PSD
    /// majorizers in the shared block, so `A_ββ` is the majorizer's operator plus
    /// their exact-minus-majorizer remainder, at the `penalty_scale` the majorizer
    /// system was assembled with. The system composes that remainder into its
    /// penalty operator and the classification geometry carries it alone, so this
    /// route prices the `A_ββ` the dense route materializes (#2515).
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
        penalty_scale: f64,
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
        let delta = self.assemble_exact_hessian_minus_b_rows(rho, target, &row_dims, border_dim)?;
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
        let border = self.border_channels_for_border_dim(border_dim)?;
        let classification_indices: std::sync::Arc<[usize]> = border
            .iter()
            .map(|channel| channel.index)
            .collect::<Vec<_>>()
            .into();
        let mut clamp_base = 0usize;
        let classification_rows: std::sync::Arc<[gam_solve::arrow_schur::ExactAClassificationRow]> =
            delta
                .into_iter()
                .zip(row_dims.iter().copied())
                .map(|(block, q)| {
                    let clamp_diag = clamp.slice(s![clamp_base..clamp_base + q]).to_owned();
                    clamp_base += q;
                    gam_solve::arrow_schur::ExactAClassificationRow {
                        delta_tt: block.tt,
                        delta_tbeta: block.tbeta,
                        border_columns: std::sync::Arc::clone(&classification_indices),
                        clamp_diag,
                    }
                })
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
                for (row_idx, (row, block)) in system
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
                            row.htbeta[[a, channel.index]] += block.delta_tbeta[[a, beta_pos]];
                        }
                    }
                }
            }
            (Some(base_forward), Some(base_transpose)) => {
                // #2627 — the composed operator's declared per-row norm bounds: the
                // majorizer's own, plus the exact-A correction placed at
                // `classification_indices`. A border index repeated `m` times sums `m`
                // correction columns, so that leg is at most `√m_max·‖delta_tbeta‖_F`.
                let Some(base_declaration) = majorizer.htbeta_declaration.clone() else {
                    return Err(
                        "SaeManifoldTerm::exact_a_evidence_system: the majorizer installed a \
                         matrix-free cross-block operator without its declaration, so the \
                         composed operator has no bound to declare (#2627)"
                            .to_string(),
                    );
                };
                let widest_index_repeat = {
                    let mut sorted: Vec<usize> = classification_indices.iter().copied().collect();
                    sorted.sort_unstable();
                    let mut widest = usize::from(!sorted.is_empty());
                    let mut run = 1usize;
                    for pair in sorted.windows(2) {
                        run = if pair[0] == pair[1] { run + 1 } else { 1 };
                        widest = widest.max(run);
                    }
                    widest
                };
                let composed_row_norm_bounds: std::sync::Arc<[f64]> = base_declaration
                    .row_norm_bounds
                    .iter()
                    .zip(classification_rows.iter())
                    .map(|(&base_bound, block)| {
                        let correction = gam_solve::arrow_schur::frobenius_norm_upper_bound(
                            block.delta_tbeta.iter().copied(),
                        );
                        let placed = gam_solve::arrow_schur::guaranteed_norm_upper_bound(
                            correction * (widest_index_repeat as f64).sqrt(),
                            2,
                        );
                        gam_solve::arrow_schur::guaranteed_norm_upper_bound(base_bound + placed, 1)
                    })
                    .collect();
                // The composed forward adds the correction's inner product over the
                // `classification_indices` entries into the base operator's output, and the
                // transpose adds each correction column into its border entry once per
                // repeat of the index: one addition past the longer of the stages.
                let composed_declaration = gam_solve::arrow_schur::RowHtbetaDeclaration {
                    row_norm_bounds: composed_row_norm_bounds,
                    apply_depth: base_declaration
                        .apply_depth
                        .max(classification_indices.len())
                        .max(widest_index_repeat + 1)
                        + 1,
                };
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
                    composed_declaration,
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
        // #2828 — leg (5): `A_ββ = B_ββ + ΔC_ββ`, at the scale the majorizer's
        // β-tier priors were installed with. Every reduced-Schur apply of this
        // system sees the composed operator; the geometry carries the remainder
        // alone so the classifier recovers `B_raw` and the border clamp.
        let remainder_op = self.decoder_prior_border_remainder_op(border_dim, penalty_scale)?;
        let border_remainder = match remainder_op {
            Some(remainder) => {
                let remainder: std::sync::Arc<dyn gam_solve::arrow_schur::BetaPenaltyOp> =
                    std::sync::Arc::new(remainder);
                system.set_penalty_op(std::sync::Arc::new(
                    gam_solve::arrow_schur::CompositePenaltyOp {
                        k: border_dim,
                        ops: vec![
                            majorizer.effective_penalty_op(),
                            std::sync::Arc::clone(&remainder),
                        ],
                    },
                ));
                Some(remainder)
            }
            None => None,
        };
        system.exact_a_classification =
            Some(gam_solve::arrow_schur::ExactAClassificationGeometry {
                rows: classification_rows,
                border_remainder,
            });
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
            .map_err(|error| format!("SaeManifoldTerm::streaming_exact_arrow_log_det_with_lane_and_system: {error}"))?;
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
    ) -> Result<(f64, Option<StreamingEvidence>), String> {
        if target.dim() != (self.n_obs(), self.output_dim()) {
            return Err(format!(
                "SaeManifoldTerm::streaming_exact_arrow_log_det_with_lane_and_system: target must be ({}, {}); got {:?}",
                self.n_obs(),
                self.output_dim(),
                target.dim()
            ));
        }
        // #9: when the rank charge is on, accumulate the per-atom Grams + effective
        // sample sizes chunk-additively alongside the log-det (single pass). Zero
        // cost / untouched when `None`.
        if let Some(ri) = rank_inputs.as_deref_mut() {
            ri.grams = self.empty_decoder_gram_accumulator();
            ri.n_eff = vec![0.0; self.k_atoms()];
        }
        let plan = self.streaming_plan()?.admitted_or_error(
            self.n_obs(),
            self.output_dim(),
            self.k_atoms(),
        )?;
        // A gradient-bearing streaming evaluation always goes through the lane,
        // even when a chunked dense Schur would fit: only the lane emits the
        // derivative bundle whose contractions are the exact derivative of its
        // value. Where the host holds the dense lane's own k×k blocks, the lane
        // takes the exact log-det off one eigendecomposition; otherwise it
        // evaluates the frozen rational surrogate (#2731). A value-only call takes the
        // chunked dense route only where it is cheaper than SLQ, as priced below.
        // #972 / #977 T1: the reduced β-Schur is over the FACTORED border when
        // frames are active (each chunk inherits the frames via
        // `materialize_chunk`, so every `chunk_schur` is `border_dim²`), matching
        // the dense path's factored log-det. Full-`B` ⇒ `border_dim == beta_dim`.
        let border_dim = if self.frames_active() {
            self.factored_border_dim()
        } else {
            self.beta_dim()
        };
        // #2731/#2900 — the chunked branch pulls every row's elimination term back into
        // a dense `border_dim²` reduced Schur, `n·q·k²` flops for `q = K·(1 + d_max)`, and
        // holds about six such blocks. The matrix-free branch prices `log|S|` by SLQ at a
        // fixed `(probes + 1)·steps` products, each about `2·p·(support + q)` flops per
        // row, with every atom's basis in the support (exact for an OBB assignment, an
        // upper bound under top-k routing). The chunked branch is taken only where the
        // plan's in-core budget holds the block and the dense route costs fewer products
        // than SLQ. Census job 599567 (`sae_manifold_k_ladder_recovery_k64`, n = 4000,
        // q = 128, k = 30720) spent its time in that pullback, about 4.8e14 flops against
        // SLQ's 2112 products of about 4.1e8.
        let rows = self.n_obs() as u64;
        let total_basis: u64 = self.atoms.iter().map(|atom| atom.basis_size() as u64).sum();
        let d_max = self.atoms.iter().map(|atom| atom.latent_dim()).max().unwrap_or(0) as u64;
        let row_block_dim = (self.k_atoms() as u64).saturating_mul(1 + d_max);
        let apply_flops = rows
            .saturating_mul(2)
            .saturating_mul(self.output_dim() as u64)
            .saturating_mul(total_basis.saturating_add(row_block_dim));
        let border = border_dim as u64;
        let pullback_flops = rows
            .saturating_mul(row_block_dim)
            .saturating_mul(border)
            .saturating_mul(border);
        let slq_products = (SCHUR_SLQ_LOGDET_PROBES + 1) * SCHUR_SLQ_LOGDET_LANCZOS_STEPS;
        let dense_reduced_schur_admitted = plan.estimated_dense_schur_bytes
            <= plan.in_core_budget_bytes
            && matches!(
                gam_solve::arrow_schur::dense_reduced_schur_route(
                    gam_solve::arrow_schur::DenseReducedSchurRoute::ChunkedEvidence {
                        pullback_flops,
                    },
                    border_dim,
                    apply_flops,
                ),
                gam_linalg::pcg::PcgAttempt::Budgeted { products } if products < slq_products
            );
        if !dense_reduced_schur_admitted || lane.is_some() {
            // #988 memory-matrix-free evidence route. The dense k×k reduced Schur
            // (≈8 GB at the K=32k manifold border) does NOT fit the in-core
            // budget, so estimate log|S| via Stochastic Lanczos Quadrature on the
            // matrix-free `schur_matvec` apply (`gam_solve::arrow_schur::
            // matrix_free_arrow_evidence_log_det_surrogate`) instead of assembling +
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
            let a_sys = chunk_term.exact_a_evidence_system(target, rho, &sys, 1.0)?;
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
                    // #2731 — the dense lane allocates several k×k blocks, so it is admitted
                    // at that size, not at the one block the chunked route above prices.
                    // Under this branch's refusing exact-A policy the dense lane is the
                    // exact-A pencil lane, admitted at its own count (#2933 F07). It is taken
                    // only where it also costs no more products than the rational surrogate
                    // would spend, priced at the reduced-Schur product `apply_flops` above
                    // (#2900 row 6.16).
                    let dense_lane_admitted =
                        gam_solve::arrow_schur::dense_lane_exact_a_pencil_peak_bytes(a_sys.k)
                            .is_some_and(|bytes| bytes <= plan.in_core_budget_bytes)
                            && gam_solve::arrow_schur::surrogate_lane_prices_dense_reduced_schur(
                                lane,
                                a_sys.k,
                                apply_flops,
                            );
                    let evaluated = gam_solve::arrow_schur::matrix_free_arrow_evidence_evaluation(
                        &a_sys,
                        0.0,
                        0.0,
                        &options,
                        SCHUR_SLQ_LOGDET_PROBES,
                        SCHUR_SLQ_LOGDET_LANCZOS_STEPS,
                        SCHUR_SLQ_LOGDET_SEED,
                        lane,
                        dense_lane_admitted,
                    )
                    .map_err(|err| {
                        format!(
                            "SaeManifoldTerm::streaming_exact_arrow_log_det_with_lane_and_system: matrix-free criterion log-det: {err:?}"
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
                            "SaeManifoldTerm::streaming_exact_arrow_log_det_with_lane_and_system: matrix-free criterion log-det: {err:?}"
                        )
                    })?;
                    (log_det_tt, log_det_schur, None)
                }
            };
            if !log_det_schur.is_finite() {
                return Err(format!(
                    "SaeManifoldTerm::streaming_exact_arrow_log_det_with_lane_and_system: matrix-free reduced-Schur \
                     log|S| non-finite ({log_det_schur})"
                ));
            }
            return Ok((
                log_det_tt + log_det_schur,
                exact_a_cache.map(|exact_a_cache| StreamingEvidence::Bundle(StreamingEvidenceArtifacts {
                    majorizer_system: sys,
                    exact_a_cache,
                })),
            ));
        }
        let n_total = self.n_obs();
        let chunk_size = plan.chunk_size.min(n_total.max(1));
        let mut schur_acc = Array2::<f64>::zeros((border_dim, border_dim));
        let mut majorizer_acc = Array2::<f64>::zeros((border_dim, border_dim));
        let mut clamp_acc = Array2::<f64>::zeros((border_dim, border_dim));
        let mut exact_a_chunks = 0usize;
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
                .map_err(|err| format!("SaeManifoldTerm::streaming_exact_arrow_log_det_with_lane_and_system: {err}"))?;
            // #2509/#2515 Phase-2b — same substitution as the matrix-free branch:
            // the Laplace normalizer is `log|A|`, and every `ΔC` channel except
            // ordered Beta-Bernoulli is row-local, so the correction is
            // chunk-additive exactly as the majorizer is. (`penalty_scale` scales
            // the β-side penalties, and leg (5) of `ΔC` with them.)
            let sys = chunk.exact_a_evidence_system(z_chunk, rho, &sys, penalty_scale)?;
            let mut streaming = StreamingArrowSchur::from_system(&sys, sys.rows.len().max(1));
            let evidence = streaming
                .evidence_schur_chunk(0.0, 0.0, &options)
                .map_err(|err| format!("SaeManifoldTerm::streaming_exact_arrow_log_det_with_lane_and_system: {err}"))?;
            log_det_tt += evidence.log_det_tt;
            match (evidence.majorizer_metric, evidence.clamp_metric) {
                (Some(majorizer), Some(clamp)) => {
                    majorizer_acc += &majorizer;
                    clamp_acc += &clamp;
                    exact_a_chunks += 1;
                }
                (None, None) => {}
                _ => {
                    return Err("SaeManifoldTerm::streaming_exact_arrow_log_det_with_lane_and_system: partial exact-A \
                                chunk carrier is not a classification"
                        .to_string());
                }
            }
            for row in 0..border_dim {
                for col in 0..border_dim {
                    schur_acc[[row, col]] += evidence.schur[[row, col]];
                }
            }
            start = end;
        }
        let expected_chunks = n_total.div_ceil(chunk_size);
        if exact_a_chunks != 0 && exact_a_chunks != expected_chunks {
            return Err("SaeManifoldTerm::streaming_exact_arrow_log_det_with_lane_and_system: partial exact-A \
                        chunk carrier would classify a different operator"
                .to_string());
        }
        let exact_a = (exact_a_chunks == expected_chunks).then_some((&majorizer_acc, &clamp_acc));
        let log_det_schur = StreamingArrowSchur::reduced_schur_log_det(
            &schur_acc,
            &options,
            exact_a.map(|metrics| metrics.0),
            exact_a.map(|metrics| metrics.1),
        )
        .map_err(|err| format!("SaeManifoldTerm::streaming_exact_arrow_log_det_with_lane_and_system: {err}"))?;
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
        // #1038/#1419 softmax: the assembled majorizer is the DIAGONAL
        // `scale·D`, `D = diag(Σ_j|H_kj|)` of the entropy block, carrying the row's
        // design weight (#991). It scales linearly with `λ_sparse = exp(ρ)`, so
        // `∂B/∂ρ = scale·D` on the free logit slots of the reduced K−1 chart, and
        // it takes the one Daleckii–Krein deflation correction below, like every
        // other family (#2916). The kept-subspace diagonal it used to contract
        // equals that correction only when no deflated direction couples to the
        // border, since `vᵢᵀ (H⁻¹)_tt vᵢ = 1 + (H_βt vᵢ)ᵀ S⁻¹ (H_βt vᵢ)`.
        let mut hdiag = match self.assignment.mode {
            AssignmentMode::Softmax {
                temperature,
                sparsity,
            } => {
                if k_atoms <= 1 {
                    return Ok(0.0);
                }
                let inv_tau = 1.0 / temperature;
                let scale = rho.lambda_sparse()? * sparsity * inv_tau * inv_tau;
                let penalty = gam_terms::analytic_penalties::SoftmaxAssignmentSparsityPenalty::new(
                    k_atoms,
                    temperature,
                );
                let row_loss_w = self.row_loss_weights.as_deref();
                let mut weighted = Array1::<f64>::zeros(self.n_obs() * k_atoms);
                for row in 0..self.n_obs() {
                    let w_row = row_loss_w.map_or(1.0, |w| w[row]);
                    let row_logits: Vec<f64> = (0..k_atoms)
                        .map(|k| self.assignment.logits[[row, k]])
                        .collect();
                    let d = penalty.psd_majorizer_abs_row_sums(&row_logits, scale);
                    for atom in 0..k_atoms.min(d.len()) {
                        weighted[row * k_atoms + atom] = w_row * d[atom];
                    }
                }
                weighted
            }
            _ => crate::assignment::assignment_prior_log_strength_hdiag_weighted(
                &self.assignment,
                rho,
                self.row_loss_weights.as_deref(),
            )?,
        };
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
        // `hdiag` differentiates the prior along whatever `log_lambda_sparse` carries:
        // the concentration when it is effectively learnable. A fixed concentration puts
        // no coordinate into the prior (#2933 F45), so `hdiag` is zero and there is nothing
        // to majorize.
        let ordered_channels = ordered_beta_bernoulli_psd_majorizer_third_channels_weighted(
            &self.assignment,
            rho,
            self.row_loss_weights.as_deref(),
        )?;
        // The integrated marginal's mass-Hessian coefficient is strictly
        // negative, so its cross-row rank-one block has the zero PSD Loewner
        // majorizer. Retain only the positive part of the row-local
        // concrete-Jacobian term, matching assembly exactly.
        if let Some(ch) = ordered_channels.as_ref()
            && self.assignment.effective_alpha_is_learnable()
        {
            for row in 0..self.n_obs() {
                for atom in 0..k_atoms {
                    let slot = row * k_atoms + atom;
                    hdiag[slot] =
                        super::construction_arrow_schur_assembly::ordered_beta_bernoulli_psd_majorized_log_alpha_hdiag(
                            ch, row, k_atoms, atom, hdiag[slot],
                        );
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
            let spectrum = cache
                .deflation_row_spectra
                .get(row)
                .and_then(Option::as_ref);
            if Self::row_deflation_is_live(dirs, spectrum) {
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
                trace -= Self::deflation_block_correction(&inv_vv, &d_mat, dirs, spectrum);
            }
        }
        Ok(0.5 * trace)
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
    /// counterpart: the full Daleckii–Krein `tr(inv_vv·(D − DΦ[D]))` on every branch.
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
        // Same predicate as the dense trace: a fixed concentration leaves `hdiag` zero.
        if let Some(channels) = ordered_channels.as_ref()
            && self.assignment.effective_alpha_is_learnable()
        {
            for row in 0..self.n_obs() {
                for atom in 0..k_atoms {
                    let index = row * k_atoms + atom;
                    hdiag[index] =
                        super::construction_arrow_schur_assembly::ordered_beta_bernoulli_psd_majorized_log_alpha_hdiag(
                            channels, row, k_atoms, atom, hdiag[index],
                        );
                }
            }
        }
        // #2915 — under the exact observed information this trace differentiates
        // `A = B + ΔC`. `assignment_prior_log_strength_hdiag_weighted` reads the
        // gate's PSD clamp `∂B/∂ρ_sparse`, so add the non-positive remainder from
        // the producer the dense delta map uses. Both are degree one in
        // `λ_sparse`, so `∂ΔC/∂ρ_sparse` is the remainder itself.
        if operator.is_exact_a()
            && matches!(self.assignment.mode, AssignmentMode::ThresholdGate { .. })
        {
            hdiag += &crate::assignment::threshold_gate_negative_hessian_remainder_weighted(
                &self.assignment,
                rho,
                self.row_loss_weights.as_deref(),
            )?;
        }
        // #2915 — a clamp-basin price moves with the clamp itself, and the
        // Daleckii–Krein map does not see that motion.
        let clamp = if operator.is_exact_a() {
            Some(self.materialize_ard_concave_clamp_diagonal_for_rows(rho, &cache.row_dims)?)
        } else {
            None
        };
        // #2915 — so does a reduced-Schur clamp-basin price, which the selected
        // inverse does not see either.
        // The border clamp is the decoder priors' remainder at the unit penalty scale of
        // the full evidence system this lane factors.
        let border_remainder = if clamp.is_some() && cache.beta_schur_conditioning.is_some() {
            self.decoder_prior_border_remainder_op(cache.k, 1.0)?
        } else {
            None
        };
        let beta_price = match clamp.as_ref() {
            Some(clamp) => Self::beta_schur_clamp_basin_price_weights(
                cache,
                clamp.view(),
                border_remainder.as_ref().map(|op| op as &dyn BetaPenaltyOp),
            )?,
            None => None,
        };
        let assignment_dim = self.assignment.assignment_coord_dim();
        let row_loss_weights = self.row_loss_weights.as_deref();
        let mut trace = 0.0_f64;
        for row in 0..self.n_obs() {
            let q = cache.row_dims[row];
            // The DEFLATED row-block selected inverse from the shared bundle
            // (#2712). The `t–β` block is not contracted here, so it is not built.
            let (mut inv_vv, _) = row_selected_inverse_from_probes(
                cache,
                row,
                probes,
                sinv_probes,
                false,
                "assignment_log_strength_hessian_trace_from_probes",
            )?;
            if let Some(weights) = beta_price.as_ref() {
                inv_vv += &weights[row].0;
            }
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
            let price = clamp.as_ref().and_then(|clamp| {
                let base = cache.row_offsets[row];
                Self::clamp_basin_price_weights(&inv_vv, clamp.slice(s![base..base + q]), spectrum)
            });

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
                                // `∂B/∂ρ_sparse` is DIAGONAL. It takes the one
                                // Daleckii–Krein correction every family takes
                                // (#2916). The kept-subspace diagonal it used to
                                // contract equals that correction only when no
                                // deflated direction couples to the border, since
                                // `vᵢᵀ inv_vv vᵢ = 1 + (H_βt vᵢ)ᵀ S⁻¹ (H_βt vᵢ)`.
                                let mut d_mat = Array2::<f64>::zeros((q, q));
                                for atom in 0..logit_dim {
                                    d_mat[[atom, atom]] = block[[atom, atom]];
                                    trace += inverse_diagonal[atom] * block[[atom, atom]];
                                }
                                if Self::row_deflation_is_live(dirs, spectrum) {
                                    trace -= Self::deflation_block_correction(
                                        &inv_vv, &d_mat, dirs, spectrum,
                                    );
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
                                if Self::row_deflation_is_live(dirs, spectrum) {
                                    trace -= Self::deflation_block_correction(
                                        &inv_vv, &d_mat, dirs, spectrum,
                                    );
                                }
                                // #2915 — softmax logits carry no clamp, so only the
                                // basin prices' response to `dA` is added.
                                if let Some((_, response)) = price.as_ref() {
                                    trace += (response * &d_mat).sum();
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
                if Self::row_deflation_is_live(dirs, spectrum) {
                    // Same Daleckii–Krein correction the dense sibling subtracts,
                    // against the same deflated `inv_vv` (#2712).
                    let mut d_mat = Array2::<f64>::zeros((q, q));
                    for slot in 0..q {
                        d_mat[[slot, slot]] = d_diag[slot];
                    }
                    trace -= Self::deflation_block_correction(&inv_vv, &d_mat, dirs, spectrum);
                }
                if let (Some((explicit, response)), Some(clamp)) = (price.as_ref(), clamp.as_ref())
                {
                    // `∂E/∂ρ_sparse` is the gate's clamp on its logit slots, degree one
                    // in `λ_sparse`; a coordinate slot's ARD clamp does not move with it.
                    // Every other family writes no clamp on its logits.
                    let base = cache.row_offsets[row];
                    let logit_slots = match self.last_row_layout {
                        Some(ref layout) => layout.active_atoms[row].len().min(q),
                        None => assignment_dim.min(q),
                    };
                    for slot in 0..logit_slots {
                        trace += explicit[slot] * clamp[base + slot];
                    }
                    for slot in 0..q {
                        trace += response[[slot, slot]] * d_diag[slot];
                    }
                }
                if let (Some(weights), Some(clamp)) = (beta_price.as_ref(), clamp.as_ref()) {
                    // The reduced-Schur basin prices read the same logit clamp.
                    let base = cache.row_offsets[row];
                    let logit_slots = match self.last_row_layout {
                        Some(ref layout) => layout.active_atoms[row].len().min(q),
                        None => assignment_dim.min(q),
                    };
                    for slot in 0..logit_slots {
                        trace += weights[row].1[slot] * clamp[base + slot];
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

    /// Whether a row's evidence factor installed a deflation map that every
    /// rho/theta trace must differentiate (#2333/#2515/#2336): a recorded
    /// spectrum, or a gauge-deflated direction. A clamp-basin classification
    /// reprices the spectrum without unit-deflating any direction, so an empty
    /// direction list is not evidence that the row's operator is the raw block.
    /// Every `deflation_block_correction` consumer gates on this one predicate.
    pub(crate) fn row_deflation_is_live(
        dirs: &[Array1<f64>],
        spectrum: Option<&RowDeflationSpectrum>,
    ) -> bool {
        spectrum.is_some() || !dirs.is_empty()
    }

    /// Fold the row's Daleckii–Krein deflation differential into a single
    /// t–t weight (#2333), as `evidence_metric_raw_weight` consumes it.
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
    /// `E_tt` fold [`Self::deflation_folded_trace_weight`].
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

    /// #2915 — contraction weights for the part of a row's clamp-basin prices
    /// that the Daleckii–Krein map does not differentiate.
    ///
    /// A clamp-basin direction `v_a` of the raw exact-A row block (raw `λ_a < 0`,
    /// conditioning `Raw`) is priced at `λ̃_a = λ_a + c_a` with `c_a = v_aᵀ E v_a`
    /// (`classify_exact_a_direction`). `DΦ[dA]` moves `λ̃_a` only through `λ_a`, so
    /// an exact-operator trace on the row is short by `Σ_a g_aa·dc_a`, where
    /// `g_aa = (UᵀGU)_aa` and
    /// `dc_a = v_aᵀ dE v_a + 2 Σ_{b≠a} (UᵀEU)_ab (UᵀdAU)_ab / (λ_a − λ_b)`.
    ///
    /// Returns `(explicit, response)`: `explicit[s] = Σ_a g_aa·v_a[s]²` contracts a
    /// diagonal `dE`, and `response = U W Uᵀ` contracts the raw derivative `dA`.
    /// A near-degenerate pair takes the gap convention of
    /// [`Self::row_deflation_frechet_coefficients`]. `None` when the row prices no
    /// clamp basin.
    pub(crate) fn clamp_basin_price_weights(
        inv_vv: &Array2<f64>,
        clamp_row: ArrayView1<'_, f64>,
        spectrum: Option<&RowDeflationSpectrum>,
    ) -> Option<(Array1<f64>, Array2<f64>)> {
        let spec = spectrum?;
        let q = inv_vv.nrows();
        let u = &spec.evecs;
        if u.nrows() != q || u.ncols() != q || clamp_row.len() != q {
            return None;
        }
        let raw = &spec.raw_evals;
        let basins: Vec<usize> = (0..q)
            .filter(|&a| spec.conditioning[a] == RowSpectralConditioning::Raw && raw[a] < 0.0)
            .collect();
        if basins.is_empty() {
            return None;
        }
        let eigen_scale = raw
            .iter()
            .chain(spec.cond_evals.iter())
            .copied()
            .fold(0.0_f64, |scale, value| scale.max(value.abs()));
        let gap_threshold = eigen_gap_threshold(eigen_scale, raw.len());
        let g = u.t().dot(inv_vv).dot(u);
        let clamp = Array2::from_diag(&clamp_row.to_owned());
        let e_rot = u.t().dot(&clamp).dot(u);
        let mut explicit = Array1::<f64>::zeros(q);
        let mut w = Array2::<f64>::zeros((q, q));
        for &a in &basins {
            let g_aa = g[[a, a]];
            for s in 0..q {
                explicit[s] += g_aa * u[[s, a]] * u[[s, a]];
            }
            for b in 0..q {
                let denom = raw[a] - raw[b];
                if b == a || denom.abs() <= gap_threshold {
                    continue;
                }
                let weight = g_aa * e_rot[[a, b]] / denom;
                w[[a, b]] += weight;
                w[[b, a]] += weight;
            }
        }
        Some((explicit, u.dot(&w).dot(&u.t())))
    }

    /// #2915 — contraction weights for the reduced-Schur clamp-basin prices, which
    /// the selected inverse does not differentiate.
    ///
    /// The evidence factor classifies each eigendirection `w` of the raw reduced
    /// Schur `S = H_ββ − Σ_i H_βt Φ_i⁻¹ H_tβ` (`Φ_i` the conditioned row blocks) and
    /// prices a clamp basin at `λ̃_w = λ_w + c_w`, `c_w = wᵀ C w`,
    /// `C = Σ_i G_iᵀ diag(E_i) G_i`, `G_i = −Φ_i⁻¹ H_tβ^(i)`
    /// (`exact_a_reduced_classification`). Contracting the selected inverse against
    /// `dΦ_i` moves `λ̃_w` only through `λ_w`, so a trace is short by `Σ_w dc_w/λ̃_w`.
    /// For a ρ that moves neither `H_tβ` nor `H_ββ`, `dS = Σ_i G_iᵀ dΦ_i G_i`,
    /// `dG_i = −Φ_i⁻¹ dΦ_i G_i` and
    /// `dc_w = wᵀ dC w + 2 Σ_{v≠w} (QᵀCQ)_wv (QᵀdSQ)_vw / (λ_w − λ_v)`.
    ///
    /// Returns, per row, `(weight, explicit)`. `weight = (G_iQ) R (G_iQ)ᵀ −
    /// (Φ_i⁻¹E_iN_i + N_iE_iΦ_i⁻¹)` adds to the row's selected inverse wherever it
    /// contracts `dΦ_i`, with `R_wv = R_vw` accumulating `(QᵀCQ)_wv / (λ̃_w (λ_w − λ_v))`
    /// over basins `w` and `N_i = Σ_w (G_i w)(G_i w)ᵀ / λ̃_w`. `explicit[s] = N_i[s, s]`
    /// contracts a diagonal `dE_i`. A near-degenerate pair takes the gap convention
    /// of [`Self::row_deflation_frechet_coefficients`]. The work is `O(n·q·K²)`, the
    /// order of the classification's own clamp metric. `None` when the factor
    /// records no reduced-Schur clamp basin.
    pub(crate) fn beta_schur_clamp_basin_price_weights(
        cache: &ArrowFactorCache,
        clamp: ArrayView1<'_, f64>,
        border_remainder: Option<&dyn BetaPenaltyOp>,
    ) -> Result<Option<Vec<(Array2<f64>, Array1<f64>)>>, String> {
        Ok(Self::beta_schur_clamp_basin_operands(cache, clamp, border_remainder)?
            .map(|operands| Self::beta_schur_basin_row_weights(cache, clamp, &operands)))
    }

    /// The operands every reduced-Schur clamp-basin weight is built from: `G_i Q`
    /// per row, `R` in the reduced-Schur eigenbasis, and each basin's eigen-index
    /// with `1/λ̃_w`. Since cf712b006 the classification's clamp metric also carries
    /// the decoder priors' border clamp `E_ββ = −remainder` (#2828), so
    /// `C = Σ_i G_iᵀ diag(E_i) G_i + E_ββ`. `border_remainder` is that remainder at the
    /// penalty scale the evidence system was assembled with. `None` when the factor
    /// records no reduced-Schur clamp basin.
    fn beta_schur_clamp_basin_operands(
        cache: &ArrowFactorCache,
        clamp: ArrayView1<'_, f64>,
        border_remainder: Option<&dyn BetaPenaltyOp>,
    ) -> Result<Option<(Vec<Array2<f64>>, Array2<f64>, Vec<(usize, f64)>)>, String> {
        let Some(spec) = cache.beta_schur_conditioning.as_ref() else {
            return Ok(None);
        };
        let k = cache.k;
        if spec.evecs.dim() != (k, k)
            || spec.raw_evals.len() != k
            || spec.cond_evals.len() != k
            || spec.conditioning.len() != k
        {
            return Err(format!(
                "beta_schur_clamp_basin_operands: the recorded reduced-Schur spectrum does not \
                 match border width {k}"
            ));
        }
        let basins: Vec<(usize, f64)> = (0..k)
            .filter(|&w| {
                spec.conditioning[w]
                    == gam_solve::arrow_schur::BetaSchurSpectralConditioning::ClampBasin
            })
            .map(|w| (w, 1.0 / spec.cond_evals[w]))
            .collect();
        if basins.is_empty() {
            return Ok(None);
        }
        if clamp.len() != cache.delta_t_len() {
            return Err(format!(
                "beta_schur_clamp_basin_operands: the clamp diagonal has {} entries for {} latent \
                 coordinates",
                clamp.len(),
                cache.delta_t_len()
            ));
        }
        let basis = &spec.evecs;
        let raw = &spec.raw_evals;
        let eigen_scale = raw
            .iter()
            .chain(spec.cond_evals.iter())
            .copied()
            .fold(0.0_f64, |scale, value| scale.max(value.abs()));
        let gap_threshold = eigen_gap_threshold(eigen_scale, k);
        let n_rows = cache.row_dims.len();
        // `G_i Q`, one q×K block per row.
        let mut graphs: Vec<Array2<f64>> = Vec::with_capacity(n_rows);
        for row in 0..n_rows {
            let q = cache.row_dims[row];
            let factor = cache.undamped_factor(row);
            let mut graph = Array2::<f64>::zeros((q, k));
            let mut coupled = Array1::<f64>::zeros(q);
            for v in 0..k {
                coupled.fill(0.0);
                if !cache.apply_htbeta_row(row, basis.column(v), &mut coupled) {
                    return Err(format!(
                        "beta_schur_clamp_basin_operands: H_tβ^({row}) apply failed"
                    ));
                }
                let solved = cholesky_solve_vector(factor, coupled.view());
                for s in 0..q {
                    graph[[s, v]] = -solved[s];
                }
            }
            graphs.push(graph);
        }
        // `(QᵀCQ)_wv` for every basin `w`.
        let mut clamp_rotated = Array2::<f64>::zeros((basins.len(), k));
        for (row, graph) in graphs.iter().enumerate() {
            let base = cache.row_offsets[row];
            for (index, &(w, _)) in basins.iter().enumerate() {
                for v in 0..k {
                    let mut acc = 0.0_f64;
                    for s in 0..graph.nrows() {
                        acc += graph[[s, w]] * clamp[base + s] * graph[[s, v]];
                    }
                    clamp_rotated[[index, v]] += acc;
                }
            }
        }
        // `E_ββ = −remainder`, the decoder priors' border clamp, joins `C`.
        if let Some(remainder) = border_remainder {
            if remainder.dim() != k {
                return Err(format!(
                    "beta_schur_clamp_basin_operands: the border remainder has width {} for \
                     border width {k}",
                    remainder.dim()
                ));
            }
            let mut applied = vec![0.0_f64; k];
            for (index, &(w, _)) in basins.iter().enumerate() {
                applied.fill(0.0);
                let direction = basis.column(w).to_owned();
                remainder.matvec(
                    direction
                        .as_slice()
                        .expect("an owned eigenvector is contiguous"),
                    &mut applied,
                );
                for v in 0..k {
                    let mut acc = 0.0_f64;
                    for b in 0..k {
                        acc += basis[[b, v]] * applied[b];
                    }
                    clamp_rotated[[index, v]] -= acc;
                }
            }
        }
        let mut response = Array2::<f64>::zeros((k, k));
        for (index, &(w, inv_price)) in basins.iter().enumerate() {
            for v in 0..k {
                let denom = raw[w] - raw[v];
                if v == w || denom.abs() <= gap_threshold {
                    continue;
                }
                let weight = clamp_rotated[[index, v]] * inv_price / denom;
                response[[w, v]] += weight;
                response[[v, w]] += weight;
            }
        }
        Ok(Some((graphs, response, basins)))
    }

    /// Per row `(weight, explicit)` from [`Self::beta_schur_clamp_basin_operands`],
    /// as documented on [`Self::beta_schur_clamp_basin_price_weights`].
    fn beta_schur_basin_row_weights(
        cache: &ArrowFactorCache,
        clamp: ArrayView1<'_, f64>,
        operands: &(Vec<Array2<f64>>, Array2<f64>, Vec<(usize, f64)>),
    ) -> Vec<(Array2<f64>, Array1<f64>)> {
        let (graphs, response, basins) = operands;
        let mut out = Vec::with_capacity(graphs.len());
        for (row, graph) in graphs.iter().enumerate() {
            let q = cache.row_dims[row];
            let base = cache.row_offsets[row];
            let factor = cache.undamped_factor(row);
            let mut weight = graph.dot(response).dot(&graph.t());
            let mut basin_gram = Array2::<f64>::zeros((q, q));
            for &(w, inv_price) in basins {
                for a in 0..q {
                    for b in 0..q {
                        basin_gram[[a, b]] += inv_price * graph[[a, w]] * graph[[b, w]];
                    }
                }
            }
            // `Φ_i⁻¹ E_i N_i`, one Cholesky solve per column.
            let mut clamped_solve = Array2::<f64>::zeros((q, q));
            let mut rhs = Array1::<f64>::zeros(q);
            for col in 0..q {
                for s in 0..q {
                    rhs[s] = clamp[base + s] * basin_gram[[s, col]];
                }
                let solved = cholesky_solve_vector(factor, rhs.view());
                for s in 0..q {
                    clamped_solve[[s, col]] = solved[s];
                }
            }
            weight -= &clamped_solve;
            weight -= &clamped_solve.t();
            out.push((weight, basin_gram.diag().to_owned()));
        }
        out
    }

    /// #2915 — the θ-adjoint's border weights for the reduced-Schur clamp-basin
    /// prices, from [`Self::beta_schur_clamp_basin_operands`].
    ///
    /// A θ also moves `H_tβ^(i)` and `H_ββ`, so `dS` gains
    /// `dH_ββ + dH_βt,i G_i + G_iᵀ dH_tβ,i` and `dG_i` gains `−Φ_i⁻¹ dH_tβ,i`. Returns
    /// per row `X_i = G_i Q R Qᵀ − Φ_i⁻¹ E_i G_i Ω` with `Ω = Σ_w w wᵀ / λ̃_w`, which
    /// adds to the row's `(H⁻¹)_tβ` block wherever it contracts `dH_tβ,i`, `Q R Qᵀ`,
    /// which contracts a direct `dH_ββ`, and `Ω`, which contracts the border clamp's
    /// own `∂E_ββ/∂β`.
    fn beta_schur_basin_border_weights(
        cache: &ArrowFactorCache,
        clamp: ArrayView1<'_, f64>,
        operands: &(Vec<Array2<f64>>, Array2<f64>, Vec<(usize, f64)>),
    ) -> Result<(Vec<Array2<f64>>, Array2<f64>, Array2<f64>), String> {
        let Some(spec) = cache.beta_schur_conditioning.as_ref() else {
            return Err(
                "beta_schur_basin_border_weights: the factor records no reduced-Schur spectrum"
                    .to_string(),
            );
        };
        let (graphs, response, basins) = operands;
        let basis = &spec.evecs;
        let mut borders = Vec::with_capacity(graphs.len());
        for (row, graph) in graphs.iter().enumerate() {
            let q = cache.row_dims[row];
            let base = cache.row_offsets[row];
            let factor = cache.undamped_factor(row);
            let mut rotated = graph.dot(response);
            let mut rhs = Array1::<f64>::zeros(q);
            for &(w, inv_price) in basins {
                for s in 0..q {
                    rhs[s] = clamp[base + s] * graph[[s, w]];
                }
                let solved = cholesky_solve_vector(factor, rhs.view());
                for s in 0..q {
                    rotated[[s, w]] -= inv_price * solved[s];
                }
            }
            borders.push(rotated.dot(&basis.t()));
        }
        let k = basis.nrows();
        let mut omega = Array2::<f64>::zeros((k, k));
        for &(w, inv_price) in basins {
            for a in 0..k {
                for b in 0..k {
                    omega[[a, b]] += inv_price * basis[[a, w]] * basis[[b, w]];
                }
            }
        }
        Ok((borders, basis.dot(response).dot(&basis.t()), omega))
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

    /// #2330 Patch D — raw basis THIRD jets `∂³φ` per atom, for the exact-A
    /// θ-adjoint's residual-curvature leg `⟨error_metric, ∂³f⟩` on the dense and
    /// from-probes routes. Each atom's evaluator must supply its analytic jet,
    /// shaped `(n_obs, basis, d, d, d)`, or certify that every third partial
    /// vanishes. An atom whose jet is unavailable, including one with no
    /// evaluator, is refused with [`ThirdJetUnavailable`] (#2933 F02): an absent
    /// derivative is not a zero one, and omitting its leg would return an
    /// incomplete gradient as exact.
    pub(crate) fn atom_third_jets(&self) -> Result<Vec<AtomThirdJet>, String> {
        let mut out = Vec::with_capacity(self.k_atoms());
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let unavailable = || {
                ThirdJetUnavailable {
                    atom: atom.name.clone(),
                }
                .to_string()
            };
            let evaluator = atom.basis_evaluator.as_ref().ok_or_else(unavailable)?;
            let coords = self.assignment.coords[atom_idx].as_matrix();
            let jet = match evaluator.third_jet_dyn(coords.view())? {
                SaeBasisThirdJetCapability::Analytic(jet) => {
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
                    AtomThirdJet::Analytic(jet)
                }
                SaeBasisThirdJetCapability::CertifiedZero => AtomThirdJet::CertifiedZero,
                SaeBasisThirdJetCapability::Unavailable => return Err(unavailable()),
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
        // #2520 - WHICH operator's diagonal is being differentiated. `B` carries
        // the PSD clamp of the ThresholdGate's signed logit curvature and `A`
        // carries the raw signed value, so the two have different theta-adjoints
        // on the concave half. This is the same split the sibling ARD pair
        // (`ard_majorized_hessian_derivative` / `ard_exact_hessian_derivative`)
        // already carries, on the one prior family that never received it.
        exact_a: bool,
    ) -> f64 {
        let SaeLocalRowVar::Logit { atom: wrt_atom } = wrt else {
            return 0.0;
        };
        // #Bug4: under TopK or frozen routing every logit is FIXED and its assembled `htt`
        // diagonal entry is ZEROED (see `assignment_prior_grad_hdiag_weighted`), so the
        // θ-adjoint third derivative of that zeroed entry must also be zero. Mirror the ordered
        // Beta--Bernoulli channel zeroing in
        // `ordered_beta_bernoulli_psd_majorizer_third_channels_weighted`.
        if self.assignment.logits_are_fixed() {
            return 0.0;
        }
        // #2080 — the gate prior's logit Jacobian adds the exact curvature `2z(1 − z)/τ²`
        // to the logit diagonal of both `B` and `A`, so both adjoints differentiate it the
        // same way. The prior's own channels follow.
        let gate_jacobian_third = if diag_atom == wrt_atom {
            crate::assignment::gate_logit_jacobian_third_weighted(
                &self.assignment,
                self.row_loss_weights.as_deref(),
                row,
                diag_atom,
            )
        } else {
            0.0
        };
        gate_jacobian_third + match self.assignment.mode {
            AssignmentMode::Softmax { .. } => {
                // #1038: the softmax entropy Hessian is now stored DENSE in
                // `block.htt` and its full θ-derivative `∂H_{k,j}/∂z_w` (diagonal
                // AND off-diagonal) is added inline in `logdet_theta_adjoint`. Returning the
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
                // #991 - this row's ThresholdGate prior curvature in `htt` carries
                // the design weight `w_row`, so its theta-derivative carries the
                // SAME `w_row` (value/logdet/adjoint stay on one weighted branch).
                let w_row = self.row_loss_weights.as_deref().map_or(1.0, |w| w[row]);
                // #1415 gave `P''(l) = (lambda/tau^2) s (1-2a)` the third
                // derivative `P'''(l) = (lambda/tau^3) s (1-6a+6a^2)`, and that is
                // still exactly right for `A`. #2520 then made `B` install the PSD
                // clamp of `P''` instead, so on the concave half (`1-2a < 0`,
                // every logit the gate has switched ON) the majorizer is a hard
                // `0` and its theta-adjoint is `0` too. Reading both derivatives
                // off the SAME seam the value path reads is what stops the two
                // from drifting apart again.
                let curvature = crate::assignment::ThresholdGateLogitCurvature::eval(
                    w_row * threshold_strength,
                    self.assignment.logits[[row, diag_atom]],
                    threshold,
                    1.0 / temperature,
                );
                if exact_a {
                    curvature.exact_hess_logit_derivative()
                } else {
                    curvature.majorized_hess_logit_derivative()
                }
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
            // Unreachable in practice: `logits_are_fixed` holds under TopK, so
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
        let periods = self.ard_axis_periods(atom);
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
        let periods = self.ard_axis_periods(atom);
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

    /// `λ·(½(P+Pᵀ) ⊗ I) C_k` written into atom `k`'s β-block of `beta`, in the cache's
    /// border layout: the IFT right-hand side of a coordinate that scales
    /// (`P = S_k`) or reshapes (`P = ∂S_k/∂κ`) atom `k`'s penalty Gram (#1556, #2935).
    fn decoder_penalty_ift_rhs_block(
        &self,
        cache: &ArrowFactorCache,
        target_atom: usize,
        lambda: f64,
        penalty: &Array2<f64>,
        beta: &mut Array1<f64>,
    ) -> Result<(), String> {
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
                    let p_sym = 0.5 * (penalty[[mu, nu]] + penalty[[nu, mu]]);
                    acc += p_sym * coeffs[[nu, channel]];
                }
                beta[off + mu * r + channel] = lambda * acc;
            }
        }
        Ok(())
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
    /// SCOPE — with a `residual_target`, the #2330 Patch-D residual
    /// THIRD-derivative legs (`⟨error_metric, ∂³f⟩`) are carried through the same
    /// `patchd_residual_third_leg` / `patchd_residual_third_leg_beta` the dense
    /// route calls, softmax cross-atom gate terms included (#2933 F01); with
    /// `None` both routes skip them. `OrderedBetaBernoulli`'s cross-row adjoint is
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
        // #2915 — on the exact operator a clamp-basin price moves with the clamp
        // itself (`E` and its θ-diagonal), which the Daleckii–Krein map does not see.
        let clamp_price_inputs = if exact_a {
            Some((
                self.materialize_ard_concave_clamp_diagonal_for_rows(rho, &cache.row_dims)?,
                self.ard_concave_clamp_dt_diagonal(rho, cache)?,
            ))
        } else {
            None
        };
        // #2915 — so does a reduced-Schur clamp-basin price, which also moves with the
        // border and `H_ββ`; the selected inverse sees none of that motion.
        // The border clamp is the decoder priors' remainder at the unit penalty scale of
        // the full evidence system this lane factors.
        let border_remainder = if exact_a && cache.beta_schur_conditioning.is_some() {
            self.decoder_prior_border_remainder_op(cache.k, 1.0)?
        } else {
            None
        };
        let beta_basin = match clamp_price_inputs.as_ref() {
            Some((clamp, _)) => {
                match Self::beta_schur_clamp_basin_operands(
                    cache,
                    clamp.view(),
                    border_remainder.as_ref().map(|op| op as &dyn BetaPenaltyOp),
                )? {
                    Some(operands) => Some((
                        Self::beta_schur_basin_row_weights(cache, clamp.view(), &operands),
                        Self::beta_schur_basin_border_weights(cache, clamp.view(), &operands)?,
                    )),
                    None => None,
                }
            }
            None => None,
        };
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
        let patchd_gate = PatchDGate::for_mode(&self.assignment.mode);
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
        let mut assignments = Array1::<f64>::zeros(self.k_atoms());
        let mut jet_window: std::collections::VecDeque<SaeRowJets> =
            std::collections::VecDeque::new();
        let mut jet_window_next = 0usize;
        let mut ordered_beta_bernoulli_logit_sites: Vec<(usize, usize, usize, f64)> = Vec::new();
        // #2933 F24 — rows holding an embedded-sphere block factor the Riemannian
        // conversion of their ambient row, which moves with θ too (`SphereRowConversion`).
        let sphere_blocks = self.sphere_tangent_blocks_by_row(&cache.row_dims)?;
        let sphere_axis_periods = self.all_ard_axis_periods();

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
            let (mut inv_vv, mut inv_vbeta) = row_selected_inverse_from_probes(
                cache,
                row,
                probes,
                sinv_probes,
                true,
                "logdet_theta_adjoint_from_probes",
            )?;
            if let Some((row_weights, (border_weights, _, _))) = beta_basin.as_ref() {
                inv_vv += &row_weights[row].0;
                inv_vbeta += &border_weights[row];
            }

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
            let defl_live = Self::row_deflation_is_live(defl_dirs, defl_spectrum);
            let clamp_price = clamp_price_inputs.as_ref().and_then(|(clamp, _)| {
                Self::clamp_basin_price_weights(
                    &inv_vv,
                    clamp.slice(s![base..base + q]),
                    defl_spectrum,
                )
            });

            // #2330 Patch D per-row residual context (#2515 port). `w_row_prior`
            // is bound below for the majorizer legs; the residual weighting is the
            // row's own design weight, read here through the one authority both
            // routes share.
            let patchd_w_row = self.row_loss_weights.as_deref().map_or(1.0, |w| w[row]);
            let patchd_error_metric: Option<Vec<f64>> = patchd_residual.map(|tgt| {
                self.patchd_row_error_metric(row, patchd_w_row, tgt, &assignments, whiten_row_jets)
            });
            let patchd_sqrt_w = patchd_w_row.sqrt();
            let patchd_reconstruction: Option<PatchDRowContractions> = patchd_error_metric
                .as_deref()
                .zip(patchd_third_jets.as_deref())
                .map(|(em, third_jets)| {
                    self.patchd_row_contractions(row, em, &assignments, &second_jets, third_jets)
                });
            let patchd_ctx: Option<PatchDResidualCtx<'_>> = patchd_error_metric
                .as_deref()
                .zip(patchd_reconstruction.as_ref())
                .map(|(em, reconstruction)| PatchDResidualCtx {
                    row,
                    error_metric: em,
                    sqrt_w: patchd_sqrt_w,
                    assignments: &assignments,
                    second_jets: &second_jets,
                    gate: patchd_gate,
                    reconstruction,
                });

            if ordered_beta_bernoulli_channels.is_some() {
                for (position, variable) in jets.vars.iter().enumerate() {
                    if let SaeLocalRowVar::Logit { atom } = *variable {
                        // Same per-slot Daleckii–Krein weight for a unit diagonal
                        // derivative the dense route records, so the shared-mass column
                        // pass below differentiates the same conditioned majorizer.
                        let diag_deflation_weight = if !defl_live {
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
                            inv_vv[[position, position]] - diag_deflation_weight
                                + clamp_price
                                    .as_ref()
                                    .map_or(0.0, |(_, response)| response[[position, position]]),
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
            // The direct `H_ββ` leg of the reduced-Schur basin prices contracts `QRQᵀ`
            // against `∂H_ββ/∂θ_w = Σ_c (∂b_c/∂θ_w b_cᵀ + b_c ∂b_cᵀ/∂θ_w)`, so the row's
            // border jets `b_c` are folded through `QRQᵀ` once.
            let beta_fold = match beta_basin.as_ref() {
                Some((_, (_, beta_weight, _))) if bjet_len > 0 => {
                    let mut coefficients = Array2::<f64>::zeros((bjet_len, k_border));
                    for (beta_pos, channel) in border.iter().enumerate() {
                        let bj = jets.beta(beta_pos);
                        for c in 0..bjet_len {
                            coefficients[[c, channel.index]] += bj[c];
                        }
                    }
                    Some(coefficients.dot(beta_weight))
                }
                _ => None,
            };

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
            let simplex_count = crate::assignment::simplex_gate_free_count(&self.assignment);
            // #2933 F24 — on a sphere row the tower keeps its ambient derivatives as
            // matrices, converts them (`SphereRowConversion`), contracts the converted
            // block, and projects the slot functional to the tangent after the loop.
            let sphere_conversion = if sphere_blocks[row].is_empty() {
                None
            } else {
                let target = residual_target.ok_or_else(|| {
                    format!(
                        "logdet_theta_adjoint_from_probes: row {row} holds an embedded-sphere \
                         block, whose Riemannian conversion reads the row residual, but no \
                         target was supplied"
                    )
                })?;
                let error_metric = self.patchd_row_error_metric(
                    row,
                    patchd_w_row,
                    target,
                    &assignments,
                    whiten_row_jets,
                );
                SphereRowConversion::for_row(
                    self,
                    row,
                    &sphere_blocks[row],
                    &jets,
                    border.len(),
                    &error_metric,
                    &ard_precisions,
                    &sphere_axis_periods,
                    exact_a,
                )
            };
            let collect_matrices = defl_live || sphere_conversion.is_some();
            let contract_converted = |d_tt: &Array2<f64>, d_tbeta: &Array2<f64>| -> f64 {
                let mut converted = 0.0_f64;
                for a in 0..q {
                    for b in 0..q {
                        converted += inv_vv[[b, a]] * d_tt[[a, b]];
                    }
                }
                if defl_live {
                    converted -= Self::deflation_block_correction(
                        &inv_vv,
                        d_tt,
                        defl_dirs,
                        defl_spectrum,
                    );
                }
                for a in 0..q {
                    for (beta_pos, channel) in border.iter().enumerate() {
                        converted +=
                            2.0 * inv_vbeta[[a, channel.index]] * d_tbeta[[a, beta_pos]];
                    }
                }
                converted
            };
            let mut sphere_slot_functional =
                Array1::<f64>::zeros(if sphere_conversion.is_some() { q } else { 0 });
            for w in 0..q {
                let mut gamma = 0.0_f64;
                let softmax_d_dw: Option<(&[f64], f64, f64, f64, usize)> =
                    match (softmax_adjoint_row, jets.vars[w]) {
                        (Some((a, mm, scale, inv_tau)), SaeLocalRowVar::Logit { atom: atom_w }) => {
                            Some((a, mm, scale, inv_tau, atom_w))
                        }
                        _ => None,
                    };
                // The exact operator's entropy block moves with the logit through the
                // dense entropy third derivative; one O(K) setup per logit variable.
                let softmax_entropy_derivative = match softmax_d_dw {
                    Some((a_soft, mm, scale, inv_tau, atom_w)) if exact_a => Some(
                        SoftmaxEntropyDerivative::new(a_soft, atom_w, mm, scale, inv_tau),
                    ),
                    _ => None,
                };
                // t–t block: reuse the dense contraction. On a deflated row the raw
                // per-slot derivative is retained as a matrix so the Daleckii–Krein
                // correction can be applied to it after the loop; on a PD row the
                // matrix stays `0×0` and nothing is allocated.
                let mut deflated_base_dh_mat = if !collect_matrices {
                    Array2::<f64>::zeros((0, 0))
                } else {
                    Array2::<f64>::zeros((q, q))
                };
                let mut dh_border = if sphere_conversion.is_some() {
                    Array2::<f64>::zeros((q, border.len()))
                } else {
                    Array2::<f64>::zeros((0, 0))
                };
                let mut beta_beta = 0.0_f64;
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
                            if exact_a {
                                // #2333 — `A` carries the exact dense entropy Hessian on
                                // the logit block: `B`'s Gershgorin majorizer `D̃` plus the
                                // ΔC remainder `h_entropy − D̃` on the diagonal and
                                // `h_entropy` off it. Its logit derivative is the dense
                                // entropy third derivative, off-diagonal pairs included.
                                if let Some(derivative) = softmax_entropy_derivative.as_ref() {
                                    dh += w_row_prior * derivative.entry(atom_a, atom_b).1;
                                }
                            } else if atom_a == atom_b {
                                dh += w_row_prior
                                    * active_softmax_majorizer_logit_derivative_entry(
                                        a_soft, atom_a, _atom_w, mm, scale, inv_tau,
                                    );
                            }
                            // #2080 — the softmax row's logit Jacobian has the exact dense
                            // curvature `c·(diag z − zzᵀ)/τ²` in both `B` and `A`.
                            if let Some(count) = simplex_count {
                                if !self.assignment.logits_are_fixed() {
                                    dh += w_row_prior
                                        * crate::assignment::simplex_gate_logit_jacobian_third(
                                            a_soft, atom_a, atom_b, _atom_w, count, inv_tau,
                                        );
                                }
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
                                        exact_a,
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
                        if collect_matrices {
                            deflated_base_dh_mat[[a, b]] = dh;
                        }
                        gamma += inv_vv[[b, a]] * dh;
                    }
                }
                if defl_live {
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
                if let (Some((explicit, response)), Some((_, clamp_dt))) =
                    (clamp_price.as_ref(), clamp_price_inputs.as_ref())
                {
                    // #2915 — `∂E/∂θ_w` is diagonal, nonzero only at slot `w`.
                    gamma += explicit[w] * clamp_dt[base + w]
                        + (response * &deflated_base_dh_mat).sum();
                }
                if let (Some((row_weights, _)), Some((_, clamp_dt))) =
                    (beta_basin.as_ref(), clamp_price_inputs.as_ref())
                {
                    // The reduced-Schur basin prices read the same clamp θ-diagonal.
                    gamma += row_weights[row].1[w] * clamp_dt[base + w];
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
                        if sphere_conversion.is_some() {
                            dh_border[[a, beta_pos]] = dh;
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
                        let contribution =
                            inv_m * (sae_dot(&rd_l, &p_probe[l]) + sae_dot(&r_probe[l], &q_l));
                        gamma += contribution;
                        beta_beta += contribution;
                    }
                }
                if let Some(fold) = beta_fold.as_ref() {
                    for (beta_pos, channel) in border.iter().enumerate() {
                        let bd = jets.beta_deriv(w, beta_pos);
                        for c in 0..bjet_len {
                            let contribution = 2.0 * bd[c] * fold[[c, channel.index]];
                            gamma += contribution;
                            beta_beta += contribution;
                        }
                    }
                }
                match sphere_conversion.as_ref() {
                    None => gamma_t[base + w] = gamma,
                    Some(conversion) => {
                        let (d_tt, d_tbeta) =
                            conversion.slot_derivative(w, &deflated_base_dh_mat, &dh_border);
                        let mut converted = contract_converted(&d_tt, &d_tbeta) + beta_beta;
                        if let (Some((explicit, response)), Some((_, clamp_dt))) =
                            (clamp_price.as_ref(), clamp_price_inputs.as_ref())
                        {
                            converted += explicit[w] * clamp_dt[base + w] + (response * &d_tt).sum();
                        }
                        if let (Some((row_weights, _)), Some((_, clamp_dt))) =
                            (beta_basin.as_ref(), clamp_price_inputs.as_ref())
                        {
                            converted += row_weights[row].1[w] * clamp_dt[base + w];
                        }
                        sphere_slot_functional[w] = converted;
                    }
                }
            }
            if let Some(conversion) = sphere_conversion.as_ref() {
                let projected = conversion.project_slot_functional(&sphere_slot_functional);
                for w in 0..q {
                    gamma_t[base + w] = projected[w];
                }
            }

            for (w_beta_pos, w_channel) in border.iter().enumerate() {
                let mut gamma = 0.0_f64;
                let mut dh_mat = if !collect_matrices {
                    Array2::<f64>::zeros((0, 0))
                } else {
                    Array2::<f64>::zeros((q, q))
                };
                let mut dh_border = if sphere_conversion.is_some() {
                    Array2::<f64>::zeros((q, border.len()))
                } else {
                    Array2::<f64>::zeros((0, 0))
                };
                for a in 0..q {
                    for b in 0..q {
                        let mut dh = sae_dot(jets.beta_l_deriv(a, w_beta_pos), jets.first(b))
                            + sae_dot(jets.first(a), jets.beta_l_deriv(b, w_beta_pos));
                        if exact_a {
                            dh += sae_dot(jets.beta(w_beta_pos), jets.second(a, b));
                        }
                        if let Some(ctx) = patchd_ctx.as_ref() {
                            dh += self.patchd_residual_third_leg_beta(
                                ctx, jets.vars[a], jets.vars[b], w_channel,
                            );
                        }
                        if collect_matrices {
                            dh_mat[[a, b]] = dh;
                        }
                        gamma += inv_vv[[b, a]] * dh;
                    }
                }
                if defl_live {
                    // The border channels differentiate the same conditioned t–t block,
                    // so they carry the same Daleckii–Krein correction (#2712).
                    gamma -= Self::deflation_block_correction(
                        &inv_vv,
                        &dh_mat,
                        defl_dirs,
                        defl_spectrum,
                    );
                }
                if let Some((_, response)) = clamp_price.as_ref() {
                    // #2915 — the clamp does not depend on β, so a border variable
                    // moves a basin price only through `dA`.
                    gamma += (response * &dh_mat).sum();
                }
                for a in 0..q {
                    for (beta_pos, channel) in border.iter().enumerate() {
                        let mut dh = sae_dot(jets.beta_l_deriv(a, w_beta_pos), jets.beta(beta_pos));
                        if exact_a {
                            dh += sae_dot(jets.beta(w_beta_pos), jets.beta_deriv(a, beta_pos));
                        }
                        if sphere_conversion.is_some() {
                            dh_border[[a, beta_pos]] = dh;
                        }
                        gamma += 2.0 * inv_vbeta[[a, channel.index]] * dh;
                    }
                }
                match sphere_conversion.as_ref() {
                    None => gamma_beta[w_channel.index] += gamma,
                    Some(conversion) => {
                        let (d_tt, d_tbeta) =
                            conversion.border_derivative(w_beta_pos, &dh_mat, &dh_border);
                        let mut converted = contract_converted(&d_tt, &d_tbeta);
                        if let Some((_, response)) = clamp_price.as_ref() {
                            converted += (response * &d_tt).sum();
                        }
                        gamma_beta[w_channel.index] += converted;
                    }
                }
            }
        }

        if exact_a && k_border > 0 {
            let prepared = self.prepare_exact_decoder_prior_third()?;
            for (probe, solved) in probes.iter().zip(sinv_probes) {
                self.exact_decoder_prior_theta_pair_add(
                    cache, &prepared, solved.view(), probe.view(), inv_m, &mut gamma_beta,
                )?;
            }
            if let Some((_, (_, beta_weight, basin_omega))) = beta_basin.as_ref() {
                // The reduced-Schur basin prices contract `QRQᵀ` against the decoder
                // prior's own `∂H_ββ/∂β`.
                let mut unit = Array1::<f64>::zeros(k_border);
                for col in 0..k_border {
                    unit[col] = 1.0;
                    self.exact_decoder_prior_theta_pair_add(
                        cache,
                        &prepared,
                        beta_weight.column(col),
                        unit.view(),
                        1.0,
                        &mut gamma_beta,
                    )?;
                    unit[col] = 0.0;
                }
                if border_remainder.is_some() {
                    // A border clamp moves with β itself: `Ω` contracts `∂E_ββ/∂β`.
                    gamma_beta += &self.decoder_prior_gap_theta_trace(cache, basin_omega.view())?;
                }
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

    /// #2933 F33 — the information a shape covariance is assembled from: the
    /// border blocks of the pseudo-inverse of the exact observed information
    /// `A = ∇²_θθ L` at this converged state, on the identified space the
    /// exact-A value path prices.
    ///
    /// The criterion prices `½log|A|` from its own dense materialization but
    /// returns only the majorizer's factor cache, whose Schur inverse is
    /// `[B⁻¹]_ββ`. `geometry` is the same `A` formed by
    /// [`Self::materialize_exact_stationarity_geometry`], the one construction the
    /// IFT solve and the fitted-response divergence (#2933 F36) read, over the
    /// operator `exact_observed_information_log_dets_with_saddle_directions`
    /// classifies, at the state `cache` factors. This reads every atom's block off
    /// its eigensystem. On the production route
    /// ([`Self::shape_information_route`]) the dispersion's divergence reads the
    /// same geometry, so a report pays one dense build and one `O(dim³)`
    /// decomposition, on the route whose admission (`direct_logdet_admitted`)
    /// already prices that block for every criterion evaluation.
    ///
    /// A resolved negative direction of `A` returns
    /// [`SaeShapeCovarianceUnavailable::IndefiniteObservedInformation`]. The value
    /// path may still price such a basin through the majorizer's concave clamp,
    /// but a clamped operator is not an observed information, and inverting it
    /// would report a regularized band under a posterior label.
    ///
    /// The same `A⁺` is the bread of the row-sandwich companion
    /// [`SaeAtomShapeUncertainty::band_sd_robust`], whose meat is the outer
    /// product of the per-row data scores at `target` together with any
    /// aggregate-mass estimating equations ([`Self::row_sandwich_meat`]).
    ///
    /// On a framed state this holds every learned frame at its fitted `U_k`, so
    /// it is the route only where the frames cannot be integrated; see
    /// [`Self::shape_information`].
    pub(crate) fn exact_observed_information_shape_covariance(
        &self,
        geometry: &ExactHessianSpectralBlock,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
    ) -> Result<SaeShapeInformation, String> {
        let frame_conditioning = self.fixed_frame_conditioning()?;
        let total_t = cache.delta_t_len();
        if geometry.eigenvalues.len() != total_t + cache.k {
            return Err(format!(
                "exact observed-information shape covariance: geometry dimension {} is not the \
                 cache's t dimension {total_t} plus border {}",
                geometry.eigenvalues.len(),
                cache.k
            ));
        }
        let sandwich = self.row_sandwich_meat(rho, cache, target)?;
        geometry.border_selected_inverse_blocks(
            total_t,
            &self.shape_covariance_border_ranges(),
            &sandwich,
            frame_conditioning,
        )
    }

    /// #2933 F35 — which operator a shape report inverts at a converged state,
    /// decided once.
    ///
    /// A learned frame is estimated, so the covariance integrates it wherever the
    /// dense observed information of the unframed decoder is admitted
    /// ([`Self::frame_marginal_shape_information`]). The route then carries that
    /// frame-integrated information, formed here under `registry`. The covariance
    /// inverts it and the dispersion's fitted-response divergence reads it, so it
    /// is formed once. Otherwise, and with no frames at all, the report holds the
    /// frames fixed and reads the fixed-frame exact stationarity geometry formed
    /// here. The dispersion's divergence reads that geometry too (#2933 F33), unless
    /// it integrates frames the covariance holds fixed.
    pub(crate) fn shape_information_route(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        registry: Option<&AnalyticPenaltyRegistry>,
        cache: &ArrowFactorCache,
    ) -> Result<ShapeInformationRoute, String> {
        if self.frames_active() && self.frame_marginal_admission()?.is_none() {
            Ok(ShapeInformationRoute::FrameMarginal(
                self.frame_marginal_information(rho, target, registry)?,
            ))
        } else {
            Ok(ShapeInformationRoute::FixedFrame(
                self.materialize_exact_stationarity_geometry(rho, target, cache)?,
            ))
        }
    }

    /// #2933 F35 — the shape information production reports on `route`: the
    /// frame-marginal covariance, or [`Self::exact_observed_information_shape_covariance`]
    /// on the route's fixed-frame geometry, tagged with why the frames are held
    /// fixed.
    pub(crate) fn shape_information(
        &self,
        route: &ShapeInformationRoute,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
    ) -> Result<SaeShapeInformation, String> {
        match route {
            ShapeInformationRoute::FrameMarginal(information) => {
                self.frame_marginal_shape_information(information, rho, target)
            }
            ShapeInformationRoute::FixedFrame(geometry) => {
                self.exact_observed_information_shape_covariance(geometry, rho, target, cache)
            }
        }
    }

    /// Why this state's learned frames cannot be integrated, or `None` when they
    /// can: the unframed `(t, vec B)` observed information must be admitted on the
    /// same dense route the criterion prices, the framed atoms' `(M_k·p)²`
    /// covariances must fit the memory governor's single-materialization cap
    /// together, and every framed decoder must have the frame's rank, where the
    /// fixed-rank manifold has a tangent space.
    pub(crate) fn frame_marginal_admission(
        &self,
    ) -> Result<Option<SaeFrameMarginalUnavailable>, String> {
        let n = self.n_obs();
        let p = self.output_dim();
        let k_atoms = self.k_atoms();
        let total_basis: usize = self.atoms.iter().map(|atom| atom.basis_size()).sum();
        let d_max = self
            .atoms
            .iter()
            .map(SaeManifoldAtom::latent_dim)
            .max()
            .unwrap_or(0);
        let unframed_plan = sae_streaming_plan_for_shape_with_available(
            n,
            total_basis,
            k_atoms,
            d_max,
            self.beta_dim(),
            self.gpu_policy,
            self.host_available_bytes,
        )?;
        if !unframed_plan
            .admitted_or_error(n, p, k_atoms)
            .is_ok_and(|plan| plan.direct_logdet_admitted())
        {
            return Ok(Some(
                SaeFrameMarginalUnavailable::UnframedObservedInformationNotAdmitted,
            ));
        }
        if !self.framed_decoder_covariance_admitted() {
            return Ok(Some(
                SaeFrameMarginalUnavailable::DecoderCovariancesExceedMaterializationCap,
            ));
        }
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let Some(frame) = atom.decoder_frame.as_ref() else {
                continue;
            };
            if atom.decoder_numerical_rank()? < frame.rank() {
                return Ok(Some(
                    SaeFrameMarginalUnavailable::FrameCoordinatesRankDeficient { atom: atom_idx },
                ));
            }
        }
        Ok(None)
    }

    /// The conditioning of a covariance that holds every learned frame fixed. A
    /// framed state whose frames can be integrated is refused: a fixed-frame
    /// covariance there would omit orientation variance production reports.
    fn fixed_frame_conditioning(&self) -> Result<SaeFrameConditioning, String> {
        if !self.frames_active() {
            return Ok(SaeFrameConditioning::NoLearnedFrames);
        }
        match self.frame_marginal_admission()? {
            Some(reason) => Ok(SaeFrameConditioning::ConditionalOnFittedFrames(reason)),
            None => Err(
                "exact observed-information shape covariance: every learned frame admits \
                 integration, so holding it fixed would omit its orientation variance; use \
                 shape_information"
                    .to_string(),
            ),
        }
    }

    /// #2933 F35 — the joint observed information of this framed state in the
    /// identified tangent coordinates `(t, ξ)` of its learned frames, classified
    /// through the lifted `B` metric. One owner for every consumer that integrates
    /// the frames densely, on the unframed evidence factor of
    /// [`Self::unframed_evidence_factorization`].
    fn frame_marginal_information(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        registry: Option<&AnalyticPenaltyRegistry>,
    ) -> Result<FrameMarginalInformation, String> {
        let (unframed, sys, cache) = self.unframed_evidence_factorization(rho, target, registry)?;
        let tangent = LearnedFrameTangentMap::new(self, sys.gb.view())?;
        let total_t = cache.delta_t_len();
        let (a, _gap_border) =
            unframed.materialize_exact_hessian_dense_with_gap_border(rho, target, &cache)?;
        let operator = tangent.joint_operator(&a, total_t)?;
        // #2933 F07 — the tangent coordinates' pencil, in the pulled-back evidence metric
        // `diag(I, liftᵀ)·Φ·diag(I, lift)`, so the pushed-forward covariance is covariant in
        // the frame coordinates it names.
        let joint = Self::exact_hessian_spectral_block(
            operator,
            &ArrowMetric::JointLifted {
                cache: &cache,
                lift: &tangent.lift,
            }
            .prepare()?,
        )?;
        Ok(FrameMarginalInformation {
            tangent,
            joint,
            cache,
            total_t,
            unframed,
        })
    }

    /// #2933 F35 — the shape information integrated over every learned Grassmann
    /// frame: the Laplace covariance on the product of the fixed-rank decoder
    /// manifolds, in the same observed information the criterion prices.
    ///
    /// The decoder `B_k` is authoritative on a framed atom, so dropping every frame
    /// leaves the state unchanged and exposes the unframed model at it. Its frozen
    /// evidence factor carries `A_B = ∇²L` in `(t, vec B)`. A framed atom moves only
    /// along `δB_k = δC_k U_kᵀ + C_k W_kᵀ U_k⊥ᵀ`, so with `vec B = T·ξ` the observed
    /// information in the identified coordinates `(t, ξ)` is
    ///
    /// ```text
    ///   [[A_tt, A_tB·T], [Tᵀ·A_Bt, Tᵀ·A_BB·T + E]],   ξᵀEξ = 2⟨∇_B L, δC·Wᵀ·U⊥ᵀ⟩.
    /// ```
    ///
    /// `E` is the curvature of the bilinear parametrization. It vanishes unless the
    /// rank constraint binds, where `∇_B L` is normal to the manifold. The
    /// `−½·C·WᵀW·Uᵀ` second-order frame term contributes nothing, because
    /// `∇_B L·U = 0` at a C-stationary point. The pseudo-inverse over the retained
    /// directions is pushed forward, `Cov(vec B_k) = T_k·Σ_ξ·T_kᵀ`, and the row
    /// sandwich uses the meat `Tᵀ·J·T`. The tangent coordinates name physical
    /// decoder motions only, so the `GL(r_k)` factorization gauge never enters.
    /// `information` is this state's [`Self::frame_marginal_information`].
    pub(crate) fn frame_marginal_shape_information(
        &self,
        information: &FrameMarginalInformation,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
    ) -> Result<SaeShapeInformation, String> {
        let tangent = &information.tangent;
        let sandwich = tangent.tangent_sandwich(information.unframed.row_sandwich_meat(
            rho,
            &information.cache,
            target,
        )?)?;
        Ok(
            match information.joint.border_selected_inverse_blocks(
                information.total_t,
                &tangent.ranges,
                &sandwich,
                SaeFrameConditioning::MarginalOverLearnedFrames,
            )? {
                SaeShapeInformation::ObservedInformation(mut covariance) => {
                    covariance.blocks = tangent.push_forward(&covariance.blocks);
                    covariance.robust_blocks = tangent.push_forward(&covariance.robust_blocks);
                    SaeShapeInformation::ObservedInformation(covariance)
                }
                unavailable => unavailable,
            },
        )
    }

    // [#780 line-count gate] reconstruction_dispersion + assemble_shape_uncertainty
    // + recompute_joint_shape_uncertainty + unavailable_shape_uncertainty
    // (the contiguous trailing methods of this impl block) were split into the
    // sibling construction_reconstruction.rs (declared in mod.rs); callers reach
    // them bare via use super::*.
}

/// #2933 F33/F35 — the operator a shape report inverts, decided once by
/// [`SaeManifoldTerm::shape_information_route`].
pub(crate) enum ShapeInformationRoute {
    /// Every learned frame admits integration: the frame-integrated information
    /// the covariance inverts and the dispersion's fitted-response divergence reads.
    FrameMarginal(FrameMarginalInformation),
    /// No learned frames, or frames held fixed: the fixed-frame exact stationarity
    /// geometry, which the dispersion's fitted-response divergence reads too.
    FixedFrame(ExactHessianSpectralBlock),
}

impl ShapeInformationRoute {
    /// The operator the dispersion shares with the covariance on this route.
    pub(crate) fn held_response_geometry(&self) -> HeldResponseGeometry<'_> {
        match self {
            Self::FrameMarginal(information) => HeldResponseGeometry::FrameMarginal(information),
            Self::FixedFrame(geometry) => HeldResponseGeometry::FixedFrame(geometry),
        }
    }
}

impl ExactHessianSpectralBlock {
    /// Border blocks `[A⁺]_ββ[r, r]` of the pseudo-inverse over the retained
    /// directions, or the typed refusal when `A` has a resolved negative
    /// direction (#2933 F33).
    ///
    /// Classification is [`Self::rank_floor`], the one predicate the value, the
    /// differential and the stationarity solve read, on the generalized eigenpairs
    /// `(μᵢ, wᵢ)` of the pencil `(A, Φ)` (#2933 F07): `μᵢ < −floor(i)` is a negative
    /// direction, `|μᵢ| ≤ floor(i)` is unidentified and carries no variance, `μᵢ > floor(i)`
    /// is retained. `A⁺ = Σ_retained wᵢwᵢᵀ/μᵢ` is the covariant pseudo-inverse, so with
    /// `F = W_β[:, retained]·M^{−½}` each block is the Gram `F_r F_rᵀ`, positive
    /// semidefinite by construction. The robust blocks are those of the row sandwich over
    /// `sandwich`, whose aggregate-mass carriers `c_k u_k` are projected here onto the
    /// retained half-inverse, `z_k = M^{−½}W_retᵀ(c_k u_k)`.
    fn border_selected_inverse_blocks(
        &self,
        total_t: usize,
        ranges: &[std::ops::Range<usize>],
        sandwich: &RowSandwichMeat,
        frame_conditioning: SaeFrameConditioning,
    ) -> Result<SaeShapeInformation, String> {
        let dim = self.eigenvalues.len();
        if self.eigenvectors.dim() != (dim, dim) || total_t > dim {
            return Err(format!(
                "exact observed-information shape covariance: eigenvectors {:?}, spectrum \
                 {dim}, t dimension {total_t}",
                self.eigenvectors.dim()
            ));
        }
        let mut negative_directions = 0usize;
        let mut most_negative_curvature = 0.0_f64;
        let mut retained = Vec::with_capacity(dim);
        for index in 0..dim {
            let lambda = self.eigenvalues[index];
            let floor = self.rank_floor(index);
            if lambda < -floor {
                negative_directions += 1;
                most_negative_curvature = most_negative_curvature.min(lambda);
            } else if lambda > floor {
                retained.push(index);
            }
        }
        if negative_directions > 0 {
            return Ok(SaeShapeInformation::Unavailable(
                SaeShapeCovarianceUnavailable::IndefiniteObservedInformation {
                    negative_directions,
                    most_negative_curvature,
                },
            ));
        }
        let border_dim = dim - total_t;
        let border_factor = Array2::from_shape_fn((border_dim, retained.len()), |(row, col)| {
            let index = retained[col];
            self.eigenvectors[[total_t + row, index]] / self.eigenvalues[index].sqrt()
        });
        let mut mass_projections =
            Array2::<f64>::zeros((retained.len(), sandwich.mass_carriers.len()));
        for (column, (coefficient, carrier)) in sandwich.mass_carriers.iter().enumerate() {
            for &(index, value) in carrier {
                if index >= total_t {
                    return Err(format!(
                        "exact observed-information shape covariance: mass carrier index \
                         {index} lies outside the t block of width {total_t}"
                    ));
                }
                for (col, &eigen_index) in retained.iter().enumerate() {
                    mass_projections[[col, column]] += coefficient
                        * value
                        * self.eigenvectors[[index, eigen_index]]
                        / self.eigenvalues[eigen_index].sqrt();
                }
            }
        }
        let (blocks, robust_blocks) = observed_information_border_blocks(
            border_factor.view(),
            mass_projections.view(),
            sandwich.meat.view(),
            ranges,
        )?;
        Ok(SaeShapeInformation::ObservedInformation(
            SaeObservedInformationCovariance {
                blocks,
                robust_blocks,
                identified_rank: retained.len(),
                ambient_dim: dim,
                frame_conditioning,
            },
        ))
    }
}

/// The observed information of a framed state integrated over its learned frames
/// (#2933 F35); see [`SaeManifoldTerm::frame_marginal_information`].
pub(crate) struct FrameMarginalInformation {
    /// `vec B = T·ξ`, the cross curvature `E` and the per-atom `ξ` ranges.
    tangent: LearnedFrameTangentMap,
    /// The joint `(t, ξ)` operator `[[A_tt, A_tB·T], [Tᵀ·A_Bt, Tᵀ·A_BB·T + E]]` and
    /// its pencil eigensystem in the pulled-back metric `ArrowMetric::JointLifted`.
    joint: ExactHessianSpectralBlock,
    /// The frozen evidence factor of the unframed assembly, in `(t, vec B)`.
    cache: ArrowFactorCache,
    /// Latent coordinate dimension of the joint layout.
    total_t: usize,
    /// The state with every frame dropped and this term's gates declared, which
    /// `A`, the decoder gradient and the row sandwich are read from.
    unframed: SaeManifoldTerm,
}

/// Rank-`r_k` tangent coordinates of the learned frames, written into the
/// unframed decoder layout (#2933 F35).
///
/// On a framed atom `ξ_k = (vec δC_k, vec W_k)` with `δC_k ∈ ℝ^{M_k×r_k}` and
/// `W_k ∈ ℝ^{(p−r_k)×r_k}`, `δB_k = δC_k U_kᵀ + C_k W_kᵀ U_k⊥ᵀ`: `M_k r_k + r_k(p − r_k)`
/// coordinates, the dimension of the fixed-rank manifold. An unframed atom keeps
/// its `M_k·p` decoder coordinates.
struct LearnedFrameTangentMap {
    /// `T` in `vec B = T·ξ`, shape `(beta_dim, ξ_dim)`, flat decoder index `b·p + c`.
    lift: Array2<f64>,
    /// `E` with `ξᵀEξ = 2⟨∇_B L, δC·Wᵀ·U⊥ᵀ⟩`, shape `(ξ_dim, ξ_dim)`.
    cross_curvature: Array2<f64>,
    /// Each atom's `ξ` range.
    ranges: Vec<std::ops::Range<usize>>,
    /// Each atom's `vec B` range.
    beta_ranges: Vec<std::ops::Range<usize>>,
}

impl LearnedFrameTangentMap {
    /// `decoder_gradient` is `∇_B L` in the unframed layout at the same state.
    fn new(term: &SaeManifoldTerm, decoder_gradient: ArrayView1<'_, f64>) -> Result<Self, String> {
        let p = term.output_dim();
        let beta_dim = term.beta_dim();
        if decoder_gradient.len() != beta_dim {
            return Err(format!(
                "learned frame tangent map: decoder gradient length {} != beta dimension {beta_dim}",
                decoder_gradient.len()
            ));
        }
        let beta_ranges = term.beta_block_offsets().to_vec();
        let mut ranges = Vec::with_capacity(term.k_atoms());
        let mut cursor = 0usize;
        for atom in &term.atoms {
            let m = atom.basis_size();
            let width = match atom.decoder_frame.as_ref() {
                Some(frame) => m * frame.rank() + frame.rank() * (p - frame.rank()),
                None => m * p,
            };
            ranges.push(cursor..cursor + width);
            cursor += width;
        }
        let mut lift = Array2::<f64>::zeros((beta_dim, cursor));
        let mut cross_curvature = Array2::<f64>::zeros((cursor, cursor));
        for (atom_idx, atom) in term.atoms.iter().enumerate() {
            let m = atom.basis_size();
            let beta_start = beta_ranges[atom_idx].start;
            let xi_start = ranges[atom_idx].start;
            let Some(frame) = atom.decoder_frame.as_ref() else {
                for index in 0..m * p {
                    lift[[beta_start + index, xi_start + index]] = 1.0;
                }
                continue;
            };
            let u = frame.frame();
            let r = u.ncols();
            let coordinates = atom.decoder_coefficients().dot(&u);
            let complement = orthonormal_frame_complement(u)?;
            let w_start = xi_start + m * r;
            for b in 0..m {
                for c in 0..p {
                    let row = beta_start + b * p + c;
                    for j in 0..r {
                        lift[[row, xi_start + b * r + j]] = u[[c, j]];
                        for i in 0..p - r {
                            lift[[row, w_start + i * r + j]] = coordinates[[b, j]] * complement[[c, i]];
                        }
                    }
                }
            }
            // `⟨G, δC·Wᵀ·U⊥ᵀ⟩ = Σ_{b,i,j} (G·U⊥)[b,i]·δC[b,j]·W[i,j]` with
            // `G = ∇_{B_k} L`; each symmetric pair is written once per triangle.
            let gradient = Array2::from_shape_fn((m, p), |(b, c)| {
                decoder_gradient[beta_start + b * p + c]
            });
            let normal_gradient = gradient.dot(&complement);
            for b in 0..m {
                for i in 0..p - r {
                    let value = normal_gradient[[b, i]];
                    for j in 0..r {
                        let c_index = xi_start + b * r + j;
                        let w_index = w_start + i * r + j;
                        cross_curvature[[c_index, w_index]] += value;
                        cross_curvature[[w_index, c_index]] += value;
                    }
                }
            }
        }
        Ok(Self {
            lift,
            cross_curvature,
            ranges,
            beta_ranges,
        })
    }

    /// The joint `(t, ξ)` operator from the joint `(t, vec B)` observed information.
    fn joint_operator(&self, a: &Array2<f64>, total_t: usize) -> Result<Array2<f64>, String> {
        let beta_dim = self.lift.nrows();
        let xi_dim = self.lift.ncols();
        if a.dim() != (total_t + beta_dim, total_t + beta_dim) {
            return Err(format!(
                "learned frame tangent map: observed information {:?} does not match t dimension \
                 {total_t} plus beta dimension {beta_dim}",
                a.dim()
            ));
        }
        let coupling = a.slice(s![..total_t, total_t..]).dot(&self.lift);
        let border = self.congruence(&a.slice(s![total_t.., total_t..]).to_owned())
            + &self.cross_curvature;
        let mut operator = Array2::<f64>::zeros((total_t + xi_dim, total_t + xi_dim));
        operator
            .slice_mut(s![..total_t, ..total_t])
            .assign(&a.slice(s![..total_t, ..total_t]));
        operator.slice_mut(s![..total_t, total_t..]).assign(&coupling);
        operator.slice_mut(s![total_t.., ..total_t]).assign(&coupling.t());
        operator.slice_mut(s![total_t.., total_t..]).assign(&border);
        Ok(operator)
    }

    /// `Tᵀ·M·T` for a border-layout matrix `M`.
    fn congruence(&self, matrix: &Array2<f64>) -> Array2<f64> {
        fast_atb(&self.lift, &fast_ab(matrix, &self.lift))
    }

    /// The row-sandwich meat in tangent coordinates: `Tᵀ·J·T` over the border,
    /// with the aggregate-mass rows and columns carried through unchanged,
    /// because the lift moves neither `t` nor the masses.
    fn tangent_sandwich(&self, sandwich: RowSandwichMeat) -> Result<RowSandwichMeat, String> {
        let beta_dim = self.lift.nrows();
        let xi_dim = self.lift.ncols();
        let mass_dim = sandwich.mass_carriers.len();
        if sandwich.meat.dim() != (beta_dim + mass_dim, beta_dim + mass_dim) {
            return Err(format!(
                "learned frame tangent map: row-sandwich meat {:?} does not match beta dimension \
                 {beta_dim} plus {mass_dim} aggregate masses",
                sandwich.meat.dim()
            ));
        }
        let mut lift = Array2::<f64>::zeros((beta_dim + mass_dim, xi_dim + mass_dim));
        lift.slice_mut(s![..beta_dim, ..xi_dim]).assign(&self.lift);
        for k in 0..mass_dim {
            lift[[beta_dim + k, xi_dim + k]] = 1.0;
        }
        Ok(RowSandwichMeat {
            meat: fast_atb(&lift, &fast_ab(&sandwich.meat, &lift)),
            mass_carriers: sandwich.mass_carriers,
        })
    }

    /// `T_k·Σ_k·T_kᵀ` per atom: tangent-coordinate blocks to decoder blocks.
    fn push_forward(&self, blocks: &[Array2<f64>]) -> Vec<Array2<f64>> {
        blocks
            .iter()
            .enumerate()
            .map(|(atom, block)| {
                let local = self
                    .lift
                    .slice(s![self.beta_ranges[atom].clone(), self.ranges[atom].clone()])
                    .to_owned();
                fast_ab(&fast_ab(&local, block), &local.t().to_owned())
            })
            .collect()
    }
}

/// Orthonormal basis `U⊥` (`p × (p − r)`) of the complement of a column-orthonormal
/// frame, read off `I − UUᵀ`, whose eigenvalues are exactly `1` on the complement
/// and `0` on the span; `½` separates the two.
fn orthonormal_frame_complement(frame: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
    let (p, r) = frame.dim();
    let projector = Array2::<f64>::eye(p) - &frame.dot(&frame.t());
    let (values, vectors) = projector
        .eigh(Side::Lower)
        .map_err(|err| format!("learned frame complement: eigendecomposition failed: {err:?}"))?;
    let columns: Vec<usize> = (0..p).filter(|&index| values[index] > 0.5).collect();
    if columns.len() != p - r {
        return Err(format!(
            "learned frame complement: I − UUᵀ has {} unit eigenvalues for p − r = {}",
            columns.len(),
            p - r
        ));
    }
    Ok(Array2::from_shape_fn((p, p - r), |(row, col)| {
        vectors[[row, columns[col]]]
    }))
}

// [#780 line-count gate] The inner convergence behind the undamped evidence log-determinant
// lives in the sibling `construction_undamped_inner.rs`, a further `impl SaeManifoldTerm` block.
include!("construction_undamped_inner.rs");
// #2228/#2822 — the evidence root refinement and its gauge-projected exact step.
include!("construction_evidence_root.rs");

#[cfg(test)]
mod shape_covariance_observed_information_2933_f33_tests {
    use super::*;
    use crate::manifold::arrow_solver::apply_cached_arrow_hessian;
    use crate::manifold::{FaerEigh, Side};

    /// The moderate-penalty softmax basin `recompute_reproduces_joint_shape_band`
    /// fits: a genuine reconstruction residual with both atoms alive, so the
    /// residual curvature `Σ (WMr)·∇²f` that separates `A` from `B` is live.
    fn converged_basin() -> (
        SaeManifoldTerm,
        Array2<f64>,
        SaeManifoldRho,
        SaeManifoldLoss,
        ArrowFactorCache,
    ) {
        let (mut term, target, mut rho) =
            crate::manifold::tests_recovery_split_780::gamma_fd_tiny_fixture();
        rho.log_lambda_sparse = 0.0;
        for value in rho.log_lambda_smooth.iter_mut() {
            *value = -1.0;
        }
        for axis in rho.log_ard.iter_mut() {
            for value in axis.iter_mut() {
                *value = -1.0;
            }
        }
        let (_value, loss, cache) = term
            .penalized_quasi_laplace_criterion_with_cache(
                target.view(),
                &rho,
                None,
                40,
                0.4,
                1.0e-6,
                1.0e-6,
            )
            .expect("the moderate-penalty basin converges with both atoms alive");
        (term, target, rho, loss, cache)
    }

    /// The analytic joint KKT gradient in the cache's `(t, β)` layout.
    fn joint_gradient(
        term: &mut SaeManifoldTerm,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
    ) -> Array1<f64> {
        let system = term
            .assemble_arrow_schur(target, rho, None)
            .expect("joint gradient assembly");
        let mut flat = Vec::new();
        for row in &system.rows {
            flat.extend(row.gt.iter().copied());
        }
        flat.extend(system.gb.iter().copied());
        Array1::from_vec(flat)
    }

    /// Central difference of the analytic joint gradient along every joint
    /// coordinate. Shares no code with the exact-Hessian applies, the majorizer,
    /// or the selected-inverse producer.
    ///
    /// The collapse-prevention gates are the ones `term` declared at its root: `A`
    /// is the Hessian of `V(ρ; w₀)` with the routing weights `w₀` held fixed
    /// (#2933 F05). A clone of a term whose gates are not declared carries none, and a
    /// clone without a gate reads the separation barrier's coactivation from its
    /// live assignments. Its logit columns would then differentiate a routing
    /// refresh the observed information does not carry, so every perturbed clone
    /// re-declares the root's gates.
    fn finite_difference_hessian(
        term: &SaeManifoldTerm,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        total_t: usize,
        k: usize,
        step: f64,
    ) -> Array2<f64> {
        let dim = total_t + k;
        let gates = term.collapse_prevention_gates();
        let mut hessian = Array2::<f64>::zeros((dim, dim));
        for column in 0..dim {
            let mut dt = Array1::<f64>::zeros(total_t);
            let mut db = Array1::<f64>::zeros(k);
            if column < total_t {
                dt[column] = 1.0;
            } else {
                db[column - total_t] = 1.0;
            }
            let mut plus = term.clone();
            plus.declare_collapse_prevention_gates(&gates);
            plus.apply_newton_step(dt.view(), db.view(), step)
                .expect("forward perturbation");
            let forward = joint_gradient(&mut plus, target, rho);
            let mut minus = term.clone();
            minus.declare_collapse_prevention_gates(&gates);
            let (neg_t, neg_b) = (-&dt, -&db);
            minus
                .apply_newton_step(neg_t.view(), neg_b.view(), step)
                .expect("backward perturbation");
            let backward = joint_gradient(&mut minus, target, rho);
            assert_eq!(forward.len(), dim, "gradient layout must match the joint cache layout");
            hessian
                .column_mut(column)
                .assign(&((&forward - &backward) / (2.0 * step)));
        }
        hessian
    }

    /// An independent classification of a dense operator in the evidence factor's
    /// pencil, with this oracle's own operands and the shared scalar band rule
    /// (#2933 F07).
    struct OracleSpectrum {
        values: Array1<f64>,
        vectors: Array2<f64>,
        retained: Vec<usize>,
        negative: usize,
        in_band: usize,
    }

    fn classify(operator: &Array2<f64>, cache: &ArrowFactorCache) -> OracleSpectrum {
        let oracle = crate::manifold::tests::PencilOracle::new(operator, cache);
        let retained = oracle.retained();
        let negative = oracle.negative().len();
        let in_band = oracle.in_band();
        OracleSpectrum {
            values: oracle.values,
            vectors: oracle.vectors,
            retained,
            negative,
            in_band,
        }
    }

    /// `[A⁺]_ββ[r, r]` from an oracle spectrum.
    fn border_block(
        spectrum: &OracleSpectrum,
        total_t: usize,
        range: &std::ops::Range<usize>,
    ) -> Array2<f64> {
        let width = range.len();
        let mut block = Array2::<f64>::zeros((width, width));
        for &index in &spectrum.retained {
            let inverse = 1.0 / spectrum.values[index];
            for row in 0..width {
                let left = spectrum.vectors[[total_t + range.start + row, index]] * inverse;
                for col in 0..width {
                    block[[row, col]] += left * spectrum.vectors[[total_t + range.start + col, index]];
                }
            }
        }
        block
    }

    fn frobenius(matrix: ArrayView2<'_, f64>) -> f64 {
        matrix.iter().map(|value| value * value).sum::<f64>().sqrt()
    }

    /// The majorizer `B` the cache factors, materialized by columns.
    fn majorizer_by_columns(cache: &ArrowFactorCache) -> Array2<f64> {
        let total_t = cache.delta_t_len();
        let k = cache.k;
        let dim = total_t + k;
        let mut b = Array2::<f64>::zeros((dim, dim));
        for column in 0..dim {
            let mut dt = Array1::<f64>::zeros(total_t);
            let mut db = Array1::<f64>::zeros(k);
            if column < total_t {
                dt[column] = 1.0;
            } else {
                db[column - total_t] = 1.0;
            }
            let applied = apply_cached_arrow_hessian(cache, dt.view(), db.view()).expect("B apply");
            for row in 0..total_t {
                b[[row, column]] = applied.t[row];
            }
            for row in 0..k {
                b[[total_t + row, column]] = applied.beta[row];
            }
        }
        (&b + &b.t()) * 0.5
    }

    /// Central-difference step for the gradient oracle, in the Newton-step units
    /// of the joint state (chart coordinates of period one, decoder coefficients
    /// of order 0.1).
    const GRADIENT_ORACLE_STEP: f64 = 1.0e-4;

    /// #2933 F33 — the reported shape covariance is the border block of the
    /// pseudo-inverse of the observed information `A`, not of the majorizer `B`.
    ///
    /// The oracle is a central difference of the analytic joint gradient, which
    /// shares no code with the exact-Hessian applies, the majorizer or the
    /// selected-inverse producer. Central error is `O(h²)`, so at steps `h` and
    /// `h/2`, `H(h) − H(h/2) ≈ ¾·err(h)` and the finer Hessian's truncation error
    /// is about `‖H(h) − H(h/2)‖/3`; the antisymmetric half of the raw difference
    /// measures the error that does not shrink with `h` and is added to it. The
    /// Neumann bound
    /// `‖δ(A⁻¹)‖_F ≤ ‖A⁻¹‖₂²‖δA‖_F / (1 − ‖A⁻¹‖₂‖δA‖_F)` turns that into a
    /// covariance tolerance with no tuned constant. The majorizer's Schur block
    /// must miss the same oracle by more than that tolerance, so the fixture can
    /// tell `A` from `B` and the assertion is free to fail against the inverse it
    /// replaced.
    #[test]
    fn shape_covariance_is_the_observed_information_selected_inverse_2933_f33() {
        let (term, target, rho, loss, cache) = converged_basin();
        let total_t = cache.delta_t_len();
        let k = cache.k;
        let ranges = term.shape_covariance_border_ranges();
        let geometry = term
            .materialize_exact_stationarity_geometry(&rho, target.view(), &cache)
            .expect("exact stationarity geometry at the converged basin");
        let information = term
            .exact_observed_information_shape_covariance(&geometry, &rho, target.view(), &cache)
            .expect("exact observed information at the converged basin");
        let SaeShapeInformation::ObservedInformation(covariance) = &information else {
            panic!(
                "the converged PD basin must yield an observed-information covariance; got \
                 {information:?}"
            );
        };

        let coarse_raw =
            finite_difference_hessian(&term, target.view(), &rho, total_t, k, GRADIENT_ORACLE_STEP);
        let fine_raw = finite_difference_hessian(
            &term,
            target.view(),
            &rho,
            total_t,
            k,
            0.5 * GRADIENT_ORACLE_STEP,
        );
        // The true Hessian is symmetric, so the antisymmetric half of the raw
        // difference is pure oracle error: roundoff and any gradient that is not
        // the derivative of one scalar. A symmetric error of the same size need not
        // cancel, so it joins the truncation estimate in the error the bar allows.
        let roundoff = 0.5 * frobenius((&fine_raw - &fine_raw.t()).view());
        let coarse = (&coarse_raw + &coarse_raw.t()) * 0.5;
        let fine = (&fine_raw + &fine_raw.t()) * 0.5;
        let step_error = frobenius((&coarse - &fine).view()) / 3.0;
        let oracle_error = step_error + roundoff;
        let oracle = classify(&fine, &cache);
        let smallest = oracle
            .retained
            .iter()
            .map(|&index| oracle.values[index])
            .fold(f64::INFINITY, f64::min);
        let inverse_norm = 1.0 / smallest;
        let contraction = inverse_norm * oracle_error;
        let tolerance = inverse_norm * inverse_norm * oracle_error / (1.0 - contraction);
        let expected: Vec<Array2<f64>> = ranges
            .iter()
            .map(|range| border_block(&oracle, total_t, range))
            .collect();
        let misses: Vec<f64> = expected
            .iter()
            .enumerate()
            .map(|(atom, block)| frobenius((&covariance.blocks[atom] - block).view()))
            .collect();
        let majorizer_gap = ranges
            .iter()
            .zip(&expected)
            .map(|(range, block)| {
                let majorizer = cache
                    .schur_inverse_block(range.clone())
                    .expect("majorizer Schur inverse block");
                frobenius((&majorizer - block).view())
            })
            .fold(0.0_f64, f64::max);
        eprintln!(
            "[#2933 F33] gradient oracle: step error {step_error:.3e}, antisymmetric roundoff \
             {roundoff:.3e} (‖H‖_F = {:.3e}), smallest curvature {smallest:.3e}, Neumann \
             contraction {contraction:.3e}, tolerance {tolerance:.3e}; per-atom misses [{}]; \
             majorizer gap {majorizer_gap:.3e}",
            frobenius(fine.view()),
            misses
                .iter()
                .map(|miss| format!("{miss:.3e}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        assert_eq!(
            (oracle.negative, oracle.in_band),
            (0, 0),
            "the oracle Hessian must be positive definite at the converged basin"
        );
        assert!(
            contraction < 1.0,
            "the gradient oracle's error {oracle_error:.3e} must be inside the Neumann radius \
             of the smallest curvature {smallest:.3e}"
        );
        assert_eq!(covariance.identified_rank, oracle.retained.len());
        assert_eq!(covariance.ambient_dim, total_t + k);
        for (atom, miss) in misses.iter().enumerate() {
            assert!(
                *miss <= tolerance,
                "atom {atom}: ‖[A⁺]_ββ − oracle‖_F = {miss:.3e} exceeds the gradient-oracle \
                 tolerance {tolerance:.3e} (‖oracle‖_F = {:.3e})",
                frobenius(expected[atom].view())
            );
        }
        assert!(
            majorizer_gap > tolerance,
            "the majorizer's Schur inverse misses the observed-information oracle by only \
             {majorizer_gap:.3e} against the tolerance {tolerance:.3e}: this fixture cannot tell \
             A from B"
        );

        let residual = term
            .reconstruction_residual(target.view(), &rho)
            .expect("reconstruction residual");
        let dispersion = term
            .reconstruction_dispersion(&loss, &cache, &rho, residual.view())
            .expect("dispersion");
        let uncertainty = term
            .assemble_shape_uncertainty(&information, dispersion)
            .expect("shape uncertainty");
        assert_eq!(
            uncertainty.operator,
            SaeShapeCovarianceOperator::ObservedInformation {
                identified_rank: covariance.identified_rank,
                ambient_dim: covariance.ambient_dim,
                frame_conditioning: covariance.frame_conditioning,
            }
        );
        let scale = dispersion.posterior_covariance_scale();
        for (atom, entry) in uncertainty.atoms.iter().enumerate() {
            let reported = entry
                .decoder_covariance
                .as_ref()
                .expect("un-framed fixture exports its decoder covariance");
            let expected = covariance.blocks[atom].mapv(|value| scale * value);
            assert!(
                frobenius((reported - &expected).view())
                    <= f64::EPSILON * frobenius(expected.view()) * reported.len() as f64,
                "atom {atom}: the exported decoder covariance must be φ̂·[A⁺]_ββ"
            );
        }
    }

    /// #2933 F33 — where the information coincides with the majorizer, the
    /// selected inverse reduces to the historical Schur covariance.
    ///
    /// `B` materialized by columns goes through the same spectral block and
    /// selected-inverse producer as `A`; the result must equal
    /// `cache.schur_inverse_block`, an arrow Schur-complement Cholesky solve that
    /// shares no code with the eigendecomposition. Both are backward stable, so
    /// they agree to the eigendecomposition's backward error amplified by the
    /// conditioning: `dim·ε·κ(B)` relative.
    #[test]
    fn observed_information_covariance_reduces_to_the_schur_inverse_when_a_equals_b_2933_f33() {
        let (term, _target, _rho, _loss, cache) = converged_basin();
        let total_t = cache.delta_t_len();
        let dim = total_t + cache.k;
        let b = majorizer_by_columns(&cache);
        let (values, _) = b.eigh(Side::Lower).expect("B eigendecomposition");
        let smallest = values.iter().copied().fold(f64::INFINITY, f64::min);
        let largest = values.iter().copied().fold(0.0_f64, f64::max);
        assert!(smallest > 0.0, "the majorizer is positive definite");
        let tolerance = dim as f64 * f64::EPSILON * (largest / smallest);
        let metric = ArrowMetric::Joint(&cache)
            .prepare()
            .expect("prepared evidence metric");
        let block = SaeManifoldTerm::exact_hessian_spectral_block(b, &metric)
            .expect("spectral block of B");
        // Only the model blocks are checked here; the row-sandwich meat is inert.
        let sandwich = RowSandwichMeat {
            meat: Array2::<f64>::zeros((cache.k, cache.k)),
            mass_carriers: Vec::new(),
        };
        let information = block
            .border_selected_inverse_blocks(
                total_t,
                &term.shape_covariance_border_ranges(),
                &sandwich,
                SaeFrameConditioning::NoLearnedFrames,
            )
            .expect("selected inverse of B");
        let SaeShapeInformation::ObservedInformation(covariance) = information else {
            panic!("a positive definite operator must yield a covariance; got {information:?}");
        };
        assert_eq!(covariance.identified_rank, dim, "every direction of B is identified");
        for (atom, range) in term.shape_covariance_border_ranges().iter().enumerate() {
            let schur = cache
                .schur_inverse_block(range.clone())
                .expect("Schur inverse block");
            let relative =
                frobenius((&covariance.blocks[atom] - &schur).view()) / frobenius(schur.view());
            assert!(
                relative <= tolerance,
                "atom {atom}: selected inverse of B misses the Schur inverse by {relative:.3e} \
                 relative against dim·ε·κ = {tolerance:.3e}"
            );
        }
    }

    /// #2933 F33 — an observed information with resolved negative curvature has
    /// no Laplace covariance.
    ///
    /// No converged fixture carries a resolved negative direction at its criterion
    /// state. The #2336 ARD saddle this test used to take, the ordered
    /// Beta--Bernoulli patchd fixture, this softmax basin and the periodic circle all
    /// classify `negative = 0` in the evidence pencil (#2933 F07 census). The
    /// specimen is therefore built exactly in that pencil. With the converged basin's
    /// smallest retained pair `(μ, w)`, `Aw = μΦw` and `wᵀΦw = 1`, the rank-one update
    /// `A′ = A − 2μ·(Φw)(Φw)ᵀ` sends `w` to `−μ` and leaves every pair `Φ`-orthogonal
    /// to `w` unchanged. `A′` has exactly one negative direction, resolved as `μ` was.
    /// It enters production's own spectral-block constructor and shape-covariance
    /// entry, so the refusal predicate under test is production's, counted against
    /// the independent pencil oracle. The two classifications must agree to the
    /// oracle's band edge, the resolution it classifies at.
    #[test]
    fn shape_covariance_refuses_indefinite_observed_information_2933_f33() {
        let (term, target, rho, loss, cache) = converged_basin();
        let (a, _gap) = term
            .materialize_exact_hessian_dense_with_gap_border(&rho, target.view(), &cache)
            .expect("exact observed information at the converged basin");
        let a = (&a + &a.t()) * 0.5;
        let dim = a.nrows();
        let basin = crate::manifold::tests::PencilOracle::new(&a, &cache);
        let flip = basin
            .retained()
            .into_iter()
            .min_by(|&left, &right| basin.values[left].total_cmp(&basin.values[right]))
            .expect("the converged basin retains a direction");
        let mu = basin.values[flip];
        let metric = ArrowMetric::Joint(&cache)
            .prepare()
            .expect("prepared evidence metric");
        let phi_w = metric
            .apply(basin.vectors.column(flip))
            .expect("evidence metric apply");
        let flipped = &a
            - &Array2::from_shape_fn((dim, dim), |(row, col)| 2.0 * mu * phi_w[row] * phi_w[col]);
        let specimen = crate::manifold::tests::PencilOracle::new(&flipped, &cache);
        let negative = specimen.negative();
        let most_negative = specimen.values.iter().copied().fold(f64::INFINITY, f64::min);
        let negative_edge = negative
            .first()
            .map_or(f64::NAN, |&index| specimen.floors[index]);
        let listing = |values: &Array1<f64>, edge: &dyn Fn(usize) -> f64| -> String {
            (0..values.len())
                .map(|index| format!("{:.3e}/{:.1e}", values[index], edge(index)))
                .collect::<Vec<_>>()
                .join(" ")
        };
        let euclidean = |operator: &Array2<f64>| -> (Array1<f64>, f64) {
            let (values, _) = operator
                .eigh(Side::Lower)
                .expect("Euclidean eigendecomposition");
            let norm = values.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
            (values, dim as f64 * f64::EPSILON * norm)
        };
        let (basin_euclidean, basin_euclidean_edge) = euclidean(&a);
        let (flipped_euclidean, flipped_euclidean_edge) = euclidean(&flipped);
        eprintln!(
            "[#2933 F33] refusal spectra, value/edge. basin A, Euclidean dim·ε·‖A‖₂: {} | basin \
             A, evidence pencil: {} | A′, Euclidean: {} | A′, evidence pencil: {}",
            listing(&basin_euclidean, &|_| basin_euclidean_edge),
            listing(&basin.values, &|index| basin.floors[index]),
            listing(&flipped_euclidean, &|_| flipped_euclidean_edge),
            listing(&specimen.values, &|index| specimen.floors[index])
        );
        eprintln!(
            "[#2933 F33] refusal specimen: basin negative {} in band {}, flipped μ {mu:.6e} \
             (edge {:.3e}); flipped operator negative {} in band {}, most negative μ \
             {most_negative:.6e} (edge {negative_edge:.3e})",
            basin.negative().len(),
            basin.in_band(),
            basin.floors[flip],
            negative.len(),
            specimen.in_band()
        );
        assert_eq!(
            (basin.negative().len(), basin.in_band()),
            (0, 0),
            "the converged basin must be positive definite in the evidence pencil"
        );
        assert_eq!(negative.len(), 1, "the flip must resolve exactly one negative direction");
        assert!(
            (most_negative + mu).abs() <= negative_edge,
            "the flipped curvature {most_negative:.6e} must be −μ = {:.6e} to the band edge \
             {negative_edge:.3e}",
            -mu
        );
        let geometry = SaeManifoldTerm::exact_hessian_spectral_block(flipped, &metric)
            .expect("production spectral block of the flipped operator");
        let information = term
            .exact_observed_information_shape_covariance(&geometry, &rho, target.view(), &cache)
            .expect("exact observed information");
        match &information {
            SaeShapeInformation::Unavailable(
                SaeShapeCovarianceUnavailable::IndefiniteObservedInformation {
                    negative_directions,
                    most_negative_curvature,
                },
            ) => {
                assert_eq!(*negative_directions, negative.len());
                assert!(
                    (most_negative_curvature - most_negative).abs() <= negative_edge,
                    "most negative curvature {most_negative_curvature:.6e} against the oracle \
                     {most_negative:.6e} (band edge {negative_edge:.3e})"
                );
            }
            other => panic!("an indefinite observed information must be refused; got {other:?}"),
        }
        let residual = term
            .reconstruction_residual(target.view(), &rho)
            .expect("reconstruction residual");
        let dispersion = term
            .reconstruction_dispersion(&loss, &cache, &rho, residual.view())
            .expect("dispersion");
        let uncertainty = term
            .assemble_shape_uncertainty(&information, dispersion)
            .expect("explicit unavailability");
        assert!(matches!(
            uncertainty.operator,
            SaeShapeCovarianceOperator::Unavailable(
                SaeShapeCovarianceUnavailable::IndefiniteObservedInformation { .. }
            )
        ));
        for entry in &uncertainty.atoms {
            assert!(
                entry.decoder_covariance.is_none()
                    && entry.band_coords.is_none()
                    && entry.band_mean.is_none()
                    && entry.band_sd.is_none(),
                "a refused covariance reports no band"
            );
        }
    }
}

#[cfg(test)]
mod smoothness_dof_exact_oracle_tests {
    use super::*;

    impl SaeManifoldTerm {
        /// Per-atom effective penalized dof of the decoder smoothness penalty
        /// (#1556): entry `k` is `tr(S_β⁻¹ · M_k)` with `M_k = (λ_smooth[k]·S_k) ⊗ I`
        /// and `S_β⁻¹ = (H⁻¹)_ββ` the Schur-complement inverse, each atom scaled by
        /// its OWN `lambda_smooth[atom_idx]`. Built on
        /// [`ArrowFactorCache::schur_inverse_apply`]: column `(k,μ,oc)` of `M_k` is
        /// `λ_k·S_k[:,μ] ⊗ e_oc` (sparse), so we apply `S_β⁻¹` to that K-vector and
        /// read back `result[col]`. The total edf is the sum of the returned vector
        /// (a uniform/broadcast λ reproduces the historical global trace).
        ///
        /// The trace is exact at every `K`: `Σ_k M_k·r_k = K` column applies of
        /// `O(K²)` cost `O(K³)`, the order of the Schur factorization the cache
        /// already holds (#2900 row 6.19).
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
    }
}

#[cfg(test)]
mod learned_frame_cross_curvature_2933_f35_tests {
    use super::*;
    use crate::basis::SaeBasisEvaluator;
    use crate::manifold::{FaerEigh, Side};
    use std::sync::Arc;

    /// Rows, outputs and periodic basis width shared by both fixtures.
    const ROWS: usize = 24;
    const OUTPUTS: usize = 12;
    const BASIS: usize = 3;

    /// Periodic basis rows `[1, sin 2πt, cos 2πt]` at evenly spaced coordinates
    /// shifted by `phase`, with a rank-2 decoder whose nonzero rows `1` and `2`
    /// load on the output axes `axes`.
    fn periodic_atom(
        name: &str,
        phase: f64,
        axes: [usize; 2],
        loadings: [[f64; 2]; 2],
    ) -> (SaeManifoldAtom, Array2<f64>) {
        let evaluator = Arc::new(
            crate::basis::PeriodicHarmonicEvaluator::new(BASIS).expect("periodic basis"),
        );
        let coords =
            Array2::from_shape_fn((ROWS, 1), |(row, _)| (row as f64 + phase) / ROWS as f64);
        let (phi, jet) = evaluator.evaluate(coords.view()).expect("periodic jets");
        let mut decoder = Array2::<f64>::zeros((BASIS, OUTPUTS));
        for (basis_row, row_loadings) in loadings.iter().enumerate() {
            for (axis, loading) in axes.iter().zip(row_loadings.iter()) {
                decoder[[basis_row + 1, *axis]] = *loading;
            }
        }
        let atom = SaeManifoldAtom::new_with_provided_function_gram(
            name,
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            decoder,
            Array2::<f64>::eye(BASIS),
        )
        .expect("atom shapes agree")
        .with_basis_second_jet(evaluator);
        (atom, coords)
    }

    /// One periodic atom in `p = 12` outputs. The decoder starts in the span of
    /// output axes 0 and 1, so the fit activates a rank-2 frame there, but the
    /// target also carries a constant along axis 2. The circle's two weighted
    /// singular values (≈ 3.2 and 2.6) exceed the constant's (≈ 1.5), so the rank-2
    /// optimum keeps the circle and leaves the constant unfitted: the rank
    /// constraint binds and `∇_B L·U⊥ ≠ 0`.
    fn fitted_binding_circle() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
        let (atom, coords) =
            periodic_atom("binding_circle", 0.25, [0, 1], [[0.9, 0.2], [-0.1, 0.8]]);
        let mut target = atom.basis_values.dot(atom.decoder_coefficients());
        for row in 0..ROWS {
            let x = row as f64;
            target[[row, 0]] += 0.02 * (1.7 * x).sin();
            target[[row, 1]] += 0.02 * (1.3 * x).cos();
            target[[row, 2]] += 0.3;
        }
        let assignment = crate::assignment::SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((ROWS, 1)),
            vec![coords],
            vec![gam_terms::latent::LatentManifold::Circle { period: 1.0 }],
            crate::assignment::AssignmentMode::softmax(1.0),
        )
        .expect("assignment shapes agree");
        let mut term = SaeManifoldTerm::new(vec![atom], assignment).expect("term");
        let rho = SaeManifoldRho::new(
            0.0,
            0.8_f64.ln(),
            vec![Array1::from_vec(vec![250.0_f64.ln()])],
        );
        term.penalized_quasi_laplace_criterion_with_cache(
            target.view(),
            &rho,
            None,
            40,
            0.4,
            1.0e-6,
            1.0e-6,
        )
        .expect("the binding rank-2 circle converges");
        (term, target, rho)
    }

    /// Two framed periodic atoms whose decoders share output axis 1 (overlap
    /// `o ≈ 0.36`, below the repulsion gate), routed by `logits`. The target is the
    /// routed reconstruction plus a constant along axis 5 that neither frame spans.
    /// The state is not a fit, so the declared-gate comparison needs no converged
    /// root.
    fn routed_overlapping_circles(
        logits: Array2<f64>,
    ) -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
        let (first, first_coords) =
            periodic_atom("first", 0.25, [0, 1], [[0.9, 0.2], [-0.1, 0.8]]);
        let (second, second_coords) =
            periodic_atom("second", 0.6, [1, 2], [[0.6, 0.7], [-0.5, 0.6]]);
        let mut target = Array2::<f64>::zeros((ROWS, OUTPUTS));
        for row in 0..ROWS {
            let top = logits[[row, 0]].max(logits[[row, 1]]);
            let weights = [
                (logits[[row, 0]] - top).exp(),
                (logits[[row, 1]] - top).exp(),
            ];
            let total = weights[0] + weights[1];
            for (atom, weight) in [&first, &second].into_iter().zip(weights) {
                let decoded = atom.basis_values.row(row).dot(atom.decoder_coefficients());
                target
                    .row_mut(row)
                    .scaled_add(weight / total, &decoded);
            }
            target[[row, 5]] += 0.2;
        }
        let assignment = crate::assignment::SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits,
            vec![first_coords, second_coords],
            vec![
                gam_terms::latent::LatentManifold::Circle { period: 1.0 },
                gam_terms::latent::LatentManifold::Circle { period: 1.0 },
            ],
            crate::assignment::AssignmentMode::softmax(1.0),
        )
        .expect("assignment shapes agree");
        let mut term = SaeManifoldTerm::new(vec![first, second], assignment).expect("term");
        term.auto_activate_decoder_frames()
            .expect("frame activation at p = 12");
        let rho = SaeManifoldRho::new(
            0.0,
            0.8_f64.ln(),
            vec![
                Array1::from_vec(vec![250.0_f64.ln()]),
                Array1::from_vec(vec![250.0_f64.ln()]),
            ],
        );
        (term, target, rho)
    }

    /// One atom's chart of its rank-`r` decoder manifold at `B = C·Uᵀ`:
    /// `B(ξ) = (C + δC)·Q(W)ᵀ` with `Q(W) = (U + U⊥W)(I + WᵀW)^{−½}`, the polar
    /// retraction, exactly column-orthonormal because `UᵀU⊥ = 0`. Its differential
    /// at `ξ = 0` is the production lift `δC·Uᵀ + C·Wᵀ·U⊥ᵀ`, in the same
    /// `(vec δC, vec W)` layout.
    struct AtomChart {
        coordinates: Array2<f64>,
        frame: Array2<f64>,
        complement: Array2<f64>,
    }

    impl AtomChart {
        fn dim(&self) -> usize {
            let (m, r) = self.coordinates.dim();
            m * r + r * self.complement.ncols()
        }

        fn decoder(&self, xi: ArrayView1<'_, f64>) -> Array2<f64> {
            let (m, r) = self.coordinates.dim();
            let q = self.complement.ncols();
            let delta_c = Array2::from_shape_fn((m, r), |(b, j)| xi[b * r + j]);
            let w = Array2::from_shape_fn((q, r), |(i, j)| xi[m * r + i * r + j]);
            let gram = Array2::<f64>::eye(r) + &w.t().dot(&w);
            let (values, vectors) = gram.eigh(Side::Lower).expect("I + WᵀW is SPD");
            let inverse_sqrt = vectors
                .dot(&Array2::from_diag(&values.mapv(|value| 1.0 / value.sqrt())))
                .dot(&vectors.t());
            let rotated = (&self.frame + &self.complement.dot(&w)).dot(&inverse_sqrt);
            (&self.coordinates + &delta_c).dot(&rotated.t())
        }
    }

    /// Every atom's chart, concatenated in atom order as the production `ξ` layout.
    struct FixedRankChart {
        atoms: Vec<AtomChart>,
    }

    impl FixedRankChart {
        fn at(term: &SaeManifoldTerm) -> Self {
            let atoms = term
                .atoms
                .iter()
                .map(|atom| {
                    let frame = atom
                        .decoder_frame
                        .as_ref()
                        .expect("every atom carries a learned frame")
                        .frame()
                        .to_owned();
                    AtomChart {
                        coordinates: atom.decoder_coefficients().dot(&frame),
                        complement: orthonormal_frame_complement(frame.view())
                            .expect("frame complement"),
                        frame,
                    }
                })
                .collect();
            Self { atoms }
        }

        fn dim(&self) -> usize {
            self.atoms.iter().map(AtomChart::dim).sum()
        }

        fn starts(&self) -> Vec<usize> {
            self.atoms
                .iter()
                .scan(0usize, |cursor, atom| {
                    let start = *cursor;
                    *cursor += atom.dim();
                    Some(start)
                })
                .collect()
        }

        fn decoders(&self, xi: &Array1<f64>) -> Vec<Array2<f64>> {
            self.atoms
                .iter()
                .zip(self.starts())
                .map(|(atom, start)| atom.decoder(xi.slice(s![start..start + atom.dim()])))
                .collect()
        }
    }

    fn penalized_objective_on_chart(
        moving: &mut SaeManifoldTerm,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        chart: &FixedRankChart,
        xi: &Array1<f64>,
    ) -> f64 {
        for (atom, decoder) in moving.atoms.iter_mut().zip(chart.decoders(xi)) {
            atom.decoder_coefficients_mut().assign(&decoder);
        }
        moving
            .penalized_objective_total(target, rho, None, 1.0)
            .expect("the penalized objective evaluates along the chart")
    }

    /// Second differences of the penalized objective in the chart coordinates,
    /// with the latent coordinates, logits and gates held at the state. Returns the
    /// Hessian and the largest objective magnitude it read.
    fn chart_hessian(
        moving: &mut SaeManifoldTerm,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        chart: &FixedRankChart,
        step: f64,
    ) -> (Array2<f64>, f64) {
        let dim = chart.dim();
        let origin = Array1::<f64>::zeros(dim);
        let centre = penalized_objective_on_chart(moving, target, rho, chart, &origin);
        let mut largest = centre.abs();
        let mut hessian = Array2::<f64>::zeros((dim, dim));
        for i in 0..dim {
            let mut plus = origin.clone();
            plus[i] = step;
            let mut minus = origin.clone();
            minus[i] = -step;
            let f_plus = penalized_objective_on_chart(moving, target, rho, chart, &plus);
            let f_minus = penalized_objective_on_chart(moving, target, rho, chart, &minus);
            largest = largest.max(f_plus.abs()).max(f_minus.abs());
            hessian[[i, i]] = (f_plus - 2.0 * centre + f_minus) / (step * step);
            for j in 0..i {
                let mut sum = 0.0_f64;
                for (sign_i, sign_j) in [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
                    let mut corner = origin.clone();
                    corner[i] = sign_i * step;
                    corner[j] = sign_j * step;
                    let value = penalized_objective_on_chart(moving, target, rho, chart, &corner);
                    largest = largest.max(value.abs());
                    sum += sign_i * sign_j * value;
                }
                let entry = sum / (4.0 * step * step);
                hessian[[i, j]] = entry;
                hessian[[j, i]] = entry;
            }
        }
        (hessian, largest)
    }

    fn frobenius(matrix: &Array2<f64>) -> f64 {
        matrix.iter().map(|value| value * value).sum::<f64>().sqrt()
    }

    /// Central-difference step in chart units: decoder coordinates of order one
    /// and frame angles in radians.
    const CHART_ORACLE_STEP: f64 = 1.0e-2;

    /// The `(ξ, ξ)` block of [`SaeManifoldTerm::frame_marginal_information`] must be
    /// the chart Hessian of the penalized objective, with every latent coordinate,
    /// logit and collapse-prevention gate held at what `term` holds.
    ///
    /// The oracle differentiates the scalar penalized objective along the polar
    /// retraction chart of each atom's rank-2 manifold, on its own unframed clone
    /// that declares `term`'s gates, and shares no code with the exact Hessian
    /// applies or the tangent map. The chart's second-order terms are
    /// `δC·Wᵀ·U⊥ᵀ − ½·C·WᵀW·Uᵀ`, so its Hessian is `TᵀA_BB·T + E + Q`, where `Q` is
    /// `−I_{p−r} ⊗ sym((∇_B L·U)ᵀC)` on each atom's `W` block. Production omits `Q`
    /// because `∇_C L = ∇_B L·U` vanishes at a C-stationary point. The oracle adds
    /// it from the assembled gradient, so the comparison also holds away from a
    /// root.
    ///
    /// Central second differences are `O(h²)`, so the Richardson combination
    /// `(4·H(h/2) − H(h))/3` cancels the leading term and `‖H(h) − H(h/2)‖_F/3`
    /// bounds what remains. Each objective sums `n·p` data scalars, so it carries at
    /// most `n·p·ε·max|f|` of rounding, and each Hessian entry at step `h` at most
    /// four such errors over `h²`. The tolerance is the sum of the truncation and
    /// rounding bounds, with no tuned constant. With `cross_curvature_material`, the
    /// same block without `E` must miss the oracle by more than that tolerance, so
    /// the assertion is free to fail against an operator that omits the curvature.
    fn assert_tangent_information_is_the_chart_hessian(
        term: &SaeManifoldTerm,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        label: &str,
        cross_curvature_material: bool,
    ) {
        let chart = FixedRankChart::at(term);
        let fitted: Vec<Array2<f64>> = term
            .atoms
            .iter()
            .map(|atom| atom.decoder_coefficients().to_owned())
            .collect();
        for (k, (decoder, expected)) in chart
            .decoders(&Array1::zeros(chart.dim()))
            .iter()
            .zip(fitted.iter())
            .enumerate()
        {
            let miss = frobenius(&(decoder - expected));
            assert!(
                miss <= 1.0e-12 * frobenius(expected),
                "{label}: atom {k}'s chart must start at its decoder: miss {miss:.3e}"
            );
        }

        let information = term
            .frame_marginal_information(rho, target, None)
            .expect("frame-marginal information at the state");
        let total_t = information.total_t;
        let with_cross = information
            .joint
            .operator
            .slice(s![total_t.., total_t..])
            .to_owned();
        let cross_curvature = information.tangent.cross_curvature.clone();
        assert_eq!(with_cross.dim(), (chart.dim(), chart.dim()), "{label}: tangent layout");

        let mut moving = term.clone();
        for atom in moving.atoms.iter_mut() {
            atom.deactivate_decoder_frame();
        }
        moving.declare_collapse_prevention_gates(&term.collapse_prevention_gates());
        let gradient = moving
            .assemble_arrow_schur(target, rho, None)
            .expect("oracle assembly at the state")
            .gb
            .clone();
        let p = moving.output_dim();
        let beta_ranges = moving.beta_block_offsets();
        let mut chart_curvature = Array2::<f64>::zeros((chart.dim(), chart.dim()));
        let mut normal_gradient_sq = 0.0_f64;
        for ((atom, start), range) in chart
            .atoms
            .iter()
            .zip(chart.starts())
            .zip(beta_ranges.iter())
        {
            let (m, r) = atom.coordinates.dim();
            let decoder_gradient = Array2::from_shape_fn((m, p), |(b, c)| gradient[range.start + b * p + c]);
            let coupling = decoder_gradient.dot(&atom.frame).t().dot(&atom.coordinates);
            let symmetric = (&coupling + &coupling.t()) * 0.5;
            normal_gradient_sq += decoder_gradient
                .dot(&atom.complement)
                .iter()
                .map(|value| value * value)
                .sum::<f64>();
            for i in 0..atom.complement.ncols() {
                for j in 0..r {
                    for jj in 0..r {
                        chart_curvature[[start + m * r + i * r + j, start + m * r + i * r + jj]] =
                            -symmetric[[j, jj]];
                    }
                }
            }
        }

        let coarse_step = CHART_ORACLE_STEP;
        let fine_step = 0.5 * coarse_step;
        let (coarse, coarse_largest) = chart_hessian(&mut moving, target, rho, &chart, coarse_step);
        let (fine, fine_largest) = chart_hessian(&mut moving, target, rho, &chart, fine_step);
        let extrapolated = (&fine * 4.0 - &coarse) / 3.0;
        let truncation = frobenius(&(&coarse - &fine)) / 3.0;
        let evaluation_rounding =
            target.len() as f64 * f64::EPSILON * coarse_largest.max(fine_largest);
        let rounding = chart.dim() as f64
            * 4.0
            * evaluation_rounding
            * (4.0 / (fine_step * fine_step) + 1.0 / (coarse_step * coarse_step))
            / 3.0;
        let tolerance = truncation + rounding;
        let expected = &with_cross + &chart_curvature;
        let miss = frobenius(&(&extrapolated - &expected));
        let miss_without_cross = frobenius(&(&extrapolated - &(&expected - &cross_curvature)));
        eprintln!(
            "[#2933 F35 {label}] ‖∇_B L·U⊥‖_F = {:.3e}, ‖Q‖_F = {:.3e}, ‖H_chart‖_F = {:.3e}, \
             miss with E {miss:.3e}, miss without E {miss_without_cross:.3e}, tolerance \
             {tolerance:.3e} (truncation {truncation:.3e}, rounding {rounding:.3e})",
            normal_gradient_sq.sqrt(),
            frobenius(&chart_curvature),
            frobenius(&extrapolated)
        );
        assert!(
            miss <= tolerance,
            "{label}: the tangent information must be the chart Hessian of the penalized \
             objective: ‖H_chart − (TᵀAT + E + Q)‖_F = {miss:.3e} against tolerance \
             {tolerance:.3e}"
        );
        if cross_curvature_material {
            assert!(
                miss_without_cross > tolerance,
                "{label}: the normal gradient must make the cross curvature material: \
                 ‖H_chart − (TᵀAT + Q)‖_F = {miss_without_cross:.3e} against tolerance \
                 {tolerance:.3e}"
            );
        }
    }

    /// #2933 F35 — at a converged framed fit, the tangent information the
    /// frame-marginal covariance inverts is the Hessian of the penalized objective
    /// on the fixed-rank decoder manifold.
    ///
    /// The target adds a constant along an axis outside the initial frame span, but
    /// the converged fit leaves only `‖∇_B L·U⊥‖_F ≈ 1.6e-4` there (L6 job 1129126).
    /// Its cross curvature `E` is below this oracle's resolution, so this arm checks
    /// agreement at a root only. The declared-gate arm, whose routed state is not a
    /// root and has `‖∇_B L·U⊥‖_F ≈ 3.8`, requires `E` to be material.
    #[test]
    fn frame_tangent_information_is_the_fixed_rank_manifold_hessian_2933_f35() {
        let (term, target, rho) = fitted_binding_circle();
        let frame = term.atoms[0]
            .decoder_frame
            .as_ref()
            .expect("the fit must activate a Grassmann frame at p = 12");
        assert_eq!(frame.rank(), 2, "the frame must carry the circle's rank");
        assert_tangent_information_is_the_chart_hessian(
            &term,
            target.view(),
            &rho,
            "binding circle",
            false,
        );
    }

    /// #2933 F35 — the frame-integrated operator reads the collapse-prevention gates
    /// the term holds, not gates re-derived from the state. The outer objective
    /// declares the gates of its first priced root and holds them for every later
    /// drive, including the final uncertainty. An unframed clone starts with no
    /// gates, so an assembly that refreshed them would price the separation barrier
    /// with the routing of the state it is called at, while the criterion and the
    /// fixed-frame covariance read the declared routing.
    ///
    /// The gates are those refreshed at uniform routing. The state routes the first
    /// half of the rows mostly to the first atom and the rest to the second, so its
    /// own coactivation `q` differs and the barrier `−½·log det(Q ∘ O)` has
    /// different curvature along the overlapping decoders.
    #[test]
    fn frame_tangent_information_reads_the_declared_collapse_gates_2933_f35() {
        let (mut uniform, _, _) = routed_overlapping_circles(Array2::<f64>::zeros((ROWS, 2)));
        uniform.refresh_decoder_repulsion_gate();
        uniform.refresh_barrier_coactivation_gate();
        uniform.refresh_amplitude_barrier_gate();
        let declared = uniform.collapse_prevention_gates();

        let routing = Array2::from_shape_fn((ROWS, 2), |(row, atom)| {
            let first_half = row < ROWS / 2;
            if first_half == (atom == 0) { 1.0 } else { -1.0 }
        });
        let (mut term, target, rho) = routed_overlapping_circles(routing);
        for (k, atom) in term.atoms.iter().enumerate() {
            let frame = atom
                .decoder_frame
                .as_ref()
                .expect("both atoms activate a Grassmann frame at p = 12");
            assert_eq!(frame.rank(), 2, "atom {k}'s frame must carry its decoder's rank");
        }
        term.declare_collapse_prevention_gates(&declared);
        let mut refreshed = term.clone();
        refreshed.refresh_decoder_repulsion_gate();
        refreshed.refresh_barrier_coactivation_gate();
        refreshed.refresh_amplitude_barrier_gate();
        let at_state = refreshed.collapse_prevention_gates();
        eprintln!(
            "[#2933 F35 declared gates] declared {:?}; re-derived at the state {:?}",
            declared.barrier_coactivation, at_state.barrier_coactivation
        );
        assert!(
            declared.barrier_coactivation.is_some(),
            "the uniform routing must co-fire both atoms"
        );
        assert_ne!(
            declared.barrier_coactivation, at_state.barrier_coactivation,
            "the state's routing must re-derive a different separation-barrier gate"
        );
        assert_tangent_information_is_the_chart_hessian(
            &term,
            target.view(),
            &rho,
            "declared routing gates",
            true,
        );
    }
}
