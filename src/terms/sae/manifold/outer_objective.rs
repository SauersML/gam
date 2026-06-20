use super::*;

pub(crate) fn reconstruction_explained_variance(
    target: ArrayView2<'_, f64>,
    fitted: ArrayView2<'_, f64>,
) -> Option<f64> {
    if target.dim() != fitted.dim() {
        return None;
    }
    let (n, p) = target.dim();
    if n == 0 || p == 0 {
        return None;
    }
    let mut means = vec![0.0_f64; p];
    for col in 0..p {
        let mut acc = 0.0;
        for row in 0..n {
            acc += target[[row, col]];
        }
        means[col] = acc / n as f64;
    }
    let mut ssr = 0.0_f64;
    let mut sst = 0.0_f64;
    for row in 0..n {
        for col in 0..p {
            let residual = target[[row, col]] - fitted[[row, col]];
            ssr += residual * residual;
            let centered = target[[row, col]] - means[col];
            sst += centered * centered;
        }
    }
    if ssr.is_finite() && sst.is_finite() && sst > f64::MIN_POSITIVE {
        Some(1.0 - ssr / sst)
    } else {
        None
    }
}

/// #1207 — observable telemetry for the amortized warm-start (Design A). The
/// warm-start is advisory (a transient atlas-build / encode refusal must not
/// abort the criterion), so its failures were previously discarded with `.ok()`
/// and a silent cold solve was indistinguishable from a successful warm-start.
/// This counter makes the warm-start outcome verifiable: how many outer evals
/// attempted it, how many certified ≥1 row (a genuine warm-start), how many
/// certified ZERO rows (a full cold fallback — degenerate atlas), and how many
/// the warm-start path errored (logged, then cold). "Uses amortized warm-start"
/// is true exactly when `warm_started_evals > 0`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AmortizedWarmStartTelemetry {
    /// Outer evals that invoked the warm-start (gradient + value-probe lanes).
    pub attempts: usize,
    /// Evals where the amortized encoder certified ≥1 row → a real warm-start.
    pub warm_started_evals: usize,
    /// Evals where the encoder certified ZERO rows → a full cold fallback.
    pub cold_fallback_evals: usize,
    /// Evals where the warm-start path returned an error (logged, then cold).
    pub failed_evals: usize,
    /// Total certified (row, atom) coords warm-started across all evals.
    pub total_rows_warm_started: usize,
}

#[derive(Debug)]
pub struct SaeIntoFittedResult {
    pub term: SaeManifoldTerm,
    pub rho: SaeManifoldRho,
    pub loss: SaeManifoldLoss,
    /// True when the settled outer state was replaced by the re-solved seeded
    /// basin at the selected rho.
    pub used_seed_basin_fallback: bool,
    /// True when the pristine construction seed beat the returned state and was
    /// restored with its original rho.
    pub used_pristine_seed_fallback: bool,
    /// True when post-fit chart canonicalization changed any atom's chart.
    pub charts_canonicalized: bool,
}

impl SaeIntoFittedResult {
    pub fn invalidates_pre_final_shape_uncertainty(&self) -> bool {
        self.used_seed_basin_fallback
            || self.used_pristine_seed_fallback
            || self.charts_canonicalized
    }
}

impl AmortizedWarmStartTelemetry {
    /// Fold one warm-start outcome into the running tally. `Ok(rows)` with
    /// `rows > 0` is a genuine warm-start; `Ok(0)` is a degenerate-atlas cold
    /// fallback; `Err` is a (logged) failure that also proceeded cold.
    pub(crate) fn record(&mut self, outcome: &Result<usize, String>) {
        self.attempts += 1;
        match outcome {
            Ok(0) => self.cold_fallback_evals += 1,
            Ok(rows) => {
                self.warm_started_evals += 1;
                self.total_rows_warm_started += rows;
            }
            Err(_) => self.failed_evals += 1,
        }
    }
}

/// Outer REML objective for the SAE-manifold term.
///
/// Routes the SAE's smoothing hyperparameters ρ
/// (`log_lambda_sparse`, `log_lambda_smooth`, per-atom/axis `log_ard`)
/// through the *one* generic [`OuterObjective`] engine + cascade that the
/// main GAM REML path uses, instead of the SAE's deleted forked
/// `update_ard_reml` fixed-point rule. Each outer eval runs the inner
/// `(t, β)` arrow-Schur Newton solve at the engine's current ρ and returns
/// the true REML criterion (see [`SaeManifoldTerm::reml_criterion`]).
///
/// The SAE's outer coordinates ρ are all penalty-like / τ (precisions and
/// log-smoothing-strengths), so `psi_dim = 0`: there are no design-moving
/// (ψ) coordinates. No analytic outer gradient/Hessian is exposed yet
/// (task v2 wires the selected-inverse block-trace ρ-gradient), so this
/// is a cost-only objective and the engine routes it to a derivative-free /
/// central-difference outer strategy per the planner.
pub struct SaeManifoldOuterObjective {
    pub(crate) term: SaeManifoldTerm,
    /// Pristine term to restore from on `reset` (multi-start baseline).
    pub(crate) baseline_term: SaeManifoldTerm,
    pub(crate) target: Array2<f64>,
    pub(crate) registry: Option<AnalyticPenaltyRegistry>,
    /// ρ template carrying the per-atom ARD dims; `from_flat` reads its
    /// layout. Updated to each evaluated ρ so `into_fitted` can report the
    /// last ρ the engine settled on.
    pub(crate) current_rho: SaeManifoldRho,
    /// Pristine ρ to restore from on `reset`.
    pub(crate) baseline_rho: SaeManifoldRho,
    pub(crate) inner_max_iter: usize,
    pub(crate) learning_rate: f64,
    pub(crate) ridge_ext_coord: f64,
    pub(crate) ridge_beta: f64,
    /// Last inner loss breakdown observed (for `into_fitted`).
    pub(crate) last_loss: Option<SaeManifoldLoss>,
    /// Optional warm-start β slot. When the cache / continuation walk seeds a
    /// β, the next inner solve opens from it instead of cold.
    pub(crate) seeded_beta: Option<Array1<f64>>,
    /// #1207 — running tally of amortized warm-start outcomes, so a silent cold
    /// fallback is observable instead of hidden behind `.ok()`.
    pub(crate) warm_start_telemetry: AmortizedWarmStartTelemetry,
}

impl SaeManifoldOuterObjective {
    pub fn new(
        mut term: SaeManifoldTerm,
        target: Array2<f64>,
        registry: Option<AnalyticPenaltyRegistry>,
        init_rho: SaeManifoldRho,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
    ) -> Self {
        term.expected_evidence_gauge_deflated_directions = None;
        term.evidence_gauge_deflation_reanchors = 0;
        term.evidence_gauge_deflation_last_delta_sign = 0;
        term.dictionary_cocollapse_reseeds = 0;
        term.best_cocollapse_incumbent = None;
        let baseline_term = term.clone();
        let baseline_rho = init_rho.clone();
        Self {
            term,
            baseline_term,
            target,
            registry,
            current_rho: init_rho,
            baseline_rho,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
            last_loss: None,
            seeded_beta: None,
            warm_start_telemetry: AmortizedWarmStartTelemetry::default(),
        }
    }

    /// #1207 — the accumulated amortized warm-start telemetry. "Uses amortized
    /// warm-start" is verifiable as `telemetry.warm_started_evals > 0`; a silent
    /// cold solve shows up as `cold_fallback_evals` / `failed_evals`.
    pub fn warm_start_telemetry(&self) -> AmortizedWarmStartTelemetry {
        self.warm_start_telemetry
    }

    /// #1207 — record the outcome of one amortized warm-start attempt, logging a
    /// failure instead of silently swallowing it. The warm-start is advisory (a
    /// transient atlas/encode refusal must not abort the criterion), so the
    /// caller still proceeds cold — but the failure is now observable in both the
    /// telemetry tally and the log, never invisible.
    fn record_warm_start(&mut self, outcome: Result<usize, String>) {
        if let Err(err) = &outcome {
            log::debug!("[SAE/#1207] amortized warm-start fell back to a cold inner solve: {err}");
        }
        self.warm_start_telemetry.record(&outcome);
    }

    /// Consume the objective, returning the inner-fitted term, the last rho the
    /// engine evaluated, and the inner loss breakdown at that rho.
    pub fn into_fitted(self) -> SaeIntoFittedResult {
        let Self {
            term,
            mut baseline_term,
            target,
            registry,
            current_rho,
            baseline_rho,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
            last_loss,
            ..
        } = self;
        let pristine_seed_term = baseline_term.clone();
        let pristine_seed_rho = baseline_rho.clone();
        let mut fitted_rho = current_rho;
        let loss = last_loss.unwrap_or_else(|| SaeManifoldLoss {
            data_fit: 0.0,
            assignment_sparsity: 0.0,
            smoothness: 0.0,
            ard: 0.0,
            evidence_gauge_deflated_directions: 0,
        });
        // Basin guard against the multi-atom routing-collapse failure mode
        // (#629 #630). The outer ρ cascade mutates `term` cumulatively across
        // candidate ρ evaluations and never restores it between evals, so a
        // single ill-conditioned ρ poll can drag the per-row routing off the
        // decisive seed basin (the EM routing-seed / decoder-projection start)
        // into the near-uniform saddle. The settled `term` then reports that
        // collapsed routing even though the seed basin reconstructs the data
        // far better. `baseline_term` preserves the pristine seeded geometry;
        // re-solve the inner joint fit from it at the SAME settled ρ the engine
        // selected (smoothing choice is untouched) and keep the seed-basin state
        // when it wins either the penalized objective OR the reconstruction EV.
        // The EV tie-breaker is deliberately load-bearing for real activations:
        // a collapsed/rank-deficient outer walk can return a lower Laplace score
        // by pinning the quotient/curvature normalizer while losing the actual
        // reconstruction the SAE is fitted to provide.
        let settled_objective =
            term.penalized_objective_total(target.view(), &fitted_rho, registry.as_ref(), 1.0);
        let mut rho_seed = fitted_rho.clone();
        let seed_solve = match baseline_term.streaming_plan().admitted_or_error(
            baseline_term.n_obs(),
            baseline_term.output_dim(),
            baseline_term.k_atoms(),
        ) {
            Ok(plan)
                if plan.streaming
                    && plan.estimated_full_batch_bytes > plan.in_core_budget_bytes
                    && plan.estimated_dense_schur_bytes <= plan.in_core_budget_bytes =>
            {
                baseline_term.fit_streaming_in_memory(
                    target.view(),
                    &mut rho_seed,
                    registry.as_ref(),
                    inner_max_iter,
                    learning_rate,
                    ridge_ext_coord,
                    ridge_beta,
                )
            }
            Ok(_) => baseline_term.run_joint_fit_arrow_schur(
                target.view(),
                &mut rho_seed,
                registry.as_ref(),
                inner_max_iter,
                learning_rate,
                ridge_ext_coord,
                ridge_beta,
            ),
            Err(err) => Err(err),
        };
        let mut seed_won = false;
        if let (Ok(settled_total), Ok(_)) = (&settled_objective, &seed_solve) {
            let seed_total = baseline_term.penalized_objective_total(
                target.view(),
                &fitted_rho,
                registry.as_ref(),
                1.0,
            );
            if let Ok(seed_total) = seed_total {
                seed_won = seed_total.is_finite() && seed_total < *settled_total;
            }
            if !seed_won
                && let (Ok(seed_fit), Ok(settled_fit)) = (
                    baseline_term.try_fitted_for_rho(&fitted_rho),
                    term.try_fitted_for_rho(&fitted_rho),
                )
                && let (Some(seed_ev), Some(settled_ev)) = (
                    reconstruction_explained_variance(target.view(), seed_fit.view()),
                    reconstruction_explained_variance(target.view(), settled_fit.view()),
                )
            {
                seed_won = seed_ev.is_finite()
                    && settled_ev.is_finite()
                    && seed_ev >= SAE_FIT_DATA_COLLAPSE_EV_FLOOR
                    && settled_ev + SAE_FINAL_EV_DEGRADATION_TOL < seed_ev;
            }
        }
        let (mut fitted, mut fitted_loss) = if seed_won {
            let seed_loss = seed_solve.expect("seed_won implies seed_solve is Ok");
            (baseline_term, seed_loss)
        } else {
            (term, loss)
        };
        let mut pristine_seed_won = false;
        if let (Ok(seed_fit), Ok(returned_fit)) = (
            pristine_seed_term.try_fitted_for_rho(&pristine_seed_rho),
            fitted.try_fitted_for_rho(&fitted_rho),
        ) && let (Some(seed_ev), Some(returned_ev)) = (
            reconstruction_explained_variance(target.view(), seed_fit.view()),
            reconstruction_explained_variance(target.view(), returned_fit.view()),
        ) && seed_ev >= SAE_FIT_DATA_COLLAPSE_EV_FLOOR
            && returned_ev + SAE_FINAL_EV_DEGRADATION_TOL < seed_ev
            && let Ok(seed_loss) = pristine_seed_term.loss(target.view(), &pristine_seed_rho)
        {
            fitted = pristine_seed_term;
            fitted_rho = pristine_seed_rho;
            fitted_loss = seed_loss;
            pristine_seed_won = true;
        }
        // #1019 — the post-fit assembly seam: canonicalize every eligible
        // atom's chart to its canonical Diff(M) representative (arc length
        // for d = 1, minimum-isometry-defect flow for d = 2 torus atoms)
        // BEFORE the fitted term is handed to the payload / residual-gauge
        // certificate. Internally objective-gated and image-frozen (the
        // fitted state is restored verbatim on any failure or tolerance
        // breach), so the fit this returns is never degraded — an error here
        // is a refused canonicalization, not a broken fit.
        let pre_canonical_flags = fitted
            .atoms
            .iter()
            .map(|atom| atom.chart_canonicalized)
            .collect::<Vec<_>>();
        if let Err(err) =
            fitted.canonicalize_charts_post_fit(target.view(), &fitted_rho, registry.as_ref())
        {
            log::debug!("into_fitted: chart canonicalization refused: {err}");
        }
        let charts_canonicalized = fitted
            .atoms
            .iter()
            .zip(pre_canonical_flags.iter())
            .any(|(atom, before)| atom.chart_canonicalized != *before);
        if fitted.assignment.persist_resolved_ibp_alpha(&fitted_rho) {
            fitted_rho.log_lambda_sparse = 0.0;
        }
        SaeIntoFittedResult {
            term: fitted,
            rho: fitted_rho,
            loss: fitted_loss,
            used_seed_basin_fallback: seed_won,
            used_pristine_seed_fallback: pristine_seed_won,
            charts_canonicalized,
        }
    }

    /// First-order optimality certificate for this fit (#934).
    ///
    /// At the converged outer optimum `ρ̂` this runs the self-audit the desync
    /// bug genus (#752/#748/#808/#901) was always diagnosed by hand: it draws
    /// one deterministic direction `v` from the problem fingerprint, central-
    /// differences the criterion **value path** at `ρ̂ ± h v` (with a Richardson
    /// `2h` step for the FD's own error bar), and compares against the analytic
    /// directional derivative `∇V(ρ̂)·v` from the production gradient path. The
    /// returned [`CriterionCertificate`] records whether the objective and its
    /// analytic gradient agree *here*, on this data shape, where #901-class
    /// desyncs actually manifest.
    ///
    /// The numerical secant is the *audit instrument*, not an estimator: it
    /// only checks the production analytic gradient against the production value
    /// path at one point after convergence, so it is fully compatible with the
    /// exact-REML-only policy (see `sae_optimality_certificate`). Cost is four
    /// criterion value-path evaluations at the single final point.
    ///
    /// The value probes are taken on a **clone of the pristine baseline term**
    /// so the production fitted state is untouched and the value caches start
    /// cold — they must not alias the gradient path's converged warm state,
    /// since that aliasing is exactly what the certificate audits. Call before
    /// [`Self::into_fitted`].
    pub fn optimality_certificate(&mut self) -> Result<CriterionCertificate, String> {
        let rho_hat_flat = self.current_rho.to_flat();
        let dir = deterministic_probe_direction(rho_hat_flat.view());
        let h = probe_step(rho_hat_flat.view());

        // Analytic directional derivative at ρ̂, from the production gradient
        // path (same code the outer optimizer consumed). Re-forming the cache
        // here re-runs the inner solve at the settled ρ — already at its
        // optimum, so it converges immediately — and reads the exact analytic
        // outer gradient with the third-order correction included.
        let rho_hat = self.current_rho.clone();
        let (_v_hat, loss_hat, cache) = self.term.reml_criterion_with_cache(
            self.target.view(),
            &rho_hat,
            self.registry.as_ref(),
            self.inner_max_iter,
            self.learning_rate,
            self.ridge_ext_coord,
            self.ridge_beta,
        )?;
        let solver = self
            .term
            .outer_gradient_arrow_solver(&cache, rho_hat.lambda_smooth())?;
        let components = self.term.analytic_outer_rho_gradient_components(
            self.target.view(),
            &rho_hat,
            &loss_hat,
            &cache,
            &solver,
        )?;
        let grad = components.gradient();
        let grad_norm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        let analytic_directional: f64 = grad.iter().zip(dir.iter()).map(|(g, d)| g * d).sum();

        // Value-path probe on a cold clone of the pristine baseline term: the
        // value path must be exercised WITHOUT the gradient path's warm caches,
        // since aliasing the two is exactly the failure the certificate audits.
        let mut probe_term = self.baseline_term.clone();
        let value_at = |term: &mut SaeManifoldTerm, mult: f64| -> Result<f64, String> {
            let flat: Array1<f64> =
                Array1::from_shape_fn(rho_hat_flat.len(), |i| rho_hat_flat[i] + mult * h * dir[i]);
            let rho = self.baseline_rho.from_flat(flat.view());
            let (cost, _loss) = term.reml_criterion(
                self.target.view(),
                &rho,
                self.registry.as_ref(),
                self.inner_max_iter,
                self.learning_rate,
                self.ridge_ext_coord,
                self.ridge_beta,
            )?;
            Ok(cost)
        };
        let plus_h = value_at(&mut probe_term, 1.0)?;
        let minus_h = value_at(&mut probe_term, -1.0)?;
        let plus_2h = value_at(&mut probe_term, 2.0)?;
        let minus_2h = value_at(&mut probe_term, -2.0)?;

        let well_posed = plus_h.is_finite()
            && minus_h.is_finite()
            && plus_2h.is_finite()
            && minus_2h.is_finite();
        let samples = DirectionalSamples {
            plus_h,
            minus_h,
            plus_2h,
            minus_2h,
            step: h,
            grad_norm,
            analytic_directional,
            well_posed,
        };
        Ok(certificate_from_samples(&samples))
    }

    /// Posterior shape uncertainty of the fitted atoms — per-atom decoder
    /// covariance and ambient bands (see
    /// [`SaeManifoldTerm::assemble_shape_uncertainty`]).
    ///
    /// Recomputes the converged joint-Hessian Laplace factor at the settled ρ
    /// — the same undamped Direct factor the REML criterion forms at the inner
    /// optimum — and reads the per-atom covariance and bands off its cached
    /// Schur factor, scaling by the Gaussian reconstruction dispersion `φ̂`.
    /// The term is already at the optimum after the outer fit, so the inner
    /// re-solve converges immediately. Call before [`Self::into_fitted`].
    /// The most recent curvature-homotopy entry walk outcome on the live term
    /// (#1007), or `None` when no walk has run. Surfaced on the objective so the
    /// arrival / bifurcation / collapse outcome is observable without consuming
    /// the objective via [`Self::into_fitted`].
    pub fn curvature_walk_report(&self) -> Option<&CurvatureWalkReport> {
        self.term.curvature_walk_report()
    }

    pub fn decoder_shape_uncertainty(&mut self) -> Result<SaeShapeUncertainty, String> {
        let rho = self.current_rho.clone();
        let plan = self.term.streaming_plan().admitted_or_error(
            self.term.n_obs(),
            self.term.output_dim(),
            self.term.k_atoms(),
        )?;
        if !plan.direct_logdet_admitted() {
            let loss = self.term.loss(self.target.view(), &rho)?;
            let n_scalar = (self.term.n_obs().saturating_mul(self.term.output_dim())).max(1) as f64;
            let dispersion = (2.0 * loss.data_fit / n_scalar).max(f64::MIN_POSITIVE);
            return Ok(self
                .term
                .shape_uncertainty_without_decoder_covariance(dispersion));
        }
        let (_cost, loss, cache) = self.term.reml_criterion_with_cache(
            self.target.view(),
            &rho,
            self.registry.as_ref(),
            self.inner_max_iter,
            self.learning_rate,
            self.ridge_ext_coord,
            self.ridge_beta,
        )?;
        let dispersion = self.term.reconstruction_dispersion(&loss, &cache, &rho)?;
        self.term.assemble_shape_uncertainty(&cache, dispersion)
    }

    /// Certified curvature-homotopy entry walk (#1007): replace the blind
    /// multi-seed multistart with one predictor-corrector walk of the basis
    /// curvature dial `η` from the Eckart-Young anchor (`η = 0`, global by
    /// construction) to the full curved basis (`η = 1`).
    ///
    /// 1. **Anchor (`η = 0`).** The curved columns are suppressed, so the decoder
    ///    sub-problem is convex and its optimum is the Eckart-Young projection
    ///    certified by [`linear_span_anchor`]; the joint corrector lands on it. A
    ///    degenerate anchor (no recoverable linear span / a non-finite target /
    ///    a failed relaxation solve) returns `Ok(false)` — the caller falls back
    ///    to the cascade.
    /// 2. **Walk `η: 0 → 1`.** Each waypoint: a *predictor* applies the IFT step
    ///    `Δβ = −H⁻¹ · ∂g_β/∂η · Δη` on the cached evidence factor
    ///    ([`ArrowFactorCache::full_inverse_apply`], β-channel; the t / gate
    ///    blocks are re-converged by the corrector), then the *corrector* (the
    ///    damped joint Newton in `reml_criterion_with_cache`) re-converges at
    ///    `η_next`. The invariant is that the arrow factor's smallest pivot stays
    ///    at or above the safe-SPD floor `√eps · max(diag_scale, 1)`; when it
    ///    shrinks the `η` step is halved and retried from the last converged
    ///    state. A pivot collapse at the minimum step is a DETECTED bifurcation
    ///    (recorded on [`CurvatureWalkReport`], never silent) and returns
    ///    `Ok(false)`.
    /// 3. **Arrival (`η = 1`).** The term is left warm at the certified branch's
    ///    `η = 1` solution; the report is recorded and the call returns
    ///    `Ok(true)`.
    ///
    /// The direct helper walks at the construction entry ρ (`baseline_rho`);
    /// the outer seed loop uses `run_curvature_homotopy_entry_at_rho` so every
    /// generated candidate gets its own entry solve before the ρ-anneal.
    pub fn run_curvature_homotopy_entry(&mut self) -> Result<bool, String> {
        let rho = self.baseline_rho.clone();
        self.run_curvature_homotopy_entry_at_rho(&rho)
    }

    /// Certified curvature-homotopy entry walk at an explicit seed ρ. The outer
    /// seed loop calls this form so each generated candidate lands on its own
    /// fixed baseline instead of every walk reusing the construction baseline.
    pub fn run_curvature_homotopy_entry_at_rho(
        &mut self,
        rho: &SaeManifoldRho,
    ) -> Result<bool, String> {
        let rho = rho.clone();
        self.current_rho = rho.clone();
        let isometry_targets = self
            .registry
            .as_ref()
            .map(AnalyticPenaltyRegistry::isometry_scalar_weights)
            .unwrap_or_default();
        self.set_isometry_homotopy_weight(0.0, &isometry_targets);
        // Eckart-Young anchor certificate at η = 0 (output-subspace coords). A
        // degenerate anchor is the cascade's job, not the walk's.
        let anchor = match linear_span_anchor(&self.term, self.target.view()) {
            Ok(anchor) => anchor,
            Err(err) => {
                log::info!(
                    "[#1007] curvature anchor degenerate ({err}); deferring to seed cascade"
                );
                self.set_isometry_homotopy_weight(1.0, &isometry_targets);
                return Ok(false);
            }
        };
        let anchor_residual_norm_sq = anchor.residual_norm_sq;

        // Anchor corrector at η = 0: the convex linear relaxation.
        let (_loss0, mut last_cache) = match self.solve_at_eta(&rho, 0.0, &isometry_targets) {
            Ok(pair) => pair,
            Err(err) => {
                log::info!(
                    "[#1007] curvature anchor solve failed at η=0 ({err}); deferring to cascade"
                );
                self.term.set_homotopy_eta(1.0).ok();
                self.set_isometry_homotopy_weight(1.0, &isometry_targets);
                return Ok(false);
            }
        };

        let mut eta = 0.0_f64;
        let mut eta_step = CURVATURE_WALK_INITIAL_ETA_STEP;
        let mut eta_steps = 0usize;
        let mut step_halvings = 0usize;
        let mut total_correctors = 0usize;
        let mut bifurcation: Option<CurvatureBifurcation> = None;

        // Identity-homotopy shortcut: with no curved basis columns anywhere
        // AND an all-zero isometry ramp, `solve_at_eta` poses the SAME problem
        // at every η — the grid legs after the anchor corrector would re-solve
        // its converged state verbatim, paying a full criterion/factorization
        // rebuild each time. The anchor + first corrector carry all the value
        // (certified Eckart-Young initialization + one full solve); arrive at
        // η = 1 directly. `set_homotopy_eta(1.0)` restores the plain-evaluate
        // fast path (η == 1 skips the dialed evaluator); the isometry weights
        // are already at target because every ramp target is zero.
        if isometry_targets.iter().all(|&target| target == 0.0)
            && self.term.curvature_homotopy_eta_is_inert()?
        {
            self.term.set_homotopy_eta(1.0)?;
            eta = 1.0;
        }

        'walk: while eta < 1.0 {
            let eta_next = (eta + eta_step).min(1.0);
            let d_eta = eta_next - eta;

            // Predictor: IFT step on the cached factor warm-starts the corrector
            // (β-channel only; `w_t = 0`). Non-fatal — on any predictor failure
            // the corrector simply opens from the previous η's converged β.
            if let Ok(dg_beta) = self
                .term
                .curvature_beta_gradient_eta_derivative(self.target.view(), &rho)
                && dg_beta.len() == last_cache.k
            {
                let w_t = Array1::<f64>::zeros(last_cache.delta_t_len());
                if let Ok((_u_t, u_beta)) =
                    last_cache.full_inverse_apply(w_t.view(), dg_beta.view())
                {
                    let mut beta = self.term.flatten_beta();
                    if beta.len() == u_beta.len() {
                        for (b, u) in beta.iter_mut().zip(u_beta.iter()) {
                            *b -= u * d_eta;
                        }
                        if beta.iter().all(|v| v.is_finite()) {
                            self.term.set_flat_beta(beta.view()).ok();
                        }
                    }
                }
            }

            // Corrector at η_next.
            let cache = match self.solve_at_eta(&rho, eta_next, &isometry_targets) {
                Ok((_loss, cache)) => cache,
                Err(err) => {
                    // Corrector struggled: treat like a pivot shrink — halve the
                    // η step and retry from the last converged state. A failure
                    // at the minimum step is a branch bifurcation.
                    if eta_step <= CURVATURE_WALK_MIN_ETA_STEP {
                        log::info!(
                            "[#1007] curvature corrector failed at η={eta_next:.4} at the minimum \
                             η-step ({err}); recording branch bifurcation"
                        );
                        bifurcation = Some(CurvatureBifurcation {
                            eta: eta_next,
                            min_pivot: 0.0,
                        });
                        break 'walk;
                    }
                    eta_step *= 0.5;
                    step_halvings += 1;
                    self.term.set_homotopy_eta(eta).ok();
                    self.set_isometry_homotopy_weight(eta, &isometry_targets);
                    continue 'walk;
                }
            };
            total_correctors += 1;

            // Pivot invariant: min pivot ≥ eps · diag_scale, measured ON THE
            // GAUGE QUOTIENT (#1095). The floor uses machine epsilon (not its
            // square root) because the undamped cache is built with
            // `with_ill_conditioning_tolerated()`, which accepts any
            // positive-definite factor regardless of condition number.
            // Sub-sqrt(eps) pivots are legitimately produced when N < beta_dim
            // (small-N fits where the decoder Gram is rank-deficient) — this
            // is NOT a branch bifurcation: the damped corrector already
            // converged above, and genuine branch collapses are caught by
            // corrector failure (the `Err` branch). Only a pivot numerically
            // indistinguishable from zero in double precision (below
            // eps * diag_scale) marks a true collapse of the smooth branch.
            //
            // A closed-form gauge null (affine chart freedom, circle rotation)
            // is constant along the η-walk, so it can never signal a branch
            // bifurcation; only a NON-gauge, data-supported pivot collapse can.
            // `outer_gradient_arrow_solver` succeeds iff the sub-floor pivots
            // are explained by gauge/null directions (Faddeev-Popov deflation)
            // and errs honestly otherwise, which is exactly the verdict needed.
            let pivot = arrow_factor_min_pivot(&cache).min_pivot.unwrap_or(0.0);
            let diag_scale = arrow_factor_max_pivot(&cache).unwrap_or(1.0);
            let floor = f64::EPSILON * diag_scale;
            let pivot_deficit_is_gauge = !(pivot.is_finite() && pivot >= floor)
                && self
                    .term
                    .outer_gradient_arrow_solver(&cache, rho.lambda_smooth())
                    .is_ok();
            if !(pivot.is_finite() && pivot >= floor) && !pivot_deficit_is_gauge {
                if eta_step > CURVATURE_WALK_MIN_ETA_STEP {
                    eta_step *= 0.5;
                    step_halvings += 1;
                    self.term.set_homotopy_eta(eta).ok();
                    self.set_isometry_homotopy_weight(eta, &isometry_targets);
                    continue 'walk;
                }
                log::info!(
                    "[#1007] curvature branch bifurcation at η={eta_next:.4}: min pivot \
                     {pivot:.3e} < floor {floor:.3e}; deferring to seed cascade"
                );
                bifurcation = Some(CurvatureBifurcation {
                    eta: eta_next,
                    min_pivot: pivot,
                });
                break 'walk;
            }

            // Accepted waypoint: advance and gently regrow the step toward the
            // nominal cadence (a clean stretch should not stay throttled).
            eta = eta_next;
            last_cache = cache;
            eta_steps += 1;
            eta_step = (eta_step * 2.0).min(CURVATURE_WALK_INITIAL_ETA_STEP);
            if total_correctors >= CURVATURE_WALK_MAX_CORRECTORS && eta < 1.0 {
                log::info!(
                    "[#1007] curvature walk hit its corrector budget at η={eta:.4}; deferring to \
                     seed cascade"
                );
                bifurcation = Some(CurvatureBifurcation {
                    eta,
                    min_pivot: pivot,
                });
                break 'walk;
            }
        }

        let mut arrived = bifurcation.is_none() && eta >= 1.0;
        // Leave the term at the real (η = 1) objective regardless of outcome so
        // an aborted walk hands the cascade the full basis.
        if !arrived {
            self.term.set_homotopy_eta(1.0).ok();
        }
        self.set_isometry_homotopy_weight(1.0, &isometry_targets);
        if arrived
            && let Ok(before_fit) = self.term.try_fitted_for_rho(&rho)
            && let Some(before_ev) =
                reconstruction_explained_variance(self.target.view(), before_fit.view())
            && before_ev < 0.9
        {
            let snapshot = self.term.snapshot_mutable_state();
            let accepted_polish = self
                .term
                .refit_decoder_least_squares_at_current_state(self.target.view(), Some(&rho))
                .and_then(|()| {
                    self.term
                        .seed_coords_by_decoder_projection(self.target.view(), 256)
                })
                .and_then(|()| {
                    self.term.refit_decoder_least_squares_at_current_state(
                        self.target.view(),
                        Some(&rho),
                    )
                })
                .and_then(|()| {
                    let after_fit = self.term.try_fitted_for_rho(&rho)?;
                    let Some(after_ev) =
                        reconstruction_explained_variance(self.target.view(), after_fit.view())
                    else {
                        return Err(
                            "curvature-homotopy decoder LSQ polish produced no EV".to_string()
                        );
                    };
                    if after_ev > before_ev {
                        self.term.loss(self.target.view(), &rho)
                    } else {
                        Err(format!(
                            "curvature-homotopy decoder LSQ polish refused: EV {after_ev:.6} \
                             did not improve from {before_ev:.6}"
                        ))
                    }
                });
            match accepted_polish {
                Ok(loss) => self.last_loss = Some(loss),
                Err(_) => self.term.restore_mutable_state(&snapshot),
            }
        }
        // Arrival quality floor (#1117). "Arrived" is only a usable certificate
        // if the η = 1 reconstruction is actually good — the predictor-corrector
        // walk from the Eckart-Young LINEAR anchor can track into a degenerate
        // basin that is stationary on the gauge/decoder-null quotient (so the
        // inner solve legitimately converges there) yet reconstructs the data
        // badly (a NEGATIVE explained variance: worse than the data mean). For a
        // K = 1 PERIODIC (circle) atom the linear anchor IS that wrong basin — a
        // straight chord through the arc — and neither the IFT predictor nor the
        // decoder-LSQ polish (which alternates a decoder LSQ with a coordinate
        // re-projection ONTO that same bad decoder) can escape it: it is a fixed
        // point. The walk then reported `arrived = true` on EV = -0.59.
        //
        // Crucially, the production outer objective can carry `inner_max_iter = 0`
        // (a value-only / frozen-inner configuration), so neither the cascade
        // `eval` NOR `into_fitted`'s basin re-solve runs a real joint Newton fit
        // — only the homotopy + polish produce any fit, and they are stuck on the
        // linear anchor. Demoting to a bifurcation alone therefore does NOT
        // recover the circle (the cascade re-freezes at the cold seed). So the
        // recovery itself must run a REAL bounded joint Newton fit from the
        // pristine baseline term (which carries the circle-aware PCA seed the
        // cold path recovers EV ≈ 0.94 from), with a nonzero budget independent
        // of the objective's frozen `inner_max_iter`. If that recovers a good
        // reconstruction we adopt it (the walk genuinely arrives on the curved
        // branch); otherwise we demote to a recorded bifurcation so the cascade
        // takes over from the pristine baseline. A genuinely good arrival (the
        // common case, every fit already passing) never enters this block.
        // #1189 — the arrival floor is RELATIVE to the certified linear ceiling.
        // The absolute `CURVATURE_WALK_ARRIVAL_EV_FLOOR` is calibrated for planted
        // synthetic harmonics (a real circle reconstructs at EV ≳ 0.9); on real
        // high-dim data whose signal sits on a long-tailed spectrum the best
        // achievable EV at K atoms is bounded by the cumulative linear (PCA)
        // ceiling — which is well under 0.5 on real LLM activations — so the
        // absolute floor rejected EVERY genuine arrival, the fit fell to the blind
        // cascade, and the cascade collapsed to the `1e12` data-collapse sentinel.
        // The Eckart-Young anchor's OWN reconstruction EV is exactly that
        // achievable linear ceiling (`anchor_ev = 1 − ‖residual‖² / SST`); a
        // curved arrival that recovers at least `CURVATURE_WALK_ARRIVAL_ANCHOR_
        // FRACTION` of it has, by construction, NOT tracked into a worse basin
        // than the convex linear optimum it started on. The effective floor is the
        // MIN of the absolute floor and this fraction of the anchor ceiling: on
        // synthetic data the absolute floor still binds (and the curved fit clears
        // both); on real data it relaxes to the data-achievable ceiling.
        let target_sst = {
            let (n, p) = self.target.dim();
            let mut means = vec![0.0_f64; p];
            for col in 0..p {
                let mut acc = 0.0;
                for row in 0..n {
                    acc += self.target[[row, col]];
                }
                means[col] = acc / (n.max(1) as f64);
            }
            let mut sst = 0.0_f64;
            for row in 0..n {
                for col in 0..p {
                    let centered = self.target[[row, col]] - means[col];
                    sst += centered * centered;
                }
            }
            sst
        };
        let anchor_ev = if target_sst > f64::MIN_POSITIVE && anchor_residual_norm_sq.is_finite() {
            1.0 - anchor_residual_norm_sq / target_sst
        } else {
            // No usable ceiling estimate: fall back to the absolute floor.
            CURVATURE_WALK_ARRIVAL_EV_FLOOR
        };
        // The relative floor relaxes the absolute one DOWN to the data-achievable
        // ceiling, but never BELOW the data-collapse floor the sentinel itself
        // keys on: a fit at or under `SAE_FIT_DATA_COLLAPSE_EV_FLOOR` is genuinely
        // degenerate (worse than a constant predictor would warrant) and must
        // always route to recovery, even when a pathological anchor estimate
        // (negative / near-zero `anchor_ev`) would otherwise drop the floor to
        // zero and wave the collapse through.
        // #1026 — the relative floor must also be PER-ATOM-SHARE aware. The
        // anchor's certified ceiling `anchor_ev` is the CUMULATIVE Eckart-Young EV
        // across ALL K atoms (sequential residual deflation in `linear_span_anchor`).
        // In a K-atom dictionary each atom carries on average ~1/K of the captured
        // variance, so a single atom that curves trades at most ~one atom's share of
        // the linear ceiling for the geometry it gains. Holding the whole-dictionary
        // curved EV to `0.9 * anchor_ev` (the full linear ceiling) therefore rejects
        // genuine arrivals whenever the curved fit recovers slightly less than the
        // cumulative linear optimum -- the K >= 2 co-collapse signature: the walk
        // reaches a real curved branch (EV = 0.2461 / 0.29 / 0.37 on real OLMo) but
        // `0.9 * anchor_ev` demotes it to a bifurcation and the cascade then
        // collapses. The principled floor discounts the linear ceiling by exactly one
        // atom's share: a curved K-atom fit need only stay within `1/K` of the linear
        // ceiling (`anchor_ev * (K - 1)/K`), never within a flat 10%. At K = 1 the
        // share floor is 0 (a single curved atom is judged only against the absolute /
        // data-collapse floors -- it already beats its linear counterpart); as K grows
        // it climbs back toward `anchor_ev`, still capped by the absolute
        // `0.9 * anchor_ev` so a large dictionary is never held above the linear
        // optimum it relaxed from.
        // The base #1189 floor (absolute vs. the certified linear-ceiling
        // fraction) governs the SINGLE-atom regime, where there is no second atom
        // to collapse onto and the share concept is undefined. For K >= 2 it is
        // ADDITIONALLY relaxed by the per-atom share so a curved K-atom fit that
        // recovers within `1/K` of the cumulative ceiling is accepted; K = 1 keeps
        // the original gate exactly (no co-collapse to forgive there).
        let k_active = self.term.k_atoms().max(1);
        let base_floor = CURVATURE_WALK_ARRIVAL_EV_FLOOR
            .min(CURVATURE_WALK_ARRIVAL_ANCHOR_FRACTION * anchor_ev)
            .max(SAE_FIT_DATA_COLLAPSE_EV_FLOOR);
        let arrival_floor = if k_active >= 2 {
            let k = k_active as f64;
            let per_atom_share_floor = anchor_ev * ((k - 1.0) / k);
            base_floor
                .min(per_atom_share_floor.max(0.0))
                .max(SAE_FIT_DATA_COLLAPSE_EV_FLOOR)
        } else {
            base_floor
        };
        if arrived
            && let Ok(final_fit) = self.term.try_fitted_for_rho(&rho)
            && let Some(final_ev) =
                reconstruction_explained_variance(self.target.view(), final_fit.view())
            && final_ev < arrival_floor
        {
            log::info!(
                "[#1007/#1189] curvature walk reached η=1 but the reconstruction is degenerate \
                 (EV={final_ev:.4} < arrival floor {arrival_floor:.4} = min(abs \
                 {CURVATURE_WALK_ARRIVAL_EV_FLOOR}, {CURVATURE_WALK_ARRIVAL_ANCHOR_FRACTION} × \
                 anchor ceiling {anchor_ev:.4})); running a \
                 bounded joint Newton fit from the pristine seed to recover the curved branch"
            );
            // Real joint Newton fit from the pristine baseline (circle-aware
            // seed), at the full η = 1 basis, with a budget that does NOT collapse
            // to the objective's frozen `inner_max_iter`.
            let recovery_iters = self.inner_max_iter.max(CURVATURE_WALK_RECOVERY_INNER_ITERS);
            let mut recovered_term = self.baseline_term.clone();
            recovered_term.set_homotopy_eta(1.0).ok();
            let mut recovery_rho = rho.clone();
            let recovery_fit = recovered_term.run_joint_fit_arrow_schur(
                self.target.view(),
                &mut recovery_rho,
                self.registry.as_ref(),
                recovery_iters,
                self.learning_rate,
                self.ridge_ext_coord,
                self.ridge_beta,
            );
            let recovered_ev = recovery_fit.as_ref().ok().and_then(|_| {
                recovered_term
                    .try_fitted_for_rho(&rho)
                    .ok()
                    .and_then(|fit| {
                        reconstruction_explained_variance(self.target.view(), fit.view())
                    })
            });
            match (recovery_fit, recovered_ev) {
                (Ok(loss), Some(ev)) if ev > final_ev && ev >= arrival_floor => {
                    // The bounded joint fit found the curved branch: adopt it and
                    // keep `arrived = true` (the walk delivered a usable fit).
                    log::info!(
                        "[#1007] curvature degenerate-basin recovery succeeded \
                         (EV {final_ev:.4} -> {ev:.4}); adopting the recovered curved branch"
                    );
                    self.term = recovered_term;
                    self.current_rho = rho.clone();
                    self.last_loss = Some(loss);
                }
                _ => {
                    // Recovery could not improve the reconstruction: demote to a
                    // recorded bifurcation so the outer seed loop resets to the
                    // pristine baseline and the documented cascade takes over.
                    log::info!(
                        "[#1007] curvature degenerate-basin recovery did not clear the arrival \
                         floor (EV stayed {final_ev:.4}); demoting to a branch bifurcation"
                    );
                    arrived = false;
                    self.term.set_homotopy_eta(1.0).ok();
                    if bifurcation.is_none() {
                        bifurcation = Some(CurvatureBifurcation {
                            eta: 1.0,
                            min_pivot: 0.0,
                        });
                    }
                }
            }
        }
        let collapse_events = self.term.collapse_events().len();
        self.term.set_curvature_walk_report(CurvatureWalkReport {
            arrived,
            anchor_residual_norm_sq,
            bifurcation,
            eta_steps,
            step_halvings,
            collapse_events,
            reseeds: 0,
        });
        Ok(arrived)
    }

    /// Curvature-homotopy corrector (#1007): install the `η` dial and re-converge
    /// the joint fit at the entry ρ, returning the converged loss and the
    /// undamped evidence cache (for the predictor IFT solve + the pivot
    /// invariant). The dial is read on the next basis refresh inside the solve.
    pub(crate) fn solve_at_eta(
        &mut self,
        rho: &SaeManifoldRho,
        eta: f64,
        isometry_targets: &[f64],
    ) -> Result<(SaeManifoldLoss, ArrowFactorCache), String> {
        self.term.set_homotopy_eta(eta)?;
        self.set_isometry_homotopy_weight(eta, isometry_targets);
        let (_cost, loss, cache) = self.term.reml_criterion_with_cache(
            self.target.view(),
            rho,
            self.registry.as_ref(),
            self.inner_max_iter,
            self.learning_rate,
            self.ridge_ext_coord,
            self.ridge_beta,
        )?;
        self.last_loss = Some(loss.clone());
        Ok((loss, cache))
    }

    pub(crate) fn set_isometry_homotopy_weight(&mut self, eta: f64, targets: &[f64]) {
        if targets.is_empty() {
            return;
        }
        if let Some(registry) = self.registry.as_mut() {
            let eta = eta.clamp(0.0, 1.0);
            let weights: Vec<f64> = targets.iter().map(|target| eta * target).collect();
            registry.set_isometry_scalar_weights(&weights);
        }
    }

    pub(crate) fn add_fit_data_collapse_penalty(
        &mut self,
        cost: f64,
        rho: &SaeManifoldRho,
    ) -> Result<f64, String> {
        let fitted = self.term.try_fitted_for_rho(rho)?;
        let assignments = self.term.assignment.assignments_for_rho(rho)?;
        let collapsed = self.term.record_fit_data_collapse_if_needed(
            self.target.view(),
            fitted.view(),
            assignments.view(),
            self.inner_max_iter,
        )?;
        // #1217 — FINITE-GUARD the outer-REML cost handed to the BFGS line search
        // (same class as the #1192 Poisson Armijo finite-floor). At a K≥2
        // co-collapse cliff the Laplace normalizer `½log|H|` of a numerically
        // singular joint Hessian can return `±∞`/`NaN`, so `reml_criterion`
        // surfaces a non-finite `v` even when the discrete `record_fit_data_
        // collapse_if_needed` detector has not (yet) classified the iterate as a
        // collapse. The `opt` BFGS reports a non-finite probe as "Line search
        // failed (nonfinite seen)" and ABORTS the whole outer solve at the
        // current iterate (observed on real OLMo K=2: stall at iter 2, |g|≈65),
        // rather than treating it as an infeasible step to backtrack from. A
        // non-finite criterion is the STRONGEST infeasibility signal — it must
        // present to the line search as the SAME finite wall a detected collapse
        // does, so the search rejects the step and backtracks toward the feasible
        // basin instead of giving up. We therefore floor any non-finite cost to
        // the finite collapse wall `SAE_FIT_DATA_COLLAPSE_COST` (a rejectable
        // barrier, not `∞`), then add the collapse penalty on top of the already
        // finite base. The non-collapsed, finite-cost path is byte-for-byte
        // unchanged.
        let base = if cost.is_finite() {
            cost
        } else {
            SAE_FIT_DATA_COLLAPSE_COST
        };
        if collapsed {
            // The wall is itself finite, so a non-finite base already floored to
            // the wall cannot be pushed back to `∞`; clamp the sum defensively in
            // case the (finite) REML base is itself near `f64::MAX`.
            Ok((base + SAE_FIT_DATA_COLLAPSE_COST).min(2.0 * SAE_FIT_DATA_COLLAPSE_COST))
        } else {
            Ok(base)
        }
    }

    pub(crate) fn is_recoverable_value_probe_refusal(err: &str) -> bool {
        err.contains("inner solve did not converge at fixed ρ")
            || err.contains(
                "undamped evidence factorization hit a non-PD per-row H_tt block before KKT",
            )
    }

    /// Shared cost path: evaluate the REML criterion at `rho_flat`, updating
    /// the cached ρ / loss and (optionally) priming the inner solve from a
    /// seeded β. Returns `(cost, β̂)`.
    ///
    /// `refine_progress_extension = false` selects the value-probe refine
    /// budget (#1029). The budget cut keeps the SAME KKT/step tolerance as the
    /// full path — a successfully returned value is converged to the identical
    /// stationarity measure, so probe values and accepted-point values are
    /// always comparable; only an expensive grind-then-refuse becomes a cheap
    /// refusal (a recoverable line-search reject).
    pub(crate) fn evaluate_with_refine_policy(
        &mut self,
        rho_flat: ArrayView1<'_, f64>,
        refine_progress_extension: bool,
        fold_cotrain: bool,
    ) -> Result<(f64, Array1<f64>), String> {
        let rho = self.baseline_rho.from_flat(rho_flat);
        if let Some(beta) = self.seeded_beta.take() {
            // Warm-start the inner decoder coefficients before the solve.
            if beta.len() == self.term.beta_dim() {
                self.term.set_flat_beta(beta.view())?;
            }
        }
        // #1154 item 2 (Design A) — warm-start the inner latent coords from the
        // amortized encoder built on the CURRENT dictionary. At outer step m this
        // seeds the inner solve from the per-chart IFT predictor of the dictionary
        // settled at step m−1, refined to the SAME stationary point (so the REML
        // λ-gradient is untouched). Best-effort: a first-build / degenerate atlas
        // certifies no rows and warm-starts nothing, leaving the cold path
        // byte-for-byte unchanged; a transient atlas-build refusal must not abort
        // the criterion evaluation, so the warm-start is advisory only. #1207 —
        // the outcome is recorded (and a failure logged) so the cold fallback is
        // observable, never silently swallowed.
        let warm_start_outcome = self
            .term
            .warm_start_latents_from_amortized_encoder(self.target.view(), &rho);
        self.record_warm_start(warm_start_outcome);
        let (reml_cost, loss) = self.term.reml_criterion_with_refine_policy(
            self.target.view(),
            &rho,
            self.registry.as_ref(),
            self.inner_max_iter,
            self.learning_rate,
            self.ridge_ext_coord,
            self.ridge_beta,
            refine_progress_extension,
        )?;
        let beta_hat = self.term.flatten_beta();
        // #1154 — co-train the amortized encoder with the dictionary + λ: rank ρ
        // by the REML criterion PLUS the amortized-encoder consistency penalty,
        // so the derivative-free outer cascade settles on a dictionary the cheap
        // one-mat-vec encoder can faithfully and certifiably invert. The inner
        // solve already converged to stationarity above, so the consistency fold
        // is evaluated at the exact fitted dictionary and the REML λ-gradient is
        // untouched (Design A).
        //
        // #1224 — the fold `c(ρ)` has NO analytic gradient, so it may only be
        // carried where the cost is never compared against the REML gradient
        // `∇f`. `fold_cotrain == true` is the CROSS-SEED RANKING / EFS lane
        // (`eval_cost`): a value-only screen across independent seeds and final
        // selection, where no `∇f` is in play, so `f+c` is the right ranking
        // criterion. `fold_cotrain == false` is the BFGS / ARC LINE-SEARCH lane
        // (`eval_with_order(Value)`): those probes accept/reject steps whose
        // search DIRECTION came from `eval`'s `∇f`, so a sufficient-decrease test
        // on `f+c` paired with `∇f` mixes two functions and can stall or wander
        // (the objective↔gradient desync class, #931/#1206). The line search must
        // see the SAME pure REML `f` the gradient lane (`eval`) reports. Either
        // way the discrete collapse barrier stays on both lanes (BFGS rejects
        // steps into it — the intended infeasibility wall, not a smooth fold).
        let cost = if fold_cotrain {
            self.fold_cotrain_consistency(reml_cost, &rho)?
        } else {
            reml_cost
        };
        let cost = self.add_fit_data_collapse_penalty(cost, &rho)?;
        self.current_rho = rho;
        self.last_loss = Some(loss);
        Ok((cost, beta_hat))
    }

    /// #1154 — add the amortized-encoder consistency fold to an already-computed
    /// REML criterion at the converged dictionary for `rho`. The fold has NO
    /// analytic gradient: under Design A the inner solve converges to the same
    /// stationary point, so the exact outer derivative is the REML λ-gradient
    /// `∇f` (the implicit-function `dβ̂/dλ` path) and nothing else.
    ///
    /// #1206 — for that reason the fold is carried ONLY by the DERIVATIVE-FREE
    /// value-probe lane (`evaluate_with_refine_policy` → `eval_cost`), where the
    /// cost is never paired with a gradient (seed screening, cross-seed final
    /// ranking, EFS backtracking). It is NOT folded into the gradient lane
    /// (`eval`/`OuterEvalOrder::ValueAndGradient`), whose `(cost, gradient)` pair
    /// must be self-consistent for the BFGS Armijo line search — folding `c` into
    /// the cost there while returning `∇f` is exactly the objective↔gradient
    /// desync bug class (#931). The consistency term thus steers the value-only
    /// ranking the cascade does between certified candidates, never the smooth
    /// descent direction.
    fn fold_cotrain_consistency(
        &self,
        reml_cost: f64,
        rho: &SaeManifoldRho,
    ) -> Result<f64, String> {
        let consistency = self
            .term
            .amortized_encoder_consistency(self.target.view(), rho)?;
        // Route through the SINGLE source of the fold arithmetic on
        // `SaeManifoldTerm` so the cascade-ranked cost and the public
        // `reml_criterion_cotrained` value can never drift (the objective↔gradient
        // desync bug class).
        Ok(SaeManifoldTerm::fold_cotrain_consistency(
            reml_cost,
            &consistency,
        ))
    }

    /// Fellner-Schall / Mackay multiplicative fixed-point step on ρ at
    /// `rho_flat`. Runs the inner `(t, β)` solve to convergence at fixed ρ
    /// (sharing the single Direct factor with the REML criterion), then
    /// returns `(cost, additive-log-steps, β̂)`.
    ///
    /// All ρ coords are log-quantities, so the engine's additive step
    /// `rho_new = rho + step` IS the multiplicative FS update. Per coord:
    /// - ARD axis (k,j): `α_new = φ̂ n / (‖t_kj‖² + tr_kj(H⁻¹))`,
    ///   `step = ln α_new − log_ard[k][j]`. The `tr_kj(H⁻¹)` posterior
    ///   variance (from the selected-inverse latent diagonal) is exactly the
    ///   term the deleted `α=n/‖t‖²` rule dropped, so α cannot collapse on a
    ///   degenerate axis: as `‖t‖²→0`, `tr_kj(H⁻¹)→1/α` bounds the
    ///   denominator and the fixed point has a finite root.
    /// - λ_smooth: `λ_new = φ̂[p·Σ_k rank S_k − tr(S_β⁻¹ M)] / βᵀ(⊕S_k⊗I_p)β`
    ///   (Wood-Fasiolo EFS), `step = ln λ_new − log_lambda_smooth`.
    /// - λ_sparse: 0.0 — the assignment-sparsity priors (softmax entropy,
    ///   gated L1, IBP) are non-quadratic, so no Gaussian-logdet FS fixed
    ///   point exists; it stays cost-driven (the cascade still moves it via
    ///   the cost path when EFS is not the active lane for that coord).
    pub(crate) fn efs_step(&mut self, rho_flat: ArrayView1<'_, f64>) -> Result<EfsEval, String> {
        let rho = self.baseline_rho.from_flat(rho_flat);
        if let Some(beta) = self.seeded_beta.take()
            && beta.len() == self.term.beta_dim()
        {
            self.term.set_flat_beta(beta.view())?;
        }
        let (cost, loss, cache) = self.term.reml_criterion_with_cache(
            self.target.view(),
            &rho,
            self.registry.as_ref(),
            self.inner_max_iter,
            self.learning_rate,
            self.ridge_ext_coord,
            self.ridge_beta,
        )?;
        self.current_rho = rho.clone();
        let dispersion = self
            .term
            .reconstruction_dispersion(&loss, &cache, &rho)
            .map_err(|e| format!("SaeManifoldOuterObjective::efs_step: dispersion: {e}"))?;
        self.last_loss = Some(loss);

        let n_obs = self.term.n_obs() as f64;
        let sumsq = self.term.ard_coord_sumsq();
        let traces = self
            .term
            .ard_inverse_traces(&cache)
            .map_err(|e| format!("SaeManifoldOuterObjective::efs_step: ARD traces: {e}"))?;

        // Build the flat step vector in `to_flat` layout:
        // [0]=log_lambda_sparse, [1]=log_lambda_smooth, then per-atom axes.
        let n_params = rho.to_flat().len();
        let mut steps = vec![0.0_f64; n_params];

        // λ_sparse (index 0): non-quadratic prior → no FS fixed point. Step 0.
        steps[0] = 0.0;

        // λ_smooth (index 1): Wood-Fasiolo EFS multiplicative update.
        let lambda_smooth = rho.lambda_smooth();
        let p_out = self.term.output_dim() as f64;
        let mut smooth_rank_total = 0usize;
        for atom in &self.term.atoms {
            smooth_rank_total += SaeManifoldTerm::symmetric_rank(&atom.smooth_penalty)?;
        }
        let rank_total = p_out * (smooth_rank_total as f64);
        let quad = self.term.decoder_smoothness_quadratic_form();
        let eff_dof = self
            .term
            .decoder_smoothness_effective_dof(&cache, lambda_smooth)
            .map_err(|e| format!("SaeManifoldOuterObjective::efs_step: smooth dof: {e}"))?;
        // λ_new = φ̂ · (penalty_rank − effective_dof) / penalty_energy. The
        // dispersion factor makes the update target the dimensionless effective
        // stiffness λ/φ̂ instead of an absolute output-unit penalty weight.
        // Guard the FS ratio against a vanishing penalty energy or a non-positive
        // numerator (which can occur transiently far from the optimum) by holding
        // λ_smooth fixed (step 0) — the cost path still moves it then.
        if quad > 0.0 && rank_total - eff_dof > 0.0 && lambda_smooth > 0.0 {
            let lambda_new = dispersion * (rank_total - eff_dof) / quad;
            if lambda_new.is_finite() && lambda_new > 0.0 {
                steps[1] = lambda_new.ln() - rho.log_lambda_smooth;
            }
        }

        // ARD axes (indices 2..): Mackay fixed point with posterior variance.
        let mut cursor = 2usize;
        for (k, axis_logard) in rho.log_ard.iter().enumerate() {
            let d = axis_logard.len();
            for j in 0..d {
                let denom = sumsq[k][j] + traces[k][j];
                if denom > 0.0 {
                    let alpha_new = dispersion * n_obs / denom;
                    if alpha_new.is_finite() && alpha_new > 0.0 {
                        steps[cursor + j] = alpha_new.ln() - axis_logard[j];
                    }
                }
            }
            cursor += d;
        }

        let beta_hat = self.term.flatten_beta();
        let cost = self.add_fit_data_collapse_penalty(cost, &rho)?;
        Ok(EfsEval {
            cost,
            steps,
            beta: Some(beta_hat),
            psi_gradient: None,
            psi_indices: None,
            inner_hessian_scale: None,
            logdet_enclosure_gap: None,
        })
    }

    /// #1273 — central-difference outer-ρ gradient of the REML value
    /// path, used ONLY as a descent-direction fallback when the analytic outer
    /// gradient is numerically undefined at a finite-cost ρ (a near-singular but
    /// valid joint Hessian — e.g. a circle/torus topology whose data is lower-
    /// dimensional than the latent basis, so a genuine flat direction lives
    /// outside both the chart gauge orbit and the penalised decoder β-null that
    /// the Faddeev-Popov deflation recovers).
    ///
    /// This differences the SAME warm value path (`reml_criterion`) whose value
    /// the gradient lane reports, so the value and its descent direction belong
    /// to one function:
    ///
    /// * Every probe is evaluated on a FRESH clone of the LIVE warm state
    ///   (`self.term`, which on the fallback arm already holds the converged
    ///   inner solve at `rho_state`). `reml_criterion` takes `&mut self` and
    ///   mutates warm-start/cache state, so a single shared probe term would
    ///   make the result depend on probe order; a fresh clone per probe keeps
    ///   every evaluation from an identical snapshot.
    /// * Each ρ-coordinate uses its own step `probe_step_for(ρ_i)`, not one
    ///   global step, so a small coordinate is not under-resolved by a large one.
    /// * The collapse penalty is a discrete wall (`SAE_FIT_DATA_COLLAPSE_COST`);
    ///   the difference is only meaningful in the smooth flat valley where the
    ///   wall is inactive, so a probe at or above the wall (or non-finite) is
    ///   treated as unmeasurable rather than differenced across the barrier.
    /// * A coordinate that cannot be measured (central ladder, then one-sided,
    ///   all unmeasurable) yields `Err` — it is NOT reported as a zero (which the
    ///   optimiser would read as a stationary coordinate and which would poison
    ///   the BFGS curvature update); the outer step must not proceed on a
    ///   corrupted direction.
    // FD-OK: descent-direction-only fallback (#1273); the analytic value path
    // (`reml_criterion`) remains the single source of truth — this never
    // produces the cost the fit consumes, only a usable direction across the
    // near-singular flat valley.
    pub(crate) fn central_difference_outer_gradient(
        &self,
        rho_state: &SaeManifoldRho,
    ) -> Result<Array1<f64>, String> {
        let rho_flat = rho_state.to_flat();
        let n = rho_flat.len();
        // Snapshot the live warm state once; every probe clones it fresh.
        let warm_base = self.term.clone();
        let value_at = |flat: &Array1<f64>| -> Option<f64> {
            let mut term = warm_base.clone();
            let rho = self.baseline_rho.from_flat(flat.view());
            term.reml_criterion(
                self.target.view(),
                &rho,
                self.registry.as_ref(),
                self.inner_max_iter,
                self.learning_rate,
                self.ridge_ext_coord,
                self.ridge_beta,
            )
            .ok()
            .map(|(cost, _loss)| cost)
            // Reject probes on/above the discrete collapse wall: differencing
            // across it is meaningless. Such a probe counts as unmeasurable.
            .filter(|v| v.is_finite() && *v < SAE_FIT_DATA_COLLAPSE_COST)
        };
        // Shrink the step toward the valley centre when a probe lands on the
        // wall / non-finite, before giving up on a coordinate.
        const LADDER: [f64; 3] = [1.0, 1.0 / 3.0, 1.0 / 10.0];
        let mut gradient = Array1::<f64>::zeros(n);
        for i in 0..n {
            let h_i = probe_step_for(rho_flat[i]);
            if !(h_i.is_finite() && h_i > 0.0) {
                return Err(format!(
                    "[SAE/#1273] outer-ρ probe step for coordinate {i} is non-finite \
                     (ρ_i={}); the point is pathological and the outer step must not \
                     proceed on a corrupted direction",
                    rho_flat[i]
                ));
            }
            let mut measured: Option<f64> = None;
            // Central difference down the step ladder.
            for &scale in LADDER.iter() {
                let h = h_i * scale;
                let mut plus = rho_flat.clone();
                plus[i] += h;
                let mut minus = rho_flat.clone();
                minus[i] -= h;
                if let (Some(vp), Some(vm)) = (value_at(&plus), value_at(&minus)) {
                    let derivative = (vp - vm) / (2.0 * h);
                    if derivative.is_finite() {
                        measured = Some(derivative);
                        break;
                    }
                }
            }
            // One-sided (forward, then backward) at the base step as a last resort.
            if measured.is_none()
                && let Some(v0) = value_at(&rho_flat)
            {
                let mut plus = rho_flat.clone();
                plus[i] += h_i;
                if let Some(vp) = value_at(&plus) {
                    let d = (vp - v0) / h_i;
                    if d.is_finite() {
                        measured = Some(d);
                    }
                }
                if measured.is_none() {
                    let mut minus = rho_flat.clone();
                    minus[i] -= h_i;
                    if let Some(vm) = value_at(&minus) {
                        let d = (v0 - vm) / h_i;
                        if d.is_finite() {
                            measured = Some(d);
                        }
                    }
                }
            }
            match measured {
                Some(d) => gradient[i] = d,
                // No fake zeros: a genuinely unmeasurable coordinate (every probe
                // non-finite or on the collapse wall) is propagated, not silently
                // reported as a stationary component.
                None => {
                    return Err(format!(
                        "[SAE/#1273] outer-ρ gradient unmeasurable in coordinate {i} \
                         at finite-cost ρ: every central and one-sided probe (step \
                         ladder from {h_i:.3e}) was non-finite or hit the collapse \
                         barrier; the outer step must not proceed on a corrupted \
                         direction"
                    ));
                }
            }
        }
        Ok(gradient)
    }
    // END-FD-OK
}

impl OuterObjective for SaeManifoldOuterObjective {
    fn capability(&self) -> OuterCapability {
        let plan = self.term.streaming_plan();
        let gradient = if plan.direct_admitted {
            Derivative::Analytic
        } else {
            Derivative::Unavailable
        };
        OuterCapability {
            // The full analytic outer-ρ gradient is assembled for every
            // assignment mode, including IBP-MAP (its empirical-π third channel
            // landed exactly in `logdet_theta_adjoint`, #1006).
            gradient,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: self.baseline_rho.to_flat().len(),
            // ρ are all penalty-like / τ coordinates: precisions and
            // log-smoothing strengths. No design-moving ψ coordinates.
            psi_dim: 0,
            // SAE's penalty coordinates are scale-coupled to the profiled
            // Gaussian reconstruction dispersion. The generic fixed-point lane
            // can drive the smoothness axis to the absolute upper boundary after
            // a good low-noise seed, collapsing the decoder to the mean (#1023).
            // Keep the analytic value/gradient lane in charge so the
            // dimensionless seed is not overwritten by an absolute-unit EFS step.
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: true,
        }
    }

    fn eval_cost(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
        // Value-only comparison path (EFS backtracking, seed validation, FD
        // certificate probes): no gradient/Hessian is ever
        // consumed at this iterate, so it takes the cheap probe refine budget
        // (#1029). Accepted points are always re-polished through
        // `eval`/`eval_with_order(ValueAndGradient|ValueGradientHessian)`
        // before any derivative consumption, and a probe value — when one is
        // returned at all — is converged to the same KKT/step tolerance as
        // the full-budget path, so all ranked comparisons stay in one measure.
        // #1224 — `eval_cost` is the value-only CROSS-SEED RANKING / EFS lane
        // (seed screening, cross-seed final selection, EFS backtracking). No `∇f`
        // is ever paired with this cost, so it is the correct place to carry the
        // derivative-free co-training fold `f+c` (`fold_cotrain = true`).
        match self.evaluate_with_refine_policy(rho.view(), false, true) {
            Ok((cost, _beta)) => Ok(cost),
            Err(err) if Self::is_recoverable_value_probe_refusal(&err) => Ok(f64::INFINITY),
            Err(err) => Err(EstimationError::RemlOptimizationFailed(err)),
        }
    }

    fn eval(&mut self, rho: &Array1<f64>) -> Result<OuterEval, EstimationError> {
        let rho_state = self.baseline_rho.from_flat(rho.view());
        if let Some(beta) = self.seeded_beta.take()
            && beta.len() == self.term.beta_dim()
        {
            self.term
                .set_flat_beta(beta.view())
                .map_err(EstimationError::RemlOptimizationFailed)?;
        }
        // #1154 — warm-start the inner latent coords from the amortized encoder
        // built on the running dictionary at this ρ (Design A), exactly as the
        // value-probe lane (`evaluate_with_refine_policy`) does. The accepted
        // iterate's inner solve then refines from the cheap one-mat-vec seed to
        // the SAME stationary point, so the exact REML λ-gradient computed below
        // is untouched — the warm-start changes only the basin entry, never the
        // root. Advisory: a degenerate atlas certifies/warm-starts nothing and
        // leaves the cold path byte-for-byte unchanged. #1207 — the outcome is
        // recorded (failure logged) so a silent cold fallback is observable.
        let warm_start_outcome = self
            .term
            .warm_start_latents_from_amortized_encoder(self.target.view(), &rho_state);
        self.record_warm_start(warm_start_outcome);
        let (cost, loss, cache) = self
            .term
            .reml_criterion_with_cache(
                self.target.view(),
                &rho_state,
                self.registry.as_ref(),
                self.inner_max_iter,
                self.learning_rate,
                self.ridge_ext_coord,
                self.ridge_beta,
            )
            .map_err(EstimationError::RemlOptimizationFailed)?;
        // #1273 — the analytic outer gradient is built from the undamped joint
        // Hessian via `outer_gradient_arrow_solver`, whose Faddeev-Popov gauge
        // deflation recovers near-null directions that lie in the closed-form
        // chart gauge orbit or the penalised decoder β-null. On a circle/torus
        // topology whose data is intrinsically lower-dimensional than the latent
        // (`atom_topology="circle"` with `d_atom=2`: a 1-D ring embedded in a 2-D
        // torus basis), the joint Hessian carries a genuine near-singular-but-
        // valid direction OUTSIDE both deflation sets — its min pivot is tiny but
        // strictly positive (≈1.2e-10) while the max is ≈2.3e5, so the analytic
        // gradient's pivot-ratio gate (`outer_gradient_conditioning_error`)
        // legitimately reports "joint Hessian numerically singular" and the solver
        // refuses. Before the fix this `?`-propagated as `RemlOptimizationFailed`,
        // aborting the WHOLE outer optimisation: every BFGS gradient point was
        // refused, the line search stalled into consecutive infeasible probes, and
        // the seed cascade rejected every seed → `RemlConvergenceError`.
        //
        // The cost at this ρ is still the EXACT REML criterion (it factorised
        // fine; only the gradient's pivot ratio tripped the gate), so the point is
        // feasible, not infeasible. Recover by descending it with a CENTRAL
        // central-difference outer gradient of the same value path — the identical
        // FD instrument the optimality certificate already uses to audit the
        // analytic gradient (`certificates::probe_*`), here used as a descent
        // direction only when the analytic path is numerically undefined. This is
        // exact-REML-policy clean: the FD does not produce the cost (the returned
        // cost is the analytic REML value), it only supplies a usable direction so
        // BFGS can cross the flat valley instead of aborting. The well-conditioned
        // path is byte-for-byte unchanged: the analytic solver succeeds there and
        // this fallback is never reached.
        let analytic = self
            .term
            .outer_gradient_arrow_solver(&cache, rho_state.lambda_smooth())
            .and_then(|solver| {
                self.term.analytic_outer_rho_gradient_components(
                    self.target.view(),
                    &rho_state,
                    &loss,
                    &cache,
                    &solver,
                )
            });
        let gradient = match analytic {
            Ok(components) => components.gradient(),
            Err(analytic_err) => {
                if !cost.is_finite() {
                    return Err(EstimationError::RemlOptimizationFailed(analytic_err));
                }
                log::info!(
                    "[SAE/#1273] analytic outer gradient undefined at a finite-cost ρ \
                     ({analytic_err}); descending with a central-difference outer \
                     gradient of the value path so the near-singular flat valley is crossed \
                     instead of aborting the outer optimisation"
                );
                self.central_difference_outer_gradient(&rho_state) // fd-ok: analytic outer gradient not cost-consistent here; descent direction only (#1273)
                    .map_err(EstimationError::RemlOptimizationFailed)?
            }
        };
        let beta_hat = self.term.flatten_beta();
        // #1206 — the gradient lane (`OuterEvalOrder::ValueAndGradient`, consumed
        // by the outer BFGS Armijo line search) MUST return a cost whose gradient
        // is the gradient we return. The amortized-encoder consistency fold `c(ρ)`
        // (#1154) has NO analytic gradient — under Design A the inner solve
        // converges to the same stationary point and the exact REML λ-gradient is
        // `∇f` only. Folding `c` into the cost here while returning `∇f` would hand
        // BFGS the value of `f+c` paired with `∇f`, so its sufficient-decrease test
        // `f(ρ+αd)+c(ρ+αd) ≤ f(ρ)+c(ρ) + c₁α·∇f·d` mixes two functions and can
        // stall or wander (the objective↔gradient desync bug class, #931). So the
        // gradient lane reports the consistent pair `(f, ∇f)`; the consistency fold
        // `c` is a DERIVATIVE-FREE ranking regularizer carried ONLY by the
        // value-probe lane (`eval_cost`/`evaluate_with_refine_policy`), where no
        // gradient is ever paired with the cost (seed screening, cross-seed final
        // ranking, EFS backtracking). The collapse penalty is a discrete
        // infeasibility wall (a huge constant on a degenerate fit), not a smooth
        // regularizer — BFGS simply rejects steps into it, which is the intended
        // barrier behaviour, so it stays on both lanes.
        let cost = self
            .add_fit_data_collapse_penalty(cost, &rho_state)
            .map_err(EstimationError::RemlOptimizationFailed)?;
        self.current_rho = rho_state;
        self.last_loss = Some(loss);
        Ok(OuterEval {
            cost,
            gradient,
            hessian: HessianResult::Unavailable,
            inner_beta_hint: Some(beta_hat),
        })
    }

    fn eval_with_order(
        &mut self,
        rho: &Array1<f64>,
        order: OuterEvalOrder,
    ) -> Result<OuterEval, EstimationError> {
        match order {
            OuterEvalOrder::Value => {
                // #1224 — the `Value` order is the BFGS / ARC LINE-SEARCH cost
                // probe (see `solver/rho_optimizer/bridges.rs`). Its cost is
                // compared against steps whose direction came from `eval`'s pure
                // REML `∇f`, so it must NOT fold in the gradient-free co-training
                // consistency penalty (`fold_cotrain = false`) — otherwise the
                // Armijo/Wolfe sufficient-decrease test mixes `f+c` with `∇f` and
                // can stall or wander. The fold is carried only by the value-only
                // cross-seed ranking lane (`eval_cost`).
                let (cost, _beta_hat) =
                    match self.evaluate_with_refine_policy(rho.view(), false, false) {
                        Ok(evaluated) => evaluated,
                        Err(err) if Self::is_recoverable_value_probe_refusal(&err) => {
                            return Ok(OuterEval::infeasible(rho.len()));
                        }
                        Err(err) => return Err(EstimationError::RemlOptimizationFailed(err)),
                    };
                Ok(OuterEval {
                    cost,
                    gradient: Array1::zeros(rho.len()),
                    hessian: HessianResult::Unavailable,
                    inner_beta_hint: None,
                })
            }
            OuterEvalOrder::ValueAndGradient | OuterEvalOrder::ValueGradientHessian => {
                self.eval(rho)
            }
        }
    }

    fn eval_efs(&mut self, rho: &Array1<f64>) -> Result<EfsEval, EstimationError> {
        self.efs_step(rho.view())
            .map_err(EstimationError::RemlOptimizationFailed)
    }

    fn reset(&mut self) {
        self.term = self.baseline_term.clone();
        self.current_rho = self.baseline_rho.clone();
        self.last_loss = None;
        self.seeded_beta = None;
    }

    fn seed_inner_state(&mut self, beta: &Array1<f64>) -> Result<SeedOutcome, EstimationError> {
        // Contract (see src/solver/reml/continuation.rs:727-737): an empty-β
        // seed means "no warm-start available, use your own cold default" and
        // MUST be accepted as a no-op. The continuation pre-warm forwards the
        // previous eval's `inner_beta_hint`, but before the first accepted eval
        // that hint is empty (`state.last_beta` starts empty). Rejecting it
        // fatally dropped every continuation seed and forced a full cold solve
        // on every outer seed — the slowness in gam#577. Only a *populated* β
        // must match the decoder dimension.
        if beta.is_empty() {
            // NoSlot is the documented continuation reply for "no usable seed;
            // proceed cold, no log" (outer_strategy.rs:1776). The real β slot
            // gets populated on the next accepted eval, which publishes
            // `inner_beta_hint`, so steps 2+ warm-start normally.
            return Ok(SeedOutcome::NoSlot);
        }
        if beta.len() != self.term.beta_dim() {
            return Err(EstimationError::RemlOptimizationFailed(format!(
                "SaeManifoldOuterObjective::seed_inner_state: β length {} != decoder dim {}",
                beta.len(),
                self.term.beta_dim()
            )));
        }
        self.seeded_beta = Some(beta.clone());
        Ok(SeedOutcome::Installed)
    }

    /// The SAE-manifold joint fit enters through the heavy-smoothing
    /// [`crate::solver::continuation_path::ContinuationPath`] WHEN there is a
    /// combinatorial inter-atom routing active-set to protect: the joint
    /// `(logits, t, β)` block has a routing component that a cold solve at ρ*
    /// can collapse — but that failure class is specifically the **K ≥ 2**
    /// routing collapse (atoms competing for assignment mass). A single-atom
    /// (`K = 1`) fit has no inter-atom routing, so the coupled ρ / τ / isometry
    /// walk has nothing to prevent and is pure overhead: the cold direct cascade
    /// solves it directly (an order of magnitude faster on tiny fixtures). Gate
    /// the walk on `K ≥ 2`. When it returns `true` every seed routes through the
    /// homotopy walk (Object 1) and the seed cascade's structural-failure
    /// handling flips from REJECT to DEMOTE-WITH-REASON so the candidate set
    /// never empties on a structural diagnosis.
    fn requires_continuation_path_entry(&self) -> bool {
        // K >= 2: routing multimodality makes blind multistart hopeless — the
        // certified walk is the entry of record. K = 1 with a curved-capable
        // chart (duchon / euclidean patch): the Eckart-Young LINEAR optimum is
        // a genuine local minimum (a straight line through an arc), and cold
        // seeds converge INTO it — the walk exists precisely to track from
        // that anchor into the curved branch, and with the gauge-quotient
        // pivot invariant it arrives in a handful of legs, replacing the
        // 12-seed cascade outright. K = 1 periodic atoms keep the cascade:
        // their circular topology is baked into the basis, so the linear
        // basin is not an attractor for them and the walk buys nothing.
        if self.term.k_atoms() >= 2 {
            return true;
        }
        self.term.atoms.iter().any(|atom| {
            matches!(
                atom.basis_kind,
                SaeAtomBasisKind::Duchon
                    | SaeAtomBasisKind::EuclideanPatch
                    | SaeAtomBasisKind::Poincare
            )
        })
    }

    /// The SAE-manifold objective has a certified anchor (#1007): its `η = 0`
    /// Eckart-Young linear relaxation is convex with a global optimum certified
    /// by [`linear_span_anchor`]. Run the predictor-corrector `η`-walk from that
    /// anchor before blind multistart. On arrival the inner state is warm
    /// at the certified `η = 1` solution for the active seed; on a
    /// degenerate anchor or a detected bifurcation the term is left at the full
    /// basis (`η = 1`) and the documented cascade takes over — the outcome is
    /// recorded on the fit payload either way.
    ///
    /// For objectives that don't require continuation entry (K=1 periodic atoms
    /// whose topology is baked into the basis), return `None` so the standard
    /// seed cascade is used directly without the curvature walk.
    fn curvature_homotopy_entry(
        &mut self,
        rho: &Array1<f64>,
    ) -> Option<Result<bool, EstimationError>> {
        // K=1 periodic atoms don't need the curvature walk: their circular
        // topology is baked into the basis, so the linear basin is not an
        // attractor and the walk would just add overhead / potential failure.
        // Return `None` to use the standard seed cascade directly.
        if !self.requires_continuation_path_entry() {
            return None;
        }
        let rho_state = self.baseline_rho.from_flat(rho.view());
        Some(
            self.run_curvature_homotopy_entry_at_rho(&rho_state)
                .map_err(EstimationError::RemlOptimizationFailed),
        )
    }
}

pub(crate) fn sae_manifold_newton_directional_decrease(
    sys: &ArrowSchurSystem,
    delta_ext_coord: ArrayView1<'_, f64>,
    delta_beta: ArrayView1<'_, f64>,
) -> f64 {
    // delta_ext_coord has variable-stride layout for heterogeneous systems.
    assert_eq!(delta_ext_coord.len(), sys.row_offsets[sys.rows.len()]);
    assert_eq!(delta_beta.len(), sys.k);
    let mut gradient_dot_step = 0.0;
    for (row_idx, row) in sys.rows.iter().enumerate() {
        let row_base = sys.row_offsets[row_idx];
        let di = sys.row_dims[row_idx];
        for axis in 0..di {
            gradient_dot_step += row.gt[axis] * delta_ext_coord[row_base + axis];
        }
    }
    for idx in 0..sys.k {
        gradient_dot_step += sys.gb[idx] * delta_beta[idx];
    }
    -gradient_dot_step
}

/// Per-atom decoder-smoothness GEMM `S_k · B_k`, batched across ALL GPUs.
///
/// Every atom contributes one dense product of its `(m_k × m_k)` smoothness
/// penalty `S_k` with its `(m_k × p)` decoder coefficients `B_k`. These products
/// are independent across atoms, so the per-atom axis is the natural batch /
/// device-fan-out dimension. This helper:
///
///   * groups atoms by identical `(m_k, p)` shape (the strided-batched cuBLAS
///     GEMM requires a uniform tile),
///   * for each group with ≥ 2 atoms whose aggregate flop count clears the
///     dispatch threshold, partitions the group's atoms across every available
///     device with [`crate::gpu::pool::scatter_batched`] and runs one
///     `try_fast_abt_strided_batched` per device tile (computing
///     `S_k · B_k = S_k · (B_kᵀ)ᵀ`),
///   * falls back, atom-by-atom, to the exact ndarray `S_k.dot(B_k)` whenever no
///     GPU runtime is present, the pool returns `None`, or a tile's batched GEMM
///     declines. The result is bit-for-bit identical to the all-CPU path (f64
///     throughout, same accumulation order per product).
///
/// Returns one `S_k · B_k` matrix per atom, in atom order. `symmetrize`
/// pre-symmetrises each `S_k` (the assembly path needs `½(S+Sᵀ)`); the value /
/// quadratic-form callers pass `false` since the quadratic form only sees the
/// symmetric part regardless.
pub(crate) fn batched_smooth_sb(
    sb_inputs: &[(ArrayView2<'_, f64>, ArrayView2<'_, f64>)],
    symmetrize: bool,
) -> Vec<Array2<f64>> {
    let n_atoms = sb_inputs.len();
    // Materialise the (optionally symmetrised) S factors once; the GPU tile and
    // the CPU fallback both read these, so a single pass keeps the two routes
    // numerically identical.
    let s_mats: Vec<Array2<f64>> = sb_inputs
        .iter()
        .map(|(s, _)| {
            if symmetrize {
                let m = s.nrows();
                let mut sym = Array2::<f64>::zeros((m, m));
                for i in 0..m {
                    for j in 0..m {
                        sym[[i, j]] = 0.5 * (s[[i, j]] + s[[j, i]]);
                    }
                }
                sym
            } else {
                s.to_owned()
            }
        })
        .collect();

    // Exact CPU fallback for a single atom, reused by both the no-GPU route and
    // per-tile decline.
    let cpu_one = |idx: usize| -> Array2<f64> { s_mats[idx].dot(&sb_inputs[idx].1) };

    let rt = match crate::gpu::device_runtime::GpuRuntime::global() {
        Some(rt) => rt,
        None => return (0..n_atoms).map(cpu_one).collect(),
    };

    // Group atom indices by uniform (m, p) shape; only same-shape groups can ride
    // a strided-batched GEMM tile.
    let mut groups: std::collections::BTreeMap<(usize, usize), Vec<usize>> =
        std::collections::BTreeMap::new();
    for (idx, (_, b)) in sb_inputs.iter().enumerate() {
        let m = s_mats[idx].nrows();
        let p = b.ncols();
        groups.entry((m, p)).or_default().push(idx);
    }

    let mut out: Vec<Option<Array2<f64>>> = (0..n_atoms).map(|_| None).collect();
    for ((m, p), members) in groups {
        // Singletons and tiny groups gain nothing from batched device launch;
        // the single-product `fast_*` shim (size-gated) already handles a large
        // lone GEMM, so route those straight through the CPU-or-shim helper.
        if members.len() < 2 || m == 0 || p == 0 {
            for &idx in &members {
                out[idx] = Some(cpu_one(idx));
            }
            continue;
        }
        // Build the per-tile batched inputs lazily inside the device closure so
        // each device only packs the atoms it owns. `items` carries the member
        // atom indices; `scatter_batched` slices it per device ordinal.
        let mut items: Vec<usize> = members.clone();
        let s_ref = &s_mats;
        // Collect per-tile results into a side channel keyed by atom index, then
        // splice them in after scatter completes (scatter's closure borrows
        // `items` immutably-per-tile and must stay `Sync`).
        let tile_results: std::sync::Mutex<Vec<(usize, Array2<f64>)>> =
            std::sync::Mutex::new(Vec::with_capacity(members.len()));
        let ok = crate::gpu::pool::scatter_batched(rt, &mut items, |_ordinal, slice| {
            if slice.is_empty() {
                return Some(());
            }
            let batch = slice.len();
            // A = stacked S_k  (batch, m, m); B = stacked B_kᵀ (batch, p, m) so
            // that `A · Bᵀ` per tile yields `S_k · B_k` (batch, m, p).
            let mut a = Array3::<f64>::zeros((batch, m, m));
            let mut bt = Array3::<f64>::zeros((batch, p, m));
            for (t, &idx) in slice.iter().enumerate() {
                let s = &s_ref[idx];
                let b = &sb_inputs[idx].1;
                for i in 0..m {
                    for j in 0..m {
                        a[[t, i, j]] = s[[i, j]];
                    }
                }
                for i in 0..p {
                    for j in 0..m {
                        bt[[t, i, j]] = b[[j, i]];
                    }
                }
            }
            let prod = crate::gpu::try_fast_abt_strided_batched(a.view(), bt.view())?;
            let mut sink = tile_results.lock().expect("tile_results mutex poisoned");
            for (t, &idx) in slice.iter().enumerate() {
                sink.push((idx, prod.slice(s![t, .., ..]).to_owned()));
            }
            Some(())
        });
        // The scatter closure has returned, so all borrows of `items`/`s_mats`/
        // `tile_results` are released; write the results back into `out`.
        match ok {
            Some(()) => {
                let sink = tile_results
                    .into_inner()
                    .expect("tile_results mutex poisoned");
                for (idx, mat) in sink {
                    out[idx] = Some(mat);
                }
                // Any member a tile silently skipped (cannot happen with the
                // contract, but keep the result total) falls back to CPU.
                for &idx in &members {
                    if out[idx].is_none() {
                        out[idx] = Some(cpu_one(idx));
                    }
                }
            }
            None => {
                for &idx in &members {
                    out[idx] = Some(cpu_one(idx));
                }
            }
        }
    }
    out.into_iter()
        .enumerate()
        .map(|(idx, slot)| slot.unwrap_or_else(|| cpu_one(idx)))
        .collect()
}

/// A detected bifurcation on the curvature-homotopy branch (#1007): the arrow
/// factor's smallest Cholesky pivot collapsed below the safe-SPD tolerance at a
/// homotopy parameter `η`, so the optimal branch the tracker was following lost
/// strict positive-definiteness. Recorded on [`CurvatureWalkReport`] and never
/// silent — the walk returns control to the documented multi-seed cascade.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurvatureBifurcation {
    /// Homotopy parameter at which the pivot collapsed.
    pub eta: f64,
    /// The smallest arrow-factor pivot observed at `eta` (Hessian-scale, i.e.
    /// squared lower-Cholesky diagonal); below the safe-SPD floor.
    pub min_pivot: f64,
}

/// Outcome of one certified curvature-homotopy entry walk (#1007).
///
/// The tracker walks the basis curvature dial `η` from the Eckart-Young anchor
/// (`η = 0`, global by construction) to the full curved basis (`η = 1`),
/// predictor-corrector style, holding the per-pivot positivity invariant. This
/// report makes the outcome observable on the fit payload: `arrived` says the
/// walk reached `η = 1` on the certified branch; `bifurcation` records the first
/// detected pivot collapse (if any); `collapse_events` mirrors the inner active
/// -mass guard's verdict at the arrival state; `eta_steps` / `step_halvings`
/// are the walk's cost. A walk that did not arrive (degenerate anchor or a
/// recorded bifurcation) hands control back to the multi-seed cascade.
#[derive(Debug, Clone)]
pub struct CurvatureWalkReport {
    /// Whether the walk reached `η = 1` on the certified optimal branch.
    pub arrived: bool,
    /// Eckart-Young anchor residual energy at `η = 0` (the certificate the
    /// linear relaxation is solved to).
    pub anchor_residual_norm_sq: f64,
    /// First detected branch bifurcation (pivot collapse), or `None` when the
    /// pivot stayed strictly positive across the whole walk.
    pub bifurcation: Option<CurvatureBifurcation>,
    /// Number of accepted `η` waypoints (anchor → 1).
    pub eta_steps: usize,
    /// Number of `η`-step halvings forced by a shrinking min-pivot.
    pub step_halvings: usize,
    /// Number of inner active-mass collapse events recorded at the arrival
    /// state (the same `#976` guard ledger the cascade reads); a clean walk
    /// arrives with this empty.
    pub collapse_events: usize,
    /// Number of scaffold re-seeds the walk itself triggered. A certified walk
    /// from the global anchor reaches `η = 1` with zero reseeds.
    pub reseeds: usize,
}

#[derive(Debug, Clone)]
pub struct LinearSpanAtomAnchor {
    pub gate_weight: f64,
    pub frame: GrassmannFrame,
    pub decoder_coordinates: Array2<f64>,
    pub singular_values: Array1<f64>,
}

#[derive(Debug, Clone)]
pub struct LinearSpanAnchor {
    pub atoms: Vec<LinearSpanAtomAnchor>,
    pub reconstruction: Array2<f64>,
    pub residual_norm_sq: f64,
}

/// Curvature-homotopy linear-span anchor at `eta = 0`.
///
/// This stage-1 primitive solves the neutral-gate linear relaxation by
/// sequential Eckart-Young residual SVDs and canonicalizes every recovered output
/// subspace through the same [`GrassmannFrame`] gauge used by the #972 frame
/// machinery. It does not mutate `term` or replace the existing seed cascade.
pub fn linear_span_anchor(
    term: &SaeManifoldTerm,
    targets: ArrayView2<'_, f64>,
) -> Result<LinearSpanAnchor, String> {
    let n = term.n_obs();
    let p = term.output_dim();
    if targets.dim() != (n, p) {
        return Err(format!(
            "linear_span_anchor: targets shape {:?} != ({n}, {p})",
            targets.dim()
        ));
    }
    if term.k_atoms() == 0 {
        return Err("linear_span_anchor: term must contain at least one atom".into());
    }
    if !targets.iter().all(|v| v.is_finite()) {
        return Err("linear_span_anchor: targets must be finite".into());
    }
    let gates = neutral_gate_weights(term.assignment.mode, term.k_atoms());
    let mut residual = targets.to_owned();
    let mut reconstruction = Array2::<f64>::zeros((n, p));
    let mut atoms = Vec::with_capacity(term.k_atoms());
    for (atom_idx, atom) in term.atoms.iter().enumerate() {
        let gate = gates[atom_idx];
        if !(gate.is_finite() && gate > 0.0) {
            return Err(format!(
                "linear_span_anchor: neutral gate for atom {atom_idx} must be positive finite; got {gate}"
            ));
        }
        let requested_rank = atom.basis_size().min(n).min(p);
        if requested_rank == 0 {
            return Err(format!(
                "linear_span_anchor: atom {atom_idx} has no recoverable linear span rank"
            ));
        }
        let weighted = residual.mapv(|v| gate * v);
        let (_u_opt, singular_values_full, vt_opt) = weighted
            .svd(false, true)
            .map_err(|err| format!("linear_span_anchor: SVD failed for atom {atom_idx}: {err}"))?;
        let vt = vt_opt.ok_or_else(|| {
            format!("linear_span_anchor: SVD returned no right factor for atom {atom_idx}")
        })?;
        let rank = requested_rank
            .min(vt.nrows())
            .min(singular_values_full.len());
        if rank == 0 {
            return Err(format!(
                "linear_span_anchor: atom {atom_idx} SVD returned rank zero"
            ));
        }
        let mut frame = Array2::<f64>::zeros((p, rank));
        for col in 0..rank {
            for row in 0..p {
                frame[[row, col]] = vt[[col, row]];
            }
        }
        let singular_values = singular_values_full.slice(s![..rank]).to_owned();
        let frame = GrassmannFrame::from_oriented(frame, singular_values.clone());
        let frame_matrix = frame.frame().to_owned();
        let mut coordinates = residual.dot(&frame_matrix);
        coordinates.mapv_inplace(|v| v / gate);
        let contribution = fast_abt(&coordinates, &frame_matrix).mapv(|v| gate * v);
        reconstruction += &contribution;
        residual -= &contribution;
        atoms.push(LinearSpanAtomAnchor {
            gate_weight: gate,
            frame,
            decoder_coordinates: coordinates,
            singular_values,
        });
    }
    let residual_norm_sq = residual.iter().map(|v| v * v).sum();
    Ok(LinearSpanAnchor {
        atoms,
        reconstruction,
        residual_norm_sq,
    })
}

pub(crate) fn sae_cholesky_solve_neg_gradient(
    h: ArrayView2<'_, f64>,
    g: ArrayView1<'_, f64>,
) -> Result<Array1<f64>, String> {
    let n = h.nrows();
    if h.ncols() != n || g.len() != n {
        return Err(format!(
            "sae_cholesky_solve_neg_gradient: shape mismatch H={:?}, g={}",
            h.dim(),
            g.len()
        ));
    }
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = h[[i, j]];
            for k in 0..j {
                sum -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if !(sum.is_finite() && sum > 0.0) {
                    return Err(format!("non-positive Cholesky pivot at {i}: {sum}"));
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut sum = -g[i];
        for k in 0..i {
            sum -= l[[i, k]] * y[k];
        }
        y[i] = sum / l[[i, i]];
    }
    let mut x = Array1::<f64>::zeros(n);
    for ii in 0..n {
        let i = n - 1 - ii;
        let mut sum = y[i];
        for k in i + 1..n {
            sum -= l[[k, i]] * x[k];
        }
        x[i] = sum / l[[i, i]];
    }
    if !x.iter().all(|v| v.is_finite()) {
        return Err("sae_cholesky_solve_neg_gradient: non-finite solution".into());
    }
    Ok(x)
}

pub(crate) fn solve_basis_transport(
    new_phi: ArrayView2<'_, f64>,
    old_phi: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, String> {
    solve_design_least_squares(new_phi, old_phi)
}

pub(crate) fn transport_smooth_penalty_for_decoder(
    decoder_transport: ArrayView2<'_, f64>,
    old_smooth_penalty: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, String> {
    let m = decoder_transport.nrows();
    if decoder_transport.ncols() != m {
        return Err(format!(
            "transport_smooth_penalty_for_decoder: decoder transport must be square; got {:?}",
            decoder_transport.dim()
        ));
    }
    if old_smooth_penalty.dim() != (m, m) {
        return Err(format!(
            "transport_smooth_penalty_for_decoder: smooth penalty shape {:?} != ({m}, {m})",
            old_smooth_penalty.dim()
        ));
    }
    let transport_inverse =
        solve_design_least_squares(decoder_transport, Array2::<f64>::eye(m).view())?;
    Ok(fast_atb(
        &transport_inverse,
        &fast_ab(&old_smooth_penalty.to_owned(), &transport_inverse),
    ))
}

pub(crate) fn solve_design_least_squares(
    design: ArrayView2<'_, f64>,
    rhs: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, String> {
    if design.nrows() != rhs.nrows() {
        return Err(format!(
            "solve_design_least_squares: row mismatch design={} rhs={}",
            design.nrows(),
            rhs.nrows()
        ));
    }
    let (u_opt, sigma, vt_opt) = design
        .to_owned()
        .svd(true, true)
        .map_err(|err| format!("solve_design_least_squares: SVD failed: {err}"))?;
    let u = u_opt.ok_or_else(|| "solve_design_least_squares: SVD omitted U".to_string())?;
    let vt = vt_opt.ok_or_else(|| "solve_design_least_squares: SVD omitted Vt".to_string())?;
    let smax = sigma.iter().fold(0.0_f64, |acc, &v| acc.max(v));
    if !(smax.is_finite() && smax > 0.0) {
        return Err("solve_design_least_squares: design has zero numerical rank".to_string());
    }
    let cutoff = smax * f64::EPSILON * (design.nrows().max(design.ncols()) as f64);
    let coeffs = u.t().dot(&rhs);
    let mut scaled = Array2::<f64>::zeros(coeffs.dim());
    for row in 0..sigma.len() {
        if sigma[row] > cutoff {
            let inv = 1.0 / sigma[row];
            for col in 0..rhs.ncols() {
                scaled[[row, col]] = inv * coeffs[[row, col]];
            }
        }
    }
    Ok(vt.t().dot(&scaled))
}
