//! The #784 block-local Gauss-Hermite marginal correction.
//!
//! Extracted verbatim from `gradient_hessian.rs`, which the repo's own
//! 10,000-line gate refuses to carry (#780). It is one self-contained unit:
//! the per-bundle cached wrapper, the compute path that runs the skewness
//! diagnostic, selects the curvature-heavy block, integrates the non-Gaussian
//! remainder over it — axis by axis plus the analytic mixed-axis term
//! ([`mixed_axis_laplace_term`]) when the block has more than one axis — and
//! assembles the four exact gradient channels the splice's objective-gradient
//! contract requires.
//!
//! Nothing here changed in the move - same `impl RemlState`, same visibility,
//! so every call site is untouched.

use super::*;

impl<'a> RemlState<'a> {
    /// Adaptive, block-local Laplace-to-sampling fallback for the inner
    /// marginalization loop (issue #784).
    ///
    /// The unified evaluator summarizes the coefficient posterior by its Laplace
    /// (Gaussian) moments. This method audits that summary per curvature
    /// direction and, where the Gaussian approximation is *not* trustworthy,
    /// replaces it with a sampling-based block marginal — keeping the cheap
    /// Laplace summary everywhere else:
    ///
    /// 1. Run the directional cubic non-Gaussianity diagnostic on the observed
    ///    penalized Hessian + the third-derivative weights `solve_c_array`,
    ///    yielding per-eigendirection standardized skewness `γ_r`.
    /// 2. Convert `γ_r` into a block-local activation set via the auto-derived
    ///    threshold `τ(n_eff)` (no flag). The flagged eigenvectors span the
    ///    curvature-heavy subspace `V_b`.
    /// 3. Importance-sample the true block marginal against the local Laplace
    ///    Gaussian (reusing the whitening) and return the additive correction
    ///    `Δ_b` to the marginal log-likelihood, together with its consistent
    ///    ρ-gradient, so the outer REML/LAML stays consistent.
    ///
    /// Returns `TkCorrectionTerms` whose `value` is added to the REML cost.
    /// Because `Δ_b` is added to the *marginal log-likelihood* it is subtracted
    /// from the cost, so the returned `value` is `−Δ_b` (likewise the gradient).
    /// The gradient is laid out over the ρ coordinates and zero-extended over
    /// external coordinates to match the unified evaluator's coordinate set in
    /// `apply_tk_to_result`.
    ///
    /// A no-op (zeros) is returned for Gaussian-identity fits (Laplace is
    /// exact), when no direction trips the threshold, or when the importance
    /// estimate is not trustworthy (low ESS) — in which case the plain Laplace
    /// summary is retained rather than splicing in a noisy correction.
    ///
    /// # Outer-consistency / continuity
    ///
    /// **The activation used to be a predicate evaluated at ρ, and the argument
    /// that this was harmless is measured false (#2748).** It read:
    ///
    /// > a direction crosses the threshold only at `|γ_r| ≈ τ = O(n^{−1/2})`,
    /// > so its contribution to `Δ_b` is `O(γ_r²) = O(1/n)` — the same order as
    /// > the Laplace floor error the criterion already carries. The correction
    /// > value therefore vanishes continuously as a direction approaches the
    /// > threshold.
    ///
    /// `(5/24)γ_r²` is the CUBIC term, and `τ` is defined as the `γ` at which
    /// exactly that term equals `1/n_eff`. But the quadrature integrates the
    /// FULL non-Gaussian remainder, so what the predicate switches is the cubic
    /// term plus every higher one. Measured on `haberman_5yr` at the crossing:
    /// `1/n_eff = 3.268e-3`, `(5/24)γ² = 3.26e-3` as designed — and
    /// `Δ_b = 3.1144e-2`, **9.5× the floor the argument bounds it by.** The
    /// correction does not vanish as a direction approaches `τ`; it arrives at
    /// full size.
    ///
    /// The consequence is not a bias, it is a fit that cannot exist. `V` jumped
    /// `1.744282e2 → 1.744593e2` — exactly `Δ_b` — between two adjacent
    /// line-search trial points at `|g| = 2.045e-2`, and no sufficient-decrease
    /// test can pass across a jump larger than `c₁·α·|gᵀd|` for any `α`. Worse,
    /// the ON region's minimum SITS on the switching surface (declining the
    /// correction raises the cost by `Δ_b`, so the descent is drawn to the
    /// boundary and stays there): `max|γ|` walked `0.141 → 0.126 → 0.125` onto
    /// `τ = 0.125` and stopped. Eleven `matern` scenarios died of a different
    /// cause; `haberman_5yr` died of this one.
    ///
    /// So the admission is now a property of the MODEL, latched on first
    /// admission and held for the fit
    /// ([`RemlState::block_correction_admission`]), and the block is the
    /// `m` largest-`|γ_r|` positive-curvature directions at each ρ rather than
    /// a set defined by a threshold crossing. The spliced objective is a
    /// function of ρ again, and the spliced gradient stays exact: the four
    /// channels differentiate `Δ_b` at a fixed block, and a ρ-dependent
    /// admission would contribute a term they do not carry — the same
    /// objective↔gradient desync this site already declines the splice over for
    /// ψ coordinates and for the Beta family.
    ///
    /// Where the admission is decided is [`RemlState::block_correction_decision`]
    /// (#1082). A search decides it once, at its certified Laplace optimum,
    /// because a latch on the first engaged evaluation made the fitted criterion
    /// a function of where the search started.
    ///
    /// Per-bundle-cached wrapper around [`Self::block_local_quadrature_correction_compute`].
    ///
    /// The block-local correction is a deterministic function of this bundle's
    /// converged inner state and ρ alone (mode-invariant, Hessian-free), but the
    /// outer loop evaluates the objective at one ρ up to three times (value,
    /// value+gradient, value+gradient+Hessian) sharing the SAME `bundle`. The
    /// expensive engaged path (dense O(p³) eigendecomposition plus the
    /// fixed-seed O(draws·n·m) importance sampler) therefore reran 2–3× per
    /// outer iteration. Hoist it onto `bundle.block_local_correction` so it is
    /// computed exactly once per inner solution and every consumer at that ρ
    /// reads the identical value+gradient (exact hoist — #784, #1082). Keyed on
    /// `n_ext`, which is fixed for a fit, so one cell suffices.
    pub(crate) fn block_local_quadrature_correction(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        n_ext: usize,
    ) -> Result<TkCorrectionTerms, EstimationError> {
        // A deferred search prices the Laplace criterion, which is not this
        // bundle's correction once the admission is decided, so it is not cached.
        if self.block_correction_admission_deferred() {
            return self.block_local_quadrature_correction_compute(rho, bundle, n_ext);
        }
        if let Some((cached_ext, terms, audit)) = bundle.block_local_correction.get()
            && *cached_ext == n_ext
        {
            // Re-publish the audit record the computing call wrote: the window
            // was cleared at the start of THIS assemble call, so without this a
            // ρ whose splice engaged reads back as declined on every assemble
            // after the first (#2623).
            if let Some(record) = audit.as_ref() {
                crate::estimate::outer_eval_capture::record_quadrature_marginal(record.clone());
            }
            return Ok((**terms).clone());
        }
        let terms = self.block_local_quadrature_correction_compute(rho, bundle, n_ext)?;
        let audit = crate::estimate::outer_eval_capture::last_quadrature_marginal_record();
        // First writer wins; a racing writer built from identical inputs, so
        // either stored object is correct. A `set` that loses the race (cell
        // already filled) is fine — both terms are equal — so the `Err` is
        // discarded by returning the freshly computed `terms` either way.
        match bundle
            .block_local_correction
            .set((n_ext, std::sync::Arc::new(terms.clone()), audit))
        {
            Ok(()) => Ok(terms),
            Err(_) => Ok(terms),
        }
    }

    fn block_correction_decision_guard(
        &self,
    ) -> std::sync::MutexGuard<'_, BlockCorrectionDecision> {
        self.block_correction_decision
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Price the Laplace criterion until the search certifies its optimum, and
    /// decide the correction's admission there (#1082,
    /// [`BlockCorrectionDecision::DeferredToOptimum`]).
    pub(crate) fn defer_block_correction_admission(&self) {
        *self.block_correction_decision_guard() = BlockCorrectionDecision::DeferredToOptimum;
    }

    /// Whether the #784 correction is latched into this fit's criterion. Its
    /// value and exact ρ-gradient are spliced, but `Δ_b` has no analytic
    /// ρ-Hessian, so a latched criterion declares none: the search continues on
    /// BFGS curvature and the smoothing correction is typed-unavailable, as for
    /// a non-canonical Firth link.
    pub(crate) fn block_correction_latched(&self) -> bool {
        self.block_correction_admission
            .load(std::sync::atomic::Ordering::Relaxed)
            > 0
    }

    pub(crate) fn block_correction_admission_deferred(&self) -> bool {
        *self.block_correction_decision_guard() == BlockCorrectionDecision::DeferredToOptimum
    }

    /// Decide the deferred admission once, at the search's certified Laplace
    /// optimum `rho` (#1082). One criterion evaluation there runs the skewness
    /// verdict and, when it engages, the order search that latches the block,
    /// exactly as a first admission does (#2748).
    ///
    /// Returns whether the correction was admitted. If it was, `rho` was
    /// certified under a criterion that is no longer the model's, and the caller
    /// continues the corrected search from it. A correction refused at `rho` is
    /// the fit's error: the verdict requires the correction at the point the fit
    /// would publish, and it cannot be evaluated there.
    pub(crate) fn decide_block_correction_admission(
        &self,
        rho: &Array1<f64>,
    ) -> Result<bool, EstimationError> {
        *self.block_correction_decision_guard() = BlockCorrectionDecision::DecidingAtOptimum;
        // Every cached evaluation priced the Laplace criterion, and the decision
        // is taken at the terminal inner mode, not a capped screening one.
        self.reset_outer_seed_state();
        self.compute_cost(rho)?;
        let mut decision = self.block_correction_decision_guard();
        // An unconditional decline (the family or the hyper-layout) returns
        // before the verdict, so the decision is still open here.
        if *decision == BlockCorrectionDecision::DecidingAtOptimum {
            *decision = BlockCorrectionDecision::DeclinedAtOptimum;
        }
        Ok(*decision == BlockCorrectionDecision::AdmittedAtOptimum)
    }

    fn block_local_quadrature_correction_compute(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        n_ext: usize,
    ) -> Result<TkCorrectionTerms, EstimationError> {
        // #1521 trait-inversion: the #784 importance-sampling correction and its
        // eigen-diagnostic live UP in the gam-inference `hmc_io` tier; gam-solve
        // calls them through the neutral `gam_problem` sampler contract instead
        // of a back-edge into the inference SCC. The pure threshold math
        // (`laplace_trustworthiness_from_skewness`) moved down outright.
        use gam_problem::estimation_error::BlockQuadratureCorrectionStage;
        use gam_problem::laplace_sampler_contract::laplace_trustworthiness_from_skewness;

        let n_rho = self.canonical_penalties.len();
        let zero = || TkCorrectionTerms {
            value: 0.0,
            gradient: Some(Array1::zeros(n_rho + n_ext)),
            hessian: None,
        };

        // A search deciding at its optimum prices the Laplace criterion until
        // then, and a correction declined there stays declined for the fit.
        if matches!(
            *self.block_correction_decision_guard(),
            BlockCorrectionDecision::DeferredToOptimum | BlockCorrectionDecision::DeclinedAtOptimum
        ) {
            return Ok(zero());
        }

        // Laplace is exact for the Gaussian-identity model: nothing to correct.
        if reml_is_gaussian_identity(&self.config.likelihood) {
            return Ok(zero());
        }
        // The mode and trace channels need one λ per canonical penalty.
        if rho.len() != n_rho || n_rho == 0 {
            return Ok(zero());
        }

        let pirls_result = bundle.pirls_result.as_ref();
        // Operate in the transformed basis, where `h_total`, `solve_c_array`,
        // `final_eta`, `finalweights`, `beta_transformed` and `x_transformed`
        // are all mutually consistent.
        let h_total = bundle.h_total.as_ref();
        let c_weights = &pirls_result.solve_c_array.to_owned();
        let x_design = &pirls_result.x_transformed;
        let p = h_total.nrows();
        if p == 0 || c_weights.len() != x_design.nrows() {
            return Ok(zero());
        }

        // The correction integrates over a dense copy of the design, which the
        // memory governor admits or refuses with a typed error. Below its cap the
        // skewness verdict decides at every n and p; no picked problem-scale cap
        // switches the criterion off (gam#2900).
        block_correction_design_admission(
            x_design.nrows(),
            p,
            gam_runtime::resource::MemoryGovernor::global().single_materialization_cap_bytes(),
        )?;

        // ── Unconditional declines, BEFORE any evidence is bought ────────────
        //
        // The two predicates below decline the whole correction, and neither
        // consults a single number the diagnostic produces: one reads the
        // hyper-layout, the other the configured response family. Both are
        // therefore constant across the entire fit.
        //
        // They used to sit AFTER `directional_cubic_diagnostic` — an `O(p³)`
        // dense factorization plus `O(n·p)` cubic contractions — so every
        // ψ-carrying model and every Beta fit paid that sweep on EVERY outer
        // evaluation and then discarded it, guaranteed, with the decline logged
        // as though it had been decided on evidence (gam#2584). Evidence is
        // worth buying only when the verdict can depend on it.
        //
        // Hoisting them is exactly value-preserving: neither predicate reads
        // `sampler`, `max_abs`, `directional` or `verdict`, and every path they
        // guard returns the same `zero()` it returned before.

        // External (ψ) hyper-coordinates present: the exact gradient of the
        // realized estimator along ψ requires the field motion of `X(ψ)`,
        // `S(ψ)` and the reparameterized basis — moments this seam does not
        // yet carry. A spliced value whose ψ-gradient entries are zeroed (or
        // truncated) is an objective↔gradient desync (#901, the #752/#748
        // bug class); per the gradient exactness contract on
        // `block_quadrature_marginal_correction`, the correct response is to
        // DECLINE the splice — value AND gradient together — rather than
        // approximate.
        if n_ext > 0 {
            log::trace!(
                "[#784] block-local fallback declined before the skewness diagnostic: \
                 {n_ext} external (ψ) coordinate(s) present and the ψ-exact gradient \
                 channels are not implemented; splicing a ψ-truncated gradient would \
                 desync objective and gradient (#901)"
            );
            return Ok(zero());
        }
        // The exact score channel relies on the exponential-family unit-
        // deviance identity dD/dμ = −2w(y−μ)/V(μ), which does not hold for
        // the Beta pseudo-family parameterization. Decline rather than splice
        // a gradient that is not the derivative of the spliced value.
        if matches!(
            reml_spec(&self.config.likelihood).response,
            ResponseFamily::Beta { .. }
        ) {
            log::trace!(
                "[#784] block-local fallback declined before the skewness diagnostic: \
                 Beta family has no exponential-family score identity for the exact \
                 gradient channels"
            );
            return Ok(zero());
        }
        // Firth/Jeffreys fits: the integrand `Gam784BlockTarget::excess` is the
        // remainder of the PLAIN penalized likelihood about its mode, but under
        // Firth β̂ is the mode of the Jeffreys-penalized objective. The plain
        // remainder then keeps a linear term (∇Φ(β̂) ≠ 0), omits the Jeffreys
        // change Φ(β̂+δ)−Φ(β̂), and subtracts only XᵀWX while the draws are
        // scaled by `h_total`, which carries −H_Φ. On separated data that
        // mis-targeted Δ_b is orders of magnitude above 1/n_eff and drags the
        // criterion off the certified Laplace surface, so the outer search it
        // is spliced into cannot certify. Decline — value and gradient
        // together — until the Jeffreys term is integrated.
        if reml_robust_jeffreys_link(&self.config).is_some() {
            log::debug!(
                "[#784] block-local fallback declined before the skewness diagnostic: \
                 Firth/Jeffreys bias reduction is active and the block target \
                 integrates the plain penalized likelihood, not the Jeffreys-penalized one"
            );
            return Ok(zero());
        }

        // Resolve the injected gam-inference corrector. When the inference tier
        // is not linked / registered, decline the correction (zero contribution) —
        // the same safe no-op as every other decline branch here.
        let Some(corrector) = gam_problem::laplace_sampler_contract::laplace_marginal_corrector()
        else {
            return Ok(zero());
        };

        // The eigensystem every step below reads: the block's directions `v_r`,
        // their curvatures `λ_r`, and the resolvent the mode and gap terms are
        // built from. It is the criterion's own spectral operator for this ρ
        // whenever the criterion priced one on the full frame. An `eigh` of the
        // assembled `H` resolves each eigenvalue only to `O(ε·‖H‖)`; with one λ
        // railed, `‖H‖` is many orders above the soft modes this block lives
        // on, so that error is a visible fraction of `λ_r` and of `v_r`, and it
        // changes with the last bits of `H` from one inner solve to the next.
        // `Δ_b` then differs between two evaluations at the same ρ by far more
        // than the outer line search can resolve, and the BFGS continuation
        // stalls on noise. The criterion's operator is priced from the root
        // `B` with `H = BᵀB` when the assembled spectrum cannot resolve
        // `log|H|` (#2644), which prices each soft mode to its own scale — the
        // same eigenpairs the Laplace term was priced on.
        //
        // The decision is published by the spectral assembly. A value-only
        // probe on the transformed route prices a Cholesky factor and publishes
        // none, so the assembly is run once at a derivative order to obtain it,
        // exactly as `criterion_rank_decision_at` does: `Δ_b` must be one
        // function of ρ whether the evaluation carries a gradient or not.
        // Only a sparse-exact backend, or a spectral operator priced on an
        // active-constraint face's free basis, leaves no full-frame decision;
        // the criterion priced no spectrum of this frame's `H` there, so the
        // assembled matrix's own is the only one.
        if bundle.criterion_rank_decision().is_none()
            && bundle.backend_kind() != GeometryBackendKind::SparseExactSpd
        {
            self.build_auto_assembly(
                rho,
                bundle,
                super::reml_outer_engine::EvalMode::ValueAndGradient,
                false,
                false,
            )?;
        }
        let sym_h = (h_total + &h_total.t()) * 0.5;
        let (evals, evecs) = match bundle.criterion_rank_decision() {
            Some(decision) => {
                let evals = Array1::from(decision.operator.raw_eigenvalues.clone());
                let evecs = match decision.frame {
                    CriterionFrame::Transformed => decision.operator.eigenvectors.clone(),
                    // `H_orig = Qs·H·Qsᵀ` with `Qs` orthogonal, so the transformed
                    // frame's eigenvectors are `Qsᵀ·V_orig` with the same eigenvalues.
                    CriterionFrame::Original => match pirls_result.coordinate_frame {
                        pirls::PirlsCoordinateFrame::TransformedQs => pirls_result
                            .reparam_result
                            .qs
                            .t()
                            .dot(&decision.operator.eigenvectors),
                        pirls::PirlsCoordinateFrame::OriginalSparseNative => {
                            decision.operator.eigenvectors.clone()
                        }
                    },
                };
                log::trace!(
                    "[#784] block eigensystem from the criterion's {:?} operator ({:?} frame)",
                    decision.predicate,
                    decision.frame,
                );
                (evals, evecs)
            }
            None => sym_h.eigh(Side::Lower).map_err(|e| {
                EstimationError::InvalidInput(format!(
                    "#784 block-local fallback eigendecomposition failed: {e}"
                ))
            })?,
        };
        if evals.len() != p || evecs.dim() != (p, p) {
            return Err(EstimationError::InvalidInput(format!(
                "#784 block eigensystem has {} eigenvalues and {}x{} eigenvectors for p={p}",
                evals.len(),
                evecs.nrows(),
                evecs.ncols()
            )));
        }

        // Step 1: per-direction skewness diagnostic γ_r, aligned to those pairs.
        let (max_abs, directional) = corrector
            .directional_cubic_diagnostic(&evals, &evecs, x_design, c_weights, false)
            .map_err(EstimationError::InvalidInput)?;
        if !max_abs.is_finite() || max_abs == 0.0 {
            return Ok(zero());
        }

        // Step 2: auto-derived, block-local activation. `n_eff` is the number of
        // observations carrying curvature; using it (not the raw n) keeps the
        // verdict tied to the actual information content.
        let n_eff = c_weights.iter().filter(|&&c| c != 0.0).count() as f64;
        let verdict = laplace_trustworthiness_from_skewness(&directional, n_eff);

        // The admission this fit already latched, if any (#2748). `0` means the
        // correction has never been admitted, so the τ predicate still decides
        // whether it is admitted HERE — which is what makes a fit that never
        // engages bit-identical to the pre-#2748 fit. Once admitted, the block
        // dimension is the model's and `τ` no longer switches anything.
        let latched_block_dim = self
            .block_correction_admission
            .load(std::sync::atomic::Ordering::Relaxed)
            .checked_sub(1);
        if latched_block_dim.is_none() && !verdict.fallback_required() {
            if *self.block_correction_decision_guard() == BlockCorrectionDecision::DecidingAtOptimum
            {
                log::debug!(
                    "[#784] block-local correction DECLINED for this fit at its certified Laplace \
                     optimum: max|γ|={:.4e} against τ={:.4e} (#1082)",
                    verdict.max_abs_skewness,
                    verdict.threshold,
                );
            }
            return Ok(zero());
        }

        // Build the block subspace V_b. Under a latched admission the block is
        // the `m` largest-|γ_r| positive-curvature directions, NOT the set that
        // happens to clear `τ` at this ρ: a set defined by a threshold crossing
        // changes cardinality as ρ moves, and every change is a jump of a whole
        // direction's contribution to `Δ_b`. Ranking is the continuous
        // extension of the same rule — it agrees with it exactly wherever the
        // flagged set has the latched size, which is every ρ the pre-#2748 fit
        // was already stable on.
        let mut admissible: Vec<usize> = (0..evals.len().min(directional.len()))
            .filter(|&r| evals[r] > 0.0 && directional[r].is_finite())
            .collect();
        let mut block_cols: Vec<usize> = match latched_block_dim {
            Some(m) => {
                // Descending |γ_r|, ties broken by index so the selection is a
                // deterministic function of (H, γ) and not of sort stability.
                admissible.sort_by(|&a, &b| {
                    directional[b]
                        .abs()
                        .partial_cmp(&directional[a].abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.cmp(&b))
                });
                admissible.truncate(m);
                admissible
            }
            None => verdict
                .untrustworthy_directions
                .iter()
                .copied()
                .filter(|&r| r < evals.len() && evals[r] > 0.0)
                .collect(),
        };
        order_block_axes_by_curvature(&mut block_cols, &evals);
        if block_cols.is_empty() {
            return Ok(zero());
        }
        let m = block_cols.len();
        let mut block_vecs = Array2::<f64>::zeros((p, m));
        let mut block_lambdas = Array1::<f64>::zeros(m);
        for (j, &r) in block_cols.iter().enumerate() {
            block_vecs.column_mut(j).assign(&evecs.column(r));
            block_lambdas[j] = evals[r];
        }

        // Penalty scores S_k β̂ in the TRANSFORMED frame, and λ_k = e^{ρ_k}.
        // β̂ = `pirls_result.beta_transformed` lives in the stable
        // reparameterized basis, so the penalties contracted against it MUST
        // be `reparam_result.canonical_transformed` — per its own doc, "the
        // single source of truth for penalty roots in the transformed frame"
        // for exactly this TK-correction path. Contracting the ORIGINAL-frame
        // `self.canonical_penalties` here (gam#2623) made the spliced
        // ρ-gradient wrong by 4.6e-2 to 1.0 relative — sign-inverted with
        // nine orders of error on the worst cells — whenever ‖Q_s − I‖ was
        // large, which is what turned one outer evaluation into 178 on the
        // fold that opened that issue. Computed once per inner solution on
        // the eval bundle and reused across every assemble call sharing this
        // bundle (exact hoist, identical values for every consumer).
        let transformed_penalties = pirls_result.reparam_result.canonical_transformed.as_slice();
        let penalty_scores = bundle.canonical_penalty_scores_at_mode(transformed_penalties)?;
        let lambdas = gam_problem::checked_exp_log_strengths(rho.iter().copied())?;

        // Scale converting the REPORTED deviance into negative log-likelihood.
        // Gaussian's reported deviance already includes a fixed dispersion;
        // Beta's deviance is defined directly as twice a saturated-loglikelihood
        // difference and already contains its precision.  Dividing either a
        // second time would change the sampled objective. Gamma and Tweedie,
        // by contrast, deliberately report unscaled deviance and need their
        // EDM dispersion here.
        //
        // The dispersion, the base rows and the row oracle must read the
        // likelihood the inner solve converged under, `pirls_result.likelihood`,
        // not the configuration it started from: PIRLS estimates the Gamma shape
        // from the warm-start η and locks it for the solve, so `β̂`, `H` and the
        // Laplace Gaussian the quadrature corrects are all at that shape. Scoring
        // the remainder `ΔF` at the configuration's shape instead measures a
        // different posterior against that Gaussian. On `y ~ te(x, z)`
        // gamma-log (n=600) the configured φ = 1 against the solve's 0.2772 made
        // `log E_q[e^{−ΔF}]` climb past 4.4 without converging, where the solve's
        // own shape gives −4.5e-3.
        let likelihood = &pirls_result.likelihood;
        let phi = match reml_spec(likelihood).response {
            ResponseFamily::Gaussian | ResponseFamily::Beta { .. } => 1.0,
            _ => reml_fixed_glm_dispersion(likelihood)?,
        };
        if !(phi.is_finite() && phi > 0.0) {
            return Err(EstimationError::InvalidInput(format!(
                "#784 block-local fallback requires finite positive dispersion; got {phi}"
            )));
        }

        let x_dense = x_design
            .try_to_dense_arc("#784 block-local fallback requires dense design access")
            .map_err(EstimationError::InvalidInput)?;

        let eta_hat = pirls_result.final_eta.to_owned();
        let inverse_link = self.runtime_inverse_link();
        let base_rows = crate::pirls::deviance_eta_rows_with_log_measure_scale(
            self.y.view(),
            &eta_hat,
            likelihood,
            &inverse_link,
            self.weights.view(),
            -phi.ln(),
        )?;
        let base_half_values: Vec<f64> = base_rows.iter().map(|row| row.half_deviance).collect();
        let base_absolute_half_deviance: f64 = base_half_values.iter().map(|value| value.abs()).sum();
        let base_scaled_half_deviance = crate::pirls::stable_finite_signed_sum(
            &base_half_values,
            "#784 base scaled half-deviance",
        )?;
        let base_neg_score_at_mode =
            Array1::from_iter(base_rows.into_iter().map(|row| row.eta_score));
        gam_linalg::matrix::FiniteSignedWeightsView::try_new(pirls_result.finalweights.view())
            .map_err(EstimationError::InvalidInput)?;
        let weights_obs = pirls_result.finalweights.to_owned();
        let weights_obs_log_abs = weights_obs.mapv(|weight| {
            if weight == 0.0 {
                f64::NEG_INFINITY
            } else {
                weight.abs().ln()
            }
        });

        let target = Gam784BlockTarget {
            x_transformed: x_dense.as_ref(),
            block_design: Gam784BlockDesign::new(x_dense.as_ref(), &block_vecs),
            block_vecs,
            block_lambdas,
            eta_hat,
            weights_obs,
            weights_obs_log_abs,
            y: self.y.to_owned(),
            prior_weights: self.weights.to_owned(),
            row_measures: crate::pirls::DevianceRowMeasure::rows(self.weights.view(), -phi.ln()),
            likelihood: likelihood.clone(),
            inverse_link,
            phi,
            penalty_scores,
            penalties: transformed_penalties,
            lambdas,
            base_scaled_half_deviance,
            base_neg_score_at_mode,
            base_absolute_half_deviance,
        };

        // The correction exists to remove the O(1/n_eff) Laplace term, so its
        // quadrature error must sit below the next-order remainder 1/n_eff²
        // (#2623). Each axis's Gauss–Hermite order is selected ONCE, at
        // admission, as the smallest order whose paired difference with the next
        // lower rule resolves min(|Δ|, 1/n_eff²), and latched beside the block
        // dimension. Under a latched admission the orders are the model's, so
        // every ρ integrates against the same nodes and the value, gradient and
        // moments share one measure. The paired lower rules then switch nothing
        // (#2748), so a latched evaluation integrates the fine rule alone and
        // carries the admission's paired errors as its certificate: on a
        // three-axis block the lower rules are five times the fine rule's nodes.
        // A one-axis piece is the exception: its composite rule adapts to every
        // evaluation's axis (below).
        let laplace_floor = if n_eff > 0.0 {
            1.0 / n_eff
        } else {
            f64::INFINITY
        };
        let next_order_remainder = laplace_floor * laplace_floor;
        let latched_quadrature = self
            .block_correction_axis_orders
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
            .filter(|latch| latch.axis_orders.len() == m);

        // ── Axis by axis, or one tensor rule ─────────────────────────────
        //
        // A tensor Gauss–Hermite rule over the whole block costs Π_r o_r nodes,
        // and each axis resolves 1/n_eff² only at o_r ≈ 9–11. On a Poisson
        // `te(x0, x1)` fit at n = 1e4 the block reached m = 7: the search sat at
        // 4^7 = 16384 nodes, each evaluation took 5.5 minutes, and its paired
        // errors (~1e-4) were four orders above the 1e-8 target, so the fit never
        // finished. No tensor rule on that block is feasible.
        //
        // The whitened coordinates z_r = √λ_r t_r are independent under the
        // Laplace Gaussian, and the ANOVA split of the block marginal is
        //
        //   Δ_b = Σ_r Δ_r + Φ + O(n_eff⁻²),
        //
        // where Δ_r is the exact one-dimensional correction along axis r (the
        // same target restricted to one column of V_b, integrated by its own
        // certified 1-D rule) and Φ is the analytic mixed-axis part of the
        // second-order Laplace expansion ([`mixed_axis_laplace_term`]). Every
        // single-axis term is in Δ_r at every order; what Φ leaves out are the
        // mixed-axis terms past O(n_eff⁻¹). The O(n_eff⁻³ᐟ²) ones are odd
        // Gaussian moments and vanish, so the truncation is O(n_eff⁻²) — the same
        // order as the remainder the quadrature is asked to resolve. The cost is
        // Σ_r o_r nodes plus O(n·m³) for Φ.
        //
        // Φ reads the likelihood's third and fourth η-derivatives of the
        // curvature, so the split requires the exported curvature to be the
        // likelihood's own: observed information, or a canonical link where it
        // coincides with the Fisher weights. It also takes the linear term of the
        // excess to cancel, which is the mode condition Sβ̂ = ∇ℓ(β̂). A block of
        // one axis has no mixed term and keeps the single rule. Whether the block
        // is split is latched with the orders, so a latched fit integrates every
        // ρ with the same rule.
        let exact_curvature = !pirls_result.derivatives_unsupported
            && matches!(pirls_result.firth, crate::pirls::FirthDiagnostics::Inactive)
            && (matches!(
                pirls_result.exported_laplace_curvature,
                crate::pirls::ExportedLaplaceCurvature::ObservedExact
            ) || matches!(
                (&reml_spec(likelihood).response, &target.inverse_link),
                (
                    ResponseFamily::Poisson,
                    InverseLink::Standard(StandardLink::Log)
                ) | (
                    ResponseFamily::Binomial,
                    InverseLink::Standard(StandardLink::Logit)
                )
            ));
        let axis_split = match &latched_quadrature {
            Some(latch) => latch.axis_split,
            None => m >= 2 && exact_curvature,
        };
        if axis_split && !exact_curvature {
            return Err(EstimationError::BlockQuadratureCorrectionRefused {
                stage: BlockQuadratureCorrectionStage::AxisSplitWithoutExactCurvature {
                    block_dim: m,
                },
            });
        }

        // Integrate each piece and contract its moments into the gradient
        // channels at once, so one axis target (and its moments) is alive at a
        // time.
        let piece_count = if axis_split { m } else { 1 };
        let mut pieces: Vec<BlockPieceQuadrature> = Vec::with_capacity(piece_count);
        let mut axis_orders: Vec<usize> = Vec::with_capacity(m);
        for k in 0..piece_count {
            let (first_axis, width) = if axis_split { (k, 1) } else { (0, m) };
            let axis_target;
            let piece_target = if axis_split {
                axis_target = block_axis_target(&target, k);
                &axis_target
            } else {
                &target
            };
            // Name a refused search's axis and orders in the block's frame.
            let order_search_refused =
                |mut refusal: gam_problem::laplace_sampler_contract::BlockQuadratureOrderRefusal| {
                    refusal.axis += first_axis;
                    let mut block_orders = axis_orders.clone();
                    block_orders.extend_from_slice(&refusal.axis_orders);
                    refusal.axis_orders = block_orders;
                    EstimationError::BlockQuadratureCorrectionRefused {
                        stage: BlockQuadratureCorrectionStage::OrderSearchRefused(refusal),
                    }
                };
            let mut quadrature = if width == 1 {
                // One axis: the composite Gauss–Kronrod rule, whose bisection resolves a
                // wall no representable Gauss–Hermite order reaches. Its partition is
                // adapted at every evaluation, latched or not: the axis is a Hessian
                // eigenvector at this ρ, whose sign is arbitrary and whose wall moves,
                // and on adult the admission's partitions carried to the next ρ left
                // every axis near 1e-3 against its 1.5e-9 target. Each evaluation's own
                // rule resolves its target, so the criterion moves by less than the
                // targets between rules, and its gradient is the derivative of the rule
                // that priced it.
                corrector
                    .composite_axis_marginal_correction(piece_target, next_order_remainder)
                    .map_err(order_search_refused)?
                    .marginal
            } else {
                match &latched_quadrature {
                    Some(latch) => corrector
                        .block_quadrature_marginal_correction_at_certified_orders(
                            piece_target,
                            &latch.axis_orders[first_axis..first_axis + width],
                            &latch.axis_quadrature_errors[first_axis..first_axis + width],
                        )
                        .map_err(|refusal| EstimationError::InvalidInput(refusal.to_string()))?,
                    None => gam_problem::laplace_sampler_contract::select_block_quadrature_orders(
                        corrector,
                        piece_target,
                        next_order_remainder,
                    )
                    .map_err(order_search_refused)?,
                }
            };
            axis_orders.extend_from_slice(&quadrature.axis_orders);
            let Some(moments) = quadrature.moments.take() else {
                // The corrector's contract reserves absent moments for the empty
                // block, and every piece has at least one axis.
                return Err(EstimationError::BlockQuadratureCorrectionRefused {
                    stage: BlockQuadratureCorrectionStage::CorrectorReturnedNoMoments {
                        block_dim: m,
                    },
                });
            };
            let channels = block_target_channel_moments(piece_target, &moments, c_weights)?;
            pieces.push(BlockPieceQuadrature {
                resolution_target: quadrature.value.abs().min(next_order_remainder),
                quadrature,
                channels,
            });
        }

        let x = x_dense.as_ref();
        // Φ and its derivatives in the whitened block coordinates
        // a_i = Λ^{-1/2} V_bᵀ x_i.
        let mixed = if axis_split {
            let (c_obs, d_obs, e_obs) = self.hessian_cde_arrays(pirls_result)?;
            let mut whitened = target.block_design.product.clone();
            for r in 0..m {
                let scale = target.block_lambdas[r].sqrt().recip();
                whitened.column_mut(r).mapv_inplace(|v| v * scale);
            }
            let term = mixed_axis_laplace_term(whitened.view(), &c_obs, &d_obs, &e_obs);
            Some((term, whitened))
        } else {
            None
        };

        let delta_b = pieces.iter().map(|piece| piece.quadrature.value).sum::<f64>()
            + mixed.as_ref().map_or(0.0, |(term, _)| term.value);
        let axis_quadrature_errors: Vec<f64> = pieces
            .iter()
            .flat_map(|piece| piece.quadrature.axis_quadrature_errors.iter().copied())
            .collect();
        // The pieces' errors add in Δ_b, so their sum bounds its error.
        let quadrature_error: f64 = pieces
            .iter()
            .map(|piece| piece.quadrature.quadrature_error)
            .sum();
        let node_count: usize = pieces.iter().map(|piece| piece.quadrature.node_count).sum();
        let abs_value = delta_b.abs();
        let relative_error = if abs_value > 0.0 {
            quadrature_error / abs_value
        } else {
            f64::INFINITY
        };

        // Trust gate: splice `Δ_b` only when every axis's paired rules agree
        // finely enough to resolve both its piece of the correction and the
        // next-order remainder the correction leaves behind. This makes admission
        // a deterministic accuracy certificate, not a Monte-Carlo efficiency
        // heuristic, and the order selection above only returns orders that pass
        // it.
        //
        // It decides ADMISSION and nothing else (#2748). Once the fit has
        // latched an admission, the latched evaluations carry the admission's
        // errors, which this test resolved, and it no longer switches: a rule that drops the whole `Δ_b` whenever the paired rules disagree
        // is a second predicate on ρ, and it re-introduces exactly the jump the
        // latch exists to remove — measured toggling 143 times against 235
        // admissions inside ONE `haberman_5yr` fit. A quadrature error that
        // varies with ρ perturbs the criterion CONTINUOUSLY, which the outer
        // loop's noise-floor machinery is built for; a criterion that drops a
        // 3e-2 term and picks it up again is not a function.
        let unresolved = pieces.iter().find(|piece| {
            !piece
                .quadrature
                .axis_quadrature_errors
                .iter()
                .all(|&error| error == 0.0 || error < piece.resolution_target)
        });
        if let Some(piece) = unresolved {
            if latched_block_dim.is_none() {
                return Err(EstimationError::BlockQuadratureCorrectionRefused {
                    stage: BlockQuadratureCorrectionStage::UnresolvedAtAdmission {
                        quadrature_error: piece.quadrature.quadrature_error,
                        resolution_target: piece.resolution_target,
                        axis_orders: axis_orders.clone(),
                        node_count,
                    },
                });
            }
            log::debug!(
                "[#784] block-local correction spliced UNRESOLVED (admission already latched, \
                 #2748): paired Gauss-Hermite error {:.4e} does not resolve \
                 min(|Δ|, 1/n_eff²)={:.4e} (|Δ_b|={abs_value:.4e}, m={m}, axis split={axis_split}, \
                 max|γ|={:.3}, τ={:.3}, axis orders={:?}, nodes={node_count}, 1/n_eff={:.3e})",
                piece.quadrature.quadrature_error,
                piece.resolution_target,
                verdict.max_abs_skewness,
                verdict.threshold,
                axis_orders,
                laplace_floor,
            );
        }

        // Latch the admission on the first evaluation that reaches here with
        // every gate cleared. Everything below this point splices, so this is
        // the exact boundary of "the correction is part of this model".
        if latched_block_dim.is_none() {
            self.block_correction_admission
                .store(m + 1, std::sync::atomic::Ordering::Relaxed);
            *self
                .block_correction_axis_orders
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(BlockQuadratureLatch {
                axis_orders: axis_orders.clone(),
                axis_quadrature_errors: axis_quadrature_errors.clone(),
                axis_split,
            });
            let mut decision = self.block_correction_decision_guard();
            if *decision == BlockCorrectionDecision::DecidingAtOptimum {
                *decision = BlockCorrectionDecision::AdmittedAtOptimum;
            }
            drop(decision);
            log::debug!(
                "[#784] block-local correction ADMITTED for this fit: block dimension m={m}, \
                 axis split={axis_split} and axis orders {:?} are now the model's, and the \
                 tau={:.3} activation no longer switches the criterion on and off along the \
                 outer search (#2748, #2623)",
                axis_orders,
                verdict.threshold,
            );
        }

        log::debug!(
            "[#784] deterministic block-local Gauss-Hermite correction ENGAGED: \
             m={m}, axis split={axis_split}, max|γ|={:.3}, τ={:.3}, Δ_b={:.4e} \
             (mixed-axis Φ={:.4e}), axis orders={:?}, nodes={node_count} \
             [paired-rule error={:.4e}, error/|Δ_b|={:.3e}, 1/n_eff={:.3e}]",
            verdict.max_abs_skewness,
            verdict.threshold,
            delta_b,
            mixed.as_ref().map_or(0.0, |(term, _)| term.value),
            axis_orders,
            quadrature_error,
            relative_error,
            laplace_floor,
        );

        // `Δ_b` is added to the marginal log-likelihood ⇒ subtracted from the
        // REML cost. The gradient ∂Δ_b/∂ρ likewise enters the cost with a
        // negative sign.
        //
        // ── Exact gradient channels (b)–(d) ─────────────────────────────
        // The explicit channel `quadrature.rho_gradient` is NOT
        // the total ρ-derivative of the realized quadrature: the fixed nodes
        // `t_s = z_s/√λ_r(ρ)` also move through the block eigenvalues
        // (node rescale, (b)), the block eigenvectors (frame
        // rotation, (c)), and the mode β̂ (mode motion, (d)). Splicing (a)
        // alone is the #752/#748/#901 objective↔gradient desync. The four
        // channels are assembled here per the gradient exactness contract on
        // `block_quadrature_marginal_correction`, contracting the corrector's
        // normalized moments against fields this evaluator already owns:
        //
        //   d(cost)/dρ_j = E_p[dΔF/dρ_j]
        //                = (a) E_p[∂ΔF/∂ρ_j]
        //                + (b)+(c) tr(Ḣ_j · (Q_b + Q_c))
        //                + (d) g_dᵀ · dβ̂/dρ_j,
        //
        // where (a) is identically zero for the row-remainder `ΔF` the target
        // integrates (ρ reaches it only through β̂), so its content lives in (d).
        //
        // with the TOTAL drift `Ḣ_j = λ_j S_j − C[v_j]`,
        // `C[v] = Xᵀ diag(c ⊙ Xv) X`, the IFT mode response
        // `dβ̂/dρ_j = −v_j = −H⁻¹ λ_j S_j β̂`, and
        //
        //   Q_b = Σ_r (M_r/λ_r) u_r u_rᵀ                       (rank m)
        //   Q_c = sym( Σ_r Σ_{q≠r} u_q (R̃_{q r}/(λ_r − σ_q)) u_rᵀ )
        //   M_r = E_p[(∂ΔF/∂t)_r · (−½ t_r)],   R̃ = Uᵀ E_p[t_r ∂ΔF/∂δ].
        //
        // Split by axis, each piece supplies its own column of R and entry of
        // M (piece r depends on the block only through (λ_r, u_r)) and its own
        // g_d and explicit gradient, which add. Φ depends on (λ_r, u_r) and β̂
        // through a_ir = u_rᵀx_i/√λ_r and (c, d)(η̂), so with G = ∂Φ/∂A its
        // cost-side channels are
        //
        //   R[:,r] −= Xᵀ G[:,r] / √λ_r,   M_r += ½ G[:,r]·A[:,r],
        //   g_d    −= Xᵀ ∂Φ/∂η,
        //
        // and it has no explicit ρ-dependence.
        //
        // Eigenvalue near-degeneracies `λ_r ≈ σ_q` are genuine
        // non-differentiability points of the eigenframe; the splice is
        // declined there rather than clamped.
        let n_rows = x.nrows();
        let mut g_d = Array1::<f64>::zeros(p);
        let mut r_mat = Array2::<f64>::zeros((p, m));
        let mut m_vec = Array1::<f64>::zeros(m);
        let mut explicit_gradient = Array1::<f64>::zeros(n_rho);
        for (k, piece) in pieces.iter().enumerate() {
            let first_axis = if axis_split { k } else { 0 };
            let width = piece.channels.m_vec.len();
            g_d += &piece.channels.g_d;
            r_mat
                .slice_mut(ndarray::s![.., first_axis..first_axis + width])
                .assign(&piece.channels.r_mat);
            m_vec
                .slice_mut(ndarray::s![first_axis..first_axis + width])
                .assign(&piece.channels.m_vec);
            for (total, &value) in explicit_gradient
                .iter_mut()
                .zip(piece.quadrature.rho_gradient.iter())
            {
                *total += value;
            }
        }
        if let Some((term, whitened)) = mixed.as_ref() {
            g_d.scaled_add(-1.0, &x.t().dot(&term.eta_gradient));
            let xt_g = x.t().dot(&term.a_gradient); // p × m
            for r in 0..m {
                r_mat
                    .column_mut(r)
                    .scaled_add(-target.block_lambdas[r].sqrt().recip(), &xt_g.column(r));
                m_vec[r] += 0.5 * term.a_gradient.column(r).dot(&whitened.column(r));
            }
        }

        // Eigenframe assembly. `block_vecs` are the `block_cols` columns of
        // `evecs`, so `Q_b`/`Q_c` are built from the same spectrum as the
        // draws — one source of truth for "the direction λ_r".
        if evals.iter().any(|&s| !(s.is_finite() && s > 0.0)) {
            // A NaN eigenvalue is reported as the minimum rather than skipped.
            let min_eigenvalue = evals
                .iter()
                .copied()
                .fold(f64::INFINITY, |min, s| if s.is_nan() || s < min { s } else { min });
            return Err(EstimationError::BlockQuadratureCorrectionRefused {
                stage: BlockQuadratureCorrectionStage::NonPositivePenalizedCurvature {
                    min_eigenvalue,
                },
            });
        }
        // When is `λ_r − σ_q` a MEASUREMENT rather than a rounding residue?
        //
        // This tolerance used to be `1e-10 · max|λ|` — 4.5e5 machine epsilons
        // against the LARGEST eigenvalue, compared with gaps between the
        // SMALLEST ones. `H_pen = XᵀWX + S_λ` spans the λ range the outer search
        // drives, so at `ρ = 21` (`λ = 1.3e9`) `max|λ|` reached `3.78e11` here
        // and the tolerance became an ABSOLUTE `37.8`: gaps of `1.3`, `37.1`,
        // `0.49` — resolved by the eigensolver to twelve significant digits —
        // were all declared degenerate. Measured on `haberman_5yr` fold 2
        // (#2748): the splice declined **10305 times against 10603
        // admissions** inside one fit, toggling a `Δ_b` of the same size the τ
        // gate used to toggle, and the fit died with `|Pg| = 6.700e-1` after
        // 362.6 s.
        //
        // The question the tolerance is about is the eigendecomposition's own
        // accuracy, and this repo measures that rather than declaring it:
        // `‖H v_r − λ_r v_r‖` is a backward error, and by Weyl each eigenvalue
        // carries at most that much uncertainty, so a gap is a measurement when
        // it exceeds TWO of them. Nothing is chosen; at a genuine crossing the
        // eigenframe really is non-differentiable and the splice still declines.
        // PER PAIR, not the maximum over the spectrum. `‖H v_q − σ_q v_q‖` is
        // the backward error of THAT eigenpair, and Weyl bounds `|σ_q − true|`
        // by it alone. `H_pen` spans the whole λ range the outer search drives,
        // so the worst pair (a fully-railed penalty direction at `λ = e^30`) is
        // no statement at all about a curvature-heavy direction at the small
        // end — and taking the max makes every gap down there unresolvable by
        // an eigenvalue that has nothing to do with it.
        // Each pair's residual is certified against its OWN evaluation error,
        // which is what bounds a residual that rounds to zero, so nothing is
        // chosen here either.
        let pair_resolution =
            match crate::estimate::smoothing_correction::eigenpair_residual_bounds(
                &sym_h, &evals, &evecs,
            ) {
                Ok(bounds) => bounds,
                Err(reason) => {
                    return Err(EstimationError::BlockQuadratureCorrectionRefused {
                        stage: BlockQuadratureCorrectionStage::EigenpairResolutionUnavailable {
                            reason: reason.to_string(),
                        },
                    });
                }
            };
        let r_tilde = evecs.t().dot(&r_mat); // p × m
        let mut g_mat = Array2::<f64>::zeros((p, m));
        for (jr, &col_r) in block_cols.iter().enumerate() {
            let lam_r = target.block_lambdas[jr];
            for q in 0..p {
                if q == col_r {
                    continue;
                }
                let gap = lam_r - evals[q];
                // Each eigenvalue carries its own uncertainty, so the gap's is
                // the sum of the two. `<=` declines an exactly degenerate pair at zero
                // resolution, whose eigenframe derivative would divide by zero (#2469).
                let degeneracy_tol = pair_resolution[col_r] + pair_resolution[q];
                if gap.abs() <= degeneracy_tol {
                    return Err(EstimationError::BlockQuadratureCorrectionRefused {
                        stage: BlockQuadratureCorrectionStage::EigenframeNearDegeneracy {
                            block_eigenvalue: lam_r,
                            other_eigenvalue: evals[q],
                            gap: gap.abs(),
                            tolerance: degeneracy_tol,
                        },
                    });
                }
                g_mat[(q, jr)] = r_tilde[(q, jr)] / gap;
            }
        }
        let q_c_raw = evecs.dot(&g_mat).dot(&target.block_vecs.t()); // p × p
        let mut q_mat = 0.5 * (&q_c_raw + &q_c_raw.t());
        for jr in 0..m {
            let u_r = target.block_vecs.column(jr);
            let scale = m_vec[jr] / target.block_lambdas[jr];
            for a in 0..p {
                for b in 0..p {
                    q_mat[(a, b)] += scale * u_r[a] * u_r[b];
                }
            }
        }

        // rowq_i = x_iᵀ Q x_i (for tr(C[v] Q) = Σ_i (c ⊙ Xv)_i rowq_i).
        let xq = x.dot(&q_mat); // n × p
        let rowq = (&xq * x).sum_axis(ndarray::Axis(1)); // n

        // Per-coordinate contraction.
        //
        // The splice ran, so channels (b), (c) and (d) below are real on this
        // evaluation. Each channel is retained per coordinate and published into
        // the ρ-block audit after the loop, so an FD row can ASSERT engagement
        // and then compare the channels SEPARATELY; without the assertion the
        // comparison is vacuous whenever the splice declines, and without the
        // split it can only compare the total, which on a near-cancelling fit
        // cannot say which channel is wrong (#2623).
        let audit_armed = crate::estimate::outer_eval_capture::rho_outer_audit_enabled();
        let mut audit_a: Vec<f64> = Vec::new();
        let mut audit_trace: Vec<f64> = Vec::new();
        let mut audit_mode: Vec<f64> = Vec::new();
        let mut audit_spliced: Vec<f64> = Vec::new();

        // WARNING (#2623) -- READ THIS BEFORE CHANGING THE SIGN IN THIS LOOP.
        //
        // The convention of the four channels is NOT settled by the contract
        // comment above, which is self-inconsistent. The authoritative
        // statement is on the type, in gam-problem laplace_sampler_contract:
        //
        //     value:        Delta_b            added to the block marginal
        //                                      log-likelihood, SUBTRACTED
        //                                      from the REML/LAML cost
        //     rho_gradient: d(Delta_b)/d(rho)  explicit channel (a) ONLY
        //
        // So channel (a) is PLUS quadrature.rho_gradient, not its negation, and a
        // sum of four Delta_b-side channels is d(Delta_b)/d(rho), not
        // d(cost)/d(rho). The formula above labels its left side d(cost)/d(rho)
        // while listing (a) in Delta_b-side form, and separately calls the
        // NEGATION of quadrature.rho_gradient channel (a). A Delta_b-side term
        // cannot appear unnegated in a cost-side total, so the label, the terms
        // and the type contract cannot all three be right.
        //
        // What is settled: value is PLUS Delta_b, confirmed independently by
        // block_quadrature_marginal_recovers_analytic_quartic_correction, which
        // checks it against a 20001-point trapezoid reference and asserts it is
        // negative for an added quartic penalty. So the value: -delta_b
        // below is correct.
        //
        // What is OPEN: whether trace_j and mode_j below are Delta_b-side or
        // cost-side. The two readings differ by exactly 2*(trace_j + mode_j),
        // which #2623 measures at about 9.65 on a fold where the true slope is
        // a three-way near-cancellation and each channel is 25-30x the sum. So
        // the wrong reading does not perturb the search, it INVERTS it: an
        // outer gradient of +9.4547 AT the cost minimum, Wolfe failure, and 178
        // evaluations at one theta.
        //
        // DO NOT resolve this by reading, in either direction. It is decided by
        // giving the typed rho-block audit (enable_rho_outer_audit, #2454) a row
        // whose fixture ASSERTS the #784 splice engaged, then comparing each
        // channel against finite differences separately. The existing FD guard
        // cannot see it: both of its rows are deliberately well-behaved, so the
        // splice declines and trace_j and mode_j are never exercised at all.
        //
        // MEASURED (#2623), and the answer is NEITHER SIGN. The channel record
        // published below drove the #2623 probe, which finite-differenced Delta_b
        // itself on fixtures where the splice
        // engages. On two well-conditioned cells whose importance sampler is
        // essentially exact (ESS 507.9/512 and 500.1/512) the FD reference is
        // stable to six digits over h from 3e-4 to 3e-3, and the envelope
        // channels agree with it to 1e-7 relative -- so the stencil is sound.
        // Against that reference the three channels below match at no sign
        // assignment. The four measured ratios of the shipped line to the truth
        // are 0.84, -1.40, -1.43 and -17.4; for the proposed flip they are -12.1,
        // 4.36, 8.88 and 27.8. Decisively, WHICH sign is closer changes between
        // the two rho coordinates of a SINGLE evaluation, and no global sign
        // convention can do that. So this is a wrong contraction, not a wrong
        // sign, and flipping it exchanges one wrong gradient for another -- which
        // is also what the flip measured end-to-end. The residual total gradient
        // error is 1e-4 to 1.3e-1 relative in these mild regimes and INVERTS the
        // search on the #2623 fold, where the true slope is a three-way
        // near-cancellation.
        let mut gradient = Array1::<f64>::zeros(n_rho + n_ext);
        for j in 0..n_rho {
            let lam_j = target.lambdas[j];
            let a_j = target.penalty_scores[j].mapv(|v| lam_j * v); // λ_j S_j β̂
            // v_j = H⁻¹ a_j through the same eigendecomposition as Q.
            let uta = evecs.t().dot(&a_j);
            let v_j = evecs.dot(&(&uta / &evals));
            // tr(A_j Q) = λ_j Σ_c (S_j Q[:,c])_c.
            let mut tr_sq = 0.0_f64;
            for c in 0..p {
                let s_col = transformed_penalty_matvec(
                    &target.penalties[j],
                    &q_mat.column(c).to_owned(),
                );
                tr_sq += s_col[c];
            }
            // tr(C[v_j] Q) = Σ_i c_i (X v_j)_i rowq_i.
            let xv_j = gam_linalg::faer_ndarray::fast_av(x, &v_j);
            let mut tr_cq = 0.0_f64;
            for i in 0..n_rows {
                tr_cq += c_weights[i] * xv_j[i] * rowq[i];
            }
            let trace_j = lam_j * tr_sq - tr_cq;
            let mode_j = -v_j.dot(&g_d);
            gradient[j] = -explicit_gradient[j] + trace_j + mode_j;
            if audit_armed {
                audit_a.push(explicit_gradient[j]);
                audit_trace.push(trace_j);
                audit_mode.push(mode_j);
                audit_spliced.push(gradient[j]);
            }
        }
        if audit_armed {
            crate::estimate::outer_eval_capture::record_quadrature_marginal(
                crate::estimate::outer_eval_capture::QuadratureMarginalAudit {
                    delta_b,
                    quadrature_error,
                    node_count,
                    axis_orders,
                    axis_quadrature_errors,
                    max_abs_skewness: verdict.max_abs_skewness,
                    skewness_threshold: verdict.threshold,
                    block_cols: block_cols.clone(),
                    explicit_a: audit_a,
                    trace_bc: audit_trace,
                    mode_d: audit_mode,
                    spliced: audit_spliced,
                },
            );
        }
        Ok(TkCorrectionTerms {
            value: -delta_b,
            gradient: Some(gradient),
            hessian: None,
        })
    }
}

/// Admit the dense design copy the #784 block-local correction integrates over,
/// against the memory governor's single-materialization cap `cap_bytes`. Past it
/// the correction refuses with a typed error instead of switching the criterion
/// off at a picked problem size (gam#2900).
fn block_correction_design_admission(
    n_obs: usize,
    p: usize,
    cap_bytes: usize,
) -> Result<(), EstimationError> {
    let requested_bytes = n_obs
        .saturating_mul(p)
        .saturating_mul(std::mem::size_of::<f64>());
    if requested_bytes > cap_bytes {
        return Err(EstimationError::DenseMaterializationRefused {
            context: "#784 block-local correction design".to_string(),
            rows: n_obs,
            cols: p,
            requested_bytes,
            cap_bytes,
        });
    }
    Ok(())
}

/// Put the block's eigendirections in ascending-curvature order, ties broken by
/// index.
///
/// The latched Gauss-Hermite orders are bound to axis *positions*, so the
/// position of each direction must be a property of the direction, not of the
/// eigensolver that produced it. `eigh` of the assembled `H` returns ascending
/// eigenvalues; the stacked-root SVD the criterion switches to once the
/// assembled spectrum cannot resolve `log|H|` (#2644) returns descending ones.
/// Ordered by index, the block's two axes traded their latched orders at that
/// switch: on the prostate `s(pc1) + s(pc2)` binomial fit the 16-node rule
/// moved to the axis certified at 9, `Δ_b` stepped by 1.2e-5 between
/// `ρ₂ = 18.4366` and `18.4473` where its smooth variation is 1e-7, and the
/// corrected BFGS continuation failed its line search on the step until
/// `StepSizeTooSmall`.
fn order_block_axes_by_curvature(block_cols: &mut [usize], evals: &Array1<f64>) {
    block_cols.sort_by(|&a, &b| evals[a].total_cmp(&evals[b]).then(a.cmp(&b)));
}

/// One integrated piece of the block marginal: the whole block under a tensor
/// rule, or one axis of it under the split, with its gradient channels.
struct BlockPieceQuadrature {
    quadrature: gam_problem::laplace_sampler_contract::BlockQuadratureMarginal,
    /// `min(|Δ_piece|, 1/n_eff²)`, the error the piece's rule must resolve.
    resolution_target: f64,
    channels: BlockTargetChannels,
}

/// The cost-side gradient-channel moments of one block target, over that
/// target's own axes: `g_d = E_p[∂ΔF/∂β̂]` (p), `R[:,r] = E_p[t_r ∂ΔF/∂δ]`
/// (p × width) and `M_r = E_p[(∂ΔF/∂t)_r (−½ t_r)]` (width).
struct BlockTargetChannels {
    g_d: Array1<f64>,
    r_mat: Array2<f64>,
    m_vec: Array1<f64>,
}

/// Contract a block target's normalized quadrature moments into the channel
/// moments the exact (b)–(d) assembly consumes.
fn block_target_channel_moments(
    target: &Gam784BlockTarget<'_>,
    moments: &gam_problem::laplace_sampler_contract::BlockQuadratureMoments,
    c_weights: &Array1<f64>,
) -> Result<BlockTargetChannels, EstimationError> {
    let x = target.x_transformed;
    let n_rows = x.nrows();
    let width = target.block_vecs.ncols();
    let xv = &target.block_design.product; // n × width
    let ngs_base = target
        .base_neg_score()
        .map_err(EstimationError::InvalidInput)?;

    // σ²_i = E_p[s_i²] and the shared n × width intermediates.
    let xv_ett = xv.dot(&moments.e_tt); // n × width
    let sigma2 = (&xv_ett * xv).sum_axis(ndarray::Axis(1)); // n
    let mut w_xv_ett = xv_ett.clone();
    for i in 0..n_rows {
        let w_i = target.weights_obs[i];
        w_xv_ett.row_mut(i).mapv_inplace(|v| v * w_i);
    }

    // Channel (d) moment: g_d = E_p[∂ΔF/∂β̂] with δ held fixed. `ΔF` is the
    // row Taylor remainder `Σ_i [ψ_i(η̂_i+s_i) − ψ_i(η̂_i) − ψ_i'(η̂_i)s_i − ½W_i s_i²]`
    // (see `Gam784BlockTarget`), and `∂ψ'/∂η = W`, `∂W/∂η = c`, so
    //   g_d = Xᵀ(E_p[ngs_disp] − ngs_base) − Xᵀ W X (V_b E_p[t])
    //         − ½ Xᵀ(c ⊙ E_p[s²]).
    // On the exact mode this is the penalty-score form's `g_d` minus
    // `H δ̄`, and `v_j·H δ̄ = λ_j (S_j β̂)·δ̄` is exactly the explicit channel
    // that form carried, so the total is unchanged there.
    let s_mean = xv.dot(&moments.e_t); // n
    let mut g_d = x.t().dot(&(&moments.e_neg_score - &ngs_base));
    g_d.scaled_add(-1.0, &x.t().dot(&(&target.weights_obs * &s_mean)));
    g_d.scaled_add(-0.5, &x.t().dot(&(c_weights * &sigma2)));

    // Channel (c) moment: R[:,r] = E_p[t_r · ∂ΔF/∂δ]
    //   = Xᵀ E_p[t_r ngs_disp] − (Xᵀ ngs_base) E_p[t_r] − Xᵀ W X V_b E_p[t tᵀ][:,r].
    let base_score_coef = x.t().dot(&ngs_base); // p
    let mut r_mat = x.t().dot(&moments.e_t_neg_score); // p × width
    for r in 0..width {
        r_mat
            .column_mut(r)
            .scaled_add(-moments.e_t[r], &base_score_coef);
    }
    r_mat -= &x.t().dot(&w_xv_ett);

    // Channel (b) moment: M_r = E_p[(∂ΔF/∂t)_r (−½ t_r)] via
    // ∂ΔF/∂t = (XV)ᵀ (ngs_disp − ngs_base − W ⊙ s).
    let xvt_etngs = xv.t().dot(&moments.e_t_neg_score); // width × width
    let base_score_block = xv.t().dot(&ngs_base); // width
    let xvt_w_xv_ett = xv.t().dot(&w_xv_ett); // width × width
    let mut m_vec = Array1::<f64>::zeros(width);
    for r in 0..width {
        m_vec[r] = -0.5
            * (xvt_etngs[(r, r)] - base_score_block[r] * moments.e_t[r] - xvt_w_xv_ett[(r, r)]);
    }
    Ok(BlockTargetChannels { g_d, r_mat, m_vec })
}

/// The block target restricted to its axis `r`: the same excess `ΔF`, with the
/// displacement confined to the block eigenvector `u_r` and its curvature `λ_r`.
fn block_axis_target<'t>(target: &Gam784BlockTarget<'t>, r: usize) -> Gam784BlockTarget<'t> {
    Gam784BlockTarget {
        x_transformed: target.x_transformed,
        block_vecs: target
            .block_vecs
            .column(r)
            .to_owned()
            .insert_axis(ndarray::Axis(1)),
        block_design: target.block_design.column(r),
        block_lambdas: Array1::from_elem(1, target.block_lambdas[r]),
        eta_hat: target.eta_hat.clone(),
        weights_obs: target.weights_obs.clone(),
        weights_obs_log_abs: target.weights_obs_log_abs.clone(),
        y: target.y.clone(),
        row_measures: target.row_measures.clone(),
        prior_weights: target.prior_weights.clone(),
        likelihood: target.likelihood.clone(),
        inverse_link: target.inverse_link.clone(),
        phi: target.phi,
        penalty_scores: std::sync::Arc::clone(&target.penalty_scores),
        penalties: target.penalties,
        lambdas: target.lambdas.clone(),
        base_scaled_half_deviance: target.base_scaled_half_deviance,
        base_neg_score_at_mode: target.base_neg_score_at_mode.clone(),
        base_absolute_half_deviance: target.base_absolute_half_deviance,
    }
}

/// The mixed-axis second-order Laplace term `Φ` and its derivatives.
struct MixedAxisLaplaceTerm {
    value: f64,
    /// `∂Φ/∂a_i`, row `i` (n × m).
    a_gradient: Array2<f64>,
    /// `∂Φ/∂η_i` at fixed `a_i`, through `(c_i, d_i)(η_i)` (n).
    eta_gradient: Array1<f64>,
}

/// The part of the second-order Laplace expansion of a block marginal that no
/// single axis carries.
///
/// With whitened rows `a_i = Λ^{-1/2} V_bᵀ x_i`, `z ~ N(0, I_m)` and
/// `s_i = a_i·z`, the excess over the Laplace Gaussian is
/// `ΔF = Σ_i (c_i/6) s_i³ + (d_i/24) s_i⁴ + …` once the mode condition cancels
/// its linear term (`c = ∂W/∂η`, `d = ∂²W/∂η²`), and
///
///   log E[e^{−ΔF}] = −E[ΔF₄] + ½ E[ΔF₃²] + O(n⁻²)
///                  = −⅛ Σ d_i |a_i|⁴ + ⅛ |u|² + (1/12) ‖T‖² + O(n⁻²),
///
/// from `E[s_i⁴] = 3|a_i|⁴` and `E[s_i³ s_j³] = 9 σ_i² σ_j² κ_ij + 6 κ_ij³`,
/// where `u = Σ c_i |a_i|² a_i` and `T = Σ c_i a_i^{⊗3}`. The same expansion of
/// one axis is `−⅛ Σ d_i a_ir⁴ + (5/24) T_rrr²`, so the mixed-axis part is
///
///   Φ = −⅛ Σ d_i (|a_i|⁴ − Σ_r a_ir⁴) + ⅛ |u|² + (1/12) ‖T‖² − (5/24) Σ_r T_rrr².
///
/// With `c`'s own η-derivative `d` and `d`'s `e = ∂³W/∂η³`,
///
///   ∂Φ/∂a_i = −½ d_i (|a_i|² a_i − a_i^{∘3}) + ¼ c_i (2 (a_i·u) a_i + |a_i|² u)
///             + ½ c_i T(a_i, a_i, ·) − (5/4) c_i T_diag ∘ a_i^{∘2},
///   ∂Φ/∂η_i = −⅛ e_i (|a_i|⁴ − Σ_r a_ir⁴) + ¼ d_i |a_i|² (a_i·u)
///             + (1/6) d_i T(a_i, a_i, a_i) − (5/12) d_i Σ_r T_rrr a_ir³.
///
/// The cost is O(n·m³).
fn mixed_axis_laplace_term(
    a: ndarray::ArrayView2<'_, f64>,
    c: &Array1<f64>,
    d: &Array1<f64>,
    e: &Array1<f64>,
) -> MixedAxisLaplaceTerm {
    let (n, m) = a.dim();
    let sq: Array1<f64> = a.map_axis(ndarray::Axis(1), |row| row.dot(&row));
    let quart: Array1<f64> =
        a.map_axis(ndarray::Axis(1), |row| row.iter().map(|v| v.powi(4)).sum());

    // u and the symmetric tensor T, accumulated on its sorted index triples
    // p ≤ q ≤ r and then filled out.
    let mut u = Array1::<f64>::zeros(m);
    let mut tensor = vec![0.0_f64; m * m * m];
    let at = |p: usize, q: usize, r: usize| (p * m + q) * m + r;
    for i in 0..n {
        let ci = c[i];
        if ci == 0.0 {
            continue;
        }
        let ai = a.row(i);
        u.scaled_add(ci * sq[i], &ai);
        for p in 0..m {
            let cp = ci * ai[p];
            for q in p..m {
                let cpq = cp * ai[q];
                for r in q..m {
                    tensor[at(p, q, r)] += cpq * ai[r];
                }
            }
        }
    }
    for p in 0..m {
        for q in p..m {
            for r in q..m {
                let value = tensor[at(p, q, r)];
                for (x, y, z) in [(p, r, q), (q, p, r), (q, r, p), (r, p, q), (r, q, p)] {
                    tensor[at(x, y, z)] = value;
                }
            }
        }
    }
    let t_diag: Vec<f64> = (0..m).map(|r| tensor[at(r, r, r)]).collect();

    let mut value = 0.125 * u.dot(&u) + tensor.iter().map(|t| t * t).sum::<f64>() / 12.0
        - (5.0 / 24.0) * t_diag.iter().map(|t| t * t).sum::<f64>();
    let mut a_gradient = Array2::<f64>::zeros((n, m));
    let mut eta_gradient = Array1::<f64>::zeros(n);
    // T(a_i, a_i, ·), reused across rows.
    let mut taa = vec![0.0_f64; m];
    for i in 0..n {
        let ai = a.row(i);
        let (ci, di, ei) = (c[i], d[i], e[i]);
        let off_axis_quartic = sq[i] * sq[i] - quart[i];
        value -= 0.125 * di * off_axis_quartic;
        for (r, slot) in taa.iter_mut().enumerate() {
            let mut total = 0.0;
            for p in 0..m {
                let mut inner = 0.0;
                for q in 0..m {
                    inner += tensor[at(p, q, r)] * ai[q];
                }
                total += ai[p] * inner;
            }
            *slot = total;
        }
        let au = ai.dot(&u);
        let taaa: f64 = taa.iter().zip(ai.iter()).map(|(t, v)| t * v).sum();
        let diag_cubic: f64 = t_diag
            .iter()
            .zip(ai.iter())
            .map(|(t, v)| t * v * v * v)
            .sum();
        let mut row = a_gradient.row_mut(i);
        for r in 0..m {
            let air = ai[r];
            row[r] = -0.5 * di * (sq[i] * air - air * air * air)
                + 0.25 * ci * (2.0 * au * air + sq[i] * u[r])
                + 0.5 * ci * taa[r]
                - 1.25 * ci * t_diag[r] * air * air;
        }
        eta_gradient[i] = -0.125 * ei * off_axis_quartic + 0.25 * di * sq[i] * au
            + di * taaa / 6.0
            - (5.0 / 12.0) * di * diag_cubic;
    }
    MixedAxisLaplaceTerm {
        value,
        a_gradient,
        eta_gradient,
    }
}

#[cfg(test)]
mod mixed_axis_laplace_term_tests {
    use super::*;

    /// The second-order Laplace expansion of `log E[e^{−ΔF}]` over the whitened
    /// rows `a`, summed pair by pair from the Gaussian moments
    /// `E[s_i⁴] = 3σ_i⁴` and `E[s_i³ s_j³] = 9σ_i²σ_j²κ_ij + 6κ_ij³`.
    fn pairwise_second_order(a: &Array2<f64>, c: &Array1<f64>, d: &Array1<f64>) -> f64 {
        let n = a.nrows();
        let mut quartic = 0.0;
        let mut cubic_square = 0.0;
        for i in 0..n {
            let si = a.row(i).dot(&a.row(i));
            quartic += d[i] * 3.0 * si * si / 24.0;
            for j in 0..n {
                let sj = a.row(j).dot(&a.row(j));
                let kij = a.row(i).dot(&a.row(j));
                cubic_square += c[i] * c[j] * (9.0 * si * sj * kij + 6.0 * kij.powi(3)) / 36.0;
            }
        }
        -quartic + 0.5 * cubic_square
    }

    fn fixture() -> (Array2<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
        let (n, m) = (7usize, 3usize);
        let a = Array2::from_shape_fn((n, m), |(i, r)| {
            ((i * 7 + r * 3) as f64 * 0.61).sin() * 0.4 + 0.05 * r as f64
        });
        let c = Array1::from_shape_fn(n, |i| (i as f64 * 1.3).cos() * 0.7);
        let d = Array1::from_shape_fn(n, |i| 0.3 + (i as f64 * 0.9).sin() * 0.2);
        let e = Array1::from_shape_fn(n, |i| (i as f64 * 2.1).cos() * 0.5);
        (a, c, d, e)
    }

    #[test]
    fn value_is_the_full_expansion_less_every_single_axis_expansion() {
        let (a, c, d, e) = fixture();
        let per_axis: f64 = (0..a.ncols())
            .map(|r| {
                let column = a.column(r).to_owned().insert_axis(ndarray::Axis(1));
                pairwise_second_order(&column, &c, &d)
            })
            .sum();
        let expected = pairwise_second_order(&a, &c, &d) - per_axis;
        let term = mixed_axis_laplace_term(a.view(), &c, &d, &e);
        assert!(
            (term.value - expected).abs() <= 1e-12 * expected.abs().max(1.0),
            "Φ={} against the pairwise expansion {expected}",
            term.value
        );
        // A single axis has no mixed part.
        let column = a.column(0).to_owned().insert_axis(ndarray::Axis(1));
        let single = mixed_axis_laplace_term(column.view(), &c, &d, &e);
        assert!(single.value.abs() <= 1e-14, "one axis gave Φ={}", single.value);
    }

    #[test]
    fn gradients_match_central_differences() {
        let (a, c, d, e) = fixture();
        let term = mixed_axis_laplace_term(a.view(), &c, &d, &e);
        let h = 1e-6;
        for i in 0..a.nrows() {
            for r in 0..a.ncols() {
                let mut plus = a.clone();
                plus[(i, r)] += h;
                let mut minus = a.clone();
                minus[(i, r)] -= h;
                let fd = (mixed_axis_laplace_term(plus.view(), &c, &d, &e).value
                    - mixed_axis_laplace_term(minus.view(), &c, &d, &e).value)
                    / (2.0 * h);
                assert!(
                    (fd - term.a_gradient[(i, r)]).abs() <= 1e-7,
                    "∂Φ/∂a[{i},{r}]: analytic {} against FD {fd}",
                    term.a_gradient[(i, r)]
                );
            }
            // η_i moves c_i and d_i at rates d_i and e_i.
            let mut c_plus = c.clone();
            let mut d_plus = d.clone();
            c_plus[i] += h * d[i];
            d_plus[i] += h * e[i];
            let mut c_minus = c.clone();
            let mut d_minus = d.clone();
            c_minus[i] -= h * d[i];
            d_minus[i] -= h * e[i];
            let fd = (mixed_axis_laplace_term(a.view(), &c_plus, &d_plus, &e).value
                - mixed_axis_laplace_term(a.view(), &c_minus, &d_minus, &e).value)
                / (2.0 * h);
            assert!(
                (fd - term.eta_gradient[i]).abs() <= 1e-7,
                "∂Φ/∂η[{i}]: analytic {} against FD {fd}",
                term.eta_gradient[i]
            );
        }
    }
}

#[cfg(test)]
mod design_admission_tests {
    use super::*;

    #[test]
    fn design_admission_admits_shapes_past_the_old_tk_scale_caps() {
        // Each shape sat past one of the deleted caps (n > 20,000, p > 2,000,
        // n·p > 5e6), where the correction used to be zero whatever its verdict.
        for (n_obs, p) in [(20_001usize, 1usize), (10, 2_001), (2_501, 2_000)] {
            assert!(
                block_correction_design_admission(n_obs, p, usize::MAX).is_ok(),
                "n={n_obs} p={p} must be decided by the skewness verdict, not refused"
            );
        }
    }

    #[test]
    fn design_admission_refuses_one_byte_past_the_cap() {
        let (n_obs, p) = (20_001usize, 3usize);
        let bytes = n_obs * p * std::mem::size_of::<f64>();
        assert!(block_correction_design_admission(n_obs, p, bytes).is_ok());
        assert!(matches!(
            block_correction_design_admission(n_obs, p, bytes - 1),
            Err(EstimationError::DenseMaterializationRefused {
                rows,
                cols,
                requested_bytes,
                cap_bytes,
                ..
            }) if (rows, cols, requested_bytes, cap_bytes) == (n_obs, p, bytes, bytes - 1)
        ));
    }
}

#[cfg(test)]
mod block_axis_order_tests {
    use super::*;

    /// The same block read from an ascending (`eigh`) and a descending
    /// (stacked-root SVD) eigensystem must give every axis position the same
    /// direction, or the latched per-axis orders land on different directions
    /// when the criterion switches route.
    #[test]
    fn axis_positions_do_not_depend_on_the_eigensolver_order() {
        let ascending = Array1::from(vec![2.092e-1, 1.545, 2.772, 1.160e1, 1.753e1, 1.054e2]);
        let descending = Array1::from_iter(ascending.iter().rev().copied());
        let n = ascending.len();
        let mut from_eigh = vec![4usize, 1];
        let mut from_root = vec![n - 1 - 4, n - 1 - 1];
        order_block_axes_by_curvature(&mut from_eigh, &ascending);
        order_block_axes_by_curvature(&mut from_root, &descending);
        let curvatures = |cols: &[usize], evals: &Array1<f64>| -> Vec<f64> {
            cols.iter().map(|&r| evals[r]).collect()
        };
        assert_eq!(curvatures(&from_eigh, &ascending), vec![1.545, 1.753e1]);
        assert_eq!(
            curvatures(&from_eigh, &ascending),
            curvatures(&from_root, &descending)
        );
    }
}
