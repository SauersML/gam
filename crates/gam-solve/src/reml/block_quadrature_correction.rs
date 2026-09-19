//! The #784 block-local Gauss-Hermite marginal correction.
//!
//! Extracted verbatim from `gradient_hessian.rs`, which the repo's own
//! 10,000-line gate refuses to carry (#780). It is one self-contained unit:
//! the per-bundle cached wrapper, the compute path that runs the skewness
//! diagnostic, selects the curvature-heavy block, integrates the non-Gaussian
//! remainder over it, and assembles the four exact gradient channels the
//! splice's objective-gradient contract requires.
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
        // The penalty-score channel needs one λ per canonical penalty.
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
            log::debug!(
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
            log::debug!(
                "[#784] block-local fallback declined before the skewness diagnostic: \
                 Beta family has no exponential-family score identity for the exact \
                 gradient channels"
            );
            return Ok(zero());
        }
        // A Firth fit's posterior carries the Jeffreys prior ½ log|I(β)|, and its
        // mode solves S β̂ = ∇ℓ(β̂) + ∇½ log|I(β̂)|. `Gam784BlockTarget` integrates
        // the flat-prior posterior −ℓ + ½ βᵀSβ and reduces its remainder through
        // the flat-prior mode condition S β̂ = ∇ℓ(β̂), so on a Firth fit its value
        // omits the Jeffreys remainder and its gradient channels are not the
        // derivative of that value. Under (quasi-)separation, the case Firth
        // exists for, the flat-prior posterior it integrates is improper along
        // the separating direction: no Gauss–Hermite order resolves it, so the
        // order search raised orders until the fit ground to a halt on a
        // perfectly separated step, and on a quasi-separated one the spliced
        // gradient left the outer search stalled at |g| = 0.23. The Firth
        // criterion's own higher-order term is the Tierney–Kadane refinement of
        // the same Jeffreys objective (`tierney_kadane_terms`), so the
        // correction is declined — value and gradient together — whenever the
        // Jeffreys term is armed.
        if reml_robust_jeffreys_link(&self.config).is_some() {
            log::debug!(
                "[#784] block-local fallback declined before the skewness diagnostic: the \
                 Jeffreys (Firth) prior is armed and the block target integrates the \
                 flat-prior posterior"
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

        // Step 1: per-direction skewness diagnostic γ_r.
        let (max_abs, directional) = corrector
            .directional_cubic_diagnostic(h_total, x_design, c_weights, false)
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
                log::info!(
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
        let sym_h = (h_total + &h_total.t()) * 0.5;
        let (evals, evecs) = sym_h.eigh(Side::Lower).map_err(|e| {
            EstimationError::InvalidInput(format!(
                "#784 block-local fallback eigendecomposition failed: {e}"
            ))
        })?;
        let mut admissible: Vec<usize> = (0..evals.len().min(directional.len()))
            .filter(|&r| evals[r] > 0.0 && directional[r].is_finite())
            .collect();
        let block_cols: Vec<usize> = match latched_block_dim {
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
                admissible.sort_unstable();
                admissible
            }
            None => verdict
                .untrustworthy_directions
                .iter()
                .copied()
                .filter(|&r| r < evals.len() && evals[r] > 0.0)
                .collect(),
        };
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
            block_vecs,
            block_lambdas,
            eta_hat,
            weights_obs,
            weights_obs_log_abs,
            y: self.y.to_owned(),
            prior_weights: self.weights.to_owned(),
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
        // lower rule resolves min(|Δ_b|, 1/n_eff²), and latched beside the block
        // dimension. Under a latched admission the orders are the model's, so
        // every ρ integrates against the same nodes and the value, gradient and
        // moments share one measure.
        let laplace_floor = if n_eff > 0.0 {
            1.0 / n_eff
        } else {
            f64::INFINITY
        };
        let next_order_remainder = laplace_floor * laplace_floor;
        let latched_axis_orders = self
            .block_correction_axis_orders
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
            .filter(|orders| orders.len() == m);
        let quadrature = match latched_axis_orders {
            Some(axis_orders) => corrector
                .block_quadrature_marginal_correction(&target, &axis_orders)
                .map_err(|refusal| EstimationError::InvalidInput(refusal.to_string()))?,
            None => match gam_problem::laplace_sampler_contract::select_block_quadrature_orders(
                corrector,
                &target,
                next_order_remainder,
            ) {
                Ok(quadrature) => quadrature,
                Err(refusal) => {
                    return Err(EstimationError::BlockQuadratureCorrectionRefused {
                        stage: BlockQuadratureCorrectionStage::OrderSearchRefused(refusal),
                    });
                }
            },
        };

        let abs_value = quadrature.value.abs();
        let relative_error = if abs_value > 0.0 {
            quadrature.quadrature_error / abs_value
        } else {
            f64::INFINITY
        };

        // Trust gate: splice `Δ_b` only when every axis's paired rules agree
        // finely enough to resolve both the correction itself and the next-order
        // remainder the correction leaves behind. This makes admission a
        // deterministic accuracy certificate, not a Monte-Carlo efficiency
        // heuristic, and the order selection above only returns orders that pass
        // it.
        //
        // It decides ADMISSION and nothing else (#2748). Once the fit has
        // latched an admission, this test is reported and no longer switches:
        // a rule that drops the whole `Δ_b` whenever the paired rules disagree
        // is a second predicate on ρ, and it re-introduces exactly the jump the
        // latch exists to remove — measured toggling 143 times against 235
        // admissions inside ONE `haberman_5yr` fit. A quadrature error that
        // varies with ρ perturbs the criterion CONTINUOUSLY, which the outer
        // loop's noise-floor machinery is built for; a criterion that drops a
        // 3e-2 term and picks it up again is not a function.
        let resolution_target = abs_value.min(next_order_remainder);
        let resolved = quadrature
            .axis_quadrature_errors
            .iter()
            .all(|&error| error == 0.0 || error < resolution_target);
        if !resolved {
            if latched_block_dim.is_none() {
                return Err(EstimationError::BlockQuadratureCorrectionRefused {
                    stage: BlockQuadratureCorrectionStage::UnresolvedAtAdmission {
                        quadrature_error: quadrature.quadrature_error,
                        resolution_target,
                        axis_orders: quadrature.axis_orders.clone(),
                        node_count: quadrature.node_count,
                    },
                });
            }
            log::info!(
                "[#784] block-local correction spliced UNRESOLVED (admission already latched, \
                 #2748): paired Gauss-Hermite error {:.4e} does not resolve \
                 min(|Δ_b|, 1/n_eff²)={resolution_target:.4e} (|Δ_b|={abs_value:.4e}, m={m}, \
                 max|γ|={:.3}, τ={:.3}, axis orders={:?}, nodes={}, 1/n_eff={:.3e})",
                quadrature.quadrature_error,
                verdict.max_abs_skewness,
                verdict.threshold,
                quadrature.axis_orders,
                quadrature.node_count,
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
                .unwrap_or_else(std::sync::PoisonError::into_inner) =
                Some(quadrature.axis_orders.clone());
            let mut decision = self.block_correction_decision_guard();
            if *decision == BlockCorrectionDecision::DecidingAtOptimum {
                *decision = BlockCorrectionDecision::AdmittedAtOptimum;
            }
            drop(decision);
            log::info!(
                "[#784] block-local correction ADMITTED for this fit: block dimension m={m} and \
                 axis orders {:?} are now the model's, and the tau={:.3} activation no longer \
                 switches the criterion on and off along the outer search (#2748, #2623)",
                quadrature.axis_orders,
                verdict.threshold,
            );
        }

        log::info!(
            "[#784] deterministic block-local Gauss-Hermite correction ENGAGED: \
             m={m}, max|γ|={:.3}, τ={:.3}, Δ_b={:.4e}, axis orders={:?}, nodes={} \
             [paired-rule error={:.4e}, error/|Δ_b|={:.3e}, 1/n_eff={:.3e}]",
            verdict.max_abs_skewness,
            verdict.threshold,
            quadrature.value,
            quadrature.axis_orders,
            quadrature.node_count,
            quadrature.quadrature_error,
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
        // with the TOTAL drift `Ḣ_j = λ_j S_j − C[v_j]`,
        // `C[v] = Xᵀ diag(c ⊙ Xv) X`, the IFT mode response
        // `dβ̂/dρ_j = −v_j = −H⁻¹ λ_j S_j β̂`, and
        //
        //   Q_b = Σ_r (M_r/λ_r) u_r u_rᵀ                       (rank m)
        //   Q_c = sym( Σ_r Σ_{q≠r} u_q (R̃_{q r}/(λ_r − σ_q)) u_rᵀ )
        //   M_r = E_p[(∂ΔF/∂t)_r · (−½ t_r)],   R̃ = Uᵀ E_p[t_r ∂ΔF/∂δ].
        //
        // Eigenvalue near-degeneracies `λ_r ≈ σ_q` are genuine
        // non-differentiability points of the eigenframe; the splice is
        // declined there rather than clamped.
        let Some(moments) = quadrature.moments.as_ref() else {
            // The corrector's contract reserves absent moments for the empty block, and
            // `block_cols` is non-empty here.
            return Err(EstimationError::BlockQuadratureCorrectionRefused {
                stage: BlockQuadratureCorrectionStage::CorrectorReturnedNoMoments {
                    block_dim: m,
                },
            });
        };
        let x = x_dense.as_ref();
        let n_rows = x.nrows();
        let xv = x.dot(&target.block_vecs); // n × m
        let ngs_base = target
            .base_neg_score()
            .map_err(EstimationError::InvalidInput)?;

        // σ²_i = E_p[s_i²] and the shared n×m intermediates.
        let xv_ett = xv.dot(&moments.e_tt); // n × m
        let sigma2 = (&xv_ett * &xv).sum_axis(ndarray::Axis(1)); // n
        let mut w_xv_ett = xv_ett.clone();
        for i in 0..n_rows {
            let w_i = target.weights_obs[i];
            w_xv_ett.row_mut(i).mapv_inplace(|v| v * w_i);
        }

        // Channel (d) moment: g_d = E_p[∂ΔF/∂β̂]
        //   = Xᵀ(E_p[ngs_disp] − ngs_base) + Σ_k λ_k S_k (V_b E_p[t])
        //     − ½ Xᵀ(c ⊙ E_p[s²]).
        let delta_mean = target.block_vecs.dot(&moments.e_t); // p
        let mut g_d = x.t().dot(&(&moments.e_neg_score - &ngs_base));
        for (pen, &lam) in target.penalties.iter().zip(target.lambdas.iter()) {
            g_d.scaled_add(lam, &transformed_penalty_matvec(pen, &delta_mean));
        }
        g_d.scaled_add(-0.5, &x.t().dot(&(c_weights * &sigma2)));

        // Channel (c) moment: R[:,r] = E_p[t_r · ∂ΔF/∂δ]
        //   = Xᵀ E_p[t_r ngs_disp] + (Σ_k λ_k S_k β̂) E_p[t_r] − Xᵀ W X V_b E_p[t tᵀ][:,r].
        let mut pen_score_total = Array1::<f64>::zeros(p);
        for (score, &lam) in target.penalty_scores.iter().zip(target.lambdas.iter()) {
            pen_score_total.scaled_add(lam, score);
        }
        let mut r_mat = x.t().dot(&moments.e_t_neg_score); // p × m
        for r in 0..m {
            r_mat
                .column_mut(r)
                .scaled_add(moments.e_t[r], &pen_score_total);
        }
        r_mat -= &x.t().dot(&w_xv_ett);

        // Channel (b) moment: M_r = E_p[(∂ΔF/∂t)_r (−½ t_r)] via
        // ∂ΔF/∂t = (XV)ᵀ ngs_disp + V_bᵀ(Σλ_k S_k β̂) − (XV)ᵀ(W ⊙ s).
        let xvt_etngs = xv.t().dot(&moments.e_t_neg_score); // m × m
        let pterm = target.block_vecs.t().dot(&pen_score_total); // m
        let xvt_w_xv_ett = xv.t().dot(&w_xv_ett); // m × m
        let mut m_vec = Array1::<f64>::zeros(m);
        for r in 0..m {
            m_vec[r] =
                -0.5 * (xvt_etngs[(r, r)] + pterm[r] * moments.e_t[r] - xvt_w_xv_ett[(r, r)]);
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
        // negative for an added quartic penalty. So the value: -quadrature.value
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
        for j in 0..n_rho.min(quadrature.rho_gradient.len()) {
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
            gradient[j] = -quadrature.rho_gradient[j] + trace_j + mode_j;
            if audit_armed {
                audit_a.push(quadrature.rho_gradient[j]);
                audit_trace.push(trace_j);
                audit_mode.push(mode_j);
                audit_spliced.push(gradient[j]);
            }
        }
        if audit_armed {
            crate::estimate::outer_eval_capture::record_quadrature_marginal(
                crate::estimate::outer_eval_capture::QuadratureMarginalAudit {
                    delta_b: quadrature.value,
                    quadrature_error: quadrature.quadrature_error,
                    node_count: quadrature.node_count,
                    axis_orders: quadrature.axis_orders.clone(),
                    axis_quadrature_errors: quadrature.axis_quadrature_errors.clone(),
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
            value: -quadrature.value,
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
