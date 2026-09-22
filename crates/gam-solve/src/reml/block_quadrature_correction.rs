//! The #784 block-local Gauss-Hermite marginal correction.
//!
//! Extracted verbatim from `gradient_hessian.rs`, which the repo's own
//! 10,000-line gate refuses to carry (#780). It is one self-contained unit:
//! the per-bundle cached wrapper, the compute path that runs the skewness
//! diagnostic, selects the curvature-heavy block, integrates the non-Gaussian
//! remainder over it — axis by axis plus the mixed-axis part
//! ([`MixedAxisRule`]) when the block has more than one axis — and
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
    /// directions at the spectral positions the admission integrated
    /// ([`BlockQuadratureLatch::block_positions`]) at each ρ rather than a set
    /// re-selected by a threshold crossing or a `|γ_r|` ranking. The spliced
    /// objective is a function of ρ again, and the spliced gradient stays exact: the four
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
    /// converged inner state and ρ alone, but the outer loop evaluates the
    /// objective at one ρ up to three times (value, value+gradient,
    /// value+gradient+Hessian) sharing the SAME `bundle`. Hoist it onto
    /// `bundle.block_local_correction` so the eigendecomposition and the
    /// quadrature run once per inner solution and every consumer at that ρ
    /// reads the identical value+gradient (exact hoist — #784, #1082).
    ///
    /// `want_hessian` says whether the evaluation asks for the ρ-Hessian. Its
    /// second-order pass over the nodes is paid only then: a value or
    /// value+gradient evaluation (a line search, the ρ-posterior sampler's
    /// leapfrog steps) never reads it.
    pub(crate) fn block_local_quadrature_correction(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        n_ext: usize,
        want_hessian: bool,
    ) -> Result<TkCorrectionTerms, EstimationError> {
        // This is the ONE seam every spliced evaluation passes through, fresh or
        // cached, so it is where the quadrature's certified error is published to
        // the certificate (#3004). Publishing at the computing site instead would
        // leave the second and third assembles at one ρ — which share the bundle
        // and hit the cache — with no quadrature term in their `band_f`, exactly
        // the way the audit record went missing before #2623.
        let publish = |quadrature_error: f64| {
            crate::estimate::outer_eval_capture::record_certificate_quadrature(
                crate::estimate::outer_eval_capture::QuadratureCharge {
                    error: quadrature_error,
                },
            );
        };
        // A deferred search prices the Laplace criterion, which is not this
        // bundle's correction once the admission is decided, so it is not cached.
        if self.block_correction_admission_deferred() {
            let (terms, quadrature_error) = self
                .block_local_quadrature_correction_compute(rho, bundle, n_ext, want_hessian)?;
            publish(quadrature_error);
            return Ok(terms);
        }
        if let Some(entry) = bundle.block_local_correction.get(n_ext, want_hessian) {
            // Re-publish the audit record the computing call wrote: the window
            // was cleared at the start of THIS assemble call, so without this a
            // ρ whose splice engaged reads back as declined on every assemble
            // after the first (#2623).
            if let Some(record) = entry.audit {
                crate::estimate::outer_eval_capture::record_quadrature_marginal(record);
            }
            publish(entry.quadrature_error);
            return Ok((*entry.terms).clone());
        }
        let (terms, quadrature_error) =
            self.block_local_quadrature_correction_compute(rho, bundle, n_ext, want_hessian)?;
        publish(quadrature_error);
        bundle
            .block_local_correction
            .store(super::BlockLocalCorrectionCache {
                n_ext,
                terms: std::sync::Arc::new(terms.clone()),
                carries_hessian: want_hessian,
                audit: crate::estimate::outer_eval_capture::last_quadrature_marginal_record(),
                quadrature_error,
            });
        Ok(terms)
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

    /// Why the latched #784 correction has no closed-form ρ-Hessian on this
    /// fit, or `None` when it is not latched or carries its exact ρ-Hessian
    /// (`block_correction_hessian`). A criterion with a refused Hessian declares
    /// none: its search runs on BFGS curvature and its smoothing-corrected
    /// covariance refuses with this reason.
    pub(crate) fn block_correction_hessian_refusal(&self) -> Option<String> {
        if self
            .block_correction_admission
            .load(std::sync::atomic::Ordering::Relaxed)
            == 0
        {
            return None;
        }
        self.block_correction_axis_orders
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .and_then(|latch| latch.hessian_refusal.clone())
    }

    pub(crate) fn block_correction_admission_deferred(&self) -> bool {
        *self.block_correction_decision_guard() == BlockCorrectionDecision::DeferredToOptimum
    }

    /// Decide the deferred admission once, at the search's certified Laplace
    /// optimum `rho` (#1082). One criterion evaluation there runs the skewness
    /// verdict and, when it engages, the order search that latches the block,
    /// exactly as a first admission does (#2748).
    ///
    /// `certified_value_band` is the certificate's own value resolution at
    /// `rho` — `band_f`, the error the evaluated `V` already carries there
    /// ([`CriterionErrorBound::value_band`](crate::model_types::CriterionErrorBound)).
    /// It is the ORDER TARGET the block's Gauss–Hermite rules are selected
    /// against (#3004): resolving `Δ_b` finer than the value the certificate can
    /// distinguish buys no decision, and the correction's own error is charged
    /// into that same `band_f`. `None` where the search certified no value bound,
    /// and the correction then declines rather than choosing a target itself.
    ///
    /// Returns whether the correction was admitted. If it was, `rho` was
    /// certified under a criterion that is no longer the model's, and the caller
    /// continues the corrected search from it. A correction refused at `rho` is
    /// the fit's error: the verdict requires the correction at the point the fit
    /// would publish, and it cannot be evaluated there.
    pub(crate) fn decide_block_correction_admission(
        &self,
        rho: &Array1<f64>,
        certified_value_band: Option<f64>,
    ) -> Result<bool, EstimationError> {
        // A family the correction never applies to has its decision without an
        // evaluation: the verdict below would only reach the same decline, after
        // a reset that discards the certified optimum's inner solve and a fresh
        // one to replace it.
        if let Some(reason) = self.block_correction_family_decline() {
            log::trace!("[#784] block-local correction declined at the optimum: {reason}");
            *self.block_correction_decision_guard() = BlockCorrectionDecision::DeclinedAtOptimum;
            return Ok(false);
        }
        *self
            .block_correction_value_band
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) =
            certified_value_band.filter(|band| band.is_finite() && *band > 0.0);
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

    /// Why the correction never applies to this fit's likelihood, if it never
    /// does. Each reason reads only the configured family, so it holds at every
    /// ρ of the fit.
    fn block_correction_family_decline(&self) -> Option<&'static str> {
        if reml_is_gaussian_identity(&self.config.likelihood) {
            return Some("Laplace is exact for the Gaussian-identity model");
        }
        // The exact score channel relies on the exponential-family unit-
        // deviance identity dD/dμ = −2w(y−μ)/V(μ), which does not hold for
        // the Beta pseudo-family parameterization. Decline rather than splice
        // a gradient that is not the derivative of the spliced value.
        if matches!(
            reml_spec(&self.config.likelihood).response,
            ResponseFamily::Beta { .. }
        ) {
            return Some(
                "the Beta family has no exponential-family score identity for the exact \
                 gradient channels",
            );
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
            return Some(
                "Firth/Jeffreys bias reduction is active and the block target integrates \
                 the plain penalized likelihood, not the Jeffreys-penalized one",
            );
        }
        None
    }

    /// The correction's terms together with the CERTIFIED ERROR of the
    /// quadrature that produced them, in `V`'s own units (#3004).
    ///
    /// The error travels with the value because the certificate charges it in
    /// `band_f` beside the channel, factor and inner-residual terms. A declined
    /// splice returns `0.0`, which is the exact error of a correction that was
    /// not taken, not a missing measurement.
    fn block_local_quadrature_correction_compute(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        n_ext: usize,
        want_hessian: bool,
    ) -> Result<(TkCorrectionTerms, f64), EstimationError> {
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
            return Ok((zero(), 0.0));
        }

        if let Some(reason) = self.block_correction_family_decline() {
            log::trace!("[#784] block-local fallback declined: {reason}");
            return Ok((zero(), 0.0));
        }
        // The mode and trace channels need one λ per canonical penalty.
        if rho.len() != n_rho || n_rho == 0 {
            return Ok((zero(), 0.0));
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
            return Ok((zero(), 0.0));
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
        // The predicate below and the family declines above decline the whole
        // correction, and none consults a single number the diagnostic
        // produces: this one reads the hyper-layout, the others the configured
        // response family. All are therefore constant across the entire fit.
        //
        // They used to sit AFTER `directional_cubic_diagnostic` — an `O(p³)`
        // dense factorization plus `O(n·p)` cubic contractions — so every
        // ψ-carrying model and every Beta fit paid that sweep on EVERY outer
        // evaluation and then discarded it, guaranteed, with the decline logged
        // as though it had been decided on evidence (gam#2584). Evidence is
        // worth buying only when the verdict can depend on it.
        //
        // Hoisting them is exactly value-preserving: no predicate reads
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
            return Ok((zero(), 0.0));
        }

        // Resolve the injected gam-inference corrector. When the inference tier
        // is not linked / registered, decline the correction (zero contribution) —
        // the same safe no-op as every other decline branch here.
        let Some(corrector) = gam_problem::laplace_sampler_contract::laplace_marginal_corrector()
        else {
            return Ok((zero(), 0.0));
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
            .directional_cubic_diagnostic(&evals, &evecs, x_design, c_weights)
            .map_err(EstimationError::InvalidInput)?;
        if !max_abs.is_finite() || max_abs == 0.0 {
            return Ok((zero(), 0.0));
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
            return Ok((zero(), 0.0));
        }

        // Build the block subspace V_b. At admission the block is the flagged
        // set, the positive-curvature directions whose |γ_r| clears `τ`. Under
        // a latched admission it is the directions at the SPECTRAL POSITIONS
        // the admission integrated, NOT a set re-selected at this ρ.
        //
        // A set defined by a threshold crossing changes cardinality as ρ moves,
        // and every change is a jump of a whole direction's contribution to
        // `Δ_b` (#2748). Re-ranking by |γ_r| at every ρ keeps the cardinality
        // but still jumps: the set changes wherever two directions' |γ| cross,
        // a codimension-one surface in ρ, and each change swaps one axis's
        // whole contribution for another's, with the latched orders silently
        // reassigned to directions they were never certified on. Measured on
        // the convergence fuzzer's `case0/binomial/n1000` (m = 5, axis split):
        // `Δ_b` took exactly two values, 4.660e-2 and 5.797e-2, across the
        // BFGS polish's trial points at |g| = 1.4e-4, so the line search could
        // not pass sufficient decrease and the fit ended `line_search_failed`.
        // The eigenvector at a fixed position is a continuous function of ρ
        // away from an eigenvalue coincidence with a neighbouring position, so
        // the latched block is too, and the frame-rotation channel (c) below
        // differentiates exactly that motion. It is not uniformly smooth:
        // near an avoided crossing the eigenvector rotates over a ρ-width
        // about the size of the relative gap, and `Δ_b` moves as steeply
        // there (gaps down to 7e-4 on the fuzzer's `case45/binomial/n1000`).
        //
        // A position is a rank in ASCENDING eigenvalue order, not an index into
        // `evals`: the criterion's operator above is an `eigh` of the assembled
        // `H` (ascending) where that resolves `log|H|` and the root SVD
        // (descending) where it does not (#2644), and the route can change
        // between two ρ of one search.
        let ascending = ascending_spectral_order(&evals);
        let latched_quadrature = self
            .block_correction_axis_orders
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
            .filter(|latch| Some(latch.block_positions.len()) == latched_block_dim);
        let mut block_cols: Vec<usize> = match (&latched_quadrature, latched_block_dim) {
            (Some(latch), _) => {
                // The spectrum's dimension is the model's, so a latched position
                // outside it is a broken latch at every ρ, not a trial point.
                let Some(cols) = latch
                    .block_positions
                    .iter()
                    .map(|&k| ascending.get(k).copied())
                    .collect::<Option<Vec<usize>>>()
                else {
                    return Err(EstimationError::BlockQuadratureCorrectionRefused {
                        stage: BlockQuadratureCorrectionStage::LatchedBlockUnavailable {
                            block_dim: latch.block_positions.len(),
                            spectrum_dim: evals.len(),
                        },
                    });
                };
                // The block is whitened by `√λ_r`, so a latched direction whose
                // curvature is not positive at this ρ has no block marginal. That
                // is a fact about this trial point, which the outer search backs
                // away from (#3113).
                if let Some(&r) = cols
                    .iter()
                    .find(|&&r| !(evals[r].is_finite() && evals[r] > 0.0))
                {
                    return Err(EstimationError::BlockQuadratureCorrectionRefused {
                        stage: BlockQuadratureCorrectionStage::NonPositivePenalizedCurvature {
                            min_eigenvalue: evals[r],
                        },
                    });
                }
                cols
            }
            // The admission and its block are latched together below, so an
            // admission without its block is a broken latch, not a model.
            (None, Some(m)) => {
                return Err(EstimationError::BlockQuadratureCorrectionRefused {
                    stage: BlockQuadratureCorrectionStage::LatchedBlockUnavailable {
                        block_dim: m,
                        spectrum_dim: evals.len(),
                    },
                });
            }
            (None, None) => verdict
                .untrustworthy_directions
                .iter()
                .copied()
                .filter(|&r| r < evals.len() && evals[r] > 0.0)
                .collect(),
        };
        order_block_axes_by_curvature(&mut block_cols, &evals);
        if block_cols.is_empty() {
            return Ok((zero(), 0.0));
        }
        let m = block_cols.len();
        let (block_vecs, block_lambdas) =
            oriented_block_axes(&evecs, &evals, &directional, &block_cols);

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

        // THE ORDER TARGET IS THE CERTIFICATE'S OWN VALUE RESOLUTION (#3004).
        //
        // Each axis's Gauss–Hermite order is selected ONCE, at admission, as the
        // smallest order whose paired difference with the next lower rule
        // resolves `min(|Δ|, target)`, and latched beside the block dimension.
        // Under a latched admission the orders are the model's, so every ρ
        // integrates against the same nodes and the value, gradient and moments
        // share one measure. The paired lower rules then switch nothing (#2748),
        // so a latched evaluation integrates the fine rule alone and carries the
        // admission's paired errors as its certificate: on a three-axis block the
        // lower rules are five times the fine rule's nodes. A one-axis piece
        // latches its composite partition the same way (below).
        //
        // The target used to be `1/n_eff²`, the next-order remainder of the
        // Laplace expansion the correction removes the `O(1/n_eff)` term of. That
        // is a STATISTICAL accuracy target, and nothing reads the criterion
        // statistically: every consumer reads it through a certificate that
        // cannot distinguish two values closer than `band_f`. On the measured
        // n = 10k binomial fit the two were five decades apart — the search drove
        // the paired error to 7.2e-10 against a `band_f` of 1.429e-3 — and paid
        // for it in nodes, which is 95% of every outer evaluation once the
        // correction engages. Resolving below `band_f` cannot change a verdict,
        // a comparison of two values, or a stall decision, because each of those
        // is decided against `band_f` itself.
        //
        // The certificate now CHARGES this rule's error in that same `band_f`
        // (`ObjectiveBand::quadrature`), so the target is not a licence to be
        // sloppy: a looser rule widens the band it is judged against, and the
        // decrement verdict tightens with it. That is the ordering this change
        // depends on — the charge exists before the target moves.
        //
        // No band, no correction. A correction admitted where the search
        // certified no value bound has no derived target at all, and picking one
        // would be choosing a number. Declining returns the exact Laplace
        // criterion, which is what the fit had before #784 and is always valid.
        // The Laplace term the correction removes, reported beside the target so
        // a reader can see how far apart the statistical and the arithmetic
        // scales are on this fit.
        let laplace_floor = if n_eff > 0.0 {
            1.0 / n_eff
        } else {
            f64::INFINITY
        };
        let Some(resolution_band) = *self
            .block_correction_value_band
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
        else {
            log::trace!(
                "[#784] block-local correction declined: the admission carried no certified \
                 value band, so its Gauss–Hermite order target is underived (#3004)"
            );
            return Ok((zero(), 0.0));
        };

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
        //   Δ_b = Σ_r Δ_r + Ψ + O(n_eff⁻²),
        //
        // where Δ_r is the exact one-dimensional correction along axis r (the
        // same target restricted to one column of V_b, integrated by its own
        // certified 1-D rule) and Ψ is the mixed-axis part: the two- and
        // three-axis interactions of the anchored ANOVA of the block marginal,
        // each integrated exactly on the closed-form three-point Gauss–Hermite
        // rule ([`MixedAxisRule`]). No term through O(n_eff⁻¹) couples more than
        // three axes, the O(n_eff⁻³ᐟ²) ones are odd and vanish, so the truncation
        // is O(n_eff⁻²) — the order the quadrature is asked to resolve. Ψ is a
        // combination of log-integrals of e^{−ΔF}, bounded along a soft axis as
        // the Δ_r are. The cost is Σ_r o_r nodes plus
        // 1 + 2m + 4·C(m,2) + 8·C(m,3) for Ψ.
        //
        // Ψ's rule is exact on the interactions only when the excess starts at
        // cubic order, so the split requires the exported curvature to be the
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
        // Whether `Δ_b` has a closed-form ρ-Hessian on this fit, and how its row
        // curvature is read off the mode. It reads only the family, the link,
        // the inner solve's curvature contract and the latched block shape, so
        // it is one answer for the whole fit.
        let hessian_support = super::block_correction_hessian::block_correction_row_curvature(
            pirls_result,
            &target.inverse_link,
            axis_split,
            m,
        );

        // Integrate each piece and contract its moments into the gradient
        // channels at once, so one axis target (and its moments) is alive at a
        // time.
        let piece_count = if axis_split { m } else { 1 };
        let mut pieces: Vec<BlockPieceQuadrature> = Vec::with_capacity(piece_count);
        let mut axis_orders: Vec<usize> = Vec::with_capacity(m);
        let mut piece_rules: Vec<LatchedPieceRule> = Vec::with_capacity(piece_count);
        let mut truncated_pieces: Vec<bool> = Vec::new();
        let mut composite_pieces: Vec<bool> = Vec::new();
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
            // A one-axis piece is integrated by the composite Gauss–Kronrod rule, whose
            // bisection resolves a wall no representable Gauss–Hermite order reaches.
            // The partition is adapted once, at admission, and latched: every later ρ
            // integrates on the same cells in the oriented standardized axis, so the
            // criterion is one rule's value and its channels are that rule's gradient.
            // A truncated axis (a positive-domain link) keeps the Gauss–Hermite rule,
            // whose truncated-normal transport integrates exactly to the cut; the
            // composite rule would meet the cut as infeasible nodes.
            let latched_piece = latched_quadrature.as_ref().map(|latch| &latch.pieces[k]);
            let composite_axis = match latched_piece {
                Some(rule) => matches!(rule, LatchedPieceRule::Composite { .. }),
                None => width == 1 && piece_target.axis_truncation().is_none(),
            };
            let (mut quadrature, piece_rule, composite_nodes) = match latched_piece {
                Some(LatchedPieceRule::Composite { breakpoints }) => {
                    let latched = corrector
                        .composite_axis_marginal_correction_on_partition(piece_target, breakpoints)
                        .map_err(order_search_refused)?;
                    (
                        latched.marginal,
                        LatchedPieceRule::Composite {
                            breakpoints: breakpoints.clone(),
                        },
                        Some(latched.nodes),
                    )
                }
                Some(LatchedPieceRule::GaussHermite {
                    axis_orders: orders,
                    certified_axis_errors,
                }) => (
                    corrector
                        .block_quadrature_marginal_correction_at_certified_orders(
                            piece_target,
                            orders,
                            certified_axis_errors,
                        )
                        .map_err(|refusal| EstimationError::InvalidInput(refusal.to_string()))?,
                    LatchedPieceRule::GaussHermite {
                        axis_orders: orders.clone(),
                        certified_axis_errors: certified_axis_errors.clone(),
                    },
                    None,
                ),
                None if composite_axis => {
                    let adapted = corrector
                        .composite_axis_marginal_correction(piece_target, resolution_band)
                        .map_err(order_search_refused)?;
                    (
                        adapted.marginal,
                        LatchedPieceRule::Composite {
                            breakpoints: adapted.breakpoints,
                        },
                        Some(adapted.nodes),
                    )
                }
                None => {
                    let selected =
                        gam_problem::laplace_sampler_contract::select_block_quadrature_orders(
                            corrector,
                            piece_target,
                            resolution_band,
                        )
                        .map_err(order_search_refused)?;
                    let rule = LatchedPieceRule::GaussHermite {
                        axis_orders: selected.axis_orders.clone(),
                        certified_axis_errors: selected.axis_quadrature_errors.clone(),
                    };
                    (selected, rule, None)
                }
            };
            if crate::estimate::outer_eval_capture::rho_outer_audit_enabled() {
                composite_pieces.push(matches!(piece_rule, LatchedPieceRule::Composite { .. }));
            }
            piece_rules.push(piece_rule);
            axis_orders.extend_from_slice(&quadrature.axis_orders);
            if crate::estimate::outer_eval_capture::rho_outer_audit_enabled() {
                truncated_pieces.push(piece_target.axis_truncation().is_some_and(|truncation| {
                    truncation.lower().is_some() || truncation.upper().is_some()
                }));
            }
            let Some(moments) = quadrature.moments.take() else {
                // The corrector's contract reserves absent moments for the empty
                // block, and every piece has at least one axis.
                return Err(EstimationError::BlockQuadratureCorrectionRefused {
                    stage: BlockQuadratureCorrectionStage::CorrectorReturnedNoMoments {
                        block_dim: m,
                    },
                });
            };
            let channels = block_target_channel_moments(piece_target, &moments, c_weights, 1.0)?;
            pieces.push(BlockPieceQuadrature {
                resolution_target: quadrature.value.abs().min(resolution_band),
                quadrature,
                channels,
                composite_nodes,
            });
        }

        let x = x_dense.as_ref();
        let mixed = if axis_split {
            Some(mixed_axis_term(&target, c_weights)?)
        } else {
            None
        };

        let delta_b = pieces.iter().map(|piece| piece.quadrature.value).sum::<f64>()
            + mixed.as_ref().map_or(0.0, |term| term.value);
        let axis_quadrature_errors: Vec<f64> = pieces
            .iter()
            .flat_map(|piece| piece.quadrature.axis_quadrature_errors.iter().copied())
            .collect();
        // The pieces' errors add in Δ_b, so their sum bounds its error.
        let quadrature_error: f64 = pieces
            .iter()
            .map(|piece| piece.quadrature.quadrature_error)
            .sum();
        let node_count: usize = pieces.iter().map(|piece| piece.quadrature.node_count).sum::<usize>()
            + mixed.as_ref().map_or(0, |term| term.node_count);
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
                 #2748): the latched rule's measured error {:.4e} does not resolve \
                 min(|Δ|, band_f)={:.4e} (|Δ_b|={abs_value:.4e}, m={m}, axis split={axis_split}, \
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
        // the exact boundary of "the correction is part of this model". The
        // block, its quadrature and its Hessian support latch together, so the
        // criterion's Hessian declaration reads this fit's own answer; an
        // admission without its block is refused where the block is chosen.
        if latched_block_dim.is_none() {
            let mut rank_of = vec![0; ascending.len()];
            for (k, &r) in ascending.iter().enumerate() {
                rank_of[r] = k;
            }
            let block_positions: Vec<usize> = block_cols.iter().map(|&r| rank_of[r]).collect();
            self.block_correction_admission
                .store(m + 1, std::sync::atomic::Ordering::Relaxed);
            *self
                .block_correction_axis_orders
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(BlockQuadratureLatch {
                block_positions: block_positions.clone(),
                pieces: piece_rules,
                axis_split,
                hessian_refusal: hessian_support.as_ref().err().cloned(),
            });
            let mut decision = self.block_correction_decision_guard();
            if *decision == BlockCorrectionDecision::DecidingAtOptimum {
                *decision = BlockCorrectionDecision::AdmittedAtOptimum;
            }
            drop(decision);
            log::debug!(
                "[#784] block-local correction ADMITTED for this fit: block dimension m={m}, \
                 spectral positions {block_positions:?}, axis split={axis_split} and axis orders \
                 {:?} are now the model's, and the tau={:.3} activation no longer switches \
                 the criterion on and off along the outer search (#2748, #2623)",
                axis_orders,
                verdict.threshold,
            );
        }

        log::debug!(
            "[#784] deterministic block-local Gauss-Hermite correction ENGAGED: \
             m={m}, axis split={axis_split}, max|γ|={:.3}, τ={:.3}, Δ_b={:.4e} \
             (mixed-axis Ψ={:.4e}), axis orders={:?}, nodes={node_count} \
             [paired-rule error={:.4e}, error/|Δ_b|={:.3e}, 1/n_eff={:.3e}]",
            verdict.max_abs_skewness,
            verdict.threshold,
            delta_b,
            mixed.as_ref().map_or(0.0, |term| term.value),
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
        // g_d and explicit gradient, which add. Ψ integrates the whole block's
        // excess at t_r = z_r/√λ_r, so its channels are the tensor rule's under
        // the signed node measure ω (∂(−Ψ) = Σ_z ω ∂ΔF), over every axis; they
        // add to the pieces'. It has no explicit ρ-dependence either.
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
        if let Some(term) = mixed.as_ref() {
            g_d += &term.channels.g_d;
            r_mat += &term.channels.r_mat;
            m_vec += &term.channels.m_vec;
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

        // The exact ρ-Hessian of `−Δ_b` on the same eigensystem, nodes and
        // mode: every pair is resolved above, so the eigenframe is twice
        // differentiable here. A fit whose `Δ_b` has no closed-form Hessian
        // declares none (`BlockQuadratureLatch::hessian_refusal`), and its
        // smoothing-corrected covariance refuses with that reason. An
        // evaluation that does not ask for the Hessian does not pay for it.
        let cost_hessian = match &hessian_support {
            Ok(curvature) if want_hessian => {
                let piece_rules: Vec<super::block_correction_hessian::PieceRule<'_>> = pieces
                    .iter()
                    .map(|piece| match &piece.composite_nodes {
                        Some(nodes) => {
                            super::block_correction_hessian::PieceRule::Composite { nodes }
                        }
                        None => super::block_correction_hessian::PieceRule::GaussHermite {
                            order: piece.quadrature.axis_orders[0],
                        },
                    })
                    .collect();
                let second_order = super::block_correction_hessian::block_correction_cost_hessian(
                    &target,
                    &super::block_correction_hessian::BlockCorrectionHessianInputs {
                        curvature: *curvature,
                        axis_split,
                        piece_rules: &piece_rules,
                        evals: &evals,
                        evecs: &evecs,
                        block_cols: &block_cols,
                        c: c_weights,
                        d: &pirls_result.solve_d_array.to_owned(),
                    },
                )?;
                log::trace!(
                    "[#784] ρ-Hessian pieces V={:?} (quadrature {:?}), Ψ={:?}",
                    second_order.piece_values,
                    pieces
                        .iter()
                        .map(|piece| piece.quadrature.value)
                        .collect::<Vec<_>>(),
                    second_order.mixed_value,
                );
                Some(second_order)
            }
            _ => None,
        };
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

        // Every channel is on the cost side: `gradient[j]` is ∂(−Δ_b)/∂ρ_j, the
        // derivative of the `value: −Δ_b` this function returns. The engaged
        // finite-difference rows in `regression_block_correction_outer_hessian_fd`
        // pin the total against cost differences on single- and multi-axis blocks
        // (#2623).
        let mut gradient = Array1::<f64>::zeros(n_rho + n_ext);
        for j in 0..n_rho {
            let lam_j = target.lambdas[j];
            let a_j = target.penalty_scores[j].mapv(|v| lam_j * v); // λ_j S_j β̂
            // v_j = H⁻¹ a_j through the same eigendecomposition as Q.
            let uta = evecs.t().dot(&a_j);
            let v_j = evecs.dot(&(&uta / &evals));
            // tr(A_j Q) = λ_j tr(S_j Q), over the penalty's own block.
            let penalty = &target.penalties[j];
            let range = penalty.col_range.clone();
            let tr_sq = (&penalty.local * &q_mat.slice(ndarray::s![range.clone(), range])).sum();
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
                    truncated_pieces,
                    composite_pieces,
                    explicit_a: audit_a,
                    trace_bc: audit_trace,
                    mode_d: audit_mode,
                    spliced: audit_spliced,
                },
            );
        }
        if let Some(second_order) = cost_hessian.as_ref() {
            log::trace!(
                "[#784] ρ-gradient spliced {:?} against the Hessian's own {:?}",
                gradient,
                second_order.implied_gradient,
            );
        }
        Ok((
            TkCorrectionTerms {
                value: -delta_b,
                gradient: Some(gradient),
                hessian: cost_hessian.map(|second_order| second_order.hessian),
            },
            quadrature_error,
        ))
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

/// The eigen indices of `evals` in ascending order of curvature, ties broken by
/// index: `order[k]` is the index of the rank-`k` eigenpair. The spectrum's own
/// index order is not a rank (it is ascending for `eigh` and descending for the
/// stacked-root SVD, see `order_block_axes_by_curvature`), so a block latched
/// by spectral position (#3113) is resolved through this order.
fn ascending_spectral_order(evals: &Array1<f64>) -> Vec<usize> {
    let mut order: Vec<usize> = (0..evals.len()).collect();
    order.sort_by(|&a, &b| evals[a].total_cmp(&evals[b]).then(a.cmp(&b)));
    order
}

/// The block axes `V_b` and their curvatures, each axis oriented so its standardized
/// skewness is positive.
///
/// An eigenvector's sign is the eigensolver's, not the model's, and `γ_r` is odd in
/// it, so `γ_r > 0` names one of the two directions from (H, γ) alone. A latched
/// composite partition is a rule in this oriented axis: in the solver's sign it would
/// be carried onto the mirror image of the integrand whenever the eigensolver flips
/// the vector between two ρ (#784). The gradient's eigenframe term `V G V_bᵀ` carries
/// the sign in both `G`'s column and `V_b`'s, so it is unchanged.
fn oriented_block_axes(
    evecs: &Array2<f64>,
    evals: &Array1<f64>,
    directional: &Array1<f64>,
    block_cols: &[usize],
) -> (Array2<f64>, Array1<f64>) {
    let mut block_vecs = Array2::<f64>::zeros((evecs.nrows(), block_cols.len()));
    let mut block_lambdas = Array1::<f64>::zeros(block_cols.len());
    for (j, &r) in block_cols.iter().enumerate() {
        let orientation = if directional[r] < 0.0 { -1.0 } else { 1.0 };
        block_vecs
            .column_mut(j)
            .assign(&evecs.column(r).mapv(|v| orientation * v));
        block_lambdas[j] = evals[r];
    }
    (block_vecs, block_lambdas)
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
///
/// This is the same comparator as the ascending spectral order the latch
/// records its block positions in, so the admission's block is sorted before
/// its ranks are taken and a latched block, read back in ascending position
/// order, is already in this order.
fn order_block_axes_by_curvature(block_cols: &mut [usize], evals: &Array1<f64>) {
    block_cols.sort_by(|&a, &b| evals[a].total_cmp(&evals[b]).then(a.cmp(&b)));
}

/// One integrated piece of the block marginal: the whole block under a tensor
/// rule, or one axis of it under the split, with its gradient channels.
struct BlockPieceQuadrature {
    quadrature: gam_problem::laplace_sampler_contract::BlockQuadratureMarginal,
    /// `min(|Δ_piece|, band_f)`, the error the piece's rule must resolve: its own
    /// value or the certificate's value resolution, whichever is smaller (#3004).
    resolution_target: f64,
    channels: BlockTargetChannels,
    /// A composite piece's nodes, which its second-order pass differentiates;
    /// `None` for a Gauss–Hermite piece, whose rule is its order.
    composite_nodes: Option<Vec<gam_problem::laplace_sampler_contract::CompositeNode>>,
}

/// The cost-side gradient-channel moments of one block target, over that
/// target's own axes: `g_d = E_p[∂ΔF/∂β̂]` (p), `R[:,r] = E_p[t_r ∂ΔF/∂δ]`
/// (p × width) and `M_r = E_p[(∂ΔF/∂t)_r (−½ t_r)]` (width).
struct BlockTargetChannels {
    g_d: Array1<f64>,
    r_mat: Array2<f64>,
    m_vec: Array1<f64>,
}

/// Contract a block target's quadrature moments into the channel moments the
/// exact (b)–(d) assembly consumes. The moments are linear in the node measure,
/// whose total `mass` is one for a normalized posterior; only the base score,
/// a constant of the nodes, reads it.
fn block_target_channel_moments(
    target: &Gam784BlockTarget<'_>,
    moments: &gam_problem::laplace_sampler_contract::BlockQuadratureMoments,
    c_weights: &Array1<f64>,
    mass: f64,
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
    let mut g_d = x.t().dot(&(&moments.e_neg_score - &(&ngs_base * mass)));
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
pub(super) fn block_axis_target<'t>(target: &Gam784BlockTarget<'t>, r: usize) -> Gam784BlockTarget<'t> {
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

/// How many block axes one term of the mixed-axis part couples.
///
/// With `z ~ N(0, I_m)` the whitened block coordinates and `ΔF = ΔF₃ + ΔF₄ + …`
/// the excess by its order in `z` (the mode condition cancels the linear term
/// and `W = ψ''(η̂)` the quadratic one), the cumulant expansion of the block
/// marginal is
///
///   log E[e^{−ΔF}] = −E[ΔF₄] + ½ E[ΔF₃²] + O(n_eff⁻²),
///
/// the `O(n_eff⁻³ᐟ²)` terms being odd Gaussian moments. `E[ΔF₄]` reads the
/// monomials `z_p² z_q²` and `E[ΔF₃²]` the products of two cubic monomials,
/// whose even part is at most `z_p² z_q² z_r²`: no term through `O(n_eff⁻¹)`
/// couples more than three axes. The anchored ANOVA of the marginal over the
/// axis subsets, truncated past three-axis interactions, is therefore exact to
/// the same `O(n_eff⁻²)` the certified single-axis rules resolve.
const MIXED_AXIS_COUPLING: usize = 3;

/// The three-point Gauss–Hermite rule for the standard normal, in closed form:
/// the nodes are the zeros of `He₃(z) = z³ − 3z`, `{0, ±√3}`, and the weights
/// `3!/(3² He₂(z_k)²)` with `He₂ = z² − 1`, `{2/3, 1/6, 1/6}`. It integrates
/// every polynomial of degree at most five exactly, which covers each axis's
/// degree (at most four) in the mixed monomials of [`MIXED_AXIS_COUPLING`]. The
/// middle node is the anchor `z = 0` exactly, so a piece's nodes restricted to
/// fewer axes are the smaller pieces' nodes.
fn three_point_rule() -> [(f64, f64); 3] {
    let root = 3.0_f64.sqrt();
    [(0.0, 2.0 / 3.0), (-root, 1.0 / 6.0), (root, 1.0 / 6.0)]
}

fn binomial_coefficient(n: usize, k: usize) -> i64 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut value: i64 = 1;
    for j in 0..k {
        value = value * (n - j) as i64 / (j + 1) as i64;
    }
    value
}

/// The coefficient `κ_T` of an axis subset of size `size` in the mixed-axis
/// part `Ψ = Σ_T κ_T f_T` of an `m`-axis block.
///
/// Möbius inversion over the subsets writes the block marginal `f_{[m]}` as
/// `Σ_U g_U` with `g_U = Σ_{V⊆U} (−1)^{|U∖V|} f_V`. Keeping `|U| ≤ K` and
/// collecting each `f_V` gives `f_{[m]} ≈ Σ_{|V|≤K} c_{|V|} f_V` with
/// `c_k = Σ_{j=0}^{K−k} (−1)^j C(m−k, j)`. The single-axis pieces are the
/// certified `Δ_r`, so the three-point rule carries `κ = c_k − [k = 1]`, and
/// `Σ_{T⊇U} κ_T = [|U| ≥ 2]` for every `1 ≤ |U| ≤ K`: every single-axis part
/// of the three-point pieces cancels, its rule error with it, and every mixed
/// part is counted once.
fn mixed_axis_piece_coefficient(m: usize, size: usize) -> i64 {
    let coupling = MIXED_AXIS_COUPLING.min(m);
    let mut c = 0_i64;
    for j in 0..=coupling - size {
        let term = binomial_coefficient(m - size, j);
        c += if j % 2 == 0 { term } else { -term };
    }
    c - i64::from(size == 1)
}

/// Call `visit` with every size-`size` subset of `0..m`, in lexicographic order.
fn for_each_axis_subset(m: usize, size: usize, mut visit: impl FnMut(&[usize])) {
    let mut subset: Vec<usize> = (0..size).collect();
    'next: loop {
        visit(&subset);
        let mut i = size;
        while i > 0 {
            i -= 1;
            if subset[i] != i + m - size {
                subset[i] += 1;
                for j in i + 1..size {
                    subset[j] = subset[j - 1] + 1;
                }
                continue 'next;
            }
        }
        break;
    }
}

/// The deterministic rule for the mixed-axis part of a split block.
///
/// Each axis subset `T` with `1 ≤ |T| ≤ min(K, m)` and `κ_T ≠ 0` is a piece:
/// the three-point tensor rule over the axes in `T`, anchored at `z = 0` on
/// the others,
///
///   f_T = log Σ_z w_T(z) e^{−ΔF(z)} − log Σ_z w_T(z),
///
/// and `Ψ = Σ_T κ_T f_T`. The pieces share their nodes, so the rule holds each
/// distinct node once: `1 + 2m + 4·C(m,2) + 8·C(m,3)` of them.
pub(super) struct MixedAxisRule {
    /// Standard-normal node coordinates `z`, one node per column (`m × q`).
    pub(super) nodes: Array2<f64>,
    pub(super) pieces: Vec<MixedAxisPiece>,
}

pub(super) struct MixedAxisPiece {
    pub(super) coefficient: f64,
    /// `(node, ln w_T(z))` for each of the piece's `3^|T|` nodes; the node
    /// count is the one record of `|T|`.
    pub(super) nodes: Vec<(usize, f64)>,
}

/// `Ψ` at one evaluation, with the measure its derivatives are taken against.
pub(super) struct MixedAxisPosterior {
    pub(super) value: f64,
    /// `ω(z) = Σ_T κ_T p_T(z)` per node, `p_T ∝ w_T e^{−ΔF}` a piece's
    /// normalized node posterior (zero on an infeasible node), so
    /// `∂(−Ψ)/∂θ = Σ_z ω(z) ∂ΔF(z)/∂θ`.
    pub(super) node_weights: Array1<f64>,
    /// `Σ_z ω(z) = Σ_T κ_T`.
    pub(super) mass: f64,
    pub(super) pieces: Vec<MixedAxisPiecePosterior>,
}

pub(super) struct MixedAxisPiecePosterior {
    pub(super) coefficient: f64,
    /// `f_T`; the posterior's value is `Σ_T κ_T f_T` over these pieces.
    pub(super) log_ratio: f64,
    /// `(node, p_T(z))` over the piece's feasible nodes.
    pub(super) nodes: Vec<(usize, f64)>,
}

impl MixedAxisRule {
    pub(super) fn new(m: usize) -> Self {
        let rule = three_point_rule();
        let mut keys: HashMap<Vec<u8>, usize> = HashMap::new();
        let mut coordinates: Vec<Vec<u8>> = Vec::new();
        let anchor = vec![0_u8; m];
        keys.insert(anchor.clone(), 0);
        coordinates.push(anchor);
        let mut pieces = Vec::new();
        for size in 1..=MIXED_AXIS_COUPLING.min(m) {
            let coefficient = mixed_axis_piece_coefficient(m, size);
            if coefficient == 0 {
                continue;
            }
            let pattern_count = 3_usize.pow(size as u32);
            for_each_axis_subset(m, size, |subset| {
                let mut nodes = Vec::with_capacity(pattern_count);
                for pattern in 0..pattern_count {
                    let mut key = vec![0_u8; m];
                    let mut log_weight = 0.0;
                    let mut digits = pattern;
                    for &axis in subset {
                        let point = digits % 3;
                        digits /= 3;
                        key[axis] = point as u8;
                        log_weight += rule[point].1.ln();
                    }
                    let next = coordinates.len();
                    let node = *keys.entry(key.clone()).or_insert(next);
                    if node == next {
                        coordinates.push(key);
                    }
                    nodes.push((node, log_weight));
                }
                pieces.push(MixedAxisPiece {
                    coefficient: coefficient as f64,
                    nodes,
                });
            });
        }
        let nodes = Array2::from_shape_fn((m, coordinates.len()), |(r, node)| {
            rule[usize::from(coordinates[node][r])].0
        });
        Self { nodes, pieces }
    }

    pub(super) fn node_count(&self) -> usize {
        self.nodes.ncols()
    }

    /// `Ψ` and its node measure from the excess `ΔF` at every node. An
    /// infeasible node (`ΔF = +∞`) is dropped from each piece it is in, as the
    /// tensor rule drops it; the anchor is the mode and always feasible.
    pub(super) fn posterior(&self, excesses: &[f64]) -> Result<MixedAxisPosterior, EstimationError> {
        let q = self.node_count();
        if excesses.len() != q {
            crate::bail_invalid_estim!(
                "#784 mixed-axis rule: {} excesses for {q} nodes",
                excesses.len()
            );
        }
        if !excesses[0].is_finite() {
            crate::bail_invalid_estim!(
                "#784 mixed-axis rule: the excess at the mode is {}",
                excesses[0]
            );
        }
        if let Some(node) = excesses.iter().position(|&e| e.is_nan() || e == f64::NEG_INFINITY) {
            crate::bail_invalid_estim!(
                "#784 mixed-axis rule: node {node} has excess {}",
                excesses[node]
            );
        }
        let mut mass = 0.0;
        let mut node_weights = Array1::<f64>::zeros(q);
        let mut pieces = Vec::with_capacity(self.pieces.len());
        for piece in &self.pieces {
            let log_norm = log_sum_exp(piece.nodes.iter().map(|&(_, lw)| lw));
            let feasible: Vec<(usize, f64)> = piece
                .nodes
                .iter()
                .filter(|&&(node, _)| excesses[node].is_finite())
                .map(|&(node, lw)| (node, lw - excesses[node]))
                .collect();
            let log_mass = log_sum_exp(feasible.iter().map(|&(_, lw)| lw));
            let log_ratio = log_mass - log_norm;
            let nodes: Vec<(usize, f64)> = feasible
                .into_iter()
                .map(|(node, lw)| (node, (lw - log_mass).exp()))
                .collect();
            for &(node, prob) in &nodes {
                node_weights[node] += piece.coefficient * prob;
            }
            mass += piece.coefficient;
            pieces.push(MixedAxisPiecePosterior {
                coefficient: piece.coefficient,
                log_ratio,
                nodes,
            });
        }
        let value = pieces.iter().map(|piece| piece.coefficient * piece.log_ratio).sum::<f64>();
        Ok(MixedAxisPosterior {
            value,
            node_weights,
            mass,
            pieces,
        })
    }
}

fn log_sum_exp(values: impl Iterator<Item = f64> + Clone) -> f64 {
    let max = values.clone().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return max;
    }
    max + values.map(|v| (v - max).exp()).sum::<f64>().ln()
}

/// Evaluate the block target at every node of `rule`, in chunks whose working
/// memory is reserved on the governor first, and hand each chunk to `visit` as
/// `(first node, its z columns, its (excess, displaced score) results)`.
/// `fixed_bytes` is what the caller holds live across the whole sweep.
pub(super) fn visit_mixed_axis_nodes<T: BlockExcessTarget + ?Sized>(
    target: &T,
    rule: &MixedAxisRule,
    with_scores: bool,
    fixed_bytes: usize,
    mut visit: impl FnMut(
        usize,
        ndarray::ArrayView2<'_, f64>,
        Vec<(f64, Option<Array1<f64>>)>,
    ) -> Result<(), EstimationError>,
) -> Result<(), EstimationError> {
    let refusal = |reason: &str| {
        EstimationError::InvalidInput(format!(
            "#784 mixed-axis rule: working memory refused: {reason}"
        ))
    };
    let m = rule.nodes.nrows();
    let q = rule.node_count();
    let inv_sqrt: Vec<f64> = target
        .block_curvatures()
        .iter()
        .map(|lambda| lambda.sqrt().recip())
        .collect();
    if inv_sqrt.len() != m {
        crate::bail_invalid_estim!(
            "#784 mixed-axis rule: {m}-axis rule on a {}-axis target",
            inv_sqrt.len()
        );
    }
    let node_bytes = target
        .node_working_bytes()
        .and_then(|bytes| bytes.checked_add(m.checked_mul(std::mem::size_of::<f64>())?))
        .ok_or_else(|| refusal("a node's working bytes overflow usize"))?;
    let governor = gam_runtime::resource::MemoryGovernor::global();
    let mut start = 0;
    let mut chunk = q;
    while start < q {
        chunk = chunk.min(q - start);
        let reservation = loop {
            let admitted = governor.remaining_bytes().saturating_sub(fixed_bytes) / node_bytes;
            chunk = chunk.min(admitted).max(1);
            let requested = node_bytes
                .checked_mul(chunk)
                .and_then(|bytes| bytes.checked_add(fixed_bytes))
                .ok_or_else(|| refusal("the chunk's working bytes overflow usize"))?;
            match governor.try_reserve(requested, "#784 mixed-axis rule chunk") {
                Ok(reservation) => break reservation,
                Err(error) if chunk == 1 => return Err(refusal(&error.to_string())),
                // Another reservation landed between reading the remainder and
                // reserving; the next width is re-read and strictly narrower.
                Err(_) => chunk -= 1,
            }
        };
        let z = rule.nodes.slice(ndarray::s![.., start..start + chunk]);
        let draws = Array2::from_shape_fn((m, chunk), |(r, k)| z[(r, k)] * inv_sqrt[r]);
        let results = if with_scores {
            target.excess_with_displaced_neg_score_batch(&draws)
        } else {
            target
                .excess_batch(&draws)
                .into_iter()
                .map(|excess| (excess, None))
                .collect()
        };
        if results.len() != chunk {
            crate::bail_invalid_estim!(
                "#784 mixed-axis rule: the excess batch returned {} nodes for {chunk}",
                results.len()
            );
        }
        visit(start, z, results)?;
        // The chunk's working memory is the governor's again before the next
        // chunk measures the pool, so the next width is read against what is
        // free and not against this chunk's own reservation. The guard is held
        // across the draws, the batch and the visit because those are what it
        // reserves for; releasing it here rather than at the end of the
        // iteration is the whole of its lifetime.
        drop(reservation);
        start += chunk;
    }
    Ok(())
}

/// Evaluate the rule's excesses and its node measure.
pub(super) fn mixed_axis_posterior<T: BlockExcessTarget + ?Sized>(
    target: &T,
    rule: &MixedAxisRule,
) -> Result<(Vec<f64>, MixedAxisPosterior), EstimationError> {
    let mut excesses = vec![f64::NAN; rule.node_count()];
    visit_mixed_axis_nodes(target, rule, false, 0, |start, _, results| {
        for (k, (excess, _)) in results.into_iter().enumerate() {
            excesses[start + k] = excess;
        }
        Ok(())
    })?;
    let posterior = rule.posterior(&excesses)?;
    Ok((excesses, posterior))
}

/// `Ψ` with its node count and cost-side gradient channels.
struct MixedAxisTerm {
    value: f64,
    node_count: usize,
    channels: BlockTargetChannels,
}

/// The mixed-axis part `Ψ` of a split block's marginal
/// ([`MixedAxisRule`]), with the `ω`-weighted moments contracted into the same
/// gradient channels as the pieces'.
///
/// Every `f_T` is bounded along a soft axis, as the certified `Δ_r` are: the
/// anchor gives `f_T ≥ |T| ln(2/3)`, and for a convex row surface at an exact
/// mode `ΔF(z) ≥ −½|z|²`, so `f_T ≤ max_z ½|z|² = 3|T|/2`. The draws are those of
/// the whole block, `t_r = z_r/√λ_r`, so the channels are the tensor rule's
/// under the signed measure `ω`, whose total is `Σ_T κ_T` rather than one.
fn mixed_axis_term(
    target: &Gam784BlockTarget<'_>,
    c_weights: &Array1<f64>,
) -> Result<MixedAxisTerm, EstimationError> {
    let m = target.block_dim();
    let n = target.eta_hat.len();
    let rule = MixedAxisRule::new(m);
    let (excesses, posterior) = mixed_axis_posterior(target, &rule)?;
    let inv_sqrt: Vec<f64> = target
        .block_lambdas
        .iter()
        .map(|lambda| lambda.sqrt().recip())
        .collect();
    let mut e_t = Array1::<f64>::zeros(m);
    let mut e_tt = Array2::<f64>::zeros((m, m));
    let mut e_neg_score = Array1::<f64>::zeros(n);
    let mut e_t_neg_score = Array2::<f64>::zeros((n, m));
    let fixed_bytes = n
        .saturating_mul(m + 1)
        .saturating_mul(std::mem::size_of::<f64>());
    visit_mixed_axis_nodes(target, &rule, true, fixed_bytes, |start, z, results| {
        for (k, (_, score)) in results.into_iter().enumerate() {
            let node = start + k;
            if !excesses[node].is_finite() {
                continue;
            }
            let Some(score) = score else {
                crate::bail_invalid_estim!(
                    "#784 mixed-axis rule: node {node} is feasible and has no displaced score"
                );
            };
            let weight = posterior.node_weights[node];
            let t = Array1::from_shape_fn(m, |r| z[(r, k)] * inv_sqrt[r]);
            e_t.scaled_add(weight, &t);
            for a in 0..m {
                for b in 0..m {
                    e_tt[(a, b)] += weight * t[a] * t[b];
                }
            }
            e_neg_score.scaled_add(weight, &score);
            for r in 0..m {
                e_t_neg_score
                    .column_mut(r)
                    .scaled_add(weight * t[r], &score);
            }
        }
        Ok(())
    })?;
    let moments = gam_problem::laplace_sampler_contract::BlockQuadratureMoments {
        e_t,
        e_tt,
        e_neg_score,
        e_t_neg_score,
    };
    let channels = block_target_channel_moments(target, &moments, c_weights, posterior.mass)?;
    Ok(MixedAxisTerm {
        value: posterior.value,
        node_count: rule.node_count(),
        channels,
    })
}

#[cfg(test)]
mod mixed_axis_rule_tests {
    use super::*;

    fn excesses(rule: &MixedAxisRule, excess: impl Fn(ndarray::ArrayView1<'_, f64>) -> f64) -> Vec<f64> {
        rule.nodes.columns().into_iter().map(excess).collect()
    }

    #[test]
    fn coefficients_count_every_mixed_part_once_and_no_single_axis_part() {
        for m in 1..=8usize {
            let coupling = MIXED_AXIS_COUPLING.min(m);
            // Σ_{T⊇U} κ_T over the subsets of size ≤ K depends only on |U|.
            for u in 1..=coupling {
                let total: i64 = (u..=coupling)
                    .map(|size| {
                        binomial_coefficient(m - u, size - u) * mixed_axis_piece_coefficient(m, size)
                    })
                    .sum();
                assert_eq!(total, i64::from(u >= 2), "m={m}, |U|={u}");
            }
            if m < 2 {
                continue;
            }
            let rule = MixedAxisRule::new(m);
            let expected = (0..=coupling)
                .map(|k| binomial_coefficient(m, k) as usize * 2_usize.pow(k as u32))
                .sum::<usize>();
            assert_eq!(rule.node_count(), expected, "m={m}");
        }
    }

    #[test]
    fn a_separable_excess_has_no_mixed_part() {
        for m in 2..=5usize {
            let rule = MixedAxisRule::new(m);
            let values = excesses(&rule, |z| {
                z.iter()
                    .enumerate()
                    .map(|(r, &v)| 0.3 * (r as f64 + 1.0) * v.powi(3) + 0.1 * v.powi(4))
                    .sum()
            });
            let psi = rule.posterior(&values).unwrap().value;
            assert!(psi.abs() <= 1e-12, "m={m}: Ψ={psi}");
        }
    }

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

    #[test]
    fn psi_is_the_mixed_part_of_the_laplace_expansion_to_its_order() {
        let (n, m) = (7usize, 3usize);
        let a = Array2::from_shape_fn((n, m), |(i, r)| {
            ((i * 7 + r * 3) as f64 * 0.61).sin() * 0.4 + 0.05 * r as f64
        });
        let c = Array1::from_shape_fn(n, |i| (i as f64 * 1.3).cos() * 0.7);
        let d = Array1::from_shape_fn(n, |i| 0.3 + (i as f64 * 0.9).sin() * 0.2);
        let single: f64 = (0..m)
            .map(|r| {
                let column = a.column(r).to_owned().insert_axis(ndarray::Axis(1));
                pairwise_second_order(&column, &c, &d)
            })
            .sum();
        let mixed = pairwise_second_order(&a, &c, &d) - single;
        assert!(mixed.abs() > 1e-3, "fixture has no mixed part: {mixed}");
        let rule = MixedAxisRule::new(m);
        // ΔF = ε ΔF₃ + ε² ΔF₄: the mixed part is ε²·mixed, the odd ε³ terms vanish,
        // and what is left is O(ε⁴).
        let error = |eps: f64| {
            let values = excesses(&rule, |z| {
                let s = a.dot(&z);
                (0..n)
                    .map(|i| eps * c[i] * s[i].powi(3) / 6.0 + eps * eps * d[i] * s[i].powi(4) / 24.0)
                    .sum()
            });
            rule.posterior(&values).unwrap().value - eps * eps * mixed
        };
        let (coarse, fine) = (error(0.2), error(0.1));
        assert!(
            coarse.abs() > 12.0 * fine.abs(),
            "Ψ less the expansion's mixed part does not fall at fourth order: {coarse:.3e} at \
             ε=0.2, {fine:.3e} at ε=0.1"
        );
    }

    /// Saturated softplus rows, whose curvature `W_i = ψ''(η̂_i)` vanishes as
    /// `|η̂|` grows, carrying both block axes against a vanishing penalty: the
    /// whitened rows grow like `W^{-1/2}` and the second-order expansion's mixed
    /// part like `1/W`. `Ψ` stays inside the bounds of its pieces.
    #[test]
    fn psi_stays_bounded_where_the_second_order_expansion_diverges() {
        let softplus = |x: f64| if x > 0.0 { x + (-x).exp().ln_1p() } else { x.exp().ln_1p() };
        let sig = |e: f64| 1.0 / (1.0 + (-e).exp());
        let x = [[1.0, 0.3], [0.2, 1.0], [0.8, -0.6], [-0.4, 0.9]];
        let rule = MixedAxisRule::new(2);
        let mut expansions = Vec::new();
        for saturation in [2.0_f64, 6.0, 10.0, 14.0] {
            let eta = [-saturation, saturation, -saturation, saturation];
            let w: Vec<f64> = eta.iter().map(|&e| sig(e) * (1.0 - sig(e))).collect();
            let c = Array1::from_shape_fn(4, |i| w[i] * (1.0 - 2.0 * sig(eta[i])));
            let d = Array1::from_shape_fn(4, |i| w[i] * (1.0 - 6.0 * w[i]));
            // Whiten by the Cholesky factor of H = XᵀWX + μI, μ → 0.
            let mut h = [[1e-12, 0.0], [0.0, 1e-12]];
            for i in 0..4 {
                for p in 0..2 {
                    for q in 0..2 {
                        h[p][q] += w[i] * x[i][p] * x[i][q];
                    }
                }
            }
            let l00 = h[0][0].sqrt();
            let l10 = h[1][0] / l00;
            let l11 = (h[1][1] - l10 * l10).sqrt();
            let a = Array2::from_shape_fn((4, 2), |(i, r)| {
                let a0 = x[i][0] / l00;
                if r == 0 { a0 } else { (x[i][1] - l10 * a0) / l11 }
            });
            let values = excesses(&rule, |z| {
                let s = a.dot(&z);
                (0..4)
                    .map(|i| {
                        let e = eta[i];
                        softplus(e + s[i]) - softplus(e) - sig(e) * s[i] - 0.5 * w[i] * s[i] * s[i]
                    })
                    .sum()
            });
            let posterior = rule.posterior(&values).unwrap();
            let mut bound = 0.0;
            for (piece, rule_piece) in posterior.pieces.iter().zip(&rule.pieces) {
                // A piece over `|T|` axes holds exactly `3^|T|` nodes.
                let size = (rule_piece.nodes.len() as f64).log(3.0).round();
                assert!(
                    piece.log_ratio >= size * (2.0_f64 / 3.0).ln() - 1e-12
                        && piece.log_ratio <= 1.5 * size + 1e-12,
                    "f_T={} outside its bounds for |T|={size} at saturation {saturation}",
                    piece.log_ratio
                );
                bound += piece.coefficient.abs() * 1.5 * size;
            }
            assert!(posterior.value.abs() <= bound, "Ψ={} past {bound}", posterior.value);
            let single: f64 = (0..2)
                .map(|r| {
                    let column = a.column(r).to_owned().insert_axis(ndarray::Axis(1));
                    pairwise_second_order(&column, &c, &d)
                })
                .sum();
            expansions.push((pairwise_second_order(&a, &c, &d) - single).abs());
        }
        assert!(
            expansions[3] > 100.0 * expansions[0].max(1e-3),
            "the fixture does not reach the divergent regime: {expansions:?}"
        );
    }

    #[test]
    fn node_measure_is_the_derivative_of_psi() {
        let m = 3usize;
        let rule = MixedAxisRule::new(m);
        // ΔF(z; θ) = θ₀ z₀z₁z₂ + θ₁ z₀²z₁² + θ₀θ₁ z₁³ + ¼(z·z)²θ₁², with node
        // derivative ∂ΔF/∂θ.
        let excess = |z: ndarray::ArrayView1<'_, f64>, theta: [f64; 2]| {
            let q = z.dot(&z);
            theta[0] * z[0] * z[1] * z[2]
                + theta[1] * z[0] * z[0] * z[1] * z[1]
                + theta[0] * theta[1] * z[1].powi(3)
                + 0.25 * q * q * theta[1] * theta[1]
        };
        let theta = [0.21, 0.13];
        let posterior = rule
            .posterior(&excesses(&rule, |z| excess(z, theta)))
            .unwrap();
        let h = 1e-6;
        for j in 0..2 {
            let mut analytic = 0.0;
            for (node, z) in rule.nodes.columns().into_iter().enumerate() {
                let (mut plus, mut minus) = (theta, theta);
                plus[j] += h;
                minus[j] -= h;
                let derivative = (excess(z, plus) - excess(z, minus)) / (2.0 * h);
                analytic += posterior.node_weights[node] * derivative;
            }
            let (mut plus, mut minus) = (theta, theta);
            plus[j] += h;
            minus[j] -= h;
            let at = |t: [f64; 2]| rule.posterior(&excesses(&rule, |z| excess(z, t))).unwrap().value;
            let fd = -(at(plus) - at(minus)) / (2.0 * h);
            assert!((fd - analytic).abs() <= 1e-7, "∂(−Ψ)/∂θ_{j}: {analytic} against {fd}");
        }
        let total: f64 = posterior.node_weights.sum();
        assert!((total - posterior.mass).abs() <= 1e-12);
    }

    #[test]
    fn infeasible_nodes_leave_their_pieces() {
        let rule = MixedAxisRule::new(2);
        let mut values = excesses(&rule, |z| 0.1 * z[0] * z[0] * z[1]);
        let dropped = (0..rule.node_count())
            .find(|&node| rule.nodes[(0, node)] > 0.0 && rule.nodes[(1, node)] > 0.0)
            .unwrap();
        values[dropped] = f64::INFINITY;
        let posterior = rule.posterior(&values).unwrap();
        assert!(posterior.value.is_finite());
        assert_eq!(posterior.node_weights[dropped], 0.0);
        values[0] = f64::INFINITY;
        assert!(rule.posterior(&values).is_err(), "an infeasible mode must refuse");
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
    /// Flipping an eigenvector's sign flips its `γ_r` (odd in the vector), and the
    /// oriented block axis is the same vector either way, so a latched composite
    /// partition lands on the same side of the integrand whichever sign the eigensolver
    /// returned (#784).
    #[test]
    fn block_axes_do_not_depend_on_the_eigenvector_sign() {
        let evecs = Array2::from_shape_fn((4, 4), |(i, j)| ((i * 5 + j * 3) as f64 * 0.7).sin());
        let evals = Array1::from(vec![0.5, 1.5, 2.5, 3.5]);
        let directional = Array1::from(vec![0.9, -1.4, 0.2, -0.05]);
        let block_cols = [1usize, 3, 0];
        let (reference, lambdas) = oriented_block_axes(&evecs, &evals, &directional, &block_cols);
        for flipped in 0..4 {
            let mut evecs_flipped = evecs.clone();
            evecs_flipped.column_mut(flipped).mapv_inplace(|v| -v);
            let mut directional_flipped = directional.clone();
            directional_flipped[flipped] = -directional_flipped[flipped];
            let (vecs, flipped_lambdas) =
                oriented_block_axes(&evecs_flipped, &evals, &directional_flipped, &block_cols);
            assert_eq!(vecs, reference, "flipping column {flipped} moved the block axes");
            assert_eq!(flipped_lambdas, lambdas);
        }
        // Every oriented axis has positive skewness: γ of the output is |γ|.
        for (j, &r) in block_cols.iter().enumerate() {
            let sign = reference.column(j).dot(&evecs.column(r)).signum();
            assert_eq!(sign * directional[r], directional[r].abs());
        }
    }

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
