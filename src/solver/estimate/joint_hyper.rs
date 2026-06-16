use super::*;

pub(crate) fn validate_joint_hyper_direction_shapes(
    x: &DesignMatrix,
    canonical_len: usize,
    theta: &Array1<f64>,
    rho_dim: usize,
    hyper_dirs: &[DirectionalHyperParam],
) -> Result<(), EstimationError> {
    if rho_dim > theta.len() {
        crate::bail_invalid_estim!(
            "rho_dim {} exceeds theta dimension {}",
            rho_dim,
            theta.len()
        );
    }

    let p = x.ncols();
    let psi_dim = theta.len() - rho_dim;
    if hyper_dirs.len() != psi_dim {
        crate::bail_invalid_estim!(
            "joint hyper-gradient derivative count mismatch: psi_dim={}, hyper_dirs={}",
            psi_dim,
            hyper_dirs.len()
        );
    }

    for (idx, hyper_dir) in hyper_dirs.iter().enumerate() {
        for component in hyper_dir.penalty_first_components() {
            if component.penalty_index >= canonical_len {
                crate::bail_invalid_estim!(
                    "penalty_index for dir {idx} out of bounds: {} >= {}",
                    component.penalty_index,
                    canonical_len
                );
            }
        }
        if hyper_dir.x_tau_original.nrows() != x.nrows() || hyper_dir.x_tau_original.ncols() != p {
            crate::bail_invalid_estim!(
                "X_tau[{idx}] must be {}x{}, got {}x{}",
                x.nrows(),
                p,
                hyper_dir.x_tau_original.nrows(),
                hyper_dir.x_tau_original.ncols()
            );
        }
        RemlState::validate_penalty_component_shapes(
            hyper_dir.penalty_first_components(),
            p,
            &format!("S_tau[{idx}]"),
        )?;
        if let Some(x2) = hyper_dir.x_tau_tau_original.as_ref() {
            if x2.len() != psi_dim {
                crate::bail_invalid_estim!(
                    "X_tau_tau[{idx}] length mismatch: expected {}, got {}",
                    psi_dim,
                    x2.len()
                );
            }
            for (j, x_ij) in x2.iter().enumerate() {
                let Some(x_ij) = x_ij.as_ref() else {
                    continue;
                };
                if x_ij.nrows() != x.nrows() || x_ij.ncols() != p {
                    crate::bail_invalid_estim!(
                        "X_tau_tau[{idx}][{j}] must be {}x{}, got {}x{}",
                        x.nrows(),
                        p,
                        x_ij.nrows(),
                        x_ij.ncols()
                    );
                }
            }
        }
        if let Some(s2) = hyper_dir.penaltysecond_componentrows() {
            if s2.len() != psi_dim {
                crate::bail_invalid_estim!(
                    "S_tau_tau[{idx}] length mismatch: expected {}, got {}",
                    psi_dim,
                    s2.len()
                );
            }
            for (j, components) in s2.iter().enumerate() {
                let Some(components) = components.as_ref() else {
                    continue;
                };
                RemlState::validate_penalty_component_shapes(
                    components,
                    p,
                    &format!("S_tau_tau[{idx}][{j}]"),
                )?;
            }
        }
    }

    Ok(())
}

pub(crate) struct ExternalJointHyperEvaluator<'a> {
    pub(crate) conditioning: ParametricColumnConditioning,
    pub(crate) penalty_shrinkage_floor: Option<f64>,
    pub(crate) kronecker_penalty_system: Option<crate::smooth::KroneckerPenaltySystem>,
    pub(crate) kronecker_factored: Option<crate::basis::KroneckerFactoredBasis>,
    pub(crate) reml_state: RemlState<'a>,
    /// Cached design revision counter from the upstream
    /// `SingleBlockExactJointDesignCache` (or n-block analogue). When the
    /// caller threads a revision through `evaluate_with_order` /
    /// `evaluate_efs` / `evaluate_cost_only`, the evaluator can detect ψ-
    /// invariant repeat calls (cost-only line-search probes, fall-through
    /// memoization) and short-circuit `reset_surface`'s O(Σ pₖ³) canonical
    /// rebuild plus the bundle/PIRLS cache wipes. `None` means "no
    /// revision yet recorded" — every subsequent call is treated as a
    /// fresh-canonical case and the slow path runs.
    pub(crate) last_canonical_revision: Option<u64>,
    /// Certified Chebyshev-in-ψ Gram tensor for the SINGLE design-moving
    /// hyperparameter (#1033b, isotropic spatial κ): when present and the
    /// trial ψ lies inside the certified window, `prepare_eval_state`
    /// installs the n-free assembled `GaussianFixedCache` after
    /// `reset_surface`, replacing the per-trial O(n·p²) Gram re-stream. Built
    /// in the conditioned frame by `build_and_set_psi_gram_tensor` (the same
    /// fixed column transform the streamed Gram uses), so the installed
    /// statistics are frame-exact against the streamed ones.
    pub(crate) psi_gram_tensor:
        Option<std::sync::Arc<crate::solver::psi_gram_tensor::PsiGramTensor>>,
    /// Frozen-weight GLM first-Fisher-step data-fit Gram `XᵀWX` staged for the
    /// CURRENT ψ-trial (#1111 / #1033 mechanism (c)), in the conditioned
    /// (`x_fit`) frame. Set per-trial by [`SpatialJointContext::eval_full`] when
    /// the frozen-W tensor covers ψ and the working weight has not drifted, then
    /// installed onto the inner REML surface inside `prepare_eval_state` (after
    /// `reset_surface`, on both the slow and design-revision fast paths) and
    /// cleared. `None` (the default) clears the surface slot so a stale
    /// previous-ψ Gram is never consumed.
    pub(crate) pending_glm_first_step_gram: Option<std::sync::Arc<Array2<f64>>>,
    /// Conditioned-frame exact ψ-derivative pair `(∂XᵀWX/∂ψ, ∂XᵀW(y−offset)/∂ψ)`
    /// staged for the CURRENT ψ-trial in the GLM frozen-W lane (#1033 / #1111),
    /// in the conditioned (`x_fit`) frame. Set per-trial by
    /// [`SpatialJointContext::eval_full`] from
    /// [`crate::solver::glm_sufficient_lane::FrozenWeightGramTensor::gradient_pair_if_sound`]
    /// when the frozen-W tensor covers ψ for the gradient and the working weight
    /// has not drifted, then installed onto the inner REML surface inside
    /// `prepare_eval_state` and cleared. Serves the GLM ψ-gradient `a_j` / `g_j`
    /// n-free; `B_j` stays the exact slab. `None` (the default) clears the
    /// surface slot so a stale previous-ψ derivative is never consumed.
    pub(crate) pending_glm_psi_gram_deriv: Option<std::sync::Arc<(Array2<f64>, Array1<f64>)>>,
}

impl<'a> ExternalJointHyperEvaluator<'a> {
    pub(crate) fn new(
        y: ArrayView1<'a, f64>,
        w: ArrayView1<'a, f64>,
        x: &DesignMatrix,
        offset: ArrayView1<'_, f64>,
        s_list: &[BlockwisePenalty],
        opts: &ExternalOptimOptions,
        context: &str,
    ) -> Result<Self, EstimationError> {
        if let Some(message) = row_mismatch_message(y.len(), w.len(), x.nrows(), offset.len()) {
            crate::bail_invalid_estim!("{}", message);
        }

        let p = x.ncols();
        let specs: Vec<PenaltySpec> = s_list.iter().map(PenaltySpec::from_blockwise_ref).collect();
        validate_penalty_specs(&specs, p, context)?;
        let (canonical, active_nullspace_dims) = crate::construction::canonicalize_penalty_specs(
            &specs,
            &opts.nullspace_dims,
            p,
            context,
        )?;
        let conditioning = ParametricColumnConditioning::infer_from_penalty_specs(x, &specs);
        let x_fit = conditioning.apply_to_design(x);
        let fit_linear_constraints =
            conditioning.transform_linear_constraints_to_internal(opts.linear_constraints.clone());
        let (config, _) = resolved_external_config(opts)?;
        let config = Arc::new(config);

        let mut reml_state = RemlState::newwith_offset_shared(
            y,
            x_fit,
            w,
            offset,
            Arc::new(canonical),
            p,
            Arc::clone(&config),
            Some(active_nullspace_dims.clone()),
            None,
            fit_linear_constraints.clone(),
        )?;
        reml_state.set_penalty_shrinkage_floor(opts.penalty_shrinkage_floor);
        reml_state.set_rho_prior(opts.rho_prior.clone());
        reml_state.set_link_states(
            config.link_kind.mixture_state().cloned(),
            config.link_kind.sas_state().copied(),
        );
        if let Some(kron) = opts.kronecker_penalty_system.clone() {
            reml_state.set_kronecker_penalty_system(kron);
        }
        if let Some(kf) = opts.kronecker_factored.clone() {
            reml_state.set_kronecker_factored(kf);
        }
        if opts.persist_warm_start_disk {
            // Caller opted into cross-process resume (#1082): engage the
            // on-disk warm-start layer. Default-false keeps replicate/CI loops
            // disk-silent.
            reml_state.enable_persistent_warm_start_disk();
        }

        Ok(Self {
            conditioning,
            penalty_shrinkage_floor: opts.penalty_shrinkage_floor,
            kronecker_penalty_system: opts.kronecker_penalty_system.clone(),
            kronecker_factored: opts.kronecker_factored.clone(),
            reml_state,
            last_canonical_revision: None,
            psi_gram_tensor: None,
            pending_glm_first_step_gram: None,
            pending_glm_psi_gram_deriv: None,
        })
    }

    /// Stage (or clear) the frozen-weight GLM first-Fisher-step Gram for the
    /// next trial eval (#1111 / #1033 mechanism (c)). The staged Gram is
    /// installed onto the inner REML surface inside `prepare_eval_state` and
    /// then cleared; passing `None` clears any previously staged Gram so a stale
    /// previous-ψ Gram is never consumed.
    pub(crate) fn stage_glm_first_step_gram(&mut self, gram: Option<Array2<f64>>) {
        self.pending_glm_first_step_gram = gram.map(std::sync::Arc::new);
    }

    /// Stage (or clear) the GLM frozen-W conditioned-frame exact ψ-derivative
    /// pair `(∂XᵀWX/∂ψ, ∂XᵀW(y−offset)/∂ψ)` for the next trial eval
    /// (#1033 / #1111). Produced by `gradient_pair_if_sound` when the frozen-W
    /// tensor covers ψ for the gradient and the working weight has not drifted;
    /// installed onto the inner REML surface inside `prepare_eval_state` (after
    /// `reset_surface`, on both the slow and design-revision fast paths) and
    /// then cleared. Serves the GLM ψ-gradient `a_j` / `g_j` n-free; the
    /// Hessian curvature `B_j` always stays the exact n-dependent slab. Passing
    /// `None` clears any previously staged pair so a stale previous-ψ
    /// derivative is never consumed.
    pub(crate) fn stage_glm_psi_gram_deriv(&mut self, deriv: Option<(Array2<f64>, Array1<f64>)>) {
        self.pending_glm_psi_gram_deriv = deriv.map(std::sync::Arc::new);
    }

    pub(crate) fn set_analytic_penalty_registry(
        &mut self,
        registry: Option<&crate::terms::AnalyticPenaltyRegistry>,
    ) {
        let fingerprint = registry
            .map(crate::solver::estimate::reml::outer_eval::analytic_penalty_registry_fingerprint)
            .unwrap_or(0);
        crate::solver::estimate::reml::RemlState::set_analytic_penalty_registry_fingerprint(
            &mut self.reml_state,
            fingerprint,
        );
    }

    pub(crate) fn set_persistent_latent_values_fingerprint(
        &mut self,
        id_mode: &crate::terms::latent::LatentIdMode,
    ) {
        let fingerprint =
            crate::solver::estimate::reml::outer_eval::latent_id_mode_cache_fingerprint(id_mode);
        crate::solver::estimate::reml::RemlState::set_persistent_latent_values_fingerprint(
            &mut self.reml_state,
            fingerprint,
        );
    }

    pub(crate) fn load_persistent_latent_values(
        &self,
        n_obs: usize,
        latent_dim: usize,
    ) -> Option<Array2<f64>> {
        crate::solver::estimate::reml::RemlState::load_persistent_latent_values(
            &self.reml_state,
            n_obs,
            latent_dim,
        )
    }

    pub(crate) fn store_persistent_latent_values(&self, values: &Array2<f64>) {
        crate::solver::estimate::reml::RemlState::store_persistent_latent_values(
            &self.reml_state,
            values,
        );
    }

    /// Build and attach a certified ψ-Gram tensor (#1033b) for the single
    /// design-moving hyperparameter ψ over `[psi_lo, psi_hi]`.
    ///
    /// `eval_raw_design(psi)` returns the EXACT realized design at `psi` in the
    /// raw (user) column frame — the same realizer the per-trial path uses.
    /// This method threads it through THIS evaluator's parametric column
    /// conditioning before the tensor sees it, so the tensor's assembled
    /// `XᵀWX(ψ)` lives in the SAME conditioned frame as the streamed
    /// `gaussian_fixed_cache_if_eligible` (which forms its Gram from
    /// `x_fit = conditioning.apply_to_design(x)`). The conditioning is a fixed,
    /// ψ-invariant column transform (means/scales frozen from the baseline
    /// design at construction), so applying it inside the build keeps the
    /// expansion analytic and the per-trial installed cache frame-exact —
    /// without restricting to identity conditioning. Returns whether a
    /// certified tensor was attached; `false` keeps the exact per-trial path.
    pub(crate) fn build_and_set_psi_gram_tensor(
        &mut self,
        mut eval_raw_design: impl FnMut(f64) -> Result<DesignMatrix, String>,
        weights: ArrayView1<'_, f64>,
        z: ArrayView1<'_, f64>,
        psi_lo: f64,
        psi_hi: f64,
    ) -> bool {
        // Clone the (cheap) conditioning so the build closure borrows it
        // without aliasing `self` while we set the field afterward.
        let conditioning = self.conditioning.clone();
        let tensor = crate::solver::psi_gram_tensor::PsiGramTensor::build(
            |psi| {
                let raw = eval_raw_design(psi)?;
                Ok(conditioning.apply_to_design(&raw).to_dense())
            },
            weights,
            z,
            psi_lo,
            psi_hi,
        );
        match tensor {
            Some(tensor) => {
                self.psi_gram_tensor = Some(std::sync::Arc::new(tensor));
                true
            }
            None => false,
        }
    }

    /// Build a certified frozen-weight GLM ψ-Gram tensor (#1111 / #1033
    /// mechanism (c)) for the single design-moving hyperparameter ψ.
    ///
    /// Mirrors [`Self::build_and_set_psi_gram_tensor`] but for the GLM
    /// design-moving lane: the working weight `w` and working response `z` are
    /// FROZEN at the warm working point, and the tensor wraps the weighted
    /// design `A(ψ) = diag(√w)·X_fit(ψ)`. Crucially `eval_raw_design` is threaded
    /// through THIS evaluator's parametric column conditioning before the tensor
    /// sees it, so the assembled frozen-`W` Gram `XᵀWX(ψ)` lives in the SAME
    /// conditioned `x_fit` frame the inner PIRLS solve forms its Gram in — the
    /// same frame-correctness contract the Gaussian lane relies on. Without this
    /// the tensor would be assembled in the raw user-column frame and silently
    /// mismatch any inner consumer.
    ///
    /// Returns the certified tensor (caller owns it, e.g. to re-use the
    /// per-trial weight-drift guard), or `None` when no Chebyshev rung certifies
    /// — the caller then keeps the exact per-trial PIRLS rebuild.
    pub(crate) fn build_frozen_glm_gram_tensor(
        &self,
        mut eval_raw_design: impl FnMut(f64) -> Result<DesignMatrix, String>,
        frozen_w: ArrayView1<'_, f64>,
        working_z: ArrayView1<'_, f64>,
        psi_lo: f64,
        psi_hi: f64,
    ) -> Option<crate::solver::glm_sufficient_lane::FrozenWeightGramTensor> {
        let conditioning = self.conditioning.clone();
        crate::solver::glm_sufficient_lane::FrozenWeightGramTensor::build(
            |psi| {
                let raw = eval_raw_design(psi)?;
                Ok(conditioning.apply_to_design(&raw).to_dense())
            },
            frozen_w,
            working_z,
            psi_lo,
            psi_hi,
        )
    }

    /// True when a certified ψ-Gram tensor is installed AND `psi` lies inside
    /// its certified GRADIENT sub-window — i.e. the n-free k-space ψ-derivatives
    /// `(∂G/∂ψ, ∂b/∂ψ)` will serve the Gaussian gradient HyperCoord, so the
    /// caller's per-trial n×k ∂X/∂ψ slab is redundant (#1033). The value lane
    /// (`contains`) spans the full window; the gradient lane is the narrower
    /// interior where the Chebyshev derivative reconstruction is bit-tight.
    pub(crate) fn psi_gram_tensor_covers_gradient(&self, psi: f64) -> bool {
        self.psi_gram_tensor
            .as_ref()
            .is_some_and(|t| t.contains_for_gradient(psi))
    }

    /// True when a certified ψ-Gram tensor is installed AND `psi` lies inside its
    /// full certified VALUE window — i.e. the n-free assembled Gaussian
    /// sufficient statistics `XᵀWX(ψ)/XᵀWz(ψ)` reproduce the streamed Gram to the
    /// certification tolerance. The caller uses this to skip the per-trial O(n·p)
    /// design realization + conditioning entirely (#1033): when the value lane is
    /// covered, `prepare_eval_state` installs the n-free `GaussianFixedCache`, so
    /// the stale realized design is never read for its rows on the inner Gaussian
    /// PLS fast path. Strictly narrower-or-equal callers also gate on
    /// `psi_gram_tensor_covers_gradient` for the gradient channel.
    pub(crate) fn psi_gram_tensor_covers(&self, psi: f64) -> bool {
        self.psi_gram_tensor
            .as_ref()
            .is_some_and(|t| t.contains(psi))
    }

    /// True when the design-revision fast path of [`Self::prepare_eval_state`]
    /// would fire for `design_revision` — i.e. a prior eval has already pinned
    /// `last_canonical_revision` to this exact realizer revision, so the next
    /// `evaluate_with_order` at this revision will SKIP `reset_surface` (and the
    /// n×k `apply_to_design` reconditioning) and instead re-install the ψ-keyed
    /// `GaussianFixedCache` onto the existing surface (#1033).
    ///
    /// The spatial κ caller (`SpatialJointContext::eval_full`) consults this
    /// BEFORE deciding to skip its own `ensure_theta` design re-realization: it
    /// may only suppress the per-trial O(n·k) design rebuild when the evaluator
    /// will take that fast path, because the fast path is exactly the lane that
    /// keeps the (now intentionally stale) reference surface while serving the
    /// trial's value + gradient n-free from the certified tensor. When this is
    /// `false` the caller MUST realize the design so the slow path's
    /// `reset_surface` rebuilds a faithful surface.
    pub(crate) fn design_revision_fast_path_armed(&self, design_revision: u64) -> bool {
        self.last_canonical_revision == Some(design_revision)
    }

    /// Return the most-recently converged inner β from the last PIRLS solve, if
    /// it is finite and the right dimension. Used by `SpatialJointContext` to
    /// warm-start successive outer evaluations instead of cold-starting PIRLS
    /// from zero every iteration — especially important for GLM families (Poisson,
    /// NB, Binomial) that cannot use the Gaussian Gram tensor n-free shortcut.
    pub(crate) fn current_beta(&self) -> Option<Array1<f64>> {
        self.reml_state.current_original_basis_beta()
    }

    /// Install the n-free per-ψ Gaussian sufficient statistics from the certified
    /// ψ-Gram tensor (#1033b), when one is present and `theta`'s single ψ lies
    /// inside the certified window. Idempotent in ψ — must be called on EVERY
    /// trial (fast-path or slow-path) because the installed `GaussianFixedCache`
    /// (and the conditioned-frame ψ-derivatives) are keyed to the current ψ, not
    /// just to the design revision: on the design-revision fast path the design
    /// did not change but ψ still moved, so the previous ψ's Gram would be stale.
    ///
    /// Off-window, multi-ψ, ineligible family, or shape mismatch all return
    /// without installing — the streamed exact path runs unchanged.
    /// Returns `true` when the n-free Gaussian ψ-GRADIENT derivative pair was
    /// installed for this trial — i.e. the certified tensor serves both the
    /// value AND the gradient n-free, so the conditioned n×k `∂X/∂ψ` slab in the
    /// hyper_dirs is provably DEAD (the gradient HyperCoord's `j==0` branch reads
    /// the k-space derivatives and never the slab). The caller uses this to skip
    /// the per-trial slab conditioning on the design-revision fast path (#1033).
    fn install_psi_gram_statistics(&mut self, theta: &Array1<f64>, rho_dim: usize) -> bool {
        let Some(tensor) = self.psi_gram_tensor.as_ref() else {
            // No tensor installed for this fit → the surface never carries a
            // ψ-keyed Gaussian Gram, so there is nothing stale to clear.
            return false;
        };
        // #1033: every early return below is a trial for which we CANNOT serve
        // the n-free per-ψ Gram (off-window, wrong shape, multi-ψ). On the
        // design-revision fast path `reset_surface` is skipped, so a Gram keyed
        // to the PREVIOUS in-window ψ would survive and be read stale by the
        // inner Gaussian PLS. Clear it on every miss so the inner solver
        // restreams the exact Gram for this trial's design.
        if theta.len() != rho_dim + 1 {
            self.reml_state.clear_gaussian_fixed_cache();
            return false;
        }
        let psi = theta[rho_dim];
        if !tensor.contains(psi) {
            self.reml_state.clear_gaussian_fixed_cache();
            return false;
        }
        // Clone the Arc handle so the immutable borrow of `self.psi_gram_tensor`
        // is released before the `&mut self.reml_state` installs below.
        let tensor = std::sync::Arc::clone(tensor);
        if !self
            .reml_state
            .install_gaussian_fixed_cache(Arc::new(tensor.gaussian_fixed_cache_at(psi)))
        {
            self.reml_state.clear_gaussian_fixed_cache();
            return false;
        }
        log::debug!(
            "[psi-gram-tensor] installed n-free Gaussian sufficient statistics at psi={psi:.6}"
        );
        // Install the conditioned-frame exact ψ-derivatives so the Gaussian
        // ψ-gradient HyperCoord is assembled from these k×k objects instead of
        // the n×k ∂X/∂ψ slab — retiring the second per-trial n-pass. Only on the
        // certified gradient SUB-window: near the ψ-window edges the Chebyshev
        // derivative reconstruction (T_d′ ∼ d²) is not bit-tight, so those
        // trials keep the exact slab gradient.
        if tensor.contains_for_gradient(psi)
            && self.reml_state.install_gaussian_psi_gram_deriv(Arc::new((
                tensor.dgram_dpsi(psi),
                tensor.drhs_dpsi(psi),
            )))
        {
            log::debug!(
                "[psi-gram-tensor] installed n-free ψ-gradient derivatives at psi={psi:.6}"
            );
            true
        } else {
            // In the VALUE window but outside the certified GRADIENT sub-window
            // (or the deriv shape refused). The value cache above is sound and
            // stays; clear any derivative pair left from a prior in-sub-window ψ
            // so the gradient lane uses the exact slab for this trial rather than
            // a stale derivative carried over on the design-revision fast path.
            self.reml_state.clear_gaussian_psi_gram_deriv();
            false
        }
    }

    fn prepare_eval_state(
        &mut self,
        x: &DesignMatrix,
        s_list: &[BlockwisePenalty],
        nullspace_dims: &[usize],
        linear_constraints: Option<crate::pirls::LinearInequalityConstraints>,
        theta: &Array1<f64>,
        rho_dim: usize,
        mut hyper_dirs: Vec<DirectionalHyperParam>,
        warm_start_beta: Option<ArrayView1<'_, f64>>,
        context: &str,
        design_revision: Option<u64>,
    ) -> Result<Vec<DirectionalHyperParam>, EstimationError> {
        let p = x.ncols();
        // Design-revision fast path: when the caller asserts that the
        // realizer-side design (X + s_list) has not changed since the last
        // `reset_surface`, we skip the canonical-penalty rebuild and the
        // `reset_surface` work entirely. Hyper-direction conditioning still
        // runs (hyper_dirs are freshly constructed per call) and the
        // warm-start beta / penalty-shrinkage floor still need refreshing.
        let fast_path = match (design_revision, self.last_canonical_revision) {
            (Some(rev), Some(last)) => rev == last,
            _ => false,
        };

        if fast_path {
            validate_joint_hyper_direction_shapes(x, s_list.len(), theta, rho_dim, &hyper_dirs)?;

            self.reml_state
                .set_penalty_shrinkage_floor(self.penalty_shrinkage_floor);
            self.reml_state.setwarm_start_original_beta(warm_start_beta);
            // #1033b: the design did not change (fast path) but ψ moved, so the
            // GaussianFixedCache and conditioned ψ-derivatives are keyed to the
            // PREVIOUS ψ and must be re-installed for this trial's ψ from the
            // certified tensor — otherwise the inner PLS reads a stale Gram. The
            // slow path below clears + reinstalls these; the fast path skips
            // `reset_surface` (which clears them), so we re-install here directly.
            // Install BEFORE conditioning so we learn whether the n-free
            // ψ-gradient was served from the tensor: if so, the conditioned n×k
            // `∂X/∂ψ` slab below is provably DEAD (the `j==0` gradient branch
            // reads the k-space derivatives, never the slab), so we skip the
            // per-trial slab conditioning — the LAST O(n·k²) pass in the κ loop.
            let gaussian_gradient_is_n_free = self.install_psi_gram_statistics(theta, rho_dim);
            let glm_gradient_is_n_free = self.install_pending_glm_trial_statistics();
            if !(gaussian_gradient_is_n_free || glm_gradient_is_n_free) {
                // The slab gradient lane is live for this trial (off the certified
                // gradient sub-window, non-Gaussian, multi-ψ, …) — condition the
                // n×k `∂X/∂ψ` slab into the inner solver's frame as before.
                for dir in &mut hyper_dirs {
                    let mut x_tau = dir.x_tau_dense();
                    self.conditioning
                        .transform_matrix_columnswith_a_inplace(&mut x_tau);
                    dir.x_tau_original =
                        crate::solver::estimate::reml::HyperDesignDerivative::from(x_tau);
                    if let Some(rows) = dir.x_tau_tau_original.as_mut() {
                        for mat in rows.iter_mut().flatten() {
                            let mut dense = mat.materialize();
                            self.conditioning
                                .transform_matrix_columnswith_a_inplace(&mut dense);
                            *mat =
                                crate::solver::estimate::reml::HyperDesignDerivative::from(dense);
                        }
                    }
                }
            }
            return Ok(hyper_dirs);
        }

        let specs: Vec<PenaltySpec> = s_list.iter().map(PenaltySpec::from_blockwise_ref).collect();
        validate_penalty_specs(&specs, p, context)?;
        let (canonical, active_nullspace_dims) =
            crate::construction::canonicalize_penalty_specs(&specs, nullspace_dims, p, context)?;
        validate_joint_hyper_direction_shapes(x, canonical.len(), theta, rho_dim, &hyper_dirs)?;

        let x_fit = self.conditioning.apply_to_design(x);
        let fit_linear_constraints = self
            .conditioning
            .transform_linear_constraints_to_internal(linear_constraints);

        for dir in &mut hyper_dirs {
            let mut x_tau = dir.x_tau_dense();
            self.conditioning
                .transform_matrix_columnswith_a_inplace(&mut x_tau);
            dir.x_tau_original = crate::solver::estimate::reml::HyperDesignDerivative::from(x_tau);
            if let Some(rows) = dir.x_tau_tau_original.as_mut() {
                for mat in rows.iter_mut().flatten() {
                    let mut dense = mat.materialize();
                    self.conditioning
                        .transform_matrix_columnswith_a_inplace(&mut dense);
                    *mat = crate::solver::estimate::reml::HyperDesignDerivative::from(dense);
                }
            }
        }

        crate::solver::estimate::reml::RemlState::reset_surface(
            &mut self.reml_state,
            x_fit,
            Arc::new(canonical),
            p,
            active_nullspace_dims,
            None,
            fit_linear_constraints,
            self.kronecker_penalty_system.clone(),
            self.kronecker_factored.clone(),
        )?;
        self.reml_state
            .set_penalty_shrinkage_floor(self.penalty_shrinkage_floor);
        self.reml_state.setwarm_start_original_beta(warm_start_beta);
        self.last_canonical_revision = design_revision;
        // #1033b: single design-moving ψ with a certified tensor — install the
        // n-free assembled Gaussian sufficient statistics so the inner PLS and
        // the sparse scatter skip the per-trial O(n·p²) Gram re-stream
        // (`reset_surface` above just cleared the slot for the new design). Same
        // installer the fast path uses, so both branches key the Gram to ψ.
        self.install_psi_gram_statistics(theta, rho_dim);
        self.install_pending_glm_trial_statistics();
        Ok(hyper_dirs)
    }

    /// Install the staged frozen-W GLM first-step Gram onto the inner REML
    /// surface for the current trial, or clear the surface slot when nothing is
    /// staged (#1111 / #1033 mechanism (c)). Called after `reset_surface` (slow
    /// path) and on the design-revision fast path, mirroring
    /// `install_psi_gram_statistics`: the Gram is ψ-keyed, so it must be
    /// (re)installed per trial and never carried over from the previous ψ.
    fn install_pending_glm_trial_statistics(&mut self) -> bool {
        let mut gradient_is_n_free = false;
        match self.pending_glm_first_step_gram.take() {
            Some(gram) => {
                if !self.reml_state.install_glm_first_step_gram(gram) {
                    // Shape mismatch against the current surface — fall back to
                    // the exact streamed first-iteration Gram.
                    self.reml_state.clear_glm_first_step_gram();
                }
            }
            None => self.reml_state.clear_glm_first_step_gram(),
        }
        // #1033 / #1111: the GLM frozen-W conditioned-frame ψ-gradient
        // derivative is keyed to the same per-trial ψ as the first-step Gram, so
        // install/clear it on the same two sites. Serves the GLM ψ-gradient
        // `a_j` / `g_j` n-free; `B_j` stays the exact slab.
        match self.pending_glm_psi_gram_deriv.take() {
            Some(deriv) => {
                if self.reml_state.install_glm_psi_gram_deriv(deriv) {
                    gradient_is_n_free = true;
                } else {
                    // Shape mismatch against the current surface — fall back to
                    // the exact streamed ∂X/∂ψ slab gradient.
                    self.reml_state.clear_glm_psi_gram_deriv();
                }
            }
            None => self.reml_state.clear_glm_psi_gram_deriv(),
        }
        gradient_is_n_free
    }

    pub(crate) fn evaluate_with_order(
        &mut self,
        x: &DesignMatrix,
        s_list: &[BlockwisePenalty],
        nullspace_dims: &[usize],
        linear_constraints: Option<crate::pirls::LinearInequalityConstraints>,
        theta: &Array1<f64>,
        rho_dim: usize,
        hyper_dirs: Vec<DirectionalHyperParam>,
        warm_start_beta: Option<ArrayView1<'_, f64>>,
        context: &str,
        order: crate::solver::rho_optimizer::OuterEvalOrder,
        design_revision: Option<u64>,
    ) -> Result<
        (
            f64,
            Array1<f64>,
            crate::solver::rho_optimizer::HessianResult,
        ),
        EstimationError,
    > {
        let order = if matches!(
            order,
            crate::solver::rho_optimizer::OuterEvalOrder::ValueGradientHessian
        ) {
            // Firth pair Hessian terms are now available via Primitive A +
            // Primitive B in the reduced Firth dense operator; the tau-tau
            // policy no longer needs the Firth+Logit gap downgrade.
            let firth_pair_terms_unavailable = false;
            let tau_tau_policy =
                crate::solver::estimate::reml::exact_tau_tau_hessian_policy_with_firth(
                    x.nrows(),
                    x.ncols(),
                    &hyper_dirs,
                    firth_pair_terms_unavailable,
                );
            if tau_tau_policy.prefer_gradient_only() {
                log::warn!(
                    "[OUTER] disabling exact tau Hessian before conditioning; using gradient-only outer eval \
                     (n={}, p={}, psi_dim={}, implicit_tau={}, implicit_multidim_duchon={}, firth_pair_gap={}, dense_tau_cache={:.1} MiB, gradient_plan={:.1} MiB, exact_hessian_plan={:.1} MiB, budget={:.1} MiB)",
                    x.nrows(),
                    x.ncols(),
                    hyper_dirs.len(),
                    tau_tau_policy.any_has_implicit,
                    tau_tau_policy.implicit_multidim_duchon,
                    tau_tau_policy.firth_pair_terms_unavailable,
                    tau_tau_policy.estimated_dense_tau_cache_bytes as f64 / (1024.0 * 1024.0),
                    tau_tau_policy.gradient_plan.total_bytes() as f64 / (1024.0 * 1024.0),
                    tau_tau_policy.hessian_plan.total_bytes() as f64 / (1024.0 * 1024.0),
                    tau_tau_policy.budget_bytes as f64 / (1024.0 * 1024.0),
                );
                crate::solver::rho_optimizer::OuterEvalOrder::ValueAndGradient
            } else {
                order
            }
        } else {
            order
        };
        let hyper_dirs = self.prepare_eval_state(
            x,
            s_list,
            nullspace_dims,
            linear_constraints,
            theta,
            rho_dim,
            hyper_dirs,
            warm_start_beta,
            context,
            design_revision,
        )?;
        crate::solver::estimate::reml::RemlState::compute_joint_hyper_eval_with_order(
            &self.reml_state,
            theta,
            rho_dim,
            &hyper_dirs,
            order,
        )
    }

    pub(crate) fn evaluate_efs(
        &mut self,
        x: &DesignMatrix,
        s_list: &[BlockwisePenalty],
        nullspace_dims: &[usize],
        linear_constraints: Option<crate::pirls::LinearInequalityConstraints>,
        theta: &Array1<f64>,
        rho_dim: usize,
        hyper_dirs: Vec<DirectionalHyperParam>,
        warm_start_beta: Option<ArrayView1<'_, f64>>,
        context: &str,
        design_revision: Option<u64>,
    ) -> Result<crate::solver::rho_optimizer::EfsEval, EstimationError> {
        let hyper_dirs = self.prepare_eval_state(
            x,
            s_list,
            nullspace_dims,
            linear_constraints,
            theta,
            rho_dim,
            hyper_dirs,
            warm_start_beta,
            context,
            design_revision,
        )?;
        let rho = theta.slice(s![..rho_dim]).to_owned();
        self.reml_state
            .compute_efs_steps_with_psi_ext(&rho, &hyper_dirs)
    }

    /// Reset the inner surface for a value-only evaluation. This is the
    /// hyper-dir-free counterpart of [`prepare_eval_state`]: it accepts the
    /// fact that the spatial design has been re-realized at the current κ
    /// (the caller guarantees this via the realizer cache), so no directional
    /// hyper-derivatives are required to produce a correct cost. Skipping
    /// the hyper_dir validation and the per-direction conditioning loop is
    /// what makes line-search probes cheap in the iso/aniso joint paths.
    pub(crate) fn prepare_eval_state_cost_only(
        &mut self,
        x: &DesignMatrix,
        s_list: &[BlockwisePenalty],
        nullspace_dims: &[usize],
        linear_constraints: Option<crate::pirls::LinearInequalityConstraints>,
        theta: &Array1<f64>,
        rho_dim: usize,
        warm_start_beta: Option<ArrayView1<'_, f64>>,
        context: &str,
        design_revision: Option<u64>,
    ) -> Result<(), EstimationError> {
        // Design-revision fast path: when ψ hasn't moved since the last
        // full `reset_surface`, the cached surface's X, canonical penalties,
        // gaussian-fixed cache, and PIRLS cache are all still keyed to the
        // exact same (X, y, w, offset) — skip the eigendecomp + cache wipe.
        let fast_path = match (design_revision, self.last_canonical_revision) {
            (Some(rev), Some(last)) => rev == last,
            _ => false,
        };
        if fast_path {
            self.reml_state
                .set_penalty_shrinkage_floor(self.penalty_shrinkage_floor);
            self.reml_state.setwarm_start_original_beta(warm_start_beta);
            // #1111 / #1033 mechanism (c): a BFGS line-search VALUE probe runs at
            // a DIFFERENT ψ than the full eval that staged the frozen-W first-step
            // Gram. On the design-revision fast path `reset_surface` is skipped, so
            // a Gram installed for a prior trial's ψ would otherwise leak into this
            // probe's inner P-IRLS first iteration — a wrong-ψ Gram. The frozen
            // first-step lane is gradient/full-eval-only (eval_full is the sole
            // stager), so unconditionally clear the slot here; the probe restreams
            // its first-iteration Gram exactly.
            self.reml_state.clear_glm_first_step_gram();
            // Same wrong-ψ-leak reasoning for the GLM frozen-W ψ-gradient
            // derivative (#1033 / #1111): clear it so a prior trial's ψ pair
            // never serves this probe's gradient; it restreams the exact slab.
            self.reml_state.clear_glm_psi_gram_deriv();
            // #1033: the Gaussian-identity `gaussian_fixed_cache` is ALSO keyed to
            // the trial's ψ (the certified ψ-Gram tensor's `XᵀWX(ψ)/XᵀWz(ψ)`), and
            // a VALUE probe runs at a different ψ than the eval that installed it.
            // On the fast path `reset_surface` is skipped, so without re-keying
            // here the inner Gaussian PLS would read the PREVIOUS ψ's Gram — a
            // stale-Gram correctness hazard. Re-install the n-free per-ψ Gram for
            // THIS probe's ψ (in-window) so the value probe is both correct AND
            // touches only k-dim sufficient statistics; off-window the installer
            // is a no-op and the surface keeps its streamed Gram. The conditioned
            // ψ-derivatives the installer also stages are gradient-channel objects
            // unused by `compute_cost`, but keying them to this ψ keeps a single
            // source of truth and avoids leaving a prior trial's pair installed.
            self.install_psi_gram_statistics(theta, rho_dim);
            return Ok(());
        }

        let p = x.ncols();
        let specs: Vec<PenaltySpec> = s_list.iter().map(PenaltySpec::from_blockwise_ref).collect();
        validate_penalty_specs(&specs, p, context)?;
        let (canonical, active_nullspace_dims) =
            crate::construction::canonicalize_penalty_specs(&specs, nullspace_dims, p, context)?;

        let x_fit = self.conditioning.apply_to_design(x);
        let fit_linear_constraints = self
            .conditioning
            .transform_linear_constraints_to_internal(linear_constraints);

        // Cost-only paths do not introduce design drift via hyper_dirs, so
        // the directional-hyper-support check is unnecessary here.
        crate::solver::estimate::reml::RemlState::reset_surface(
            &mut self.reml_state,
            x_fit,
            Arc::new(canonical),
            p,
            active_nullspace_dims,
            None,
            fit_linear_constraints,
            self.kronecker_penalty_system.clone(),
            self.kronecker_factored.clone(),
        )?;
        self.reml_state
            .set_penalty_shrinkage_floor(self.penalty_shrinkage_floor);
        self.reml_state.setwarm_start_original_beta(warm_start_beta);
        self.last_canonical_revision = design_revision;
        Ok(())
    }

    /// Cost-only evaluation at the current κ-realized design. Used by the
    /// joint [ρ, ψ] BFGS line-search cost callback so probes pay neither the
    /// `try_build_spatial_log_kappa_hyper_dirs` cost nor the gradient assembly
    /// cost. The gradient callback continues to use [`evaluate_with_order`].
    ///
    /// Contract: the caller MUST have already realized the design at the κ
    /// implied by `theta`'s ψ tail (typically via the
    /// `SingleBlockExactJointDesignCache::ensure_theta` path). The penalty
    /// gradients w.r.t. ρ are independent of κ for the spatial single-block
    /// path, but correction gates still need to know that the objective lives
    /// on a joint `[ρ, ψ]` surface. Pass the ψ-tail count into the shared cost
    /// bridge so value-only probes decline the same ext-coordinate-incomplete
    /// corrections as the analytic joint path.
    pub(crate) fn evaluate_cost_only(
        &mut self,
        x: &DesignMatrix,
        s_list: &[BlockwisePenalty],
        nullspace_dims: &[usize],
        linear_constraints: Option<crate::pirls::LinearInequalityConstraints>,
        theta: &Array1<f64>,
        rho_dim: usize,
        warm_start_beta: Option<ArrayView1<'_, f64>>,
        context: &str,
        design_revision: Option<u64>,
    ) -> Result<f64, EstimationError> {
        if rho_dim > theta.len() {
            crate::bail_invalid_estim!(
                "rho_dim {} exceeds theta dimension {}",
                rho_dim,
                theta.len()
            );
        }
        self.prepare_eval_state_cost_only(
            x,
            s_list,
            nullspace_dims,
            linear_constraints,
            theta,
            rho_dim,
            warm_start_beta,
            context,
            design_revision,
        )?;
        let rho = theta.slice(s![..rho_dim]).to_owned();
        self.reml_state
            .compute_cost_with_ext_count(&rho, theta.len() - rho_dim)
    }
}

// canonicalize_active_penalties removed — replaced by
// crate::construction::canonicalize_penalty_specs.
