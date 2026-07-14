use super::*;

#[derive(Clone)]
pub(crate) struct TransformationExactGeometryCache {
    pub(crate) key: Vec<u64>,
    pub(crate) covariate_spec_resolved: TermCollectionSpec,
    pub(crate) covariate_design: TermCollectionDesign,
    pub(crate) family: TransformationNormalFamily,
    pub(crate) blocks: Vec<ParameterBlockSpec>,
    pub(crate) hyper_layout: SharedCustomFamilyHyperLayout,
}

#[derive(Default)]
pub(crate) struct TransformationExactModeBranch {
    carried_mode: Option<CustomFamilyWarmStart>,
    anchor: Option<CustomFamilyWarmStart>,
    frozen: bool,
}

impl TransformationExactModeBranch {
    /// Before the branch is frozen, compare a cold solve with the carried mode
    /// at every value-only objective trial. This is a deterministic
    /// mode-selection rule based on the profiled criterion, not coefficient
    /// magnitude or cache distance. Freezing makes the carried candidate
    /// immutable; cold remains a candidate so a worse anchor can never define
    /// the profiled surface.
    pub(crate) fn candidates(
        &mut self,
        eval_mode: gam_problem::EvalMode,
        rho: &Array1<f64>,
    ) -> (bool, Vec<Option<CustomFamilyWarmStart>>) {
        let froze = self.prepare(eval_mode);
        let carried = self
            .frozen
            .then_some(&self.anchor)
            .unwrap_or(&self.carried_mode)
            .as_ref()
            .filter(|warm| warm.compatible_with_rho(rho))
            .cloned();
        match carried {
            Some(warm) => (froze, vec![None, Some(warm)]),
            None => (froze, vec![None]),
        }
    }

    /// Freeze the INPUT mode for the first derivative-bearing seed evaluation.
    /// Freezing before the solve makes that seed evaluation and every later
    /// evaluation at the same theta restart from bit-identical coefficients.
    pub(crate) fn prepare(&mut self, eval_mode: gam_problem::EvalMode) -> bool {
        if self.frozen || matches!(eval_mode, gam_problem::EvalMode::ValueOnly) {
            return false;
        }
        self.anchor = self.carried_mode.take();
        self.frozen = true;
        true
    }

    /// Value-only objective trials are the sole phase allowed to advance the
    /// carried mode.
    pub(crate) fn record_value(
        &mut self,
        eval_mode: gam_problem::EvalMode,
        warm_start: CustomFamilyWarmStart,
    ) {
        if !self.frozen && matches!(eval_mode, gam_problem::EvalMode::ValueOnly) {
            self.carried_mode = Some(warm_start);
        }
    }
}

impl TransformationExactGeometryCache {
    pub(crate) fn update_block_log_lambdas(
        &mut self,
        log_lambdas: &Array1<f64>,
    ) -> Result<(), String> {
        let spec = self
            .blocks
            .first_mut()
            .ok_or_else(|| "missing transformation block spec".to_string())?;
        if log_lambdas.len() != spec.initial_log_lambdas.len() {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "transformation final fit rho length mismatch: got {}, expected {}",
                    log_lambdas.len(),
                    spec.initial_log_lambdas.len()
                ),
            }
            .into());
        }
        gam_problem::validate_log_strengths(log_lambdas.iter().copied()).map_err(|error| {
            TransformationNormalError::InvalidInput {
                reason: format!("invalid transformation smoothing strength: {error}"),
            }
        })?;
        spec.initial_log_lambdas = log_lambdas.clone();
        Ok(())
    }
}

pub(crate) fn transformation_spatial_geometry_key(
    spec: &TermCollectionSpec,
    spatial_terms: &[usize],
) -> Result<Vec<u64>, String> {
    let mut key = Vec::new();
    key.push(spatial_terms.len() as u64);
    for &term_idx in spatial_terms {
        let term = spec.smooth_terms.get(term_idx).ok_or_else(|| {
            format!(
                "transformation spatial geometry key term index {term_idx} out of range for {} smooth terms",
                spec.smooth_terms.len()
            )
        })?;
        key.push(term_idx as u64);

        // The CTN exact-family cache is valid only for an identical covariate
        // geometry. Length-scale and anisotropy scalars are not enough: the
        // family also embeds frozen centers, input standardization scales,
        // identifiability transforms, and active penalty topology. Serialize
        // the already-frozen term and store the exact bytes in the key so a
        // cache hit means the saved prediction design will replay the same
        // matrix used by the final inner fit.
        let payload = serde_json::to_vec(term).map_err(|err| {
            format!("failed to serialize transformation spatial geometry term {term_idx}: {err}")
        })?;
        key.push(payload.len() as u64);
        for chunk in payload.chunks(8) {
            let mut bytes = [0u8; 8];
            for (dst, src) in bytes.iter_mut().zip(chunk.iter().copied()) {
                *dst = src;
            }
            key.push(u64::from_le_bytes(bytes));
        }
    }
    Ok(key)
}

// ---------------------------------------------------------------------------
// Top-level fit function
// ---------------------------------------------------------------------------

/// Result of `fit_transformation_normal`.
#[derive(Clone)]
pub struct TransformationNormalFitResult {
    pub family: TransformationNormalFamily,
    pub fit: UnifiedFitResult,
    pub covariate_spec_resolved: TermCollectionSpec,
    pub covariate_design: TermCollectionDesign,
    pub score_calibration: TransformationScoreCalibration,
}

/// Fit a conditional transformation model with N-block spatial length-scale
/// optimization over the covariate side.
///
/// The response-direction basis is built once (it does not depend on κ).
/// If no spatial length-scale terms are present in the covariate spec, the
/// model is fit directly. Otherwise, the N-block joint hyper-parameter
/// optimizer is used with a single block (the covariate spec).
pub fn fit_transformation_normal(
    response: &Array1<f64>,
    weights: &Array1<f64>,
    offset: &Array1<f64>,
    covariate_data: ArrayView2<'_, f64>,
    covariate_spec: &TermCollectionSpec,
    config: &TransformationNormalConfig,
    options: &BlockwiseFitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
    warm_start: Option<&TransformationWarmStart>,
) -> Result<TransformationNormalFitResult, String> {
    let mut options = options.clone();
    // CTN advertises profiled outer-Hessian HVP support and supplies the
    // callback derivative kernel consumed by the unified REML/LAML evaluator.
    // Keep analytic curvature enabled here: the evaluator routes CTN Hessians
    // through the matrix-free operator path instead of dense pairwise assembly.
    let covariate_spec = covariate_spec.clone();

    // 1. Build a bootstrap covariate design first so the response basis can
    // adapt to the tensor width instead of always using the global default.
    let boot_design = build_term_collection_design(covariate_data, &covariate_spec)
        .map_err(|e| format!("failed to build bootstrap covariate design: {e}"))?;
    let boot_spec = freeze_term_collection_from_design(&covariate_spec, &boot_design)
        .map_err(|e| format!("failed to freeze bootstrap covariate spatial basis centers: {e}"))?;
    let mut effective_config = config.clone();
    // When the caller has already resolved the knot count (cross-fit pins it
    // once at the smallest fold complement so every fold shares one p_resp),
    // use it verbatim — re-applying the data-driven complexity cap on this
    // fold's response subsample would round to a different count and break the
    // fold-invariant `p₁` the cross-fit OOF assembly requires.
    if !config.response_num_internal_knots_pinned {
        effective_config.response_num_internal_knots = effective_response_num_internal_knots(
            config,
            response.len(),
            boot_design.design.ncols(),
            response.view(),
        );
    }

    // 2. Build response basis ONCE — it is independent of κ once the effective
    // response complexity has been chosen.
    let (resp_val, resp_deriv, resp_penalties, resp_knots, resp_transform) =
        build_response_basis(response, &effective_config)?;

    // Scope the custom-family inner exact-Newton cycle budget to CTN's
    // bounded-dimension, strictly convex (double-penalty) coefficient block.
    // The realized tensor width is `p_resp · p_cov`; the cap grows with it so a
    // genuinely high-dimensional nonlinear transformation keeps headroom, but a
    // near-Gaussian shift can no longer spin the production large-scale cap (#720).
    // Only ever *lower* the caller's cap so a deliberately tightened budget
    // (screening / CI overrides) is respected.
    let realized_p_total = resp_val.ncols().saturating_mul(boot_design.design.ncols());
    let ctn_inner_cap = CTN_INNER_MAX_CYCLES_BASE
        .saturating_add(realized_p_total.saturating_mul(CTN_INNER_MAX_CYCLES_PER_DIM))
        .min(CTN_INNER_MAX_CYCLES_CEILING);
    options.inner_max_cycles = options.inner_max_cycles.min(ctn_inner_cap);

    // 3. Check whether spatial κ optimization is needed.
    let spatial_terms = spatial_length_scale_term_indices(&covariate_spec);

    if spatial_terms.is_empty() || !kappa_options.enabled {
        // ------------------------------------------------------------------
        // NO κ: build family directly, fit, return.
        // ------------------------------------------------------------------
        let cov_design = boot_design;
        let cov_spec_resolved = boot_spec;
        let effective_offset = cov_design
            .compose_offset(offset.view(), "transformation-normal fit")
            .map_err(|error| error.to_string())?;

        let family = TransformationNormalFamily::from_prebuilt_response_basis(
            response,
            resp_val,
            resp_deriv,
            resp_penalties,
            resp_knots.clone(),
            effective_config.response_degree,
            resp_transform,
            weights,
            &effective_offset,
            cov_design.design.clone(),
            cov_design
                .penalties
                .iter()
                .map(|bp| bp.to_penalty_matrix(cov_design.design.ncols()))
                .collect(),
            &effective_config,
            warm_start,
        )?;
        let rho0 = family.penalty_scale_log_lambdas()?;
        let blocks = vec![family.block_spec(&rho0)?];
        let fit = fit_custom_family(&family, &blocks, &options)
            .map_err(|e| format!("transformation fit failed: {e}"))?;
        let (fit, score_calibration) = calibrate_transformation_scores(&family, fit)?;

        return Ok(TransformationNormalFitResult {
            family,
            fit,
            covariate_spec_resolved: cov_spec_resolved,
            covariate_design: cov_design,
            score_calibration,
        });
    }

    // ------------------------------------------------------------------
    // YES κ: use the N-block spatial length-scale optimizer (1 block).
    // ------------------------------------------------------------------

    let kappa0 = SpatialLogKappaCoords::from_length_scales_aniso(
        &covariate_spec,
        &spatial_terms,
        kappa_options,
    )
    .reseed_from_data(
        covariate_data,
        &covariate_spec,
        &spatial_terms,
        kappa_options,
    )
    .map_err(|error| error.to_string())?;
    let kappa_dims = kappa0.dims_per_term().to_vec();
    let kappa_lower = SpatialLogKappaCoords::lower_bounds_aniso_from_data(
        covariate_data,
        &covariate_spec,
        &spatial_terms,
        &kappa_dims,
        kappa_options,
    )
    .map_err(|error| error.to_string())?;
    let kappa_upper = SpatialLogKappaCoords::upper_bounds_aniso_from_data(
        covariate_data,
        &covariate_spec,
        &spatial_terms,
        &kappa_dims,
        kappa_options,
    )
    .map_err(|error| error.to_string())?;
    // Project seed onto bounds; spec.length_scale is a hint, not a constraint.
    let kappa0 = kappa0.clamp_to_bounds(&kappa_lower, &kappa_upper);

    // Check analytic derivative capability.
    let analytic_psi_available =
        build_block_spatial_psi_derivatives(covariate_data, &boot_spec, &boot_design)?.is_some();

    // Rebuild from the frozen `boot_spec` so the probe's penalty topology
    // matches the topology produced by every other build path in this
    // optimization. The outer optimizer's own bootstrap
    // (`build_term_collection_designs_and_freeze_joint(data, &[boot_spec])`)
    // and the geometry cache's `build_term_collection_design(_, &effective_spec)`
    // both feed the basis builder a frozen `FrozenTransform` identifiability,
    // while `boot_design` was built from the raw `covariate_spec` with
    // identifiability computed from scratch. Applying the captured
    // `FrozenTransform` changes the exact coefficient chart of the penalty
    // blocks. Without this rebuild, `n_penalties` is taken from the raw build
    // but every subsequent
    // evaluator measures the frozen build, and `evaluate_custom_family_joint_hyper`
    // refuses with a `joint hyper rho dimension mismatch`.
    let probe_design = build_term_collection_design(covariate_data, &boot_spec)
        .map_err(|e| format!("failed to rebuild frozen probe covariate design: {e}"))?;
    let probe_offset = probe_design
        .compose_offset(offset.view(), "transformation-normal spatial probe")
        .map_err(|error| error.to_string())?;

    // Build an initial family + blocks for capability probing.
    let probe_family = TransformationNormalFamily::from_prebuilt_response_basis(
        response,
        resp_val.clone(),
        resp_deriv.clone(),
        resp_penalties.clone(),
        resp_knots.clone(),
        effective_config.response_degree,
        resp_transform.clone(),
        weights,
        &probe_offset,
        probe_design.design.clone(),
        probe_design
            .penalties
            .iter()
            .map(|bp| bp.to_penalty_matrix(probe_design.design.ncols()))
            .collect(),
        &effective_config,
        warm_start,
    )?;
    let rho0 = probe_family.penalty_scale_log_lambdas()?;
    let probe_block = probe_family.block_spec(&rho0)?;
    let n_penalties = probe_block.initial_log_lambdas.len();
    log::info!(
        "[transformation-normal] exact joint setup: rho_dim={} log_kappa_dim={} dims_per_term={:?}",
        n_penalties,
        kappa0.len(),
        kappa_dims,
    );
    let rho_floor = -12.0;
    let rho_lower = Array1::<f64>::from_elem(n_penalties, rho_floor);
    let rho_upper = Array1::<f64>::from_elem(n_penalties, 12.0);
    let probe_blocks = vec![probe_block.clone()];
    let (_, cap_hessian) = crate::custom_family::custom_family_outer_derivatives(
        &probe_family,
        &probe_blocks,
        &options,
    );
    let analytic_gradient = analytic_psi_available;
    let analytic_hessian_supported = analytic_gradient && cap_hessian.is_analytic();
    let analytic_hessian = analytic_hessian_supported;
    if analytic_hessian {
        log::info!(
            "[transformation-normal] CTN exact joint analytic outer Hessian is available for spatial kappa optimization; using exact second-order outer geometry"
        );
    }

    let (rho0_min, rho0_max) = if rho0.is_empty() {
        (0.0, 0.0)
    } else {
        (
            rho0.iter().copied().fold(f64::INFINITY, f64::min),
            rho0.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        )
    };
    log::info!(
        "[transformation-normal] skipping baseline custom-family prefit before exact joint optimization \
         (rho_dim={}, log_kappa_dim={}, rho0_range=[{:.3}, {:.3}]); using CTN warm start and penalty-scale rho seed",
        n_penalties,
        kappa0.len(),
        rho0_min,
        rho0_max,
    );

    if !analytic_psi_available {
        return Err(
            "transformation-normal spatial length-scale optimization requires analytic spatial psi derivatives"
                .to_string(),
        );
    }

    // SCOP's squared shape coordinates make the coefficient objective
    // non-convex. Value-only trials compare cold and carried modes;
    // the first derivative-bearing evaluation freezes the selected mode's
    // INPUT as the branch anchor. Every later trial restarts from that fixed
    // anchor, making the profile independent of rejected-trial cache history.
    let exact_mode_branch: RefCell<TransformationExactModeBranch> =
        RefCell::new(TransformationExactModeBranch::default());

    let joint_setup =
        ExactJointHyperSetup::new(rho0, rho_lower, rho_upper, kappa0, kappa_lower, kappa_upper);

    // Clone response basis parts for use inside closures.
    let rv = resp_val.clone();
    let rd = resp_deriv.clone();
    let rp = resp_penalties.clone();
    let rk = resp_knots.clone();
    let rt = resp_transform.clone();
    let rdeg = effective_config.response_degree;
    let cfg = effective_config.clone();
    let ws = warm_start.cloned();

    // Helper: build family from prebuilt response basis + covariate design.
    let make_family =
        |cov_design: &TermCollectionDesign| -> Result<TransformationNormalFamily, String> {
            let effective_offset = cov_design
                .compose_offset(offset.view(), "transformation-normal spatial fit")
                .map_err(|error| error.to_string())?;
            TransformationNormalFamily::from_prebuilt_response_basis(
                response,
                rv.clone(),
                rd.clone(),
                rp.clone(),
                rk.clone(),
                rdeg,
                rt.clone(),
                weights,
                &effective_offset,
                cov_design.design.clone(),
                cov_design
                    .penalties
                    .iter()
                    .map(|bp| bp.to_penalty_matrix(cov_design.design.ncols()))
                    .collect(),
                &cfg,
                ws.as_ref(),
            )
        };

    let block_specs_slice = [boot_spec.clone()];
    let block_term_indices_slice = [spatial_terms.clone()];
    let exact_geometry_cache: RefCell<Option<TransformationExactGeometryCache>> =
        RefCell::new(None);
    let spatial_terms_for_cache = spatial_terms.clone();

    let ensure_exact_geometry = |spec: &TermCollectionSpec,
                                 design: &TermCollectionDesign,
                                 rho: &Array1<f64>,
                                 hyper_values: &Array1<f64>|
     -> Result<(), String> {
        let effective_spec = freeze_term_collection_from_design(spec, design)
            .map_err(|e| format!("failed to freeze transformation geometry key: {e}"))?;
        let key = transformation_spatial_geometry_key(&effective_spec, &spatial_terms_for_cache)?;
        let needs_rebuild = exact_geometry_cache
            .borrow()
            .as_ref()
            .map(|cached| cached.key != key)
            .unwrap_or(true);
        if !needs_rebuild {
            let mut cache = exact_geometry_cache.borrow_mut();
            let cached = cache
                .as_mut()
                .ok_or_else(|| "missing transformation exact geometry cache".to_string())?;
            if cached.hyper_layout.values().len() != hyper_values.len()
                || cached
                    .hyper_layout
                    .values()
                    .iter()
                    .zip(hyper_values)
                    .any(|(cached, current)| cached.to_bits() != current.to_bits())
            {
                return Err(
                    "transformation exact geometry key reused across distinct hypercoordinate values"
                        .to_string(),
                );
            }
            return cached.update_block_log_lambdas(rho);
        }

        let geom_start = std::time::Instant::now();
        let exact_design = build_term_collection_design(covariate_data, &effective_spec)
            .map_err(|e| format!("failed to rebuild frozen transformation geometry: {e}"))?;
        let family = make_family(&exact_design)?;
        let cov_psi_derivs =
            build_block_spatial_psi_derivatives(covariate_data, &effective_spec, &exact_design)?
                .ok_or_else(|| {
                    "missing covariate spatial psi derivatives for transformation model".to_string()
                })?;
        let tensor_derivs = build_tensor_psi_derivatives(&family, &cov_psi_derivs)?;

        log::debug!(
            "[transformation-normal] rebuilt exact geometry cache for {} spatial terms in {:.3}s",
            spatial_terms_for_cache.len(),
            geom_start.elapsed().as_secs_f64(),
        );

        exact_geometry_cache.replace(Some(TransformationExactGeometryCache {
            key,
            covariate_spec_resolved: effective_spec,
            covariate_design: exact_design,
            blocks: vec![family.block_spec(rho)?],
            family,
            hyper_layout: Arc::new(CustomFamilyHyperLayout::new(
                vec![tensor_derivs],
                Vec::new(),
                hyper_values.clone(),
            )?),
        }));
        Ok(())
    };

    let exact_mode_candidates = |eval_mode: gam_problem::EvalMode,
                                 rho: &Array1<f64>|
     -> Vec<Option<CustomFamilyWarmStart>> {
        let (froze, candidates) = exact_mode_branch.borrow_mut().candidates(eval_mode, rho);
        if froze {
            log::info!(
                "[transformation-normal] froze deterministic exact coefficient-mode branch at the first derivative-bearing outer seed evaluation"
            );
        }
        candidates
    };

    log::info!(
        "[transformation-normal] entering exact joint outer optimization \
         (analytic_gradient={}, analytic_hessian={})",
        analytic_gradient,
        analytic_hessian,
    );
    // Outer derivative policy (P2.3): consult the family's CTN-specific
    // override so the cost gate uses the Khatri–Rao row-streamed shape
    // (`O(n · (rho + psi) · p)` gradient; `min(dense, mfree)` Hessian)
    // rather than the generic `coefficient_*_cost × K` default.
    let outer_derivative_policy =
        probe_family.outer_derivative_policy(&probe_blocks, joint_setup.log_kappa_dim(), &options);
    let solved = optimize_spatial_length_scale_exact_joint(
        covariate_data,
        &block_specs_slice,
        &block_term_indices_slice,
        kappa_options,
        &joint_setup,
        gam_solve::seeding::SeedRiskProfile::Gaussian,
        analytic_gradient,
        analytic_hessian,
        // Transformation-normal has β-dependent H (through 1/h'²), so the
        // EFS Wood-Fasiolo PSD invariant fails. Keep fixed-point disabled while
        // exposing the exact outer Hessian to ARC: hiding it forces this
        // three-coordinate problem onto BFGS, whose Strong-Wolfe probes each
        // repeat a full CTN inner solve and caused every large-scale lane to
        // exhaust the 2400-second command budget before marginal-slope began.
        true,
        None,
        outer_derivative_policy,
        // fit_fn
        |theta,
         specs: &[TermCollectionSpec],
         designs: &[TermCollectionDesign],
         provenance: SpatialFitProvenance<'_, CustomFamilyJointHyperModeSelection>| {
            let rho = theta.slice(s![..joint_setup.rho_dim()]).to_owned();
            let hyper_values = theta.slice(s![joint_setup.rho_dim()..]).to_owned();
            ensure_exact_geometry(&specs[0], &designs[0], &rho, &hyper_values)?;
            let mut cache_ref = exact_geometry_cache.borrow_mut();
            let geometry = cache_ref
                .as_mut()
                .ok_or_else(|| "missing transformation exact geometry cache".to_string())?;
            let final_options = crate::outer_subsample::exact_outer_options_for_row_set(
                &options,
                &gam_problem::outer_subsample::RowSet::All,
            );
            let fit = match provenance {
                SpatialFitProvenance::NoOuterOptimization => {
                    let warm_starts =
                        exact_mode_candidates(gam_problem::EvalMode::ValueOnly, &rho);
                    let selection = evaluate_custom_family_joint_hyper_best_mode_shared(
                        &geometry.family,
                        &geometry.blocks,
                        &final_options,
                        &rho,
                        Arc::clone(&geometry.hyper_layout),
                        &warm_starts,
                        gam_problem::EvalMode::ValueOnly,
                    )
                    .map_err(|e| format!("transformation fixed mode profile: {e}"))?;
                    log::info!(
                        "[transformation-normal] user-fixed coefficient mode selected candidate={} objective={:.16e}",
                        selection.selected_candidate,
                        selection.result.objective,
                    );
                    fit_custom_family_user_fixed_log_lambdas_from_mode_selection(
                        &geometry.family,
                        &geometry.blocks,
                        &final_options,
                        selection,
                    )
                }
                SpatialFitProvenance::Certified { outer, mode: selection } => {
                    log::info!(
                        "[transformation-normal] consuming certified terminal coefficient mode candidate={} objective={:.16e} without profile replay",
                        selection.selected_candidate,
                        selection.result.objective,
                    );
                    fit_custom_family_fixed_log_lambdas_from_mode_selection(
                        &geometry.family,
                        &geometry.blocks,
                        &final_options,
                        selection,
                        theta,
                        outer,
                    )
                }
            }
            .map_err(|e| format!("transformation fit_fn: {e}"))?;
            if let Some(block) = fit.block_states.first() {
                *geometry
                    .family
                    .row_quantity_cache
                    .lock()
                    .expect("CTN row quantity cache mutex poisoned") = None;
                let final_rows = geometry.family.row_quantities(&block.beta)?;
                let max_abs_h = final_rows
                    .h
                    .iter()
                    .copied()
                    .map(f64::abs)
                    .fold(0.0, f64::max);
                let cov_chunk = geometry
                    .family
                    .covariate_design
                    .try_row_chunk(0..response.len())
                    .map_err(|err| {
                        format!("final CTN covariate design validation failed: {err}")
                    })?;
                let max_abs_cov = cov_chunk.iter().copied().map(f64::abs).fold(0.0, f64::max);
                log::info!(
                    "[transformation-normal] final fixed-rho CTN validation: max_abs_h={:.6e} max_abs_covariate_basis={:.6e}",
                    max_abs_h,
                    max_abs_cov
                );
            }
            Ok(TransformationNormalFitResult {
                family: geometry.family.clone(),
                fit,
                covariate_spec_resolved: geometry.covariate_spec_resolved.clone(),
                covariate_design: geometry.covariate_design.clone(),
                score_calibration: TransformationScoreCalibration::finite_support_pit(),
            })
        },
        // exact_fn
        |theta,
         specs: &[TermCollectionSpec],
         designs: &[TermCollectionDesign],
         eval_mode,
         row_set| {
            let rho = theta.slice(s![..joint_setup.rho_dim()]).to_owned();
            let hyper_values = theta.slice(s![joint_setup.rho_dim()..]).to_owned();
            ensure_exact_geometry(&specs[0], &designs[0], &rho, &hyper_values)?;
            let mut cache_ref = exact_geometry_cache.borrow_mut();
            let geometry = cache_ref
                .as_mut()
                .ok_or_else(|| "missing transformation exact geometry cache".to_string())?;
            let warm_starts = exact_mode_candidates(eval_mode, &rho);
            let competing_modes = warm_starts.len() > 1;
            // `row_set` is the outer driver's authoritative measure. Rebuild
            // the family-facing option on every evaluation so a pilot mask
            // cannot survive the driver's rotation back to full data.
            let eval_options =
                crate::outer_subsample::exact_outer_options_for_row_set(&options, row_set);
            let selection = evaluate_custom_family_joint_hyper_best_mode_shared(
                &geometry.family,
                &geometry.blocks,
                &eval_options,
                &rho,
                Arc::clone(&geometry.hyper_layout),
                &warm_starts,
                eval_mode,
            )
            .map_err(|e| format!("transformation exact joint mode profile: {e}"))?;
            for (candidate_idx, rejection) in selection.rejected_candidates.iter().enumerate() {
                if let Some(rejection) = rejection {
                    log::warn!(
                        "[transformation-normal] rejected exact coefficient-mode candidate mode_candidate={candidate_idx}: {rejection}"
                    );
                }
            }
            if competing_modes {
                for (candidate_idx, objective) in selection.screened_objectives.iter().enumerate() {
                    if let Some(objective) = objective {
                        let source = if candidate_idx == 0 {
                            "cold"
                        } else {
                            "carried"
                        };
                        log::info!(
                            "[transformation-normal] exact coefficient-mode screen source={} objective={:.16e}",
                            source,
                            objective,
                        );
                    }
                }
                let selected_source = if selection.selected_candidate == 0 {
                    "cold"
                } else {
                    "carried"
                };
                log::info!(
                    "[transformation-normal] selected exact coefficient mode source={} objective={:.16e}",
                    selected_source,
                    selection.result.objective,
                );
            }
            let objective = selection.result.objective;
            let gradient = selection.result.gradient.clone();
            let outer_hessian = selection.result.outer_hessian.clone();
            exact_mode_branch
                .borrow_mut()
                .record_value(eval_mode, selection.result.warm_start.clone());

            Ok(ExactJointEvaluation {
                objective,
                gradient,
                hessian: outer_hessian,
                mode: selection,
            })
        },
        |_theta,
         _specs: &[TermCollectionSpec],
         _designs: &[TermCollectionDesign],
         _row_set| {
            Err::<ExactJointEfsEvaluation<crate::custom_family::CustomFamilyJointHyperModeSelection>, String>("transformation-normal EFS callback invoked even though fixed-point optimization is disabled for beta-dependent exact curvature".to_string())
        },
        |_beta: &Array1<f64>| Ok(gam_solve::rho_optimizer::SeedOutcome::NoSlot),
    )?;

    let mut fit = solved.fit;
    let (calibrated_fit, score_calibration) =
        calibrate_transformation_scores(&fit.family, fit.fit.clone())?;
    fit.fit = calibrated_fit;
    fit.score_calibration = score_calibration;
    Ok(fit)
}
