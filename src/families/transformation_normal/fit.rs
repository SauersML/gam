use super::*;

#[derive(Clone)]
struct TransformationExactGeometryCache {
    key: Vec<u64>,
    covariate_spec_resolved: TermCollectionSpec,
    covariate_design: TermCollectionDesign,
    family: TransformationNormalFamily,
    blocks: Vec<ParameterBlockSpec>,
    derivative_blocks: Vec<Vec<CustomFamilyBlockPsiDerivative>>,
}


#[derive(Clone)]
struct TransformationExactWarmStart {
    theta: Array1<f64>,
    warm_start: CustomFamilyWarmStart,
}


impl TransformationExactWarmStart {
    fn is_compatible_with(&self, theta: &Array1<f64>, rho: &Array1<f64>) -> bool {
        const MAX_THETA_DISTANCE: f64 = 1.5;

        self.theta.len() == theta.len()
            && self
                .theta
                .iter()
                .zip(theta.iter())
                .all(|(&a, &b)| (a - b).abs() <= MAX_THETA_DISTANCE)
            && self.warm_start.compatible_with_rho(rho)
    }
}


impl TransformationExactGeometryCache {
    fn update_initial_log_lambdas(&mut self, log_lambdas: &Array1<f64>) -> Result<(), String> {
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
        spec.initial_log_lambdas = log_lambdas.clone();
        Ok(())
    }
}


fn transformation_spatial_geometry_key(
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

        let family = TransformationNormalFamily::from_prebuilt_response_basis(
            response,
            resp_val,
            resp_deriv,
            resp_penalties,
            resp_knots.clone(),
            effective_config.response_degree,
            resp_transform,
            weights,
            offset,
            cov_design.design.clone(),
            cov_design
                .penalties
                .iter()
                .map(|bp| bp.to_penalty_matrix(cov_design.design.ncols()))
                .collect(),
            &effective_config,
            warm_start,
        )?;
        let blocks = vec![family.block_spec()];
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
    );
    let kappa_dims = kappa0.dims_per_term().to_vec();
    let kappa_lower = SpatialLogKappaCoords::lower_bounds_aniso_from_data(
        covariate_data,
        &covariate_spec,
        &spatial_terms,
        &kappa_dims,
        kappa_options,
    );
    let kappa_upper = SpatialLogKappaCoords::upper_bounds_aniso_from_data(
        covariate_data,
        &covariate_spec,
        &spatial_terms,
        &kappa_dims,
        kappa_options,
    );
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
    // `FrozenTransform` to the same Duchon kernel can land the structural
    // null-space block on either side of `build_nullspace_shrinkage_penalty`'s
    // spectral tolerance, so the raw and frozen builds disagree on whether
    // the trend ridge survives as an active penalty candidate. Without this
    // rebuild, `n_penalties` is taken from the raw build but every subsequent
    // evaluator measures the frozen build, and `evaluate_custom_family_joint_hyper`
    // refuses with a `joint hyper rho dimension mismatch`.
    let probe_design = build_term_collection_design(covariate_data, &boot_spec)
        .map_err(|e| format!("failed to rebuild frozen probe covariate design: {e}"))?;

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
        offset,
        probe_design.design.clone(),
        probe_design
            .penalties
            .iter()
            .map(|bp| bp.to_penalty_matrix(probe_design.design.ncols()))
            .collect(),
        &effective_config,
        warm_start,
    )?;
    let probe_block = probe_family.block_spec();
    let n_penalties = probe_block.initial_log_lambdas.len();
    log::info!(
        "[transformation-normal] exact joint setup: rho_dim={} log_kappa_dim={} dims_per_term={:?}",
        n_penalties,
        kappa0.len(),
        kappa_dims,
    );
    let rho0 = probe_block.initial_log_lambdas.clone();
    let rho_floor = -12.0;
    let rho_lower = Array1::<f64>::from_elem(n_penalties, rho_floor);
    let rho_upper = Array1::<f64>::from_elem(n_penalties, 12.0);
    let probe_blocks = vec![probe_block.clone()];
    let (_, cap_hessian) = crate::families::custom_family::custom_family_outer_derivatives(
        &probe_family,
        &probe_blocks,
        &options,
    );
    let analytic_gradient = analytic_psi_available;
    let analytic_hessian_supported = analytic_gradient && cap_hessian.is_analytic();
    let analytic_hessian = false;
    if analytic_hessian_supported {
        log::info!(
            "[transformation-normal] CTN exact joint analytic outer Hessian is available but disabled for spatial kappa optimization; using analytic-gradient outer solves to avoid callback logdet trace work"
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

    // Shared mutable state for warm-starting across optimizer iterations.
    let exact_warm_start: RefCell<Option<TransformationExactWarmStart>> = RefCell::new(None);

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
            TransformationNormalFamily::from_prebuilt_response_basis(
                response,
                rv.clone(),
                rd.clone(),
                rp.clone(),
                rk.clone(),
                rdeg,
                rt.clone(),
                weights,
                offset,
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
                                 design: &TermCollectionDesign|
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
            return Ok(());
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
            blocks: vec![family.block_spec()],
            family,
            derivative_blocks: vec![tensor_derivs],
        }));
        Ok(())
    };

    let compatible_warm_start =
        |theta: &Array1<f64>, rho: &Array1<f64>| -> Option<CustomFamilyWarmStart> {
            exact_warm_start
                .borrow()
                .as_ref()
                .filter(|warm| warm.is_compatible_with(theta, rho))
                .map(|warm| warm.warm_start.clone())
        };
    let store_warm_start = |theta: &Array1<f64>, warm_start: CustomFamilyWarmStart| {
        exact_warm_start
            .borrow_mut()
            .replace(TransformationExactWarmStart {
                theta: theta.clone(),
                warm_start,
            });
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
        crate::seeding::SeedRiskProfile::Gaussian,
        analytic_gradient,
        analytic_hessian,
        // Transformation-normal has β-dependent H (through 1/h'²), so the
        // EFS Wood-Fasiolo PSD invariant fails. Keep fixed-point disabled,
        // but do not expose CTN's analytic Hessian to ARC: its callback
        // trace route applies full-rank logdet operators at large-scale shape.
        true,
        None,
        outer_derivative_policy,
        // fit_fn
        |theta, specs: &[TermCollectionSpec], designs: &[TermCollectionDesign]| {
            ensure_exact_geometry(&specs[0], &designs[0])?;
            let mut cache_ref = exact_geometry_cache.borrow_mut();
            let geometry = cache_ref
                .as_mut()
                .ok_or_else(|| "missing transformation exact geometry cache".to_string())?;
            let rho = theta.slice(s![..joint_setup.rho_dim()]).to_owned();
            geometry.update_initial_log_lambdas(&rho)?;
            let warm_start = compatible_warm_start(theta, &rho);
            let fit = fit_custom_family_fixed_log_lambdas(
                &geometry.family,
                &geometry.blocks,
                &options,
                warm_start.as_ref(),
                0,
                None,
                true,
            )
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
         _row_set| {
            ensure_exact_geometry(&specs[0], &designs[0])?;
            let mut cache_ref = exact_geometry_cache.borrow_mut();
            let geometry = cache_ref
                .as_mut()
                .ok_or_else(|| "missing transformation exact geometry cache".to_string())?;
            let rho = theta.slice(s![..joint_setup.rho_dim()]).to_owned();
            let warm_start = compatible_warm_start(theta, &rho);

            let eval = evaluate_custom_family_joint_hyper(
                &geometry.family,
                &geometry.blocks,
                &options,
                &rho,
                &geometry.derivative_blocks,
                warm_start.as_ref(),
                eval_mode,
            )
            .map_err(|e| format!("transformation exact_fn: {e}"))?;

            if !eval.objective.is_finite() {
                log::warn!(
                    "transformation exact joint returned non-finite objective: eval_mode={:?} rho={:?} gradient_len={}",
                    eval_mode,
                    rho,
                    eval.gradient.len(),
                );
            }

            if eval.objective.is_finite() && eval.gradient.iter().all(|value| value.is_finite()) {
                store_warm_start(theta, eval.warm_start.clone());
            }

            if !eval.inner_converged {
                return Err(format!(
                    "transformation exact joint inner solve did not converge for eval_mode={eval_mode:?}; cached warm start for retry"
                ));
            }

            Ok((eval.objective, eval.gradient, eval.outer_hessian))
        },
        |theta, specs: &[TermCollectionSpec], designs: &[TermCollectionDesign]| {
            ensure_exact_geometry(&specs[0], &designs[0])?;
            let mut cache_ref = exact_geometry_cache.borrow_mut();
            let geometry = cache_ref
                .as_mut()
                .ok_or_else(|| "missing transformation exact geometry cache".to_string())?;
            let rho = theta.slice(s![..joint_setup.rho_dim()]).to_owned();
            let warm_start = compatible_warm_start(theta, &rho);
            let eval = evaluate_custom_family_joint_hyper_efs(
                &geometry.family,
                &geometry.blocks,
                &options,
                &rho,
                &geometry.derivative_blocks,
                warm_start.as_ref(),
            )
            .map_err(|e| format!("transformation exact_efs_fn: {e}"))?;
            store_warm_start(theta, eval.warm_start.clone());
            if !eval.inner_converged {
                return Err(
                    "transformation exact joint EFS inner solve did not converge; cached warm start for retry"
                        .to_string(),
                );
            }
            Ok(eval.efs_eval)
        },
        |_beta: &Array1<f64>| Ok(()),
    )?;

    let mut fit = solved.fit;
    let (calibrated_fit, score_calibration) =
        calibrate_transformation_scores(&fit.family, fit.fit.clone())?;
    fit.fit = calibrated_fit;
    fit.score_calibration = score_calibration;
    Ok(fit)
}
