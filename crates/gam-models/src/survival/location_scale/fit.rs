use super::*;

/// Derive the survival location-scale outer-evaluation options from the exact
/// row measure selected by the spatial optimizer.
///
/// `row_set` is the sole authority for this evaluation.  In particular, an
/// inherited automatic/staged pilot must not survive into a full-data replay:
/// that would let the inner mode, objective gradient, and Hessian describe
/// different Horvitz--Thompson measures.  `All` therefore clears every stale
/// mask just as deliberately as `Subsample` installs the selected weighted
/// rows.
pub(crate) fn survival_location_scale_exact_outer_options(
    options: &BlockwiseFitOptions,
    row_set: &crate::row_kernel::RowSet,
) -> BlockwiseFitOptions {
    let mut effective = options.clone();
    effective.auto_outer_subsample = false;
    effective.outer_score_subsample = match row_set {
        crate::row_kernel::RowSet::All => None,
        crate::row_kernel::RowSet::Subsample { rows, n_full } => Some(Arc::new(
            crate::outer_subsample::OuterScoreSubsample::from_weighted_rows(
                rows.as_ref().clone(),
                *n_full,
                0,
            ),
        )),
    };
    effective
}

/// Run the direct parametric-AFT MLE for a fully reduced constant-scale model
/// and assemble the same [`UnifiedFitResult`] the coupled path would produce.
///
/// Every block is unpenalized (zero ρ) — the reduced affine time-warp, the
/// location intercept/covariate, and the constant log-σ identify the AFT MLE
/// directly, and `survival_reduced_parametric_aft_regime` has already dropped
/// any default parametric shrinkage ridge — so `log_lambdas`/`lambdas` are
/// empty, the stable penalty term is zero, and the penalized objective is just
/// `−ℓ̂`. The conditional covariance is the inverse of the observed information
/// `H` (the joint negative-log-likelihood Hessian at the MLE), and the
/// geometry's penalized Hessian is `H` itself — matching the exact-Newton joint
/// geometry the coupled survival path stores (`working_weights`/`working_response`
/// are the zero-length convention used by exact-Newton joint families). The
/// shared [`crate::custom_family::blockwise_fit_from_parts`] assembler then
/// computes EDF (= parameter count, since unpenalized) and the inference block
/// exactly as for any custom-family fit.
pub(crate) fn fit_reduced_parametric_aft(
    prepared: &PreparedSurvivalLocationScaleModel,
    options: &BlockwiseFitOptions,
) -> Result<UnifiedFitResult, String> {
    use gam_linalg::faer_ndarray::FaerCholesky;

    let specs = &prepared.blockspecs;
    let (states, log_likelihood, h) = prepared.family.fit_parametric_aft_direct_mle(
        specs,
        options.inner_max_cycles.max(1),
        options.inner_tol.max(REDUCED_AFT_OBJ_TOL_FLOOR),
    )?;

    let p_total = h.nrows();
    // Conditional covariance Var(θ | λ) = H⁻¹ in the reduced coordinate system.
    // `finalize_survival_location_scale_fit` lifts it back to the raw block
    // coordinates (time null-space expansion + leading-fixed-column padding).
    let identity = Array2::<f64>::eye(p_total);
    let covariance_conditional = match h.cholesky(faer::Side::Lower) {
        Ok(chol) => {
            let cov = chol.solve_mat(&identity);
            if cov.iter().all(|v| v.is_finite()) {
                // Symmetrize away round-off so the lifted covariance is exactly
                // symmetric, as the conditional covariance must be.
                let mut symm = cov.clone();
                for i in 0..p_total {
                    for j in (i + 1)..p_total {
                        let avg = 0.5 * (cov[[i, j]] + cov[[j, i]]);
                        symm[[i, j]] = avg;
                        symm[[j, i]] = avg;
                    }
                }
                Some(symm)
            } else {
                None
            }
        }
        Err(_) => None,
    };

    let geometry = Some(FitGeometry {
        coefficient_gauge: gam_problem::gauge::Gauge::identity(
            &specs
                .iter()
                .map(|spec| spec.design.ncols())
                .collect::<Vec<_>>(),
        ),
        penalized_hessian: h.into(),
        working_weights: Array1::<f64>::zeros(0),
        working_response: Array1::<f64>::zeros(0),
    });

    // The block states carry their η in the family's native row layout — the
    // stacked `[exit; entry; deriv]` channels (`solver_design().nrows()` rows)
    // for the time block, exactly as `refresh_all_block_etas` produces and as
    // the family's `validate_joint_states` / `offset_channel_geometry` require.
    // `blockwise_fit_from_parts` validates each block's `η.len()` against
    // `spec.design.nrows()`, so present it the row-matching `solver_design()`
    // as `design` (same coefficients, penalties, name, role — only the row
    // count differs). All other fields are unchanged, so the assembled result
    // is identical to the coupled path's.
    let assembly_specs: Vec<ParameterBlockSpec> = specs
        .iter()
        .map(|spec| {
            let mut s = spec.clone();
            s.design = spec.solver_design().clone();
            s.offset = spec.solver_offset().clone();
            s.stacked_design = None;
            s.stacked_offset = None;
            s
        })
        .collect();

    crate::custom_family::blockwise_fit_from_parts(
        crate::custom_family::BlockwiseFitResultParts {
            block_states: states,
            log_likelihood,
            log_lambdas: Array1::<f64>::zeros(0),
            lambdas: Array1::<f64>::zeros(0),
            covariance_conditional,
            stable_penalty_term: 0.0,
            // No penalties and no smoothing parameters: the reported objective
            // is the plain negative log-likelihood at the MLE.
            penalized_objective: -log_likelihood,
            outer_iterations: 0,
            outer_gradient_norm: Some(0.0),
            criterion_certificate: None,
            inner_cycles: 0,
            outer_converged: true,
            geometry,
            precomputed_edf: None,
            joint_log_lambdas: None,
        },
        &assembly_specs,
    )
    .map_err(String::from)
}

/// Variant that also returns the offset-channel residuals + curvatures at the
/// converged β̂. We have to extract these *before* `finalize_survival_location_scale_fit`
/// runs, because the location-scale finalizer empties `UnifiedFitResult::block_states`
/// (see `survival_fit_from_parts` — `block_states: Vec::new()`), and the family's
/// `offset_channel_geometry` method needs the raw, populated per-block state.
fn fit_survival_location_scale_with_geometry_authority(
    spec: SurvivalLocationScaleSpec,
    certified_outer: Option<(
        &Array1<f64>,
        &gam_solve::rho_optimizer::CertifiedOuterResult,
        Option<&CustomFamilyWarmStart>,
    )>,
) -> Result<(UnifiedFitResult, SurvivalLocationScaleConvergedGeometry), String> {
    let prepared = prepare_survival_location_scale_model(&spec)?;
    let options = survival_blockwise_fit_options(&spec);
    // Fully reduced constant-scale PARAMETRIC AFT regime (issue #736/#735/#721):
    // every block is parametric and unpenalized, so REML/LAML smoothing
    // selection is vacuous and the coupled exact-joint REML optimizer is the
    // wrong tool — it oscillates and never certifies stationarity on this tiny
    // unpenalized likelihood. Route directly to a damped, line-searched joint
    // Newton MLE (converges in a handful of iterations like survreg/lifelines),
    // then assemble the identical `UnifiedFitResult` so finalize / predict /
    // CRPS / the `offset_channel_geometry` consumer all work unchanged. Any
    // genuinely flexible or penalized survival LS fit keeps the full coupled
    // path below.
    let fit = if prepared.is_reduced_parametric_aft() {
        if certified_outer.is_some() {
            return Err(SurvivalLocationScaleError::InternalInvariant {
                reason: "a reduced unpenalized AFT fit cannot carry an optimized smoothing certificate"
                    .to_string(),
            }
            .into());
        }
        fit_reduced_parametric_aft(&prepared, &options)?
    } else if let Some((theta, outer, warm_start)) = certified_outer {
        let exact_options = survival_location_scale_exact_outer_options(
            &options,
            &crate::row_kernel::RowSet::All,
        );
        fit_custom_family_fixed_log_lambdas_from_outer(
            &prepared.family,
            &prepared.blockspecs,
            &exact_options,
            warm_start,
            theta,
            outer,
        )?
    } else {
        fit_custom_family(&prepared.family, &prepared.blockspecs, &options)?
    };
    // `finalize_survival_location_scale_fit` indexes the populated block
    // states directly, so an empty result from the inner fit violates this
    // path's contract and must fail before finalization.
    if fit.block_states.is_empty() {
        return Err(SurvivalLocationScaleError::InternalInvariant {
            reason: "fit_survival_location_scale_with_geometry: fit_custom_family returned a fit \
                     with empty block_states"
                .to_string(),
        }
        .into());
    }
    let (residuals, curvatures) = prepared.family.offset_channel_geometry(&fit.block_states)?;
    let link_param_data_fit_gradient = prepared
        .family
        .link_param_data_fit_gradient(&fit.block_states)?;
    let finalized = finalize_survival_location_scale_fit(&prepared, &fit)?;
    Ok((
        finalized,
        (residuals, curvatures, link_param_data_fit_gradient),
    ))
}

pub(crate) fn fit_survival_location_scale_with_geometry(
    spec: SurvivalLocationScaleSpec,
) -> Result<(UnifiedFitResult, SurvivalLocationScaleConvergedGeometry), String> {
    fit_survival_location_scale_with_geometry_authority(spec, None)
}

fn fit_survival_location_scale_with_geometry_from_outer(
    spec: SurvivalLocationScaleSpec,
    theta: &Array1<f64>,
    outer: &gam_solve::rho_optimizer::CertifiedOuterResult,
    warm_start: Option<&CustomFamilyWarmStart>,
) -> Result<(UnifiedFitResult, SurvivalLocationScaleConvergedGeometry), String> {
    fit_survival_location_scale_with_geometry_authority(
        spec,
        Some((theta, outer, warm_start)),
    )
}

/// Converged-fit geometry returned alongside the finalized location-scale fit:
/// the offset-channel residuals + curvatures (for the baseline-θ gradient/Hessian)
/// and the exact inverse-link data-fit θ-gradient (`None` when the link has no
/// free parameters).
pub(crate) type SurvivalLocationScaleConvergedGeometry = (
    OffsetChannelResiduals,
    OffsetChannelCurvatures,
    Option<Array1<f64>>,
);

pub(crate) fn select_survival_link_wiggle_basis_from_pilot(
    pilot: &SurvivalLocationScaleTermFitResult,
    wiggle_cfg: &WiggleBlockConfig,
    wiggle_penalty_orders: &[usize],
) -> Result<SelectedWiggleBasis, String> {
    let eta_threshold = pilot
        .threshold_design
        .apply(pilot.fit.beta_threshold().view())
        .map_err(|error| error.to_string())?;
    let eta_log_sigma = pilot
        .log_sigma_design
        .apply(pilot.fit.beta_log_sigma().view())
        .map_err(|error| error.to_string())?;
    let q_seed = Array1::from_iter(
        eta_threshold
            .iter()
            .zip(eta_log_sigma.iter())
            .map(|(&threshold, &ls)| survival_q0_from_eta(threshold, ls)),
    );
    select_wiggle_basis_from_seed(q_seed.view(), wiggle_cfg, wiggle_penalty_orders)
}

pub(crate) fn linkwiggle_block_input_from_selected_basis(
    selected_wiggle_basis: SelectedWiggleBasis,
) -> LinkWiggleBlockInput {
    let crate::wiggle::SelectedWiggleBasis {
        block,
        knots,
        degree,
        ..
    } = selected_wiggle_basis;
    let crate::parameter_block::ParameterBlockInput {
        design,
        penalties,
        nullspace_dims,
        initial_log_lambdas,
        initial_beta,
        ..
    } = block;
    LinkWiggleBlockInput {
        design,
        knots,
        degree,
        penalties,
        nullspace_dims,
        initial_log_lambdas,
        initial_beta,
    }
}

pub(crate) fn fit_survival_location_scale_terms_with_selected_wiggle(
    data: ndarray::ArrayView2<'_, f64>,
    mut spec: SurvivalLocationScaleTermSpec,
    selected_wiggle_basis: SelectedWiggleBasis,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<SurvivalLocationScaleTermFitResult, String> {
    spec.linkwiggle_block = Some(linkwiggle_block_input_from_selected_basis(
        selected_wiggle_basis,
    ));
    fit_survival_location_scale_terms(data, spec, kappa_options)
}

pub(crate) fn fit_survival_location_scale_terms(
    data: ndarray::ArrayView2<'_, f64>,
    spec: SurvivalLocationScaleTermSpec,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<SurvivalLocationScaleTermFitResult, String> {
    let threshold_boot_design =
        build_term_collection_design(data, &spec.thresholdspec).map_err(|e| e.to_string())?;
    let log_sigma_boot_design =
        build_term_collection_design(data, &spec.log_sigmaspec).map_err(|e| e.to_string())?;
    let threshold_bootspec =
        freeze_term_collection_from_design(&spec.thresholdspec, &threshold_boot_design)
            .map_err(|e| e.to_string())?;
    let log_sigma_bootspec =
        freeze_term_collection_from_design(&spec.log_sigmaspec, &log_sigma_boot_design)
            .map_err(|e| e.to_string())?;

    let threshold_boot_derivs = build_survival_covariate_block_psi_derivatives(
        data,
        &threshold_bootspec,
        &threshold_boot_design,
        &spec.threshold_template,
    )?;
    let log_sigma_boot_derivs = build_survival_covariate_block_psi_derivatives(
        data,
        &log_sigma_bootspec,
        &log_sigma_boot_design,
        &spec.log_sigma_template,
    )?;
    let analytic_joint_gradient_available =
        threshold_boot_derivs.is_some() && log_sigma_boot_derivs.is_some();
    let analytic_joint_hessian_available = threshold_boot_derivs
        .as_ref()
        .is_some_and(|derivs| survival_psi_derivatives_support_exact_joint_hessian(derivs))
        && log_sigma_boot_derivs
            .as_ref()
            .is_some_and(|derivs| survival_psi_derivatives_support_exact_joint_hessian(derivs));

    let wiggle_rho0 = spec
        .linkwiggle_block
        .as_ref()
        .and_then(|w| w.initial_log_lambdas.clone())
        .unwrap_or_else(|| Array1::zeros(0));
    // Outer time-warp ρ count. In the reduced constant-scale-AFT regime the
    // time block collapses to its unpenalized affine null space (see
    // `prepare_identified_time_block`), so it carries NO smoothing parameter and
    // must contribute no ρ coordinate to the outer REML search — otherwise the
    // outer optimizer spends a full inner blockwise fit per step crawling a
    // dead-flat time-smoothing dimension until `outer_max_iter` (issue
    // #736/#735/#721). `survival_time_rho_count` is the single source of truth
    // shared with the inner block preparation so the two layouts always agree.
    let constant_scale = log_sigma_boot_design.penalties.is_empty();
    let protected_timewiggle_cols = spec.timewiggle_block.as_ref().map_or(0, |w| w.ncols);
    // Full warp exit design + log-t at exit (same `SURVIVAL_TIME_FLOOR` map the
    // inner `prepare_survival_location_scale_model` uses), so the OUTER ρ count
    // and the reduced-parametric dispatch consult the SAME log-t-baseline
    // collapse predicate as the inner block preparation and stay in lock-step.
    let time_design_exit = spec.time_block.design_exit.to_dense();
    let log_time_exit = spec.age_exit.mapv(|t| {
        t.max(crate::survival::construction::SURVIVAL_TIME_FLOOR)
            .ln()
    });
    let k_time = survival_time_rho_count(
        &spec.time_block.penalties,
        spec.time_block.design_exit.ncols(),
        constant_scale,
        protected_timewiggle_cols,
        &time_design_exit,
        log_time_exit.view(),
    );
    let time_rho0 = if k_time == 0 {
        // Reduced parametric AFT: the time block is unpenalized, so any caller-
        // supplied per-penalty time seed is irrelevant and the outer search
        // carries no time coordinate.
        Array1::<f64>::zeros(0)
    } else {
        spec.time_block
            .initial_log_lambdas
            .clone()
            .unwrap_or_else(|| Array1::zeros(k_time))
    };
    // Reduced parametric-AFT regime (issue #736/#735/#721): when the location
    // (and scale) carry only full-rank parametric shrinkage ridges
    // (`nullspace_dim == 0`, e.g. the linear-term `LinearTermRidge` on `age`)
    // and the time-warp has reduced to affine with no wiggle, those ridges are
    // dropped — the inner `prepare_survival_location_scale_model` applies the
    // IDENTICAL predicate to the same boot-design penalties, so the inner and
    // outer ρ counts stay provably in lock-step. Dropping them takes the
    // threshold/log_sigma ρ counts to 0, so the outer search carries ZERO
    // coordinates and the fit is a single direct unpenalized parametric-AFT MLE
    // (`fit_parametric_aft_direct_mle`) — milliseconds, and numerically the
    // `survreg`/`lifelines` MLE — instead of crawling a flat, vacuous ρ surface.
    let drop_parametric_ridges = survival_reduced_parametric_aft_regime(
        &spec.time_block.penalties,
        spec.time_block.design_exit.ncols(),
        constant_scale,
        protected_timewiggle_cols,
        &threshold_boot_design.nullspace_dims,
        threshold_boot_design.penalties.len(),
        &log_sigma_boot_design.nullspace_dims,
        log_sigma_boot_design.penalties.len(),
        spec.linkwiggle_block.is_some(),
        &time_design_exit,
        log_time_exit.view(),
    );
    let layout = SurvivalLambdaLayout::new(
        k_time,
        if drop_parametric_ridges {
            0
        } else {
            threshold_boot_design.penalties.len()
        },
        if drop_parametric_ridges {
            0
        } else {
            log_sigma_boot_design.penalties.len()
        },
        wiggle_rho0.len(),
    );
    // This is the same structural predicate consumed by
    // `PreparedSurvivalLocationScaleModel::is_reduced_parametric_aft`: every
    // smoothing coordinate is absent, and neither time nor link wiggles exist.
    // Persist the result while the fit topology is still available; saved
    // replay must not attempt to recover it from fitted coefficient values.
    let time_parameterization =
        if layout.total() == 0 && protected_timewiggle_cols == 0 && spec.linkwiggle_block.is_none()
        {
            SurvivalLocationScaleTimeParameterization::ReducedParametricAft
        } else {
            SurvivalLocationScaleTimeParameterization::MonotoneWarp
        };
    let mut rho0 = Array1::<f64>::zeros(layout.total());
    if layout.k_time > 0 {
        if time_rho0.len() != layout.k_time {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "survival time initial_log_lambdas length mismatch: got {}, expected {}",
                    time_rho0.len(),
                    layout.k_time
                ),
            }
            .into());
        }
        let range = layout.time_range();
        rho0.slice_mut(s![range.start..range.end])
            .assign(&time_rho0);

        // Parametric-AFT regime: strong-smoothing seed for the time-warp
        // penalty.
        //
        // When the scale block carries no penalties (a single constant σ) the
        // residual distribution `z = (h(t) - η)/σ` is a fixed parametric shape
        // with a single global spread, so the data identifies the baseline
        // *only* through the affine `1 + log t` transform that IS the parametric
        // AFT transform. The flexible deviation of the monotone I-spline
        // time-warp `h(t)` away from its penalty nullspace (that affine
        // baseline) is then statistically unidentified, and the REML/LAML
        // profile in the time smoothing parameter is a long flat ridge that
        // climbs monotonically toward strong smoothing.
        //
        // This unidentifiability is a property of the SCALE block alone, not of
        // the mean. A smooth mean `~ s(z)` adds flexibility in *covariate*
        // space — it bends η as a function of the covariates — but it carries no
        // information about the *time* baseline shape, because the time-warp
        // enters only through `h(t)` and is identified solely by how the event
        // times distribute against a single global σ. So whether the mean is
        // rigid (`~ age`) or smooth (`~ s(z)`), a constant-scale Gaussian AFT
        // leaves the time-warp's non-affine deviation unidentified and the time
        // ridge flat. Gating the seed on `rigid_mean` therefore wrongly excluded
        // the smooth-mean constant-scale case (#735), whose threshold block
        // carries penalties: it fell through to the weak default time seed and
        // its exact-joint outer search crawled the flat time ridge forever.
        //
        // Seeding the weak default (`time_smooth_lambda ≈ 1e-2`) drops the
        // inner REML search into the *interior* of that ridge, where it crawls
        // toward the strong-smoothing boundary one short, ill-conditioned step
        // at a time and never terminates in reasonable time (#736, #735, #721).
        //
        // The previous fix seeded the *interior* point ρ = 8. That did NOT cure
        // the hang: the inner blockwise REML optimizer re-optimizes ρ_time
        // freely from its seed against an inner per-coordinate ρ box bound of
        // ±10 (`fit_custom_family_with_rho_prior`'s `.with_rho_bound(10.0)`).
        // λ = exp(8) ≈ 3·10³ already sits INSIDE the "dead-flat region" that
        // very bound exists to fence off (see the `with_rho_bound` rationale in
        // `custom_family.rs`): with a flat REML gradient and near-singular
        // curvature there, the optimizer wanders between ρ = 8 and the ρ = 10
        // boundary one micro-step at a time and the retry-stall detector spins
        // on the flat surface — producing the >200s no-iteration-log hang. A
        // seed strictly interior to the box can never certify, because the
        // unconstrained projected-gradient stationarity test it would need is
        // exactly the test the flat ridge makes ill-posed.
        //
        // Seed instead at the inner ρ box bound itself. At the bound the
        // box-constraint KKT condition (the REML gradient pushes further into
        // strong smoothing, against an active bound) certifies stationarity
        // *immediately* at iteration 0 for the time coordinate — there is no
        // interior flat region left to wander, because the optimizer is pinned
        // at the wall. λ = exp(10) ≈ 22k is the affine-nullspace limit (the
        // bound's own rationale calls this "statistically indistinguishable
        // from shrunk to nullspace"), i.e. exactly the parametric-AFT affine
        // baseline. This is a regime-specific *initialization*, not a cap or a
        // tolerance change: the I-spline basis dimensions are untouched, so any
        // independent rebuild of the time basis (predictor reconstruction) is
        // unaffected, and a genuinely flexible regime never reaches this branch.
        //
        // The seed is gated on `constant_scale` ONLY. The genuinely flexible
        // time-warp regime is a smooth scale (`noise_formula = s(...)`): a
        // varying σ lets the residual spread change with covariates, which DOES
        // supply identifying information for a non-affine baseline, so those
        // fits carry log_sigma penalties and keep the full weak-seed search.
        // Smooth-mean penalties on the threshold block are still selected
        // normally — only the TIME-WARP block's seed changes here.
        //
        // NOTE: reaching here with `constant_scale == true` already implies the
        // affine reduction did NOT fire (otherwise `k_time == 0` and this whole
        // `if layout.k_time > 0` arm is skipped — the reduced block is
        // unpenalized and carries no ρ at all). This seed therefore only covers
        // the residual constant-scale case where the time penalty has no affine
        // null space to collapse onto (or a timewiggle keeps the flexibility),
        // pinning that surviving time ρ at the strong-smoothing limit.
        if constant_scale {
            // ρ = 10 == the inner blockwise solver's per-coordinate ρ box bound
            // (`custom_family.rs` `with_rho_bound(10.0)`). Seeding AT the bound
            // (not interior, as the prior ρ = 8 seed did) makes the box
            // constraint active from iteration 0, so outer stationarity
            // certifies immediately instead of crawling the flat ridge.
            const PARAMETRIC_AFT_TIME_RHO_SEED: f64 = 10.0;
            let mut time_seed = rho0.slice_mut(s![range.start..range.end]);
            for v in time_seed.iter_mut() {
                *v = PARAMETRIC_AFT_TIME_RHO_SEED;
            }
        }
    }
    // Warm-start: inject converged ρ seeds from a previous fit if supplied. The values are
    // clamped to the outer ρ bounds (±12) so that "dead" coordinates returned at extremes
    // by a prior fit don't crowd the optimizer's box bound on the next probe.
    if layout.k_threshold > 0
        && let Some(seed) = spec.initial_threshold_log_lambdas.as_ref()
    {
        if seed.len() != layout.k_threshold {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "survival threshold initial_log_lambdas length mismatch: got {}, expected {}",
                    seed.len(),
                    layout.k_threshold
                ),
            }
            .into());
        }
        let range = layout.threshold_range();
        let mut slice = rho0.slice_mut(s![range.start..range.end]);
        for (dst, src) in slice.iter_mut().zip(seed.iter()) {
            if src.is_finite() {
                *dst = src.clamp(-12.0, 12.0);
            }
        }
    }
    if layout.k_log_sigma > 0
        && let Some(seed) = spec.initial_log_sigma_log_lambdas.as_ref()
    {
        if seed.len() != layout.k_log_sigma {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "survival log_sigma initial_log_lambdas length mismatch: got {}, expected {}",
                    seed.len(),
                    layout.k_log_sigma
                ),
            }
            .into());
        }
        let range = layout.log_sigma_range();
        let mut slice = rho0.slice_mut(s![range.start..range.end]);
        for (dst, src) in slice.iter_mut().zip(seed.iter()) {
            if src.is_finite() {
                *dst = src.clamp(-12.0, 12.0);
            }
        }
    }
    if layout.k_wiggle > 0 {
        let range = layout.wiggle_range();
        rho0.slice_mut(s![range.start..range.end])
            .assign(&wiggle_rho0);
    }
    let joint_setup = build_survival_two_block_exact_joint_setup(
        data.view(),
        &spec.thresholdspec,
        &spec.log_sigmaspec,
        rho0,
        kappa_options,
    );

    let time_beta_hint = std::cell::RefCell::new(spec.time_block.initial_beta.clone());
    let threshold_beta_hint = std::cell::RefCell::new(None::<Array1<f64>>);
    let log_sigma_beta_hint = std::cell::RefCell::new(None::<Array1<f64>>);
    let wiggle_beta_hint = std::cell::RefCell::new(
        spec.linkwiggle_block
            .as_ref()
            .and_then(|w| w.initial_beta.clone()),
    );
    let exact_warm_start = std::cell::RefCell::new(None::<CustomFamilyWarmStart>);
    // Outer ρ-cache β-seed staging slot. See BMS/SMS for the contract: stash
    // the flat β here on cache hit, promote to a real `CustomFamilyWarmStart`
    // once per-block widths are known from `prepare_survival_location_scale_model`.
    let pending_beta_seed = std::cell::RefCell::new(None::<Array1<f64>>);
    // Stash the geometry from the most recent inner fit. Updated on every
    // value-closure call by the spatial optimizer; the last one written
    // corresponds to the converged outer point. This avoids redoing
    // `prepare_survival_location_scale_model` + a second fit pass after the
    // optimizer returns, and (critically) avoids the post-finalize
    // `block_states` wipe that would make the geometry call error out.
    let last_geometry: std::cell::RefCell<Option<SurvivalLocationScaleConvergedGeometry>> =
        std::cell::RefCell::new(None);

    let build_spec = |rho: &Array1<f64>,
                      _: &TermCollectionSpec,
                      _: &TermCollectionSpec,
                      threshold_design: &TermCollectionDesign,
                      log_sigma_design: &TermCollectionDesign|
     -> Result<SurvivalLocationScaleSpec, String> {
        layout.validate_rho(rho, "survival term fit")?;
        let time_beta = filtered_initial_beta(
            time_beta_hint.borrow().as_ref(),
            spec.time_block.design_exit.ncols(),
        );
        // In the reduced parametric-AFT regime the layout carries no
        // threshold/log_sigma ρ (`drop_parametric_ridges`), yet the boot design
        // still carries the parametric ridge as a penalty. Passing the empty
        // layout slice as the seed would mismatch that penalty count; instead
        // pass `None` so the block defaults to a length-matched zero seed, which
        // the inner `prepare_survival_location_scale_model` then drops along with
        // the ridge. Outside the regime the layout slice length equals the
        // design penalty count, so `Some(slice)` is exact.
        let threshold_block = build_survival_covariate_block_from_design(
            threshold_design,
            &spec.threshold_template,
            &spec.threshold_offset,
            if drop_parametric_ridges {
                None
            } else {
                Some(layout.threshold_from(rho))
            },
            filtered_initial_beta(
                threshold_beta_hint.borrow().as_ref(),
                match &spec.threshold_template {
                    SurvivalCovariateTermBlockTemplate::Static => threshold_design.design.ncols(),
                    SurvivalCovariateTermBlockTemplate::TimeVarying {
                        time_basis_exit, ..
                    } => threshold_design.design.ncols() * time_basis_exit.ncols(),
                },
            ),
        )?;
        let log_sigma_block = build_survival_covariate_block_from_design(
            log_sigma_design,
            &spec.log_sigma_template,
            &spec.log_sigma_offset,
            if drop_parametric_ridges {
                None
            } else {
                Some(layout.log_sigma_from(rho))
            },
            filtered_initial_beta(
                log_sigma_beta_hint.borrow().as_ref(),
                match &spec.log_sigma_template {
                    SurvivalCovariateTermBlockTemplate::Static => log_sigma_design.design.ncols(),
                    SurvivalCovariateTermBlockTemplate::TimeVarying {
                        time_basis_exit, ..
                    } => log_sigma_design.design.ncols() * time_basis_exit.ncols(),
                },
            ),
        )?;
        let linkwiggle_block = spec
            .linkwiggle_block
            .as_ref()
            .map(|wiggle| LinkWiggleBlockInput {
                design: wiggle.design.clone(),
                knots: wiggle.knots.clone(),
                degree: wiggle.degree,
                penalties: wiggle.penalties.clone(),
                nullspace_dims: wiggle.nullspace_dims.clone(),
                initial_log_lambdas: layout.wiggle_from(rho),
                initial_beta: filtered_initial_beta(
                    wiggle_beta_hint.borrow().as_ref(),
                    wiggle.design.ncols(),
                ),
            });
        Ok(SurvivalLocationScaleSpec {
            age_entry: spec.age_entry.clone(),
            age_exit: spec.age_exit.clone(),
            event_target: spec.event_target.clone(),
            weights: spec.weights.clone(),
            inverse_link: spec.inverse_link.clone(),
            derivative_guard: spec.derivative_guard,
            max_iter: spec.max_iter,
            tol: spec.tol,
            time_block: TimeBlockInput {
                design_entry: spec.time_block.design_entry.clone(),
                design_exit: spec.time_block.design_exit.clone(),
                design_derivative_exit: spec.time_block.design_derivative_exit.clone(),
                offset_entry: spec.time_block.offset_entry.clone(),
                offset_exit: spec.time_block.offset_exit.clone(),
                derivative_offset_exit: spec.time_block.derivative_offset_exit.clone(),
                time_monotonicity: spec.time_block.time_monotonicity,
                penalties: spec.time_block.penalties.clone(),
                nullspace_dims: spec.time_block.nullspace_dims.clone(),
                // `initial_log_lambdas` is the per-penalty seed for THIS block's
                // (still un-reduced) `penalties`, validated against that list's
                // length by `validate_time_block`. In the flexible regime the
                // outer layout carries one time ρ per penalty, so `time_from`
                // returns exactly `penalties.len()` entries. In the reduced
                // constant-scale-AFT regime (`layout.k_time == 0`) the outer
                // search carries NO time coordinate, so `time_from` is empty —
                // but `penalties` here is the un-reduced length-`k` list (the
                // collapse to the unpenalized affine null space happens later,
                // inside `prepare_identified_time_block`). Emitting the empty
                // outer slice against the un-reduced penalties would make
                // `initial_log_lambdas.len() (0) != penalties.len() (k)` and
                // trip the block's length-consistency check. The downstream
                // reduction re-derives (and drops) this seed for the collapsed
                // block, so any length-`k` value is fine here; carry the
                // caller's original per-penalty seed to stay length-consistent
                // with the un-reduced penalty list (issue #736/#735/#721).
                initial_log_lambdas: if layout.k_time > 0 {
                    Some(layout.time_from(rho))
                } else {
                    spec.time_block.initial_log_lambdas.clone()
                },
                initial_beta: time_beta,
            },
            threshold_block,
            log_sigma_block,
            timewiggle_block: spec.timewiggle_block.clone(),
            linkwiggle_block,
            cache_session: spec.cache_session.clone(),
            cache_mirror_sessions: spec.cache_mirror_sessions.clone(),
        })
    };

    let threshold_terms = spatial_length_scale_term_indices(&spec.thresholdspec);
    let log_sigma_terms = spatial_length_scale_term_indices(&spec.log_sigmaspec);
    // Survival location-scale is a multi-block family with β-dependent
    // joint Hessian: disable EFS/HybridEFS at plan time so the outer never
    // pays for a stalled fixed-point attempt before landing on BFGS.
    let outer_policy = {
        let capability = if analytic_joint_hessian_available {
            crate::custom_family::ExactOuterDerivativeOrder::Second
        } else {
            crate::custom_family::ExactOuterDerivativeOrder::First
        };
        // Honest per-eval work model so the route planner has a real cost
        // signal for the exact gradient / joint-Hessian routes (#721). The
        // survival likelihood couples every block, so a single coefficient
        // Hessian assembly costs `n · (Σ p_b)²` (matching
        // `joint_coupled_coefficient_hessian_cost`), and each outer
        // coordinate (every penalty ρ, spatial log-κ, and auxiliary axis)
        // propagates one analytic directional derivative through the inner
        // solve. Leaving these at 0 left the planner blind and it never
        // down-routed the heavyweight exact-joint path.
        let n_work = data.nrows() as u64;
        let p_total = (spec.time_block.design_exit.ncols()
            + threshold_boot_design.design.ncols()
            + log_sigma_boot_design.design.ncols()
            + spec
                .linkwiggle_block
                .as_ref()
                .map_or(0, |w| w.design.ncols())) as u64;
        let hess_cost = n_work.saturating_mul(p_total.saturating_mul(p_total));
        let grad_cost = hess_cost / 2;
        let outer_coords =
            (joint_setup.rho_dim() + joint_setup.log_kappa_dim() + joint_setup.auxiliary_dim())
                .max(1) as u128;
        let predicted_hessian_work = (hess_cost as u128).saturating_mul(outer_coords);
        let predicted_gradient_work = (grad_cost as u128).saturating_mul(outer_coords);
        crate::custom_family::OuterDerivativePolicy {
            capability,
            predicted_gradient_work,
            predicted_hessian_work,
            // Survival location-scale consumes `outer_score_subsample` on its
            // outer-only LL, joint-Hessian, and ψ workspace paths.
            subsample_capable: true,
        }
    };
    let solved = optimize_spatial_length_scale_exact_joint(
        data,
        &[spec.thresholdspec.clone(), spec.log_sigmaspec.clone()],
        &[threshold_terms, log_sigma_terms],
        kappa_options,
        &joint_setup,
        crate::seeding::SeedRiskProfile::Survival,
        analytic_joint_gradient_available,
        analytic_joint_hessian_available,
        true,
        None,
        outer_policy,
        |theta, specs: &[TermCollectionSpec], designs: &[TermCollectionDesign], provenance| {
            let rho = theta.slice(s![..joint_setup.rho_dim()]).to_owned();
            let assembled = build_spec(
                &rho,
                &specs[0],
                &specs[1],
                &designs[0],
                &designs[1],
            )?;
            let (fit, geom) = match provenance {
                SpatialFitProvenance::NoOuterOptimization => {
                    fit_survival_location_scale_with_geometry(assembled)?
                }
                SpatialFitProvenance::Certified(outer) => {
                    let warm_start = exact_warm_start.borrow();
                    fit_survival_location_scale_with_geometry_from_outer(
                        assembled,
                        theta,
                        outer,
                        warm_start.as_ref(),
                    )?
                }
            };
            time_beta_hint.replace(Some(fit.beta_time()));
            threshold_beta_hint.replace(Some(fit.beta_threshold()));
            log_sigma_beta_hint.replace(Some(fit.beta_log_sigma()));
            wiggle_beta_hint.replace(fit.beta_link_wiggle());
            *last_geometry.borrow_mut() = Some(geom);
            Ok(fit)
        },
        |theta,
         specs: &[TermCollectionSpec],
         designs: &[TermCollectionDesign],
         eval_mode,
         row_set: &crate::row_kernel::RowSet| {
            use gam_problem::EvalMode;
            if !analytic_joint_gradient_available {
                return Err(SurvivalLocationScaleError::InvalidConfiguration { reason: "analytic spatial psi derivatives are unavailable for survival exact two-block path"
                        .to_string(), }.into());
            }
            let rho = theta.slice(s![..joint_setup.rho_dim()]).to_owned();
            let assembled = build_spec(&rho, &specs[0], &specs[1], &designs[0], &designs[1])?;
            let prepared = prepare_survival_location_scale_model(&assembled)?;
            if let Some(beta_seed) = pending_beta_seed.borrow_mut().take() {
                let widths: Vec<usize> = prepared
                    .blockspecs
                    .iter()
                    .map(|b| b.design.ncols())
                    .collect();
                match CustomFamilyWarmStart::from_cached_beta(&widths, &beta_seed) {
                    Ok(ws) => {
                        exact_warm_start.replace(Some(ws));
                    }
                    Err(e) => {
                        log::warn!(
                            "[survival-LS] outer ρ-cache β-warm-start rejected: {e}; falling back to cold β"
                        );
                    }
                }
            }
            let threshold_derivs = build_survival_covariate_block_psi_derivatives(
                data,
                &specs[0],
                &designs[0],
                &spec.threshold_template,
            )?
            .ok_or_else(|| "missing survival threshold spatial psi derivatives".to_string())?;
            let log_sigma_derivs = build_survival_covariate_block_psi_derivatives(
                data,
                &specs[1],
                &designs[1],
                &spec.log_sigma_template,
            )?
            .ok_or_else(|| "missing survival log-sigma spatial psi derivatives".to_string())?;
            let mut derivative_blocks = vec![Vec::new(), threshold_derivs, log_sigma_derivs];
            if prepared.family.x_link_wiggle.is_some() {
                derivative_blocks.push(Vec::new());
            }
            // If the caller asked for a Hessian but the family can't provide
            // an analytic one, downgrade the request to ValueAndGradient.
            // ValueOnly stays ValueOnly so cost-only line-search probes skip
            // gradient assembly entirely.
            let effective_mode = match eval_mode {
                EvalMode::ValueGradientHessian if !analytic_joint_hessian_available => {
                    EvalMode::ValueAndGradient
                }
                other => other,
            };
            let eval_options = survival_location_scale_exact_outer_options(
                &survival_blockwise_fit_options(&assembled),
                row_set,
            );
            let eval = evaluate_custom_family_joint_hyper(
                &prepared.family,
                &prepared.blockspecs,
                &eval_options,
                &rho,
                &derivative_blocks,
                exact_warm_start.borrow().as_ref(),
                effective_mode,
            )
            .map_err(|e| e.to_string())?;
            exact_warm_start.replace(Some(eval.warm_start.clone()));
            if !eval.inner_converged {
                return Err(
                    "survival location-scale exact joint inner solve did not converge".to_string(),
                );
            }
            Ok((eval.objective, eval.gradient, eval.outer_hessian))
        },
        |theta,
         specs: &[TermCollectionSpec],
         designs: &[TermCollectionDesign],
         row_set: &crate::row_kernel::RowSet| {
            if !analytic_joint_gradient_available {
                return Err(SurvivalLocationScaleError::InvalidConfiguration { reason: "analytic spatial psi derivatives are unavailable for survival exact two-block path"
                        .to_string(), }.into());
            }
            let rho = theta.slice(s![..joint_setup.rho_dim()]).to_owned();
            let assembled = build_spec(&rho, &specs[0], &specs[1], &designs[0], &designs[1])?;
            let prepared = prepare_survival_location_scale_model(&assembled)?;
            if let Some(beta_seed) = pending_beta_seed.borrow_mut().take() {
                let widths: Vec<usize> = prepared
                    .blockspecs
                    .iter()
                    .map(|b| b.design.ncols())
                    .collect();
                match CustomFamilyWarmStart::from_cached_beta(&widths, &beta_seed) {
                    Ok(ws) => {
                        exact_warm_start.replace(Some(ws));
                    }
                    Err(e) => {
                        log::warn!(
                            "[survival-LS] outer ρ-cache β-warm-start rejected (efs): {e}; falling back to cold β"
                        );
                    }
                }
            }
            let threshold_derivs = build_survival_covariate_block_psi_derivatives(
                data,
                &specs[0],
                &designs[0],
                &spec.threshold_template,
            )?
            .ok_or_else(|| "missing survival threshold spatial psi derivatives".to_string())?;
            let log_sigma_derivs = build_survival_covariate_block_psi_derivatives(
                data,
                &specs[1],
                &designs[1],
                &spec.log_sigma_template,
            )?
            .ok_or_else(|| "missing survival log-sigma spatial psi derivatives".to_string())?;
            let mut derivative_blocks = vec![Vec::new(), threshold_derivs, log_sigma_derivs];
            if prepared.family.x_link_wiggle.is_some() {
                derivative_blocks.push(Vec::new());
            }
            let eval_options = survival_location_scale_exact_outer_options(
                &survival_blockwise_fit_options(&assembled),
                row_set,
            );
            let eval = evaluate_custom_family_joint_hyper_efs(
                &prepared.family,
                &prepared.blockspecs,
                &eval_options,
                &rho,
                &derivative_blocks,
                exact_warm_start.borrow().as_ref(),
            )
            .map_err(|e| e.to_string())?;
            exact_warm_start.replace(Some(eval.warm_start.clone()));
            if !eval.inner_converged {
                return Err(
                    "survival location-scale exact joint EFS inner solve did not converge"
                        .to_string(),
                );
            }
            Ok(eval.efs_eval)
        },
        crate::marginal_slope_shared::make_beta_seed_validator(&pending_beta_seed),
    )?;

    let mut resolved_specs = solved.resolved_specs;
    let mut designs = solved.designs;
    // Fast path: the value closure stashed the offset geometry from the
    // *converged* inner fit (computed pre-finalize while `block_states` was
    // still populated). No extra family prep / refit needed here.
    //
    // Fallback: if for some reason no value-closure call ran (or the
    // optimizer's last evaluation happened through the gradient/EFS path
    // without touching the value closure at the final ρ), recompute by
    // redoing one inner fit at the final ρ̂. This pays an extra fit only when
    // the cache is cold — the common location-scale path always populates it.
    let (baseline_offset_residuals, baseline_offset_curvatures, link_param_data_fit_gradient) =
        match last_geometry.borrow_mut().take() {
            Some(geom) => geom,
            None => {
                let rho_final = solved.fit.log_lambdas.clone();
                let final_assembled = build_spec(
                    &rho_final,
                    &resolved_specs[0],
                    &resolved_specs[1],
                    &designs[0],
                    &designs[1],
                )?;
                match fit_survival_location_scale_with_geometry(final_assembled) {
                    Ok((_refit, geom)) => geom,
                    Err(e) => return Err(e),
                }
            }
        };
    Ok(SurvivalLocationScaleTermFitResult {
        fit: solved.fit,
        time_parameterization,
        threshold_time_basis: spec.threshold_template.resolved_time_basis().cloned(),
        log_sigma_time_basis: spec.log_sigma_template.resolved_time_basis().cloned(),
        resolved_thresholdspec: resolved_specs.remove(0),
        resolved_log_sigmaspec: resolved_specs.remove(0),
        threshold_design: designs.remove(0),
        log_sigma_design: designs.remove(0),
        baseline_offset_residuals,
        baseline_offset_curvatures,
        link_param_data_fit_gradient,
    })
}
