use super::*;
use crate::fit_orchestration::FitFailure;

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
/// geometry the coupled survival path stores. Exact-Newton curvature has no
/// single diagonal row representation, so `working` is explicitly `None`. The
/// shared [`crate::custom_family::blockwise_fit_from_parts`] assembler then
/// computes EDF (= parameter count, since unpenalized) and the inference block
/// exactly as for any custom-family fit.
pub(crate) fn fit_reduced_parametric_aft(
    prepared: &PreparedSurvivalLocationScaleModel,
    options: &BlockwiseFitOptions,
) -> Result<UnifiedFitResult, FitFailure> {
    use gam_linalg::faer_ndarray::FaerCholesky;

    let specs = &prepared.blockspecs;
    let (states, log_likelihood, h) = prepared.family.fit_parametric_aft_direct_mle(
        specs,
        options.inner_max_cycles.max(1),
        options.inner_tol,
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
        constrained_posterior: None,
        working: None,
    });

    // The block states carry their η in the family's native row layout — the
    // stacked `[exit; entry; deriv]` channels (`solver_design().nrows()` rows)
    // for the time block, exactly as `refresh_all_block_etas` produces and as
    // the family's `validate_joint_states` / `offset_channel_geometry` require.
    // `blockwise_fit_from_parts` validates each block's `η.len()` against
    // `spec.solver_design().nrows()`, which already resolves through
    // `stacked_design`, so the specs are handed over untouched.
    //
    // This used to flatten `solver_design()` into `design` (clearing
    // `stacked_design`) to satisfy that check back when it read `design`
    // directly. The check moved; the workaround did not, and it broke the
    // sibling invariant that every block's `design` carries one row per
    // ORIGINAL experimental unit: the time block became 3n while log_sigma
    // stayed n, so `blockwise_fit_from_parts` refused the fit and named the
    // block that was still correct. It also fed `training_sample_size` the
    // stacked channel count (3n) rather than the sample size.
    let assembly_specs: Vec<ParameterBlockSpec> = specs.to_vec();

    crate::custom_family::blockwise_fit_from_parts(
        crate::custom_family::BlockwiseFitResultParts {
            block_states: states,
            log_likelihood,
            // A censored continuous law has no finite saturated point, so the
            // survival location-scale fit keeps `−2·log_likelihood` as its
            // reported deviance (#2786).
            deviance: None,
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
            smoothing_corrected: None,
            smoothing_correction_absence: None,
        },
        &assembly_specs,
    )
    .map_err(FitFailure::from)
}

/// Variant that also returns the offset-channel residuals + curvatures at the
/// converged β̂. We have to extract these *before* `finalize_survival_location_scale_fit`
/// runs, because the location-scale finalizer empties `UnifiedFitResult::block_states`
/// (see `survival_fit_from_parts` — `block_states: Vec::new()`), and the family's
/// `offset_channel_geometry` method needs the raw, populated per-block state.
enum SurvivalLocationScaleFitAuthority<'a> {
    Direct,
    Certified {
        theta: &'a Array1<f64>,
        outer: &'a gam_solve::rho_optimizer::CertifiedOuterResult,
        mode: crate::custom_family::CustomFamilyOwnedMode,
    },
}

fn fit_survival_location_scale_with_geometry_authority(
    spec: SurvivalLocationScaleSpec,
    authority: SurvivalLocationScaleFitAuthority<'_>,
) -> Result<(UnifiedFitResult, SurvivalLocationScaleConvergedGeometry), FitFailure> {
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
    let fit = match authority {
        SurvivalLocationScaleFitAuthority::Direct if prepared.is_reduced_parametric_aft() => {
            fit_reduced_parametric_aft(&prepared, &options)?
        }
        SurvivalLocationScaleFitAuthority::Direct => {
            fit_custom_family_arming_on_evidence(&prepared.family, &prepared.blockspecs, &options)?
        }
        // A reduced fit carries no smoothing coordinate, so it reaches this arm
        // only through the inverse-link shape axes (#2904): the certificate is
        // over those axes, and the fit replays the owned mode it was measured at.
        SurvivalLocationScaleFitAuthority::Certified { theta, outer, mode } => {
            let exact_options = crate::outer_subsample::exact_outer_options(&options);
            fit_custom_family_fixed_log_lambdas_from_owned_mode(
                &prepared.family,
                &prepared.blockspecs,
                &exact_options,
                mode,
                theta,
                outer,
            )?
        }
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
    let finalized = finalize_survival_location_scale_fit(&prepared, &fit)
        .map_err(|reason| FitFailure::raised(gam_problem::FailureCategory::Invariant, reason))?;
    Ok((finalized, (residuals, curvatures)))
}

pub(crate) fn fit_survival_location_scale_with_geometry(
    spec: SurvivalLocationScaleSpec,
) -> Result<(UnifiedFitResult, SurvivalLocationScaleConvergedGeometry), FitFailure> {
    fit_survival_location_scale_with_geometry_authority(
        spec,
        SurvivalLocationScaleFitAuthority::Direct,
    )
}

fn fit_survival_location_scale_with_geometry_from_outer(
    spec: SurvivalLocationScaleSpec,
    theta: &Array1<f64>,
    outer: &gam_solve::rho_optimizer::CertifiedOuterResult,
    mode: crate::custom_family::CustomFamilyOwnedMode,
) -> Result<(UnifiedFitResult, SurvivalLocationScaleConvergedGeometry), FitFailure> {
    fit_survival_location_scale_with_geometry_authority(
        spec,
        SurvivalLocationScaleFitAuthority::Certified { theta, outer, mode },
    )
}

/// Converged-fit geometry returned alongside the finalized location-scale fit:
/// the offset-channel residuals + curvatures (for the baseline-θ gradient/Hessian).
pub(crate) type SurvivalLocationScaleConvergedGeometry =
    (OffsetChannelResiduals, OffsetChannelCurvatures);

/// The inverse link's free shape parameters, in the order its parameter partials
/// use: `(ε, log δ)` for SAS and BetaLogistic, the free ρ for a mixture, none for
/// any other link. They are family-owned outer coordinates that the LAML selects
/// together with ρ (#2904).
fn inverse_link_shape(link: &InverseLink) -> Array1<f64> {
    match link {
        InverseLink::Sas(state) | InverseLink::BetaLogistic(state) => {
            Array1::from_vec(vec![state.epsilon, state.log_delta])
        }
        InverseLink::Mixture(state) => state.rho.clone(),
        InverseLink::Standard(_) | InverseLink::LatentCLogLog(_) => Array1::zeros(0),
    }
}

/// `link` with its free shape parameters set to `shape` (see [`inverse_link_shape`]).
fn inverse_link_with_shape(
    link: &InverseLink,
    shape: ArrayView1<'_, f64>,
) -> Result<InverseLink, String> {
    let expected = inverse_link_shape(link).len();
    if shape.len() != expected {
        return Err(format!(
            "inverse-link shape has {} coordinates, expected {expected}",
            shape.len()
        ));
    }
    match link {
        InverseLink::Sas(_) => gam_solve::mixture_link::state_from_sasspec(gam_problem::SasLinkSpec {
            initial_epsilon: shape[0],
            initial_log_delta: shape[1],
        })
        .map(InverseLink::Sas),
        InverseLink::BetaLogistic(_) => {
            gam_solve::mixture_link::state_from_beta_logisticspec(gam_problem::SasLinkSpec {
                initial_epsilon: shape[0],
                initial_log_delta: shape[1],
            })
            .map(InverseLink::BetaLogistic)
        }
        InverseLink::Mixture(state) => {
            gam_solve::mixture_link::state_fromspec(&gam_problem::MixtureLinkSpec {
                components: state.components.clone(),
                initial_rho: shape.to_owned(),
            })
            .map(InverseLink::Mixture)
        }
        InverseLink::Standard(_) | InverseLink::LatentCLogLog(_) => Ok(link.clone()),
    }
}

pub(crate) fn select_survival_link_wiggle_basis_from_pilot(
    pilot: &SurvivalLocationScaleTermFitResult,
    age_exit: ArrayView1<'_, f64>,
    wiggle_cfg: &WiggleBlockConfig,
    wiggle_penalty_orders: &[usize],
) -> Result<SelectedWiggleBasis, FitFailure> {
    // The pilot's own designs applied to its own coefficients (#2937).
    let pilot_invariant =
        |error: String| FitFailure::raised(gam_problem::FailureCategory::Invariant, error);
    let eta_threshold = pilot
        .threshold_design
        .apply(pilot.fit.beta_threshold().view())
        .map_err(|error| pilot_invariant(error.to_string()))?;
    let eta_log_sigma = pilot
        .log_sigma_design
        .apply(pilot.fit.beta_log_sigma().view())
        .map_err(|error| pilot_invariant(error.to_string()))?;
    // The seed is the index the wiggle is composed on. In the reduced
    // parametric-AFT regime the `−log t` baseline rides the location before
    // `q₀` (`LocationLogTimeOffset`), so the wiggle sees `q₀(η_t − log t, η_ls)`;
    // seeding on `η_t` alone places the knots over the wrong range, and over a
    // single point once the location is constant (#3006).
    let reduced_parametric_aft = matches!(
        pilot.time_parameterization,
        SurvivalLocationScaleTimeParameterization::ReducedParametricAft
    );
    let q_seed = Array1::from_iter(
        eta_threshold
            .iter()
            .zip(eta_log_sigma.iter())
            .zip(age_exit.iter())
            .map(|((&threshold, &ls), &t)| {
                let location = if reduced_parametric_aft {
                    threshold - t.max(crate::survival::construction::SURVIVAL_TIME_FLOOR).ln()
                } else {
                    threshold
                };
                survival_q0_from_eta(location, ls)
            }),
    );
    // The composed link warp: `q = q₀ + Σ βw_j·I_j(q₀)` with `q₀` moving with β,
    // so the inner objective differentiates the basis and a clamped boundary
    // knot's multiplicity is a step in `H` and hence in the objective (gam#2695).
    crate::wiggle::select_wiggle_basis_from_seed_with_knots(
        q_seed.view(),
        wiggle_cfg,
        wiggle_penalty_orders,
        crate::wiggle::WarpKnotEnds::Simple,
    )
    .map_err(crate::gamlss::wiggle_basis_failure)
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
) -> Result<SurvivalLocationScaleTermFitResult, FitFailure> {
    spec.linkwiggle_block = Some(linkwiggle_block_input_from_selected_basis(
        selected_wiggle_basis,
    ));
    fit_survival_location_scale_terms(data, spec, kappa_options)
}

pub(crate) fn fit_survival_location_scale_terms(
    data: ndarray::ArrayView2<'_, f64>,
    spec: SurvivalLocationScaleTermSpec,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<SurvivalLocationScaleTermFitResult, FitFailure> {
    let threshold_boot_design = build_term_collection_design(data, &spec.thresholdspec)?;
    let log_sigma_boot_design = build_term_collection_design(data, &spec.log_sigmaspec)?;
    let threshold_bootspec =
        freeze_term_collection_from_design(&spec.thresholdspec, &threshold_boot_design)?;
    let log_sigma_bootspec =
        freeze_term_collection_from_design(&spec.log_sigmaspec, &log_sigma_boot_design)?;

    // The psi derivatives differentiate the designs built above, so failing to
    // build them is an engine defect (#2937).
    let derivative_invariant =
        |reason: String| FitFailure::raised(gam_problem::FailureCategory::Invariant, reason);
    let threshold_boot_derivs = build_survival_covariate_block_psi_derivatives(
        data,
        &threshold_bootspec,
        &threshold_boot_design,
        &spec.threshold_template,
    )
    .map_err(derivative_invariant)?;
    let log_sigma_boot_derivs = build_survival_covariate_block_psi_derivatives(
        data,
        &log_sigma_bootspec,
        &log_sigma_boot_design,
        &spec.log_sigma_template,
    )
    .map_err(derivative_invariant)?;
    let analytic_joint_gradient_available =
        threshold_boot_derivs.is_some() && log_sigma_boot_derivs.is_some();
    // The inverse-link shape parameters are outer coordinates of their own
    // (#2904). The family serves their first-order terms and declines the
    // second-order ones, so a fit that carries them runs a gradient-only outer.
    let link_shape0 = inverse_link_shape(&spec.inverse_link);
    let analytic_joint_hessian_available = link_shape0.is_empty()
        && threshold_boot_derivs
            .as_ref()
            .is_some_and(|derivs| survival_psi_derivatives_support_exact_joint_hessian(derivs))
        && log_sigma_boot_derivs
            .as_ref()
            .is_some_and(|derivs| survival_psi_derivatives_support_exact_joint_hessian(derivs));

    // The wiggle seed is sized by its penalties, the rule every other block's
    // seed follows (`initial_log_lambdas`); a selected wiggle basis carries no
    // caller seed, and a zero-length seed would give the outer layout no wiggle
    // ρ beside a realized block that carries its penalties (#3006).
    let wiggle_rho0 = match spec.linkwiggle_block.as_ref() {
        Some(wiggle) => {
            initial_log_lambdas(&wiggle.penalties, wiggle.initial_log_lambdas.clone())
                .map_err(SurvivalLocationScaleError::from)?
        }
        None => Array1::zeros(0),
    };
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
        // Seeding the weak default (a fixed `1e-2` time seed) drops the
        // inner REML search into the *interior* of that ridge, where it crawls
        // toward the strong-smoothing boundary one short, ill-conditioned step
        // at a time and never terminates in reasonable time (#736, #735, #721).
        //
        // The previous fix seeded the *interior* point ρ = 8. That did NOT cure
        // the hang: the inner blockwise REML optimizer re-optimizes ρ_time
        // freely from its seed against a per-coordinate ρ box bound (then ±10;
        // today `gam_custom_family::EFFECTIVE_DF_CEILING`, tightened per term
        // by the effective-df floor).
        // λ = exp(8) ≈ 3·10³ already sits INSIDE the "dead-flat region" that
        // very bound exists to fence off (see the ceiling's rationale in
        // `gam-custom-family/src/fit.rs`): with a flat REML gradient and near-singular
        // curvature there, the optimizer wanders between ρ = 8 and the ρ = 10
        // boundary one micro-step at a time and the retry-stall detector spins
        // on the flat surface — producing the >200s no-iteration-log hang. A
        // seed strictly interior to the box can never certify, because the
        // unconstrained projected-gradient stationarity test it would need is
        // exactly the test the flat ridge makes ill-posed.
        //
        // Seed instead at the ρ box bound itself. At the bound the
        // box-constraint KKT condition (the REML gradient pushes further into
        // strong smoothing, against an active bound) certifies stationarity
        // *immediately* at iteration 0 for the time coordinate — there is no
        // interior flat region left to wander, because the optimizer is pinned
        // at the wall. λ = exp(EFFECTIVE_DF_CEILING = 12) ≈ 1.6·10⁵ is deep in
        // the affine-nullspace limit (the ceiling's own rationale calls this
        // regime "statistically indistinguishable from shrunk to nullspace"),
        // i.e. exactly the parametric-AFT affine
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
            // Seed AT the custom-family outer box ceiling (not interior, as the
            // prior ρ = 8 seed was) so the box constraint is active from
            // iteration 0 and outer stationarity certifies immediately instead
            // of crawling the flat ridge. The invariant is "seed sits ON the
            // coordinate's realized wall", not any numeric value: a stranded
            // local `10.0` re-opened a two-e-fold ridge crawl when the ceiling
            // moved to 12 (#2356). Referencing the ceiling itself is exact even
            // when the term's realized upper bound is tighter (the
            // effective-df-floor tightening): `run_plan` projects every seed
            // onto the realized per-coordinate box before use.
            // The time warp is seeded at its null space: the ρ at which its
            // penalty switches the term off to working precision (#2812), read
            // off the exit-time design and the penalty themselves.
            let mut time_seed = rho0.slice_mut(s![range.start..range.end]);
            for (k, v) in time_seed.iter_mut().enumerate() {
                *v = spec
                    .time_block
                    .penalties
                    .get(k)
                    .and_then(|penalty| {
                        gam_custom_family::penalized_term_switch_off_rho(
                            &spec.time_block.design_exit,
                            penalty,
                        )
                    })
                    .unwrap_or(gam_solve::estimate::rho_domain::precision_box().1);
            }
        }
    }
    // Warm-start: inject converged ρ seeds from a previous fit if supplied. The carried
    // strengths are seeds, not bounds: the joint setup and `run_plan` project every seed
    // onto the coordinate's derived domain (#2812), so no private box is applied here.
    // A converged fit's log strength is finite; a non-finite carry is refused rather than
    // silently replaced by the default seed.
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
        if let Some(bad) = seed.iter().position(|value| !value.is_finite()) {
            return Err(SurvivalLocationScaleError::InvalidConfiguration {
                reason: format!(
                    "survival threshold initial_log_lambdas[{bad}] is non-finite ({})",
                    seed[bad]
                ),
            }
            .into());
        }
        let range = layout.threshold_range();
        rho0.slice_mut(s![range.start..range.end]).assign(seed);
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
        if let Some(bad) = seed.iter().position(|value| !value.is_finite()) {
            return Err(SurvivalLocationScaleError::InvalidConfiguration {
                reason: format!(
                    "survival log_sigma initial_log_lambdas[{bad}] is non-finite ({})",
                    seed[bad]
                ),
            }
            .into());
        }
        let range = layout.log_sigma_range();
        rho0.slice_mut(s![range.start..range.end]).assign(seed);
    }
    if layout.k_wiggle > 0 {
        let range = layout.wiggle_range();
        rho0.slice_mut(s![range.start..range.end])
            .assign(&wiggle_rho0);
    }
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
    // Stash the geometry and inverse link of the most recent inner fit. Updated on every
    // value-closure call by the spatial optimizer; the last one written
    // corresponds to the converged outer point. This avoids redoing
    // `prepare_survival_location_scale_model` + a second fit pass after the
    // optimizer returns, and (critically) avoids the post-finalize
    // `block_states` wipe that would make the geometry call error out.
    let last_geometry: std::cell::RefCell<
        Option<(SurvivalLocationScaleConvergedGeometry, InverseLink)>,
    > = std::cell::RefCell::new(None);

    let build_spec = |rho: &Array1<f64>,
                      inverse_link: InverseLink,
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
            inverse_link,
            derivative_guard: spec.derivative_guard,
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
            persistent_warm_start_store: spec.persistent_warm_start_store.clone(),
            cache_mirror_sessions: spec.cache_mirror_sessions.clone(),
        })
    };
    // The ρ domain of every block that owns a ρ coordinate (time, threshold,
    // log-sigma, link wiggle), realized at the seed by the same preparation every
    // evaluation runs, under the #2812 law `fit_custom_family` applies to those
    // blocks (#2902 item 15). The threshold and log-sigma term collections alone
    // do not carry the time and wiggle penalties.
    //
    // The same seed preparation fixes the time parameterization saved replay
    // dispatches on. It is fit topology (the time design, the log-σ penalty
    // count and the wiggles), identical at every ρ and κ, and is persisted here
    // because replay must never recover it from fitted coefficient values.
    let (rho_lower, rho_upper, time_parameterization) = {
        let seed_spec = build_spec(
            &rho0,
            spec.inverse_link.clone(),
            &spec.thresholdspec,
            &spec.log_sigmaspec,
            &threshold_boot_design,
            &log_sigma_boot_design,
        )
        .map_err(|reason| FitFailure::raised(gam_problem::FailureCategory::Invariant, reason))?;
        let prepared = prepare_survival_location_scale_model(&seed_spec)?;
        let (rho_lower, rho_upper) = crate::fit_orchestration::drivers::realized_blocks_rho_domain(
            &prepared.blockspecs,
            &survival_blockwise_fit_options(&seed_spec),
            layout.total(),
        )?;
        (rho_lower, rho_upper, prepared.time_parameterization())
    };
    let joint_setup = build_survival_two_block_exact_joint_setup(
        data.view(),
        &spec.thresholdspec,
        &spec.log_sigmaspec,
        rho0,
        rho_lower,
        rho_upper,
    )?;
    // θ = [ρ | log κ | inverse-link shape]. A shape coordinate may be any finite
    // real, so the only bound it has is the one working precision imposes.
    let joint_setup = if link_shape0.is_empty() {
        joint_setup
    } else {
        let (lower, upper) = gam_solve::estimate::rho_domain::precision_box();
        joint_setup.with_auxiliary(
            link_shape0.clone(),
            Array1::from_elem(link_shape0.len(), lower),
            Array1::from_elem(link_shape0.len(), upper),
        )
    };
    let link_shape_start = joint_setup.rho_dim() + joint_setup.log_kappa_dim();
    let inverse_link_at = |theta: &Array1<f64>| -> Result<InverseLink, String> {
        inverse_link_with_shape(&spec.inverse_link, theta.slice(s![link_shape_start..]))
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
        crate::custom_family::OuterDerivativePolicy { capability }
    };
    let solved = optimize_spatial_length_scale_exact_joint_typed(
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
        None,
        outer_policy,
        // The final fit: the solver's error is carried whole (#2937). Its link and
        // spec are rebuilt at a theta the driver produced, from specs validated
        // at entry, so failing to rebuild them is an engine defect.
        |theta, specs: &[TermCollectionSpec], designs: &[TermCollectionDesign], provenance| {
            let rho = theta.slice(s![..joint_setup.rho_dim()]).to_owned();
            let rebuild_invariant = |reason: String| {
                FitFailure::raised(gam_problem::FailureCategory::Invariant, reason)
            };
            let inverse_link = inverse_link_at(theta).map_err(rebuild_invariant)?;
            let assembled = build_spec(
                &rho,
                inverse_link.clone(),
                &specs[0],
                &specs[1],
                &designs[0],
                &designs[1],
            )
            .map_err(rebuild_invariant)?;
            let (fit, geom) = match provenance {
                SpatialFitProvenance::NoOuterOptimization => {
                    fit_survival_location_scale_with_geometry(assembled)?
                }
                SpatialFitProvenance::Certified { outer, mode } => {
                    fit_survival_location_scale_with_geometry_from_outer(
                        assembled, theta, outer, mode,
                    )?
                }
            };
            time_beta_hint.replace(Some(fit.beta_time()));
            threshold_beta_hint.replace(Some(fit.beta_threshold()));
            log_sigma_beta_hint.replace(Some(fit.beta_log_sigma()));
            wiggle_beta_hint.replace(fit.beta_link_wiggle());
            *last_geometry.borrow_mut() = Some((geom, inverse_link));
            Ok(fit)
        },
        |theta,
         specs: &[TermCollectionSpec],
         designs: &[TermCollectionDesign],
         eval_mode,
         _| {
            use gam_problem::EvalMode;
            if !analytic_joint_gradient_available {
                return Err(SurvivalLocationScaleError::InvalidConfiguration { reason: "analytic spatial psi derivatives are unavailable for survival exact two-block path"
                        .to_string(), }.into());
            }
            let rho = theta.slice(s![..joint_setup.rho_dim()]).to_owned();
            let assembled = build_spec(
                &rho,
                inverse_link_at(theta)?,
                &specs[0],
                &specs[1],
                &designs[0],
                &designs[1],
            )?;
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
                        log::debug!(
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
            let hyper_layout = CustomFamilyHyperLayout::new(
                derivative_blocks,
                (0..link_shape0.len()).collect(),
                theta.slice(s![joint_setup.rho_dim()..]).to_owned(),
            )?;
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
            let eval_options = crate::outer_subsample::exact_outer_options(
                &survival_blockwise_fit_options(&assembled),
            );
            let owned = evaluate_custom_family_joint_hyper_owned(
                &prepared.family,
                &prepared.blockspecs,
                &eval_options,
                &rho,
                &hyper_layout,
                exact_warm_start.borrow().as_ref(),
                effective_mode,
            )
            .map_err(|e| e.to_string())?;
            // An unconverged inner state (a SlowGeometricRate or stall exit) is
            // neither a fit nor a seed (#2902): the next trial warm-starts from
            // the last converged mode.
            if !owned.result.inner_converged {
                return Err(
                    "survival location-scale exact joint inner solve did not converge".to_string(),
                );
            }
            exact_warm_start.replace(Some(owned.result.warm_start.clone()));
            Ok(ExactJointEvaluation {
                objective: owned.result.objective,
                gradient: owned.result.gradient,
                hessian: owned.result.outer_hessian,
                mode: owned.mode,
            })
        },
        |theta,
         specs: &[TermCollectionSpec],
         designs: &[TermCollectionDesign]| {
            if !analytic_joint_gradient_available {
                return Err(SurvivalLocationScaleError::InvalidConfiguration { reason: "analytic spatial psi derivatives are unavailable for survival exact two-block path"
                        .to_string(), }.into());
            }
            let rho = theta.slice(s![..joint_setup.rho_dim()]).to_owned();
            let assembled = build_spec(
                &rho,
                inverse_link_at(theta)?,
                &specs[0],
                &specs[1],
                &designs[0],
                &designs[1],
            )?;
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
                        log::debug!(
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
            let hyper_layout = CustomFamilyHyperLayout::new(
                derivative_blocks,
                (0..link_shape0.len()).collect(),
                theta.slice(s![joint_setup.rho_dim()..]).to_owned(),
            )?;
            let eval_options = crate::outer_subsample::exact_outer_options(
                &survival_blockwise_fit_options(&assembled),
            );
            let owned = evaluate_custom_family_joint_hyper_efs_owned(
                &prepared.family,
                &prepared.blockspecs,
                &eval_options,
                &rho,
                &hyper_layout,
                exact_warm_start.borrow().as_ref(),
            )
            .map_err(|e| e.to_string())?;
            if !owned.result.inner_converged {
                return Err(
                    "survival location-scale exact joint EFS inner solve did not converge"
                        .to_string(),
                );
            }
            exact_warm_start.replace(Some(owned.result.warm_start.clone()));
            Ok(ExactJointEfsEvaluation {
                evaluation: owned.result.efs_eval,
                mode: owned.mode,
            })
        },
        crate::marginal_slope_shared::make_beta_seed_validator(&pending_beta_seed),
    )?;

    let mut resolved_specs = solved.resolved_specs;
    let mut designs = solved.designs;
    // The driver's fit is the fit closure's return value, and that closure
    // stashed the offset geometry (computed pre-finalize while `block_states`
    // was still populated) and the inverse link it fitted at. A refit at the
    // final ρ alone would drop the fitted inverse-link shape.
    let ((baseline_offset_residuals, baseline_offset_curvatures), inverse_link) = last_geometry
        .borrow_mut()
        .take()
        .ok_or_else(|| {
            FitFailure::raised(
                gam_problem::FailureCategory::Invariant,
                "survival location-scale exact-joint driver returned a fit its fit closure did not produce",
            )
        })?;
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
        inverse_link,
    })
}
