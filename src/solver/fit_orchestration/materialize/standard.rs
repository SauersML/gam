use super::*;

pub(crate) fn materialize_standard<'a>(
    parsed: &ParsedFormula,
    data: &'a Dataset,
    col_map: &HashMap<String, usize>,
    config: &FitConfig,
) -> Result<MaterializedModel<'a>, WorkflowError> {
    if config.noise_offset_column.is_some() {
        return Err(
            "noise_offset_column requires a location-scale model with noise_formula"
                .to_string()
                .into(),
        );
    }
    let y_col = resolve_role_col(col_map, &parsed.response, "response")?;
    let y = data.values.column(y_col).to_owned();
    let y_kind = response_column_kind(data, y_col);
    let mut inference_notes = Vec::new();

    let link_choice = parse_link_choice(config.link.as_deref(), config.flexible_link)?;
    let family = resolve_family(
        config.family.as_deref(),
        config.negative_binomial_theta,
        link_choice.as_ref(),
        y.view(),
        y_kind,
        &parsed.response,
    )?;

    // Per-family response-support validation (#335 Gamma requires y > 0;
    // #337 Poisson/NegativeBinomial require y ≥ 0; mirrors the Beta
    // (0,1)-support check in the external-design GLM path). The family
    // itself owns the check — see `ResponseFamily::validate_response_support`
    // — so adding a new family that constrains its support is a single edit
    // on the type, not a coordinated update across every materializer.
    family
        .response
        .validate_response_support(y.view())
        .map_err(|violation| violation.message_for(&parsed.response))?;

    // Per-family response-distribution degeneracy (#331 all-0/all-1 Bernoulli,
    // #332 near-constant Gaussian). Symmetric to validate_response_support —
    // each `ResponseFamily` variant owns its own degeneracy classifier, the
    // workflow only forwards the column name.
    family
        .response
        .validate_response_degeneracy(y.view())
        .map_err(|deg| deg.message_for(&parsed.response))?;

    // An explicit `linkwiggle(...)` term is only wired into the fit below for a
    // binomial family; reject it for a non-binomial response rather than drop
    // it silently (#371).
    reject_explicit_linkwiggle_for_nonbinomial(parsed, &family)?;

    let effective_linkwiggle =
        effectivelinkwiggle_formulaspec(parsed.linkwiggle.as_ref(), link_choice.as_ref());

    let latent_prepared = prepare_standard_latent_coord(parsed, data, y.view(), config)?;
    let (latent_dataset, latent_parsed, mut latent_coord) = match latent_prepared {
        Some((dataset, parsed, coord)) => (Some(dataset), Some(parsed), Some(coord)),
        None => (None, None, None),
    };
    let term_data = latent_dataset.as_ref().unwrap_or(data);
    let term_parsed = latent_parsed.as_ref().unwrap_or(parsed);
    let term_col_map = term_data.column_map();

    let policy = resolved_resource_policy(
        config,
        term_data,
        crate::solver::resource::ProblemHints::default(),
    );
    let spec = build_termspec_with_geometry_and_overrides(
        &term_parsed.terms,
        term_data,
        &term_col_map,
        &mut inference_notes,
        config.scale_dimensions,
        &policy,
        config.smooth_overrides.as_ref(),
    )?;

    // Sample size vs basis-rank gate (#309). Each smooth basis answers
    // `min_sample_rows()` for itself; this helper just sums and compares.
    // Runs *after* `build_termspec_with_geometry_and_overrides` so the lower bound is
    // computed on the fully resolved basis spec (e.g. tensor-product columns,
    // knot counts inferred at materialization time).
    check_smooth_capacity(&spec, y.len(), &parsed.response)?;
    if let Some(coord) = latent_coord.as_mut() {
        let resolved_idx = spec
            .smooth_terms
            .iter()
            .position(|term| {
                smooth_basis_feature_cols_for_latent(&term.basis)
                    .is_some_and(|cols| cols == coord.feature_cols)
            })
            .ok_or_else(|| {
                "latent-coordinate smooth term disappeared during formula materialization"
                    .to_string()
            })?;
        coord.term_index = crate::types::SmoothTermIdx::new(resolved_idx);
        if coord.manifold_auto {
            let inferred = natural_latent_manifold_for_basis(
                &spec.smooth_terms[coord.term_index.get()].basis,
                coord.feature_cols.len(),
            );
            coord.manifold = inferred.clone();
            coord.values = Arc::new(coord.values.with_manifold(inferred));
        }
    }

    let weights = resolve_weight_column(data, col_map, config.weight_column.as_deref())?;
    let offset = resolve_offset_column(data, col_map, config.offset_column.as_deref())?;
    let latent_cloglog = if family.is_latent_cloglog() {
        let sigma = match config.frailty.clone().unwrap_or(FrailtySpec::None) {
            FrailtySpec::HazardMultiplier {
                sigma_fixed: Some(sigma),
                loading: crate::families::survival::lognormal_kernel::HazardLoading::Full,
            } => sigma,
            FrailtySpec::HazardMultiplier {
                sigma_fixed: Some(_),
                loading,
            } => {
                return Err(WorkflowError::MissingDependency {
                    reason: format!(
                        "latent-cloglog-binomial requires HazardLoading::Full, got {loading:?}"
                    ),
                }
                .into());
            }
            FrailtySpec::HazardMultiplier {
                sigma_fixed: None, ..
            } => {
                return Err(WorkflowError::MissingDependency {
                    reason:
                        "latent-cloglog-binomial currently requires a fixed hazard-multiplier sigma"
                            .to_string(),
                }
                .into());
            }
            FrailtySpec::GaussianShift { .. } => {
                return Err(WorkflowError::InvalidConfig {
                    reason: "latent-cloglog-binomial does not support GaussianShift frailty"
                        .to_string(),
                }
                .into());
            }
            FrailtySpec::None => {
                return Err(WorkflowError::MissingDependency {
                    reason:
                        "latent-cloglog-binomial requires config.frailty=HazardMultiplier with a fixed sigma"
                            .to_string(),
                }
                .into());
            }
        };
        Some(
            LatentCLogLogState::new(sigma)
                .map_err(|e| format!("invalid latent_cloglog state: {e}"))?,
        )
    } else {
        if config.frailty.is_some() {
            return Err(WorkflowError::InvalidConfig {
                reason: format!(
                    "config.frailty is not supported for standard family {:?}; use a frailty-aware family instead",
                    family
                ),
            }
            .into());
        }
        None
    };
    let options = FitOptions {
        latent_cloglog,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
        // Formula/workflow fits are the interactive/default path. Keep
        // coefficient covariance and smoothing correction, but do not run the
        // optional live-rho posterior certificate/escalation here: escalation can
        // launch NUTS over rho and turns ordinary quality gates into sampler
        // benchmarks. Lower-level callers that explicitly need the rho posterior
        // can still opt in through `FitOptions`.
        skip_rho_posterior_inference: true,
        max_iter: config.outer_max_iter.unwrap_or(200),
        // Outer REML/LAML smoothing-selection tolerance. The outer convergence
        // test (`outer_gradient_tolerance`) uses a `rel_cost` criterion whose
        // effective projected-gradient threshold is ≈ `tol · (1 + |V(ρ)|)`. The
        // LAML cost grows like O(n), so at the old `1e-7` the effective gradient
        // threshold was ≈ `1e-7 · |V|` ≈ 1e-4 for a typical fit — far too coarse
        // to *resolve* the smoothing parameter: the descent halted while λ̂ could
        // still move several percent. That under-resolution is benign for a
        // single fit but breaks an exact invariance — for a fixed-dispersion
        // family a uniform prior weight `w = c` is exact `c`-fold replication, so
        // the two encodings share a byte-identical LAML surface and an identical
        // optimum, yet their (replication-equal) surfaces carry O(1e-7)
        // floating-point differences AWAY from the optimum. With the coarse
        // threshold the descent stopped at those encoding-dependent points,
        // systematically over-smoothing the weighted encoding (gam#893; up to a
        // ~22× λ ratio across seeds). Tightening to `1e-10` (effective gradient
        // threshold ≈ 1e-7, ~100× below the FP-noise floor) drives both
        // encodings to the shared optimum, restoring `w=c ⇔ c-fold replication`
        // in smoothing selection to optimiser precision. Max-iter is handled
        // best-effort (the optimiser returns its best ρ on budget exhaustion, it
        // does not hard-fail), so a harder problem that cannot reach 1e-10 in
        // `outer_max_iter` is no worse off than before — just better-resolved
        // when it can.
        tol: 1e-10,
        nullspace_dims: vec![],
        linear_constraints: None,
        firth_bias_reduction: config.firth,
        adaptive_regularization: standard_adaptive_regularization_options(config),
        penalty_shrinkage_floor: Some(1e-6),
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persist_warm_start_disk: config.persist_warm_start_disk,
    };
    let kappa_options = SpatialLengthScaleOptimizationOptions::default();

    let wiggle = effective_linkwiggle.as_ref().and_then(|cfg| {
        if !family.is_binomial() {
            return None;
        }
        let link_kind = match link_choice.as_ref() {
            Some(c) => match StandardLink::try_from(c.link) {
                Ok(std_link) => InverseLink::Standard(std_link),
                // linkwiggle is gated by `linkname_supports_joint_wiggle` which
                // rejects Sas / BetaLogistic upstream, so reaching this arm
                // means the gate was bypassed.
                Err(_) => return None,
            },
            None => {
                if let Some(state) = latent_cloglog {
                    InverseLink::LatentCLogLog(state)
                } else {
                    InverseLink::Standard(StandardLink::Logit)
                }
            }
        };
        Some(StandardBinomialWiggleConfig {
            link_kind,
            wiggle: LinkWiggleConfig {
                degree: cfg.degree,
                num_internal_knots: cfg.num_internal_knots,
                penalty_orders: cfg.penalty_orders.clone(),
                double_penalty: cfg.double_penalty,
            },
            // The second-stage refit options live inside the wiggle config so
            // the pilot can't be configured without them (see
            // `StandardBinomialWiggleConfig` doc + #320). Magic-by-default:
            // no caller-supplied options are required for the Python /
            // formula-DSL path.
            refit_options: BlockwiseFitOptions::default(),
        })
    });

    Ok(MaterializedModel {
        request: FitRequest::Standard(StandardFitRequest {
            data: term_data.values.clone(),
            y,
            weights,
            offset,
            spec,
            family,
            options,
            kappa_options,
            wiggle,
            coefficient_groups: config.coefficient_groups.clone(),
            penalty_block_gamma_priors: config.penalty_block_gamma_priors.clone(),
            latent_coord,
            _marker: std::marker::PhantomData,
        }),
        inference_notes,
    })
}
