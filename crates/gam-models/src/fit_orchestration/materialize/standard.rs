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

    let link_choice = effective_link_choice_for_materialize(parsed, config)?;
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

    // Per-family response-distribution degeneracy (#331 all-0/all-1 Bernoulli).
    // Symmetric to validate_response_support —
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
    reject_flexible_link_for_nonbinomial(link_choice.as_ref(), &family)?;

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
        gam_runtime::resource::ProblemHints::default(),
    );
    let mut spec = build_termspec_with_geometry_and_overrides(
        &term_parsed.terms,
        term_data,
        &term_col_map,
        &mut inference_notes,
        config.scale_dimensions,
        &policy,
        config.smooth_overrides.as_ref(),
    )?;
    // #1074: the Duchon default penalty is a Hilbert scale (curvature +
    // mass/tension operator dials). REML deselects the lower orders faithfully
    // only in the ProfiledGaussian arm; under a fixed-dispersion GLM the LAML
    // criterion mis-rewards the near-full-rank operator-Gram blocks for
    // over-shrinking the mean. Drop them for non-Gaussian-identity fits so the
    // default matches mgcv's single-curvature `bs="ds"`; the Gaussian path is
    // untouched and keeps the full scale.
    gate_duchon_operator_penalties_for_family(&mut spec, &family);

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
        coord.term_index = gam_problem::SmoothTermIdx::new(resolved_idx);
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
                loading: crate::survival::lognormal_kernel::HazardLoading::Full,
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
        if config.frailty.as_ref().is_some_and(FrailtySpec::is_active) {
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
    // Standard-fit `FitOptions` are built through the single shared policy
    // source (#1196) so the formula/Python path and the `gam` CLI cannot
    // resolve a different outer-REML optimization policy for the same model.
    // The SAS link state is folded into the `family`/`link` by `resolve_family`
    // and rebuilt from `FitOptions.sas_link` by the standard path, so the
    // formula path leaves `sas_link` empty.
    //
    // The blended/mixture link, however, is NOT self-sufficient on the family
    // alone: the binomial-mixture solver guard (`fit_gamwith_penalty_specs...`
    // in gam-solve) requires `FitOptions.mixture_link` to be `Some` for any
    // `is_binomial_mixture()` family, and the joint mixing-weight solve reads
    // its component set + initial ρ from that field. The `gam` CLI populates it
    // (run_fit.rs `mixture_linkspec`); the formula/Python path used to leave it
    // `None`, so a `link(type=blended(...))` formula fit aborted immediately
    // with `BinomialMixture requires mixture_link specification` before reaching
    // the solver (#1598, the Python-side analogue of #1128/#1160). Thread the
    // parsed `blended(...)`/`mixture(...)` component spec into `mixture_link`
    // here so the formula/Python path reaches the same solver the CLI does.
    //
    // The formula path rejects `link(rho=...)`
    // (`effective_link_choice_for_materialize`), so the initial ρ is always the
    // canonical zero seed of length `n_components - 1`, exactly mirroring what
    // `resolve_family` embeds into the mixture link state.
    let mixture_link = link_choice.as_ref().and_then(|choice| {
        choice.mixture_components.as_ref().map(|components| {
            let free = components.len().saturating_sub(1);
            MixtureLinkSpec {
                components: components.clone(),
                initial_rho: Array1::<f64>::zeros(free),
            }
        })
    });
    // A blended/mixture link is only an actual mixture when its mixing weights
    // are jointly optimized; with `optimize_mixture=false` the solver freezes ρ
    // at the zero seed and the fit collapses to the first component's pure link.
    // The CLI sets `optimize_mixture: true` whenever it builds a mixture spec
    // (run_fit.rs); mirror that so the formula/Python path actually fits the
    // mixture rather than silently degrading to plain logit (#1598).
    let optimize_mixture = mixture_link.is_some();
    let options = crate::fit_orchestration::canonical_standard_fit_options(
        config,
        crate::fit_orchestration::StandardFitOptionsInputs {
            latent_cloglog,
            mixture_link,
            optimize_mixture,
            firth_bias_reduction: config.firth,
            adaptive_regularization: standard_adaptive_regularization_options(config),
            persist_warm_start_disk: config.persist_warm_start_disk,
            ..Default::default()
        },
    );
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
