use super::*;

pub fn fit_gamwith_heuristic_lambdas<X>(
    x: X,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    s_list: &[BlockwisePenalty],
    heuristic_lambdas: Option<&[f64]>,
    family: gam_problem::LikelihoodSpec,
    opts: &FitOptions,
) -> Result<UnifiedFitResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    fit_gamwith_heuristic_lambdas_andwarm_start(
        x,
        y,
        weights,
        offset,
        s_list,
        heuristic_lambdas,
        None,
        family,
        opts,
    )
}

pub(crate) fn fit_gamwith_heuristic_lambdas_andwarm_start<X>(
    x: X,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    s_list: &[BlockwisePenalty],
    heuristic_lambdas: Option<&[f64]>,
    warm_start_beta: Option<ArrayView1<'_, f64>>,
    family: gam_problem::LikelihoodSpec,
    opts: &FitOptions,
) -> Result<UnifiedFitResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    let specs: Vec<PenaltySpec> = s_list.iter().map(PenaltySpec::from_blockwise_ref).collect();
    fit_gamwith_penalty_specs_andwarm_start(
        x,
        y,
        weights,
        offset,
        specs,
        opts.nullspace_dims.clone(),
        heuristic_lambdas,
        warm_start_beta,
        family,
        opts,
    )
}

pub fn fit_gam_with_penalty_specs<X>(
    x: X,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    penalty_specs: Vec<PenaltySpec>,
    nullspace_dims: Vec<usize>,
    family: gam_problem::LikelihoodSpec,
    opts: &FitOptions,
) -> Result<UnifiedFitResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    fit_gamwith_penalty_specs_andwarm_start(
        x,
        y,
        weights,
        offset,
        penalty_specs,
        nullspace_dims,
        None,
        None,
        family,
        opts,
    )
}

fn fit_gamwith_penalty_specs_andwarm_start<X>(
    x: X,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    specs: Vec<PenaltySpec>,
    nullspace_dims: Vec<usize>,
    heuristic_lambdas: Option<&[f64]>,
    warm_start_beta: Option<ArrayView1<'_, f64>>,
    family: gam_problem::LikelihoodSpec,
    opts: &FitOptions,
) -> Result<UnifiedFitResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    let x = x.into();
    // Reject empty designs (no observations or no coefficients) up front. An
    // empty design has no well-defined fit and downstream indexing / linear
    // solves would otherwise panic on the zero-sized dimensions, so bail with a
    // clean error before any such work happens.
    if x.nrows() == 0 || x.ncols() == 0 {
        crate::bail_invalid_estim!(
            "empty design matrix: cannot fit a model with {} rows and {} columns",
            x.nrows(),
            x.ncols(),
        );
    }
    if family.is_binomial_mixture() && opts.mixture_link.is_none() {
        crate::bail_invalid_estim!("BinomialMixture requires mixture_link specification");
    }
    let effective_sas_link = effective_sas_link_for_family(&family, opts.sas_link);
    if opts.mixture_link.is_some() && opts.sas_link.is_some() {
        crate::bail_invalid_estim!("mixture_link and sas_link cannot both be set");
    }
    // sas_link only makes sense when the family already declares an adaptive
    // SAS-style link (BinomialSas / BinomialBetaLogistic).  Reject any attempt
    // to use sas_link with a fixed standard link family, since the caller
    // declared a fixed link contract and silently upgrading it to an adaptive
    // family is a footgun.  effective_sas_link auto-fills defaults only for
    // adaptive families, so any non-None value seen here together with a
    // standard family link came from the caller and is inconsistent.
    if let Some(_sas_spec) = opts.sas_link.as_ref() {
        let link_supports_sas = matches!(
            &family.link,
            InverseLink::Sas(_) | InverseLink::BetaLogistic(_)
        );
        if !link_supports_sas {
            crate::bail_invalid_estim!(
                "sas_link options are only valid for adaptive SAS link families \
                 (BinomialSas / BinomialBetaLogistic); family '{}' uses a fixed link \
                 and cannot accept sas_link parameters",
                family.pretty_name(),
            );
        }
    }
    let resolved_family: gam_problem::LikelihoodSpec = if let Some(mix_spec) =
        opts.mixture_link.as_ref()
    {
        if !family.is_binomial() {
            crate::bail_invalid_estim!("mixture_link is only supported for binomial families");
        }
        match &family.link {
            InverseLink::Standard(StandardLink::Logit)
            | InverseLink::Standard(StandardLink::Probit)
            | InverseLink::Standard(StandardLink::CLogLog)
            | InverseLink::Mixture(_) => {
                let mixture_state = crate::mixture_link::state_fromspec(mix_spec).map_err(|e| {
                    EstimationError::InvalidInput(format!("invalid mixture link: {e}"))
                })?;
                LikelihoodSpec::new(
                    ResponseFamily::Binomial,
                    InverseLink::Mixture(mixture_state),
                )
            }
            _ => {
                crate::bail_invalid_estim!("mixture_link is only supported for binomial families");
            }
        }
    } else if let Some(latent_state) = opts.latent_cloglog.as_ref() {
        // When a latent_cloglog state is supplied alongside a Binomial family
        // whose link is either Standard(CLogLog) or LatentCLogLog(_), upgrade
        // the resolved family link to LatentCLogLog so the parameterized state
        // is carried through into ExternalOptimResult.likelihood_family and
        // any downstream consumer (predict, save/load, summary).
        if !family.is_binomial() {
            crate::bail_invalid_estim!("latent_cloglog is only supported for Binomial families");
        }
        match &family.link {
            InverseLink::Standard(StandardLink::CLogLog) | InverseLink::LatentCLogLog(_) => {
                LikelihoodSpec::new(
                    ResponseFamily::Binomial,
                    InverseLink::LatentCLogLog(*latent_state),
                )
            }
            _ => {
                crate::bail_invalid_estim!(
                    "latent_cloglog is only supported with the Binomial CLogLog / LatentCLogLog link"
                );
            }
        }
    } else if let Some(sas_spec) = effective_sas_link {
        if !family.is_binomial() {
            crate::bail_invalid_estim!("sas_link is only supported for binomial families");
        }
        let use_beta_logistic = family.is_binomial_beta_logistic();
        match &family.link {
            InverseLink::Sas(_) | InverseLink::BetaLogistic(_) => {
                if use_beta_logistic {
                    let st = crate::mixture_link::state_from_beta_logisticspec(sas_spec).map_err(
                        |e| {
                            EstimationError::InvalidInput(format!(
                                "invalid Beta-Logistic link: {e}"
                            ))
                        },
                    )?;
                    LikelihoodSpec::new(ResponseFamily::Binomial, InverseLink::BetaLogistic(st))
                } else {
                    let st = crate::mixture_link::state_from_sasspec(sas_spec).map_err(|e| {
                        EstimationError::InvalidInput(format!("invalid SAS link: {e}"))
                    })?;
                    LikelihoodSpec::new(ResponseFamily::Binomial, InverseLink::Sas(st))
                }
            }
            _ => {
                crate::bail_invalid_estim!(
                    "sas_link options are only valid for adaptive SAS link families"
                );
            }
        }
    } else {
        family.clone()
    };
    if resolved_family.is_royston_parmar() {
        crate::bail_invalid_estim!(
            "fit_gam external design path does not support RoystonParmar; use survival training APIs"
        );
    }
    // Validate the external-design family/link policy before looking at response
    // support so an unsupported family/link (for example a non-canonical link)
    // reports the routing problem instead of a secondary y-domain violation.
    super::external_options::resolve_external_family(
        &resolved_family,
        Some(opts.firth_bias_reduction),
    )?;
    // Per-family response-support validation, owned by the family type.
    // Gamma `y > 0`, Poisson / NegativeBinomial / Tweedie `y ≥ 0`, Beta
    // `y ∈ (0, 1)`. Centralising the rule on `ResponseFamily` means the
    // external-design GLM path and the formula path share the same
    // family-owned domain rule, while this external path appends the routing
    // context the formula path does not have. The response column name is
    // unknown on the external-design path (the caller passes a bare
    // `y: ArrayView1<f64>`) so we surface it as the generic "y".
    if let Err(violation) = resolved_family.response.validate_response_support(y.view()) {
        crate::bail_invalid_estim!(
            "{}; external-design GLM routing accepted the family/link, but the response values are outside that GLM family's support",
            violation
        );
    }
    validate_penalty_specs(&specs, x.ncols(), "fit_gam")?;
    let ext_opts = ExternalOptimOptions {
        family: resolved_family,
        latent_cloglog: opts.latent_cloglog,
        mixture_link: opts.mixture_link.clone(),
        optimize_mixture: opts.optimize_mixture,
        sas_link: effective_sas_link,
        optimize_sas: opts.optimize_sas,
        compute_inference: opts.compute_inference,
        skip_rho_posterior_inference: opts.skip_rho_posterior_inference,
        max_iter: opts.max_iter,
        tol: opts.tol,
        nullspace_dims,
        linear_constraints: opts.linear_constraints.clone(),
        firth_bias_reduction: Some(opts.firth_bias_reduction),
        penalty_shrinkage_floor: opts.penalty_shrinkage_floor,
        // Propagate caller's rho_prior so inner outer-REML minimizes the
        // same objective as paths that build ExternalOptimOptions directly.
        rho_prior: opts.rho_prior.clone(),
        kronecker_penalty_system: opts.kronecker_penalty_system.clone(),
        kronecker_factored: opts.kronecker_factored.clone(),
        persist_warm_start_disk: opts.persist_warm_start_disk,
    };

    let result = optimize_external_designwith_heuristic_lambdas_andwarm_start(
        y,
        weights,
        &x,
        offset,
        specs.clone(),
        heuristic_lambdas,
        warm_start_beta,
        &ext_opts,
    )?;
    // `log_lambdas` is the canonical optimizer coordinate and `lambdas` was
    // derived from it through the shared closed-domain conversion. Preserve
    // that exact pair; result-time floors or ceilings would change the fitted
    // model and sever the value/derivative identity.
    let log_lambdas = result.log_lambdas;
    let result_lambdas = result.lambdas;
    let edf = result
        .inference
        .as_ref()
        .map(|inf| inf.edf_total)
        .unwrap_or(0.0);
    let geometry = result.inference.as_ref().map(|inf| FitGeometry {
        coefficient_gauge: gam_problem::Gauge::identity(&[result.beta.len()]),
        penalized_hessian: inf.penalized_hessian.clone(),
        working_weights: inf.working_weights.clone(),
        working_response: inf.working_response.clone(),
    });
    let covariance_conditional = result
        .inference
        .as_ref()
        .and_then(|inf| inf.beta_covariance.as_ref().map(|c| c.as_array().clone()));
    let covariance_corrected = result
        .inference
        .as_ref()
        .and_then(|inf| inf.beta_covariance_corrected.clone());
    let penalized_objective = result.reml_score;
    let outer_cost_evals = result.outer_cost_evals;
    let inner_pirls_solves = result.inner_pirls_solves;
    UnifiedFitResult::try_from_parts(UnifiedFitResultParts {
        blocks: vec![FittedBlock {
            beta: result.beta.clone(),
            role: BlockRole::Mean,
            edf,
            lambdas: result_lambdas.clone(),
        }],
        log_lambdas,
        lambdas: result_lambdas,
        likelihood_family: Some(result.likelihood_family),
        likelihood_scale: result.likelihood_scale,
        log_likelihood_normalization: result.log_likelihood_normalization,
        log_likelihood: result.log_likelihood,
        deviance: result.deviance,
        reml_score: result.reml_score,
        stable_penalty_term: result.stable_penalty_term,
        penalized_objective,
        used_device: result.used_device,
        outer_iterations: result.iterations,
        outer_converged: result.outer_converged,
        outer_gradient_norm: Some(result.finalgrad_norm),
        standard_deviation: result.standard_deviation,
        covariance_conditional,
        covariance_corrected,
        inference: result.inference,
        fitted_link: result.fitted_link,
        geometry,
        block_states: Vec::new(),
        pirls_status: result.pirls_status,
        max_abs_eta: result.max_abs_eta,
        constraint_kkt: result.constraint_kkt,
        artifacts: result.artifacts,
        inner_cycles: 0,
    })
    .map(|mut unified| {
        // Surface the optimizer's outer cost-eval count (not carried by the
        // parts builder) so callers/tests can guard outer work (#1575).
        unified.outer_cost_evals = outer_cost_evals;
        // Surface the actual full-n inner P-IRLS solve count — the true #1575
        // cost metric — so callers/tests can guard that the warm-start /
        // parsimony-waiver / PSIS-optin economy does not regress.
        unified.inner_pirls_solves = inner_pirls_solves;
        unified
    })
}

/// External-design GAM entrypoint for GLM-style families supported by
/// `optimize_external_design`.
/// Survival families such as `RoystonParmar` use survival-specific training APIs.
pub fn fit_gam<X>(
    x: X,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    s_list: &[BlockwisePenalty],
    family: gam_problem::LikelihoodSpec,
    opts: &FitOptions,
) -> Result<UnifiedFitResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    fit_gamwith_heuristic_lambdas(x, y, weights, offset, s_list, None, family, opts)
}
