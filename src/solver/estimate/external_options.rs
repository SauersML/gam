use super::*;

pub struct ExternalOptimResult {
    pub beta: Array1<f64>,
    pub lambdas: Array1<f64>,
    pub likelihood_family: LikelihoodSpec,
    pub likelihood_scale: LikelihoodScaleMetadata,
    pub log_likelihood_normalization: LogLikelihoodNormalization,
    pub log_likelihood: f64,
    /// Residual scale on the response scale.
    ///
    /// Contract: Gaussian identity models store the residual standard
    /// deviation sigma here. Non-Gaussian families keep the response-scale
    /// summary used by their explicit likelihood-scale metadata.
    pub standard_deviation: f64,
    pub iterations: usize,
    pub finalgrad_norm: f64,
    /// True iff the outer optimizer reached a stationary point (gradient
    /// norm below tolerance), as reported by the optimizer itself. False
    /// when the run exhausted its iteration budget without reaching the
    /// gradient tolerance. Downstream consumers should NOT assume that a
    /// fit with `outer_converged == false` is unusable — it may still be
    /// the best basin reached given the budget — but they must not treat
    /// it as certified-converged either.
    pub outer_converged: bool,
    pub pirls_status: crate::pirls::PirlsStatus,
    pub deviance: f64,
    /// Stable quadratic penalty term βᵀSβ, including any solver ridge quadratic.
    pub stable_penalty_term: f64,
    pub max_abs_eta: f64,
    pub constraint_kkt: Option<crate::pirls::ConstraintKktDiagnostics>,
    pub artifacts: FitArtifacts,
    pub inference: Option<FitInference>,
    /// Complete REML/LAML objective value used for smoothing selection.
    pub reml_score: f64,
    pub fitted_link: FittedLinkState,
}

#[derive(Clone)]
pub struct ExternalOptimOptions {
    pub family: crate::types::LikelihoodSpec,
    pub latent_cloglog: Option<LatentCLogLogState>,
    pub mixture_link: Option<MixtureLinkSpec>,
    pub optimize_mixture: bool,
    pub sas_link: Option<SasLinkSpec>,
    pub optimize_sas: bool,
    pub compute_inference: bool,
    /// Internal lifecycle knob for fits whose result will be immediately
    /// superseded. Keeps ordinary inference work but skips the live-objective
    /// rho posterior certificate/escalation until the returned model is known.
    pub skip_rho_posterior_inference: bool,
    pub max_iter: usize,
    pub tol: f64,
    pub nullspace_dims: Vec<usize>,
    pub linear_constraints: Option<crate::pirls::LinearInequalityConstraints>,
    /// Optional explicit Firth override for external fitting families that
    /// support Jeffreys/Firth bias reduction.
    /// - `Some(true)`: force Firth on
    /// - `Some(false)`: force Firth off
    /// - `None`: use family default behavior
    pub firth_bias_reduction: Option<bool>,
    /// Relative shrinkage floor for penalized block eigenvalues.
    /// See [`FitOptions::penalty_shrinkage_floor`] for details.
    pub penalty_shrinkage_floor: Option<f64>,
    /// Fixed prior on smoothing parameters for explicit joint HMC sampling
    /// flows. Standard fitting stays on the REML/Laplace path.
    pub rho_prior: crate::types::RhoPrior,
    /// Kronecker-factored penalty system for tensor-product smooth terms.
    pub kronecker_penalty_system: Option<crate::smooth::KroneckerPenaltySystem>,
    /// Full Kronecker factored basis for P-IRLS factored reparameterization.
    pub kronecker_factored: Option<crate::basis::KroneckerFactoredBasis>,
    /// Engage the cross-process ON-DISK persistent warm-start layer for this
    /// fit. Default `false`: only the in-memory warm start runs, so throwaway /
    /// replicate / CI-coverage loops pay no disk I/O (#1082). A caller that
    /// wants cross-process resume threads `true` down from
    /// `FitConfig::persist_warm_start_disk`; the standard `RemlState`
    /// constructor then calls `enable_persistent_warm_start_disk()`.
    pub persist_warm_start_disk: bool,
}

pub(crate) fn resolve_external_family(
    family: &crate::types::LikelihoodSpec,
    firth_override: Option<bool>,
) -> Result<(GlmLikelihoodSpec, bool), EstimationError> {
    if family.is_royston_parmar() {
        crate::bail_invalid_estim!(
            "optimize_external_design does not support RoystonParmar; use survival training APIs"
                .to_string(),
        );
    }

    let supports_firth = family.supports_firth();
    if firth_override == Some(true) && !supports_firth {
        crate::bail_invalid_estim!(
            "firth_bias_reduction requires a Binomial inverse link with a Fisher-weight jet; {} does not support it",
            family.pretty_name(),
        );
    }

    if let ResponseFamily::Tweedie { p } = &family.response {
        if !crate::types::is_valid_tweedie_power(*p) {
            crate::bail_invalid_estim!("optimize_external_design requires a GLM family; Tweedie variance power must be finite and strictly between 1 and 2; use PoissonLog or GammaLog for boundary cases"
                    .to_string(),);
        }
    }
    if matches!(family.response, ResponseFamily::RoystonParmar) {
        crate::bail_invalid_estim!("optimize_external_design requires a GLM family; RoystonParmar is survival-specific and not a GLM likelihood"
                .to_string(),);
    }
    Ok((
        GlmLikelihoodSpec::canonical(family.clone()),
        firth_override.unwrap_or(false) && supports_firth,
    ))
}

#[inline]
pub(crate) fn effective_sas_link_for_family(
    family: &crate::types::LikelihoodSpec,
    sas_link: Option<SasLinkSpec>,
) -> Option<SasLinkSpec> {
    if (family.is_binomial_sas() || family.is_binomial_beta_logistic()) && sas_link.is_none() {
        Some(SasLinkSpec {
            initial_epsilon: 0.0,
            initial_log_delta: 0.0,
        })
    } else {
        sas_link
    }
}

#[inline]
pub(crate) fn resolved_external_inverse_link(
    link: LinkFunction,
    latent_cloglog: Option<LatentCLogLogState>,
    mixture_link: Option<&MixtureLinkSpec>,
    sas_link: Option<SasLinkSpec>,
) -> Result<InverseLink, EstimationError> {
    if let Some(state) = latent_cloglog {
        return Ok(InverseLink::LatentCLogLog(state));
    }
    if let Some(spec) = mixture_link {
        return Ok(InverseLink::Mixture(state_fromspec(spec).map_err(|e| {
            EstimationError::InvalidInput(format!("invalid blended inverse link: {e}"))
        })?));
    }
    if let Some(spec) = sas_link {
        return Ok(match link {
            LinkFunction::BetaLogistic => {
                InverseLink::BetaLogistic(state_from_beta_logisticspec(spec).map_err(|e| {
                    EstimationError::InvalidInput(format!("invalid Beta-Logistic link: {e}"))
                })?)
            }
            _ => InverseLink::Sas(
                state_from_sasspec(spec)
                    .map_err(|e| EstimationError::InvalidInput(format!("invalid SAS link: {e}")))?,
            ),
        });
    }
    Ok(InverseLink::Standard(StandardLink::try_from(link).map_err(|e| {
        EstimationError::InvalidInput(format!(
            "inverse link resolution: {e}; supply `sas_link` or `latent_cloglog` configuration for state-bearing links"
        ))
    })?))
}

#[inline]
pub(crate) fn resolved_external_config(
    opts: &ExternalOptimOptions,
) -> Result<(RemlConfig, Option<SasLinkSpec>), EstimationError> {
    if opts.latent_cloglog.is_some() && (opts.mixture_link.is_some() || opts.sas_link.is_some()) {
        crate::bail_invalid_estim!(
            "latent_cloglog cannot be combined with mixture_link or sas_link"
        );
    }
    if opts.mixture_link.is_some() && opts.sas_link.is_some() {
        crate::bail_invalid_estim!("mixture_link and sas_link are mutually exclusive");
    }
    if opts.family.is_latent_cloglog() && opts.latent_cloglog.is_none() {
        crate::bail_invalid_estim!("BinomialLatentCLogLog requires latent_cloglog state");
    }
    if opts.latent_cloglog.is_some() && !opts.family.is_latent_cloglog() {
        crate::bail_invalid_estim!("latent_cloglog is only supported with BinomialLatentCLogLog");
    }
    let effective_sas_link = effective_sas_link_for_family(&opts.family, opts.sas_link);
    let (likelihood, firth_active) =
        resolve_external_family(&opts.family, opts.firth_bias_reduction)?;
    let link = likelihood.link_function();
    let mut cfg = RemlConfig::external(likelihood, opts.tol, firth_active);
    cfg.link_kind = resolved_external_inverse_link(
        link,
        opts.latent_cloglog,
        opts.mixture_link.as_ref(),
        effective_sas_link,
    )?;
    Ok((cfg, effective_sas_link))
}

/// Shape/bounds validation for a single [`PenaltySpec`] against the total
/// coefficient width `p`. Canonical home for the block/dense shape checks that
/// were duplicated inline in `terms::construction`'s fused validate-and-
/// destructure path; both call this so the diagnostics stay identical.
pub(crate) fn validate_penalty_spec_shape(
    idx: usize,
    spec: &PenaltySpec,
    p: usize,
    context: &str,
) -> Result<(), EstimationError> {
    match spec {
        PenaltySpec::Block {
            local, col_range, ..
        } => {
            let bd = col_range.len();
            if local.nrows() != bd || local.ncols() != bd {
                crate::bail_invalid_estim!(
                    "{context}: block penalty {idx} local matrix must be {bd}x{bd}, got {}x{}",
                    local.nrows(),
                    local.ncols()
                );
            }
            if col_range.end > p {
                crate::bail_invalid_estim!(
                    "{context}: block penalty {idx} col_range {}..{} exceeds p={p}",
                    col_range.start,
                    col_range.end
                );
            }
        }
        PenaltySpec::Dense(m) => {
            if m.nrows() != p || m.ncols() != p {
                crate::bail_invalid_estim!(
                    "{context}: dense penalty {idx} must be {p}x{p}, got {}x{}",
                    m.nrows(),
                    m.ncols()
                );
            }
        }
        PenaltySpec::DenseWithMean { matrix, .. } => {
            if matrix.nrows() != p || matrix.ncols() != p {
                crate::bail_invalid_estim!(
                    "{context}: dense penalty {idx} must be {p}x{p}, got {}x{}",
                    matrix.nrows(),
                    matrix.ncols()
                );
            }
        }
    }
    Ok(())
}
