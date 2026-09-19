use super::*;

pub struct ExternalOptimResult {
    pub beta: Array1<f64>,
    /// Canonical optimized log-strengths. Physical `lambdas` are derived from
    /// this vector through the shared exact-domain conversion.
    pub log_lambdas: Array1<f64>,
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
    pub used_device: bool,
    pub max_abs_eta: f64,
    pub constraint_kkt: Option<crate::pirls::ConstraintKktDiagnostics>,
    pub artifacts: FitArtifacts,
    pub geometry: Option<FitGeometry>,
    pub inference: Option<FitInference>,
    /// Conditional coefficient covariance `Vb`, the only store of it (#2955):
    /// the fit's standard errors derive from it.
    pub covariance_conditional: Option<Array2<f64>>,
    /// Smoothing-corrected coefficient covariance `Vp`, the only store of it.
    pub covariance_corrected: Option<Array2<f64>>,
    /// Complete REML/LAML objective value used for smoothing selection, or
    /// `None` when the converged fit sits on the zero-dispersion Gaussian
    /// boundary and therefore has no finite criterion value at all. Same
    /// contract as [`crate::estimate::UnifiedFitResult::reml_score`]; the
    /// boundary is DETECTED here, where the dispersion is actually estimated,
    /// rather than predicted by an entry-level shape rule (#2595).
    pub reml_score: Option<f64>,
    pub fitted_link: FittedLinkState,
    /// Number of outer REML cost-only evaluations executed during the fit
    /// (the count the outer optimizer's trust-region/line-search probes drive,
    /// each paying an inner P-IRLS solve). Surfaced for regression guards on
    /// outer work (#1575); not part of the statistical contract.
    pub outer_cost_evals: usize,
    /// Number of *actual* full-n inner P-IRLS solves performed (cache-missing
    /// `prepare_eval_bundlewithkey` calls). This is the true cost driver the
    /// #1575 slowdown is measured in ("~150 outer cost evals each running a
    /// full n-sized P-IRLS"): unlike `outer_cost_evals`, it excludes single-slot
    /// cache hits and prior short-circuits and includes every solve done by the
    /// seed-grid prepass, screening, multistart, and finalize phases. Surfaced
    /// for a regression guard that pins the warm-start / parsimony-waiver /
    /// PSIS-optin economy (#1575); not part of the statistical contract.
    pub inner_pirls_solves: usize,
}

#[derive(Clone)]
pub struct ExternalOptimOptions {
    pub family: gam_problem::LikelihoodSpec,
    pub latent_cloglog: Option<LatentCLogLogState>,
    pub mixture_link: Option<MixtureLinkSpec>,
    pub optimize_mixture: bool,
    pub sas_link: Option<SasLinkSpec>,
    pub optimize_sas: bool,
    pub compute_inference: bool,
    /// Internal lifecycle knob for fits whose result will be immediately
    /// superseded. Keeps ordinary inference work but skips the live-objective
    /// rho posterior adequacy diagnostic/escalation until the returned model is known.
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
    /// Fixed prior on smoothing parameters for explicit joint HMC sampling
    /// flows. Standard fitting stays on the REML/Laplace path.
    pub rho_prior: gam_problem::RhoPrior,
    /// Explicit cross-process warm-start capability for this fit. `None` is
    /// disk-silent; clones share one caller-configured store handle.
    pub persistent_warm_start_store: Option<gam_runtime::warm_start::ConfiguredWarmStartStore>,
}

pub(crate) fn resolve_external_family(
    family: &gam_problem::LikelihoodSpec,
    firth_override: Option<bool>,
) -> Result<(GlmLikelihoodSpec, bool), EstimationError> {
    let external_glm_supported = match (&family.response, family.link_function()) {
        (ResponseFamily::Gaussian, LinkFunction::Identity)
        | (ResponseFamily::Poisson, LinkFunction::Log)
        | (ResponseFamily::Gamma, LinkFunction::Log)
        | (ResponseFamily::Tweedie { .. }, LinkFunction::Log)
        | (ResponseFamily::NegativeBinomial { .. }, LinkFunction::Log)
        | (ResponseFamily::Binomial, LinkFunction::Logit)
        | (ResponseFamily::Binomial, LinkFunction::Probit)
        | (ResponseFamily::Binomial, LinkFunction::CLogLog)
        // LogLog and Cauchit are ordinary state-less probability links: they
        // narrow into `StandardLink` (gam-spec), carry a full 5-jet Fisher
        // weight (`fisher_weight_jet5` → `component_fisher_weight_jet5` for
        // LinkComponent::{LogLog,Cauchit}), and their inverse-link jets live
        // in `mixture_link.rs` — the exact same external-design/P-IRLS
        // machinery probit/cloglog ride. #2104 un-gated them at validation
        // (`link_legal_for_family`); this is the fitting half of that wiring.
        | (ResponseFamily::Binomial, LinkFunction::LogLog)
        | (ResponseFamily::Binomial, LinkFunction::Cauchit)
        | (ResponseFamily::Binomial, LinkFunction::Sas)
        | (ResponseFamily::Binomial, LinkFunction::BetaLogistic) => true,
        // Beta regression with a constant precision φ is a genuine-dispersion
        // mean family on par with Gamma/Tweedie/Negative-Binomial: the inner
        // P-IRLS carries its full fixed-φ Fisher information and the outer loop
        // estimates φ by the Pearson moment estimator (`estimate_beta_phi_from_eta`,
        // mirroring the Tweedie φ / Gamma shape / NegBin θ locks). A
        // `noise_formula` upgrades it to a dispersion-location-scale model that
        // smooths log φ; without one, the external GLM route fits the mean with
        // a single estimated φ exactly as betareg does by default.
        (ResponseFamily::Beta { .. }, LinkFunction::Logit) => true,
        _ => false,
    };
    if !external_glm_supported {
        crate::bail_invalid_estim!(
            "the external-design route requires a supported standard GLM family/link; got {}. \
             The external-design route supports Gaussian(identity), Binomial(logit/probit/cloglog/loglog/cauchit/SAS/Beta-Logistic), \
             Beta(logit), and Poisson/Gamma/Tweedie/Negative-Binomial(log). For Beta precision modeling \
             add a noise_formula to upgrade to the dispersion-location-scale route",
            family.pretty_name(),
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
        if !gam_problem::is_valid_tweedie_power(*p) {
            crate::bail_invalid_estim!("the external-design route requires a GLM family; Tweedie variance power must be finite and strictly between 1 and 2; use PoissonLog or GammaLog for boundary cases"
                    .to_string(),);
        }
    }
    Ok((
        GlmLikelihoodSpec::canonical(family.clone()),
        firth_override.unwrap_or(false) && supports_firth,
    ))
}

#[inline]
pub(crate) fn effective_sas_link_for_family(
    family: &gam_problem::LikelihoodSpec,
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
/// coefficient width `p`. The checks live in `gam-terms` beside the
/// `PenaltySpec` definition they validate; re-exported here so `estimate`'s
/// existing paths keep resolving and both crates emit identical diagnostics.
pub(crate) use gam_terms::validate_penalty_spec_shape;
