use super::*;

const SCALAR_FAMILY_NAMES_HELP: &str = "auto, gaussian, binomial/bernoulli, \
binomial-logit/bernoulli-logit, binomial-probit/bernoulli-probit, \
binomial-cloglog/bernoulli-cloglog, latent-cloglog-binomial, poisson, gamma, \
beta/beta-regression, tweedie/tw, negative-binomial/negbin/nb, \
royston-parmar, transformation-normal";

/// Project an ingest-layer [`ColumnKindTag`] (plus the column's level table)
/// onto the [`ResponseColumnKind`] consumed by the family layer.
///
/// `Categorical` carries the source-string levels through so the
/// auto-inference refusal can echo them; `Binary` short-circuits the
/// numeric scan inside [`ResponseFamily::infer_from_response`]; `Continuous`
/// maps to `Numeric` and the family layer scans `y` itself to decide
/// Gaussian vs. Binomial.
pub fn response_column_kind(data: &Dataset, y_col: usize) -> ResponseColumnKind {
    match data.column_kinds.get(y_col) {
        Some(ColumnKindTag::Categorical) => ResponseColumnKind::Categorical {
            levels: data
                .schema
                .columns
                .get(y_col)
                .map(|sc| sc.levels.clone())
                .unwrap_or_default(),
        },
        Some(ColumnKindTag::Binary) => ResponseColumnKind::Binary,
        Some(ColumnKindTag::Continuous) | None => ResponseColumnKind::Numeric,
    }
}

/// Legality of a `(response family, link)` pairing.
///
/// This is the single source of truth for which links a given response family
/// accepts. It is consulted only when the caller supplied an *explicit* family
/// together with a link (`family=..., link(type=...)`): the link must be
/// validated against that family rather than the family re-inferred from the
/// link. The legal pairings are:
///
/// * `Gaussian` + `Identity`
/// * `{Poisson, Gamma, Tweedie, NegativeBinomial}` + `Log`
/// * `Beta` + `Logit`
/// * `Binomial` + `{Logit, Probit, CLogLog, LogLog, Cauchit, Sas,
///   BetaLogistic}` (and the Logit-shaped `Mixture`, handled by the caller via
///   `mixture_components`). `LogLog` (`μ = exp(−exp(−η))`, the reflected
///   extreme-value link) and `Cauchit` (`μ = ½ + atan(η)/π`) are fully wired
///   binomial inverse links — closed-form μ in the kernel plus a full IRLS
///   d1..d5 / Fisher-weight jet in the solver — so they are legal here (#2104).
///
/// `RoystonParmar` is a flexible-parametric survival family whose link is fixed
/// at construction and is never reached through the scalar link-choice path, so
/// it accepts no link override here.
fn link_legal_for_family(response: &ResponseFamily, link: LinkFunction) -> bool {
    match response {
        ResponseFamily::Gaussian => matches!(link, LinkFunction::Identity),
        ResponseFamily::Poisson
        | ResponseFamily::Gamma
        | ResponseFamily::Tweedie { .. }
        | ResponseFamily::NegativeBinomial { .. } => matches!(link, LinkFunction::Log),
        ResponseFamily::Beta { .. } => matches!(link, LinkFunction::Logit),
        ResponseFamily::Binomial => matches!(
            link,
            LinkFunction::Logit
                | LinkFunction::Probit
                | LinkFunction::CLogLog
                | LinkFunction::LogLog
                | LinkFunction::Cauchit
                | LinkFunction::Sas
                | LinkFunction::BetaLogistic
        ),
        ResponseFamily::RoystonParmar => false,
    }
}

/// Apply an explicit mgcv-style `family(link)` link argument to an
/// already-resolved family spec.
///
/// `base` is the `(spec, link_pinned)` pair the bare family head resolved to
/// (e.g. `poisson` → `(Poisson/Log, false)`); `link_str` is the parenthesized
/// link argument (e.g. `"log"`, `"probit"`); `name` is the original
/// user-supplied family string, used only for error messages.
///
/// The link is parsed with the shared [`parse_linkname`] vocabulary, validated
/// against the family with [`link_legal_for_family`], and applied to the
/// family's response variant (preserving e.g. NB θ, Tweedie p, Beta φ). The
/// result is pinned (`link_pinned = true`): an explicit link spelled into the
/// family name pins it exactly as the hyphen spelling `binomial-probit` does,
/// so a later contradictory `link(type=...)` is rejected downstream.
///
/// This is the single seam that makes *every* legal `family(link)` pairing —
/// the canonical default-link spellings `poisson(log)` / `gamma(log)` /
/// `gaussian(identity)` as much as the link-changing `binomial(probit)` —
/// resolve uniformly, and rejects illegal links (`gaussian(logit)`) and unknown
/// link names (`poisson(banana)`) with a precise message (#1129).
fn apply_paren_link(
    base: (LikelihoodSpec, bool),
    link_str: &str,
    name: &str,
) -> Result<(LikelihoodSpec, bool), String> {
    let (base_spec, base_pinned) = base;
    let link = gam_terms::inference::formula_dsl::parse_linkname(link_str).map_err(|_| {
        let reason: String = WorkflowError::InvalidConfig {
            reason: format!(
                "family '{name}' names an unknown link '{link_str}'; \
                 use one of identity|log|logit|probit|cloglog|sas|beta-logistic"
            ),
        }
        .into();
        reason
    })?;
    if !link_legal_for_family(&base_spec.response, link) {
        return Err(WorkflowError::InvalidConfig {
            reason: format!(
                "link '{}' is not supported for family '{}'",
                link.name(),
                base_spec.response.name()
            ),
        }
        .into());
    }
    // A head that already pinned its own link (only reachable via the malformed
    // double-spec `binomial-logit(probit)`) may not be re-pointed at a different
    // link — mirror the `link(type=...)` pin-conflict guard.
    if base_pinned && base_spec.link.link_function() != link {
        return Err(WorkflowError::InvalidConfig {
            reason: format!(
                "family '{}' pins link '{}', which conflicts with requested link '{}'",
                base_spec.name(),
                base_spec.link.link_function().name(),
                link.name(),
            ),
        }
        .into());
    }
    // Build the inverse link. State-less links narrow into `StandardLink`; the
    // state-bearing `Sas` / `BetaLogistic` links (legal only for Binomial, which
    // `link_legal_for_family` already enforced) carry the canonical zero seed,
    // exactly as the `link(type=...)` path constructs them — their effective
    // state is rebuilt later from `FitOptions`.
    let inverse_link = match link {
        LinkFunction::Sas => {
            let state = state_from_sasspec(SasLinkSpec {
                initial_epsilon: 0.0,
                initial_log_delta: 0.0,
            })
            .map_err(|err| format!("SAS link initial state: {err}"))?;
            InverseLink::Sas(state)
        }
        LinkFunction::BetaLogistic => {
            let state = state_from_beta_logisticspec(SasLinkSpec {
                initial_epsilon: 0.0,
                initial_log_delta: 0.0,
            })
            .map_err(|err| format!("Beta-Logistic link initial state: {err}"))?;
            InverseLink::BetaLogistic(state)
        }
        // The remaining links are state-less and narrow into `StandardLink`;
        // `try_from` only rejects the two state-bearing links handled above, so
        // its error is surfaced (not panicked) to keep this seam total.
        standard => InverseLink::Standard(StandardLink::try_from(standard).map_err(|err| {
            let reason: String = WorkflowError::InvalidConfig {
                reason: format!(
                    "link '{}' has no state-less representation: {err}",
                    standard.name()
                ),
            }
            .into();
            reason
        })?),
    };
    Ok((LikelihoodSpec::new(base_spec.response, inverse_link), true))
}

/// #2026: whether a resolved Tweedie family should have its variance power `p`
/// estimated by profile likelihood (mgcv `tw()` semantics) instead of held at
/// the interior fallback baked in by [`resolve_family`].
///
/// mgcv's `tw()` family has no canonical `p` — it profiles `p` over the open
/// interval `(1, 2)`. This returns `true` exactly when the user named the bare
/// Tweedie family (`tweedie` / `tw` / `tweedie-log`, case- and separator-
/// insensitive) WITHOUT an explicit numeric power argument. An explicit power
/// (`tweedie(1.6)` / `tweedie(p=1.6)`) pins `p` and returns `false`; a
/// non-numeric parenthesized argument (`tweedie(log)`) is a link, not a power,
/// so it still estimates. The head/argument split mirrors the parser inside
/// [`resolve_family`] so the two decisions cannot drift.
pub fn tweedie_power_is_estimated(family: Option<&str>) -> bool {
    let Some(name) = family else {
        return false;
    };
    let lowered = name.to_ascii_lowercase().replace('_', "-");
    let (head, arg): (&str, Option<&str>) = if let Some(open) = lowered.find('(')
        && lowered.ends_with(')')
    {
        let head = lowered[..open].trim_end_matches('-').trim();
        let inner = lowered[open + 1..lowered.len() - 1].trim();
        if head.is_empty() || inner.is_empty() {
            (lowered.as_str(), None)
        } else {
            (head, Some(inner))
        }
    } else {
        (lowered.as_str(), None)
    };
    if !matches!(head, "tweedie" | "tw" | "tweedie-log") {
        return false;
    }
    // An explicit numeric power (`1.6` or `p=1.6`) pins `p`; a missing argument
    // or a non-numeric link argument leaves `p` to be estimated from the data.
    match arg {
        Some(a) => {
            let numeric = a.strip_prefix("p=").unwrap_or(a).trim();
            numeric.parse::<f64>().is_err()
        }
        None => true,
    }
}

/// Resolve a family from an optional name, optional link choice, and response data.
///
/// `y_kind` describes the *source* representation of the response column
/// (string-valued `Categorical`, numeric `Binary` short-circuit, or generic
/// `Numeric`). It is consulted only on the auto-detect path — explicit
/// `family=...` always wins — but is required there because the same numeric
/// `y = [0.0, 1.0, ...]` payload may come from a real binary outcome or from
/// a categorical column whose two levels happened to encode to those indices.
/// Routing the kind through [`ResponseFamily::infer_from_response`] is what
/// stops the auto-detector from silently inferring Binomial off encoded
/// strings (see `tests/issues/issue_304`).
pub fn resolve_family(
    family: Option<&str>,
    negative_binomial_theta: Option<f64>,
    link_choice: Option<&LinkChoice>,
    y: ArrayView1<'_, f64>,
    y_kind: ResponseColumnKind,
    response_name: &str,
) -> Result<LikelihoodSpec, String> {
    // Resolve the optional theta only inside a structurally selected NB arm.
    // Outside NB, the option is an invalid configuration rather than a global
    // scalar that happens to be fabricated as one.
    let resolve_negative_binomial_theta = || -> Result<(f64, bool), String> {
        const ESTIMATED_THETA_SEED: f64 = 1.0;
        let (theta, fixed) = match negative_binomial_theta {
            Some(theta) => (theta, true),
            None => (ESTIMATED_THETA_SEED, false),
        };
        if !(theta.is_finite() && theta > 0.0) {
            return Err(format!(
                "negative-binomial theta must be finite and > 0; got {theta}"
            ));
        }
        Ok((theta, fixed))
    };
    // `link_pinned = true` means the family name carried a specific link suffix
    // (e.g. "binomial-probit"); `false` means the user only declared the response
    // family (e.g. "binomial") and any link_choice may legally refine the link
    // without being treated as a contradiction.
    let explicit: Option<(LikelihoodSpec, bool)> = match family {
        Some(name) => {
            // Accept both '-' and '_' as separators so e.g. "binomial_logit" and
            // "negative-binomial" resolve identically. Also accept mgcv's
            // parenthesized form `family(link)` (e.g. "binomial(logit)",
            // "Binomial(Probit)") which is how mgcv writes a GLM family with an
            // explicit link in R. Canonicalize all forms to `family-link`.
            let lowered = name.to_ascii_lowercase().replace('_', "-");
            // mgcv writes a GLM family carrying an explicit link as
            // `family(link)` (e.g. "poisson(log)", "Gamma(log)",
            // "gaussian(identity)", "binomial(probit)"). Parse that form
            // *structurally* — separate the family head from the link argument —
            // rather than flattening it to a `family-link` string and depending
            // on a hand-written match arm existing for that exact pair.
            // Flattening is why the canonical default-link spellings
            // `poisson(log)` / `gamma(log)` / `gaussian(identity)` were rejected
            // as "unknown family": those families only ever had a bare arm, never
            // a `poisson-log` / `gamma-log` / `gaussian-identity` arm (#1129).
            // Resolving the head as a family and validating the link against it
            // (`apply_paren_link`) makes every legal pairing accept uniformly and
            // rejects illegal ones with a precise message. Non-parenthesized
            // names — bare (`poisson`) and the historical hyphen spellings
            // (`binomial-probit`) — match the table directly as before.
            let (head_name, paren_link): (&str, Option<&str>) = if let Some(open) =
                lowered.find('(')
                && lowered.ends_with(')')
            {
                let head = lowered[..open].trim_end_matches('-').trim();
                let inner = lowered[open + 1..lowered.len() - 1].trim();
                if head.is_empty() || inner.is_empty() {
                    // Malformed parens ("()", "poisson()", "(log)") — match the
                    // whole lowered string, which falls through to the standard
                    // "unknown family" error below.
                    (lowered.as_str(), None)
                } else {
                    (head, Some(inner))
                }
            } else {
                (lowered.as_str(), None)
            };
            // mgcv's `tw()` carries the Tweedie variance power as its
            // parenthesized argument (`tweedie(1.6)` / `tweedie(p=1.6)`), NOT a
            // link name. When the head is Tweedie and the argument parses as a
            // number, interpret it as the power `p` and consume the argument so
            // it is not misrouted to the link resolver — which previously
            // rejected `tweedie(1.5)` as `unknown link '1.5'` (#2026), leaving
            // no user-facing way to set `p`. A non-numeric argument
            // (e.g. `tweedie(log)`) still flows through to the link resolver.
            let (paren_link, tweedie_p_override): (Option<&str>, Option<f64>) =
                if matches!(head_name, "tweedie" | "tw" | "tweedie-log")
                    && let Some(arg) = paren_link
                {
                    let numeric = arg.strip_prefix("p=").unwrap_or(arg).trim();
                    match numeric.parse::<f64>() {
                        Ok(p) => {
                            // Reuse the single Tweedie-power validity gate
                            // (`p` finite and strictly in (1, 2)) that the
                            // latent FFI and PIRLS deviance paths enforce, so a
                            // bad power fails here with an actionable message
                            // instead of an opaque downstream NaN deviance.
                            if !gam_spec::is_valid_tweedie_power(p) {
                                return Err(WorkflowError::InvalidConfig {
                                    reason: format!(
                                        "tweedie power p must be finite and strictly \
                                         between 1 and 2; got {p}"
                                    ),
                                }
                                .into());
                            }
                            (None, Some(p))
                        }
                        Err(_) => (Some(arg), None),
                    }
                } else {
                    (paren_link, None)
                };
            let resolved = match head_name {
                "gaussian" => (
                    LikelihoodSpec::new(
                        ResponseFamily::Gaussian,
                        InverseLink::Standard(StandardLink::Identity),
                    ),
                    false,
                ),
                "binomial" | "bernoulli" => (
                    LikelihoodSpec::new(
                        ResponseFamily::Binomial,
                        InverseLink::Standard(StandardLink::Logit),
                    ),
                    false,
                ),
                "binomial-logit" | "bernoulli-logit" => (
                    LikelihoodSpec::new(
                        ResponseFamily::Binomial,
                        InverseLink::Standard(StandardLink::Logit),
                    ),
                    true,
                ),
                "binomial-probit" | "bernoulli-probit" => (
                    LikelihoodSpec::new(
                        ResponseFamily::Binomial,
                        InverseLink::Standard(StandardLink::Probit),
                    ),
                    true,
                ),
                "binomial-cloglog" | "bernoulli-cloglog" => (
                    LikelihoodSpec::new(
                        ResponseFamily::Binomial,
                        InverseLink::Standard(StandardLink::CLogLog),
                    ),
                    true,
                ),
                "latent-cloglog-binomial" => (
                    LikelihoodSpec::new(
                        ResponseFamily::Binomial,
                        InverseLink::LatentCLogLog(
                            LatentCLogLogState::new(1.0)
                                .map_err(|err| format!("latent cloglog default state: {err}"))?,
                        ),
                    ),
                    true,
                ),
                "poisson" => (
                    LikelihoodSpec::new(
                        ResponseFamily::Poisson,
                        InverseLink::Standard(StandardLink::Log),
                    ),
                    false,
                ),
                // #983: a user-supplied `--negative-binomial-theta` holds θ
                // fixed at exactly that value (`theta_fixed = true` →
                // `FixedNegBinTheta` scale → the PIRLS refresh gate
                // `negbin_theta_is_estimated()` stays closed). With no flag,
                // θ is the running ML estimate (the #802 default seed 1.0).
                "nb" | "negbin" | "negative-binomial" => {
                    let (theta, theta_fixed) = resolve_negative_binomial_theta()?;
                    (
                        LikelihoodSpec::new(
                            ResponseFamily::NegativeBinomial { theta, theta_fixed },
                            InverseLink::Standard(StandardLink::Log),
                        ),
                        false,
                    )
                }
                "negative-binomial-log" => {
                    let (theta, theta_fixed) = resolve_negative_binomial_theta()?;
                    (
                        LikelihoodSpec::new(
                            ResponseFamily::NegativeBinomial { theta, theta_fixed },
                            InverseLink::Standard(StandardLink::Log),
                        ),
                        true,
                    )
                }
                "beta" | "beta-regression" => (
                    LikelihoodSpec::new(
                        ResponseFamily::Beta { phi: 1.0 },
                        InverseLink::Standard(StandardLink::Logit),
                    ),
                    false,
                ),
                "beta-logit" | "beta-regression-logit" => (
                    LikelihoodSpec::new(
                        ResponseFamily::Beta { phi: 1.0 },
                        InverseLink::Standard(StandardLink::Logit),
                    ),
                    true,
                ),
                "gamma" => (
                    LikelihoodSpec::new(
                        ResponseFamily::Gamma,
                        InverseLink::Standard(StandardLink::Log),
                    ),
                    false,
                ),
                // Royston-Parmar flexible-parametric survival and the
                // transformation-normal response model are CLI/formula families
                // whose materialization is dispatched before the scalar GLM
                // family resolver runs (survival via `Surv(...)`, transformation
                // via the dedicated transformation-normal path). They are listed
                // here so this resolver is the single total source of truth for
                // every family name the surface accepts: `royston-parmar` maps to
                // the canonical flexible-parametric likelihood, and
                // `transformation-normal` shares Gaussian-identity scalar
                // semantics (the transformation is learned outside this spec).
                "royston-parmar" => (LikelihoodSpec::royston_parmar(), true),
                "transformation-normal" => (
                    LikelihoodSpec::new(
                        ResponseFamily::Gaussian,
                        InverseLink::Standard(StandardLink::Identity),
                    ),
                    true,
                ),
                // Tweedie compound-Poisson-Gamma family. The variance power p
                // must lie strictly in (1, 2). mgcv's `tw()` has NO canonical
                // p — it *estimates* p by profile likelihood; here p can be set
                // explicitly via the mgcv-style `tweedie(1.6)` / `tweedie(p=1.6)`
                // parenthesized argument (parsed above into `tweedie_p_override`,
                // #2026). Absent an explicit power we fall back to p = 1.5, a
                // neutral interior default; the fitted mean (log-link
                // quasi-likelihood) is robust to a misspecified p, but the
                // observation-interval calibration depends on it, so callers on
                // data whose true p != 1.5 should set it. The link is fixed to
                // log (the only link wired through the Tweedie working-response
                // and dispersion machinery). "tw" matches mgcv's family alias.
                "tweedie" | "tw" => (
                    LikelihoodSpec::new(
                        ResponseFamily::Tweedie {
                            p: tweedie_p_override.unwrap_or(1.5),
                        },
                        InverseLink::Standard(StandardLink::Log),
                    ),
                    false,
                ),
                "tweedie-log" => (
                    LikelihoodSpec::new(
                        ResponseFamily::Tweedie {
                            p: tweedie_p_override.unwrap_or(1.5),
                        },
                        InverseLink::Standard(StandardLink::Log),
                    ),
                    true,
                ),
                "multinomial" | "multinomial-logit" | "categorical" | "categorical-logit"
                | "softmax" => {
                    // Multinomial-logit is a vector-response family with K-1
                    // active linear predictors and a per-row dense Fisher
                    // block — it cannot be represented by the scalar
                    // `LikelihoodSpec` (one `ResponseFamily` × one
                    // `InverseLink`) that this entry point produces.
                    //
                    // The principled coefficient-space solver lives in
                    // `crate::multinomial::fit_penalized_multinomial`,
                    // which routes the canonical
                    // `MultinomialLogitLikelihood: VectorLikelihood` through
                    // `gam_solve::pirls::dense_block_xtwx` in output-major
                    // coefficient ordering. The forthcoming
                    // `gamfit.fit_multinomial(...)` Python entry exposes that
                    // path with formula → design wiring; until that wrapper
                    // lands, callers reach the driver directly through the
                    // FFI surface.
                    return Err(WorkflowError::InvalidConfig {
                        reason: format!(
                            "family '{name}' is a vector-response family; use \
                             the dedicated multinomial entry point \
                             (`crate::multinomial::fit_penalized_multinomial` \
                             in Rust, or `gamfit.fit_multinomial(...)` in Python) \
                             rather than the scalar `fit(family=...)` path"
                        ),
                    }
                    .into());
                }
                _ => {
                    return Err(WorkflowError::InvalidConfig {
                        reason: format!(
                            "unknown family '{name}'; expected one of: {SCALAR_FAMILY_NAMES_HELP}"
                        ),
                    }
                    .into());
                }
            };
            // Apply an explicit mgcv-style `(link)` argument to the resolved
            // family, validating legality. A bare family name leaves the
            // family's default link untouched.
            let resolved = match paren_link {
                Some(link_str) => apply_paren_link(resolved, link_str, name)?,
                None => resolved,
            };
            Some(resolved)
        }
        None => {
            if negative_binomial_theta.is_some() {
                return Err(WorkflowError::InvalidConfig {
                    reason: "negative_binomial_theta requires family='negative-binomial'"
                        .to_string(),
                }
                .into());
            }
            None
        }
    };

    if let Some(choice) = link_choice {
        let from_link: LikelihoodSpec = if let Some(components) = choice.mixture_components.as_ref()
        {
            let n = components.len();
            let free = n.saturating_sub(1);
            let mix_spec = MixtureLinkSpec {
                components: components.clone(),
                initial_rho: Array1::<f64>::zeros(free),
            };
            let state = state_fromspec(&mix_spec)
                .map_err(|err| format!("mixture link initial state: {err}"))?;
            LikelihoodSpec::new(ResponseFamily::Binomial, InverseLink::Mixture(state))
        } else {
            match choice.link {
                LinkFunction::Identity => LikelihoodSpec::new(
                    ResponseFamily::Gaussian,
                    InverseLink::Standard(StandardLink::Identity),
                ),
                LinkFunction::Log => {
                    if y.iter()
                        .all(|&yi| yi.is_finite() && yi >= 0.0 && (yi - yi.round()).abs() <= 1e-9)
                    {
                        LikelihoodSpec::new(
                            ResponseFamily::Poisson,
                            InverseLink::Standard(StandardLink::Log),
                        )
                    } else {
                        LikelihoodSpec::new(
                            ResponseFamily::Gamma,
                            InverseLink::Standard(StandardLink::Log),
                        )
                    }
                }
                LinkFunction::Logit => LikelihoodSpec::new(
                    ResponseFamily::Binomial,
                    InverseLink::Standard(StandardLink::Logit),
                ),
                LinkFunction::Probit => LikelihoodSpec::new(
                    ResponseFamily::Binomial,
                    InverseLink::Standard(StandardLink::Probit),
                ),
                LinkFunction::CLogLog => LikelihoodSpec::new(
                    ResponseFamily::Binomial,
                    InverseLink::Standard(StandardLink::CLogLog),
                ),
                LinkFunction::LogLog => LikelihoodSpec::new(
                    ResponseFamily::Binomial,
                    InverseLink::Standard(StandardLink::LogLog),
                ),
                LinkFunction::Cauchit => LikelihoodSpec::new(
                    ResponseFamily::Binomial,
                    InverseLink::Standard(StandardLink::Cauchit),
                ),
                LinkFunction::Sas => {
                    // The SAS initial state (epsilon, log_delta) is carried into
                    // the fit through `FitOptions.sas_link`, not the family spec:
                    // the standard path's `effective_sas_link_for_family` rebuilds
                    // the inverse link from that option, overriding whatever the
                    // family embeds here. The canonical zero seed is therefore the
                    // correct, link-only placeholder for family resolution.
                    let state = state_from_sasspec(SasLinkSpec {
                        initial_epsilon: 0.0,
                        initial_log_delta: 0.0,
                    })
                    .map_err(|err| format!("SAS link initial state: {err}"))?;
                    LikelihoodSpec::new(ResponseFamily::Binomial, InverseLink::Sas(state))
                }
                LinkFunction::BetaLogistic => {
                    let state = state_from_beta_logisticspec(SasLinkSpec {
                        initial_epsilon: 0.0,
                        initial_log_delta: 0.0,
                    })
                    .map_err(|err| format!("Beta-Logistic link initial state: {err}"))?;
                    LikelihoodSpec::new(ResponseFamily::Binomial, InverseLink::BetaLogistic(state))
                }
            }
        };
        if let Some((explicit_spec, link_pinned)) = explicit.as_ref() {
            // An explicit response family was supplied: never re-infer the
            // family from the link. Validate that the requested link is legal
            // for *this* family, then apply the link (carrying any embedded
            // Sas/BetaLogistic/Mixture state, which `from_link.link` already
            // holds) to the explicit family's response variant (preserving e.g.
            // NB theta, Tweedie p, Beta phi).
            if matches!(
                choice.mode,
                gam_terms::inference::formula_dsl::LinkMode::Flexible
            ) && !matches!(explicit_spec.response, ResponseFamily::Binomial)
            {
                return Err(WorkflowError::InvalidConfig {
                    reason: format!(
                        "flexible(...) links (the jointly-fit anchored spline link offset) are \
                         implemented only for a binomial response; the resolved family is {} (a \
                         non-binomial family), for which the link offset has no solver and would \
                         otherwise be silently discarded. Use the plain base link, or fit a binomial \
                         response.",
                        explicit_spec.pretty_name()
                    ),
                }
                .into());
            }
            let mixture_requested = choice.mixture_components.is_some();
            let legal = if mixture_requested {
                // The mixture link is a Binomial latent construct; it has no
                // legal pairing with any other response family.
                matches!(explicit_spec.response, ResponseFamily::Binomial)
            } else {
                link_legal_for_family(&explicit_spec.response, choice.link)
            };
            if !legal {
                return Err(WorkflowError::InvalidConfig {
                    reason: format!(
                        "link '{}' is not supported for family '{}'",
                        choice.link.name(),
                        explicit_spec.response.name()
                    ),
                }
                .into());
            }
            // A family name that pinned its own link (e.g. "binomial-probit")
            // may not be re-pointed at a different link by `link(type=...)`.
            if *link_pinned && explicit_spec.link.link_function() != from_link.link.link_function()
            {
                return Err(WorkflowError::InvalidConfig {
                    reason: format!(
                        "family '{}' pins link '{}', which conflicts with requested link '{}'",
                        explicit_spec.name(),
                        explicit_spec.link.link_function().name(),
                        choice.link.name(),
                    ),
                }
                .into());
            }
            return Ok(LikelihoodSpec::new(
                explicit_spec.response.clone(),
                from_link.link,
            ));
        }
        return Ok(from_link);
    }

    if let Some((spec, _)) = explicit {
        return Ok(spec);
    }

    // Auto-detect: delegate to `ResponseFamily::infer_from_response` so the
    // refusal policy for non-numeric response columns lives in one place
    // (the family layer), not duplicated across every entry point. The link
    // is derived from the inferred response: Binomial → Logit, Poisson → Log,
    // Gaussian → Identity. The link_choice branch above already covered the case where
    // the user pinned a link without a family.
    let response = ResponseFamily::infer_from_response(y, y_kind).map_err(|refusal| {
        let err: String = WorkflowError::InvalidConfig {
            reason: refusal.message_for(response_name),
        }
        .into();
        err
    })?;
    let link = match response {
        ResponseFamily::Binomial => InverseLink::Standard(StandardLink::Logit),
        ResponseFamily::Poisson => InverseLink::Standard(StandardLink::Log),
        _ => InverseLink::Standard(StandardLink::Identity),
    };
    Ok(LikelihoodSpec::new(response, link))
}

#[cfg(test)]
mod tweedie_power_tests {
    //! #2026: the mgcv-style parenthesized Tweedie power `tweedie(p)` must be
    //! parsed as the variance power (not misrouted to the link resolver), so
    //! callers whose true `p != 1.5` can set it and get calibrated observation
    //! intervals (`Var(Y|x) = phi * mu^p`).
    use super::*;
    use ndarray::array;

    /// Resolve `family` and return the Tweedie variance power it carries.
    fn tweedie_p(family: &str) -> f64 {
        let y = array![0.0, 1.2, 3.4, 0.0, 5.6];
        let spec = resolve_family(
            Some(family),
            None,
            None,
            y.view(),
            ResponseColumnKind::Numeric,
            "y",
        )
        .expect("family should resolve");
        match spec.response {
            ResponseFamily::Tweedie { p } => p,
            _ => panic!("expected a Tweedie response family from `{family}`"),
        }
    }

    #[test]
    fn tweedie_paren_power_parses() {
        // Each of these was rejected before #2026 as `unknown link '<num>'`.
        assert_eq!(tweedie_p("tweedie(1.7)"), 1.7);
        assert_eq!(tweedie_p("tw(1.3)"), 1.3);
        assert_eq!(tweedie_p("tweedie(p=1.6)"), 1.6);
        assert_eq!(tweedie_p("Tweedie(1.25)"), 1.25);
        assert_eq!(tweedie_p("tweedie-log(1.9)"), 1.9);
    }

    #[test]
    fn tweedie_bare_defaults_to_interior_power() {
        // No explicit power → the neutral interior default (documented as a
        // fallback, not a canonical value).
        assert_eq!(tweedie_p("tweedie"), 1.5);
        assert_eq!(tweedie_p("tw"), 1.5);
    }

    #[test]
    fn binomial_loglog_and_cauchit_links_are_legal() {
        // #2104: `loglog` (μ = exp(−exp(−η))) and `cauchit` (μ = ½ + atan(η)/π)
        // are fully-implemented binomial inverse links — closed-form μ in the
        // kernel plus a full IRLS d1..d5 / Fisher-weight jet in the solver — and
        // are advertised by the parser vocabulary, but the legality gate omitted
        // them, so `binomial(loglog)` / `binomial(cauchit)` were rejected as
        // "not supported for family 'binomial'". Exercise the real legality
        // predicate directly (it is private to this module) and the end-to-end
        // resolver seam through which the user reaches it.
        assert!(
            link_legal_for_family(&ResponseFamily::Binomial, LinkFunction::LogLog),
            "binomial + loglog must be a legal pairing"
        );
        assert!(
            link_legal_for_family(&ResponseFamily::Binomial, LinkFunction::Cauchit),
            "binomial + cauchit must be a legal pairing"
        );
        // The other three canonical binomial links stay legal (no regression),
        // and a non-binomial family still rejects these two links.
        assert!(link_legal_for_family(
            &ResponseFamily::Binomial,
            LinkFunction::CLogLog
        ));
        assert!(!link_legal_for_family(
            &ResponseFamily::Gaussian,
            LinkFunction::LogLog
        ));
        assert!(!link_legal_for_family(
            &ResponseFamily::Gaussian,
            LinkFunction::Cauchit
        ));

        // End-to-end resolver path (mgcv-style `family(link)`) must now accept
        // both links and carry the requested inverse link into the spec.
        let y = array![0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        for (raw, want) in [
            ("binomial(loglog)", LinkFunction::LogLog),
            ("binomial(cauchit)", LinkFunction::Cauchit),
            ("Binomial(LogLog)", LinkFunction::LogLog),
            ("bernoulli(cauchit)", LinkFunction::Cauchit),
        ] {
            let spec = resolve_family(
                Some(raw),
                None,
                None,
                y.view(),
                ResponseColumnKind::Numeric,
                "y",
            )
            .unwrap_or_else(|err| panic!("resolve_family({raw:?}) must succeed, got: {err}"));
            assert!(
                matches!(spec.response, ResponseFamily::Binomial),
                "{raw}: expected Binomial response"
            );
            assert_eq!(
                spec.link.link_function(),
                want,
                "{raw}: expected {want:?} link"
            );
        }
    }

    #[test]
    fn tweedie_paren_power_rejects_out_of_range() {
        let y = array![0.0, 1.2, 3.4];
        for bad in [
            "tweedie(1.0)",
            "tweedie(2.0)",
            "tweedie(2.5)",
            "tweedie(0.5)",
        ] {
            let err = resolve_family(
                Some(bad),
                None,
                None,
                y.view(),
                ResponseColumnKind::Numeric,
                "y",
            )
            .expect_err("power outside (1, 2) must be rejected");
            assert!(
                err.contains("tweedie power"),
                "unexpected error for `{bad}`: {err}"
            );
        }
    }
}
