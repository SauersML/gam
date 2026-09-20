use super::*;

const SCALAR_FAMILY_NAMES_HELP: &str = "auto, gaussian, gaussian-identity, \
binomial/bernoulli, binomial-logit/bernoulli-logit/logistic, \
binomial-probit/bernoulli-probit/probit, \
binomial-cloglog/bernoulli-cloglog/cloglog, latent-cloglog-binomial, \
poisson, poisson-log, gamma, gamma-log, \
inverse-gaussian, beta/beta-regression, \
beta-logit/beta-regression-logit, tweedie, tweedie-log, \
negative-binomial, negative-binomial-log, \
student-t, royston-parmar, transformation-normal; any family also accepts \
a parenthesized link argument, e.g. gamma(inverse)";

/// Every scalar family head [`scalar_family_from_name`] resolves, in the
/// spelling it accepts. Used to point an underscore spelling at its hyphen
/// form; `every_listed_scalar_family_head_resolves` keeps it in sync with the
/// resolver's match.
const SCALAR_FAMILY_HEADS: &[&str] = &[
    "gaussian",
    "gaussian-identity",
    "binomial",
    "bernoulli",
    "binomial-logit",
    "bernoulli-logit",
    "logistic",
    "binomial-probit",
    "bernoulli-probit",
    "probit",
    "binomial-cloglog",
    "bernoulli-cloglog",
    "cloglog",
    "latent-cloglog-binomial",
    "poisson",
    "poisson-log",
    "negative-binomial",
    "negative-binomial-log",
    "beta",
    "beta-regression",
    "beta-logit",
    "beta-regression-logit",
    "student-t",
    "gamma",
    "gamma-log",
    "inverse-gaussian",
    "royston-parmar",
    "transformation-normal",
    "tweedie",
    "tweedie-log",
];

/// Family spellings that are not accepted, each with the one spelling that
/// names the same family (SPEC R25: one spelling per behavior).
const REMOVED_FAMILY_SPELLINGS: &[(&str, &str)] = &[
    ("tw", "tweedie"),
    ("nb", "negative-binomial"),
    ("negbin", "negative-binomial"),
    ("negbin-log", "negative-binomial-log"),
    ("t", "student-t"),
    ("inverse.gaussian", "inverse-gaussian"),
    ("inversegaussian", "inverse-gaussian"),
    ("inv-gauss", "inverse-gaussian"),
    ("invgauss", "inverse-gaussian"),
    ("inv-gaussian", "inverse-gaussian"),
];

/// The accepted spelling for an unaccepted family head, if there is exactly
/// one: a removed alias, or an underscore spelling of a hyphenated family.
fn canonical_family_head(head: &str) -> Option<&'static str> {
    let lookup = |spelling: &str| -> Option<&'static str> {
        REMOVED_FAMILY_SPELLINGS
            .iter()
            .find(|(removed, _)| *removed == spelling)
            .map(|(_, canonical)| *canonical)
            .or_else(|| SCALAR_FAMILY_HEADS.iter().copied().find(|h| *h == spelling))
    };
    if let Some(canonical) = REMOVED_FAMILY_SPELLINGS
        .iter()
        .find(|(removed, _)| *removed == head)
        .map(|(_, canonical)| *canonical)
    {
        return Some(canonical);
    }
    if head.contains('_') {
        return lookup(&head.replace('_', "-"));
    }
    None
}

/// Project an ingest-layer [`ColumnKindTag`] (plus the column's level table)
/// onto the [`ResponseColumnKind`] consumed by the family layer.
///
/// `Categorical` carries the source-string levels through so a two-level
/// column can be coded as a binary outcome
/// ([`code_two_level_label_response`]) and any other level count's
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

/// Code a two-level label response to the `0`/`1` outcome a Binomial family
/// models, in canonical sorted level order.
///
/// A string or categorical response arrives as level indices whose order is
/// the ingestion path's (encounter order for an Arrow string column, the
/// declared categories for a pandas categorical, sorted for a CSV). The event
/// must not depend on which of those carried the data, so the two levels are
/// ranked by [`gam_data::natural_level_cmp`]: the first is coded `0`, the
/// second `1`, and the fit models `P(y = second level)`. `{"no", "yes"}` models
/// `P(yes)`; `{"0", "1"}` models `P(1)`. The coding is recorded as an
/// informational fit note so the summary states which level is the event.
///
/// A no-op unless `family` is Binomial and `y_kind` is a two-level categorical
/// column; every other response is already on its family's scale.
pub(crate) fn code_two_level_label_response(
    family: &LikelihoodSpec,
    y_kind: &ResponseColumnKind,
    y: &mut Array1<f64>,
    response_name: &str,
    notes: &mut FitNotes,
) {
    let ResponseColumnKind::Categorical { levels } = y_kind else {
        return;
    };
    if !matches!(family.response, ResponseFamily::Binomial) || levels.len() != 2 {
        return;
    }
    let event_code = match gam_data::natural_level_cmp(&levels[0], &levels[1]) {
        std::cmp::Ordering::Greater => 0.0,
        _ => 1.0,
    };
    let (reference, event) = if event_code == 1.0 {
        (&levels[0], &levels[1])
    } else {
        (&levels[1], &levels[0])
    };
    y.mapv_inplace(|code| if code == event_code { 1.0 } else { 0.0 });
    notes.inform(format!(
        "response '{response_name}' has two levels, coded in sorted order as \
         '{reference}' = 0 and '{event}' = 1; the binomial fit models \
         P({response_name} = '{event}')"
    ));
}

/// Reject a `(response family, link)` pairing the likelihood legality table
/// ([`LikelihoodSpec::is_legal_cell`]) does not admit.
///
/// Consulted only when the caller supplied an *explicit* family together with
/// a link (`family=..., link(type=...)` or `family(link)`): the link is
/// validated against that family rather than the family re-inferred from the
/// link. The message lists the family's legal links, generated from the same
/// table by [`LikelihoodSpec::legal_links_for`], so it cannot drift from what
/// the solver accepts.
fn require_legal_link(response: &ResponseFamily, link: LinkFunction) -> Result<(), String> {
    let legal = LikelihoodSpec::legal_links_for(response);
    if legal.contains(&link) {
        return Ok(());
    }
    Err(WorkflowError::InvalidConfig {
        reason: format!(
            "link `{}` is not supported for family `{}`; {}{}",
            link.name(),
            response.name(),
            LikelihoodSpec::legal_links_clause(response),
            LikelihoodSpec::illegal_cell_hint(response, link)
                .map(|hint| format!("; {hint}"))
                .unwrap_or_default()
        ),
    }
    .into())
}

/// Apply an explicit parenthesized `family(link)` link argument to an
/// already-resolved family spec.
///
/// `base` is the `(spec, link_pinned)` pair the bare family head resolved to
/// (e.g. `poisson` → `(Poisson/Log, false)`); `link_str` is the parenthesized
/// link argument (e.g. `"log"`, `"probit"`); `name` is the original
/// user-supplied family string, used only for error messages.
///
/// The link is parsed with the shared [`parse_linkname`] vocabulary, validated
/// against the family with [`require_legal_link`], and applied to the
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
    let link = LinkFunction::from_name(link_str).ok_or_else(|| {
        let reason: String = WorkflowError::InvalidConfig {
            reason: format!(
                "family '{name}' names an unknown link: {}",
                gam_problem::UnknownLinkName(link_str.trim().to_string())
            ),
        }
        .into();
        reason
    })?;
    require_legal_link(&base_spec.response, link)?;
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
    // `require_legal_link` already enforced) carry the canonical zero seed,
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

/// Nuisance parameters that a family NAME cannot carry on its own.
///
/// The response family is fully determined by its name; `theta`, the Tweedie
/// variance power and the Beta precision are not. Every surface that names a
/// family supplies them through this one struct, so "the user pinned it"
/// versus "estimate it from the data" is a distinction the type can express
/// (`Some` vs `None`) rather than one each caller re-invents — the FFI
/// previously collapsed both onto a bare `f64` default and therefore had to
/// treat every supplied theta as a mere seed. Each override belongs to
/// exactly one family and is refused for any other.
#[derive(Clone, Copy, Debug, Default)]
pub struct FamilyNuisanceOverrides {
    /// `Some(theta)` pins the negative-binomial theta; `None` seeds it and
    /// leaves it to be estimated (#983).
    pub negative_binomial_theta: Option<f64>,
    /// Tweedie variance power supplied out of band. A power written into the
    /// name itself (`tweedie(1.6)`) wins over this; absent both, a Tweedie
    /// family is refused (the power is never profiled).
    pub tweedie_power: Option<f64>,
    /// Beta-regression precision. `None` is the neutral 1.0.
    pub beta_phi: Option<f64>,
}

/// Spellings (after lowercasing and `_` → `-`) that name the vector-response
/// multinomial-logit family.
const MULTINOMIAL_FAMILY_NAMES: &[&str] = &[
    "multinomial",
    "multinomial-logit",
    "categorical",
    "categorical-logit",
    "softmax",
];

/// Whether `name` denotes the multinomial-logit family. The one predicate the
/// CLI, the Python `fit` entry point and the latent fitters route on, so every
/// surface accepts the same spellings.
pub fn is_multinomial_family_name(name: &str) -> bool {
    let lowered = name.to_ascii_lowercase().replace('_', "-");
    MULTINOMIAL_FAMILY_NAMES.contains(&lowered.as_str())
}

/// Resolve a scalar family NAME to its likelihood spec, plus whether the name
/// pinned a link.
///
/// This is the single total source of truth for every family spelling the
/// surface accepts — the CLI reaches it through [`resolve_family`], and the
/// Python FFI calls it directly. It deliberately takes no response data: a
/// name either denotes a family or it does not, and only the auto-detect path
/// in [`resolve_family`] needs `y`.
pub fn scalar_family_from_name(
    name: &str,
    overrides: FamilyNuisanceOverrides,
) -> Result<(LikelihoodSpec, bool), String> {
    let FamilyNuisanceOverrides {
        negative_binomial_theta,
        tweedie_power,
        beta_phi,
    } = overrides;
    let beta_phi_supplied = beta_phi.is_some();
    // The Beta precision is a nuisance parameter of the family, not of the
    // name; validate it at the one place the family is minted so every
    // surface rejects the same values.
    let beta_phi = match beta_phi {
        Some(phi) => {
            if !(phi.is_finite() && phi > 0.0) {
                return Err(WorkflowError::InvalidConfig {
                    reason: format!("beta phi must be finite and > 0; got {phi}"),
                }
                .into());
            }
            phi
        }
        None => 1.0,
    };
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
    // Family names are case-insensitive and hyphen-separated; an underscore
    // spelling (`student_t`) is refused with the hyphen spelling named. A
    // family may carry an explicit link as `family(link)` (e.g.
    // "poisson(log)", "Gamma(log)", "gaussian(identity)",
    // "binomial(probit)"). Parse that form
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
    let lowered = name.to_ascii_lowercase();
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
    // Tweedie carries its variance power as the
    // parenthesized argument (`tweedie(1.6)` / `tweedie(p=1.6)`), NOT a
    // link name. When the head is Tweedie and the argument parses as a
    // number, interpret it as the power `p` and consume the argument so
    // it is not misrouted to the link resolver — which previously
    // rejected `tweedie(1.5)` as `unknown link '1.5'` (#2026), leaving
    // no user-facing way to set `p`. A non-numeric argument
    // (e.g. `tweedie(log)`) still flows through to the link resolver.
    let (paren_link, tweedie_p_override): (Option<&str>, Option<f64>) =
        if matches!(head_name, "tweedie" | "tweedie-log")
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
                Err(_) => (Some(arg), tweedie_power),
            }
        } else {
            (paren_link, tweedie_power)
        };
    // A power supplied out of band (the FFI's `tweedie_p` argument) passes the
    // same validity gate as the parenthesized `tweedie(p=…)` form above, so one
    // bad power fails identically whichever surface named the family. A power
    // written into the name wins, because it is the more specific statement.
    if matches!(head_name, "tweedie" | "tweedie-log")
        && tweedie_p_override.is_none()
    {
        return Err(WorkflowError::InvalidConfig {
            reason: "Tweedie family requires an explicit variance power p strictly between 1 and 2 (for example, tweedie(p=1.5)); automatic power profiling is derivative-free hyperparameter search and is forbidden by SPEC.md"
                .to_string(),
        }
        .into());
    }
    if let Some(p) = tweedie_p_override
        && !gam_spec::is_valid_tweedie_power(p)
    {
        return Err(WorkflowError::InvalidConfig {
            reason: format!("tweedie power p must be finite and strictly between 1 and 2; got {p}"),
        }
        .into());
    }
    let resolved = match head_name {
        "gaussian" => (
            LikelihoodSpec::new(
                ResponseFamily::Gaussian,
                InverseLink::Standard(StandardLink::Identity),
            ),
            false,
        ),
        "gaussian-identity" => (
            LikelihoodSpec::new(
                ResponseFamily::Gaussian,
                InverseLink::Standard(StandardLink::Identity),
            ),
            true,
        ),
        "binomial" | "bernoulli" => (
            LikelihoodSpec::new(
                ResponseFamily::Binomial,
                InverseLink::Standard(StandardLink::Logit),
            ),
            false,
        ),
        "binomial-logit" | "bernoulli-logit" | "logistic" => (
            LikelihoodSpec::new(
                ResponseFamily::Binomial,
                InverseLink::Standard(StandardLink::Logit),
            ),
            true,
        ),
        "binomial-probit" | "bernoulli-probit" | "probit" => (
            LikelihoodSpec::new(
                ResponseFamily::Binomial,
                InverseLink::Standard(StandardLink::Probit),
            ),
            true,
        ),
        "binomial-cloglog" | "bernoulli-cloglog" | "cloglog" => (
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
        "poisson-log" => (
            LikelihoodSpec::new(
                ResponseFamily::Poisson,
                InverseLink::Standard(StandardLink::Log),
            ),
            true,
        ),
        // #983: a user-supplied `--negative-binomial-theta` holds θ
        // fixed at exactly that value (`theta_fixed = true` →
        // `FixedNegBinTheta` scale → the PIRLS refresh gate, which opens only
        // for `EstimatedNegBinTheta`, stays closed). With no flag,
        // θ is the running ML estimate (the #802 default seed 1.0).
        "negative-binomial" => {
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
                ResponseFamily::Beta { phi: beta_phi },
                InverseLink::Standard(StandardLink::Logit),
            ),
            false,
        ),
        "beta-logit" | "beta-regression-logit" => (
            LikelihoodSpec::new(
                ResponseFamily::Beta { phi: beta_phi },
                InverseLink::Standard(StandardLink::Logit),
            ),
            true,
        ),
        // The Student-t scale σ and degrees of freedom ν are LAML
        // hyperparameters: the optimizer replaces this placeholder with its
        // data-derived seed (σ = weighted MAD of y, ν = 1) before the first
        // evaluation, so the values written here never reach a fit.
        "student-t" => (
            LikelihoodSpec::new(
                ResponseFamily::StudentT {
                    sigma: 1.0,
                    nu: 1.0,
                },
                InverseLink::Standard(StandardLink::Identity),
            ),
            false,
        ),
        "gamma" => (
            LikelihoodSpec::new(
                ResponseFamily::Gamma,
                InverseLink::Standard(StandardLink::Log),
            ),
            false,
        ),
        "gamma-log" => (
            LikelihoodSpec::new(
                ResponseFamily::Gamma,
                InverseLink::Standard(StandardLink::Log),
            ),
            true,
        ),
        // Inverse-Gaussian with its canonical `1/μ²` link. The log link is
        // reached through `inverse-gaussian(log)`.
        "inverse-gaussian" => (
            LikelihoodSpec::new(
                ResponseFamily::InverseGaussian,
                InverseLink::Standard(StandardLink::InverseSquared),
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
        // must lie strictly in (1, 2) and callers set it explicitly via
        // `tweedie(1.6)` / `tweedie(p=1.6)`: profiling it without an
        // analytic p-derivative would violate SPEC.md's ban on
        // derivative-free hyperparameter search. The link is fixed to
        // log (the only link wired through the Tweedie working-response
        // and dispersion machinery).
        "tweedie" => (
            LikelihoodSpec::new(
                ResponseFamily::Tweedie {
                    p: tweedie_p_override.expect("explicit Tweedie power validated above"),
                },
                InverseLink::Standard(StandardLink::Log),
            ),
            false,
        ),
        "tweedie-log" => (
            LikelihoodSpec::new(
                ResponseFamily::Tweedie {
                    p: tweedie_p_override.expect("explicit Tweedie power validated above"),
                },
                InverseLink::Standard(StandardLink::Log),
            ),
            true,
        ),
        head if MULTINOMIAL_FAMILY_NAMES.contains(&head.replace('_', "-").as_str()) => {
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
            // coefficient ordering. The table fit entry points (CLI
            // `fit`, Python `fit_table`) route these names there via
            // `is_multinomial_family_name` before reaching this scalar
            // resolver.
            return Err(WorkflowError::InvalidConfig {
                reason: format!(
                    "family '{name}' is a vector-response family; fit it \
                     from a table (`gamfit.fit(data, formula, \
                     family='multinomial')`, or `gam fit --family \
                     multinomial`) so it reaches the multinomial driver \
                     rather than the scalar family resolver"
                ),
            }
            .into());
        }
        head => {
            return Err(WorkflowError::InvalidConfig {
                reason: match canonical_family_head(head) {
                    Some(canonical) => format!("unknown family `{head}`; use `{canonical}`"),
                    None => format!(
                        "unknown family '{name}'; expected one of: {SCALAR_FAMILY_NAMES_HELP}"
                    ),
                },
            }
            .into());
        }
    };
    // A nuisance override the resolved family does not carry is a
    // contradictory request, not a value to drop: `family="poisson"` with a
    // negative-binomial theta must not silently fit an equidispersed Poisson.
    // Refuse it here, where the family is minted, so every surface (CLI,
    // `gamfit.fit`, the latent FFI) refuses the same pairs.
    for (supplied, carries, option, family_name) in [
        (
            negative_binomial_theta.is_some(),
            matches!(resolved.0.response, ResponseFamily::NegativeBinomial { .. }),
            "negative_binomial_theta",
            "negative-binomial",
        ),
        (
            beta_phi_supplied,
            matches!(resolved.0.response, ResponseFamily::Beta { .. }),
            "beta_phi",
            "beta",
        ),
        (
            tweedie_power.is_some(),
            matches!(resolved.0.response, ResponseFamily::Tweedie { .. }),
            "tweedie_power",
            "tweedie",
        ),
    ] {
        if supplied && !carries {
            return Err(WorkflowError::InvalidConfig {
                reason: format!(
                    "{option} applies only to family='{family_name}'; family '{name}' \
                     does not carry it"
                ),
            }
            .into());
        }
    }
    // Apply an explicit parenthesized `(link)` argument to the resolved
    // family, validating legality. A bare family name leaves the
    // family's default link untouched.
    let resolved = match paren_link {
        Some(link_str) => apply_paren_link(resolved, link_str, name)?,
        None => resolved,
    };
    Ok(resolved)
}

/// Resolve a family from an optional name, optional link choice, and response data.
///
/// `y_kind` describes the *source* representation of the response column
/// (string-valued `Categorical`, numeric `Binary` short-circuit, or generic
/// `Numeric`). It is consulted only on the auto-detect path — explicit
/// `family=...` always wins — but is required there because the same numeric
/// `y = [0.0, 1.0, ...]` payload may come from a real binary outcome or from
/// a categorical column whose levels happened to encode to those indices.
/// Routing the kind through [`ResponseFamily::infer_from_response`] keeps the
/// auto-detector from reading level indices as values: a two-level label
/// column is a binary outcome (Binomial, coded by
/// [`code_two_level_label_response`]), and any other label column is refused.
pub fn resolve_family(
    family: Option<&str>,
    negative_binomial_theta: Option<f64>,
    link_choice: Option<&LinkChoice>,
    y: ArrayView1<'_, f64>,
    y_kind: ResponseColumnKind,
    response_name: &str,
) -> Result<LikelihoodSpec, String> {
    // `link_pinned = true` means the family name carried a specific link suffix
    // (e.g. "binomial-probit"); `false` means the user only declared the response
    // family (e.g. "binomial") and any link_choice may legally refine the link
    // without being treated as a contradiction.
    let explicit: Option<(LikelihoodSpec, bool)> = match family {
        Some(name) => Some(scalar_family_from_name(
            name,
            FamilyNuisanceOverrides {
                negative_binomial_theta,
                // The CLI/config surface carries the Tweedie power inside the
                // name (`tweedie(p=1.6)`) and has no Beta-precision knob yet.
                tweedie_power: None,
                beta_phi: None,
            },
        )?),
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
                // `log`, `sqrt`, `1/μ` and `1/μ²` are each legal for several
                // families (the generic variance × link cells carry the
                // non-canonical ones), and nothing in the link distinguishes
                // them: a variance function is a modelling choice, not
                // something to read off whether `y` happens to be
                // integer-valued. The caller names the family. With an
                // explicit family only `from_link.link` is carried below, so
                // the response here is immaterial.
                LinkFunction::Log if explicit.is_some() => LikelihoodSpec::new(
                    ResponseFamily::Gamma,
                    InverseLink::Standard(StandardLink::Log),
                ),
                LinkFunction::Sqrt if explicit.is_some() => LikelihoodSpec::new(
                    ResponseFamily::Gamma,
                    InverseLink::Standard(StandardLink::Sqrt),
                ),
                LinkFunction::Inverse if explicit.is_some() => LikelihoodSpec::new(
                    ResponseFamily::Gamma,
                    InverseLink::Standard(StandardLink::Inverse),
                ),
                LinkFunction::InverseSquared if explicit.is_some() => LikelihoodSpec::new(
                    ResponseFamily::Gamma,
                    InverseLink::Standard(StandardLink::InverseSquared),
                ),
                link @ (LinkFunction::Log
                | LinkFunction::Sqrt
                | LinkFunction::Inverse
                | LinkFunction::InverseSquared) => {
                    return Err(WorkflowError::InvalidConfig {
                        reason: format!(
                            "link '{}' does not determine a response family; name one \
                             with family=: {}",
                            link.name(),
                            LikelihoodSpec::families_admitting(link).join("|")
                        ),
                    }
                    .into());
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
            if mixture_requested {
                // The mixture link is a Binomial latent construct; it has no
                // legal pairing with any other response family.
                if !LikelihoodSpec::is_legal_cell(&explicit_spec.response, &from_link.link) {
                    return Err(WorkflowError::InvalidConfig {
                        reason: format!(
                            "a mixture link is not supported for family `{}`; {}",
                            explicit_spec.response.name(),
                            LikelihoodSpec::legal_links_clause(&explicit_spec.response)
                        ),
                    }
                    .into());
                }
            } else {
                require_legal_link(&explicit_spec.response, choice.link)?;
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
    //! #2026: the parenthesized Tweedie power `tweedie(p)` must be
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
        assert_eq!(tweedie_p("tweedie(p=1.6)"), 1.6);
        assert_eq!(tweedie_p("Tweedie(1.25)"), 1.25);
        assert_eq!(tweedie_p("tweedie-log(1.9)"), 1.9);
    }

    /// `SCALAR_FAMILY_HEADS` feeds the underscore hint below; every entry must
    /// be a head the resolver's match accepts.
    #[test]
    fn every_listed_scalar_family_head_resolves() {
        for head in SCALAR_FAMILY_HEADS {
            // Only a Tweedie head carries (and requires) a variance power.
            let tweedie_power = head.starts_with("tweedie").then_some(1.5);
            scalar_family_from_name(
                head,
                FamilyNuisanceOverrides {
                    tweedie_power,
                    ..FamilyNuisanceOverrides::default()
                },
            )
            .unwrap_or_else(|err| panic!("listed head `{head}` must resolve: {err}"));
        }
    }

    /// A nuisance override supplied for a family that does not carry it is
    /// refused, never silently dropped: `poisson` + theta must not fit a
    /// Poisson as though the theta had not been asked for.
    #[test]
    fn a_nuisance_override_for_a_family_that_does_not_carry_it_is_refused() {
        let theta = FamilyNuisanceOverrides {
            negative_binomial_theta: Some(2.0),
            ..FamilyNuisanceOverrides::default()
        };
        let phi = FamilyNuisanceOverrides {
            beta_phi: Some(7.5),
            ..FamilyNuisanceOverrides::default()
        };
        let power = FamilyNuisanceOverrides {
            tweedie_power: Some(1.6),
            ..FamilyNuisanceOverrides::default()
        };
        for (name, overrides, option) in [
            ("poisson", theta, "negative_binomial_theta"),
            ("gaussian(log)", theta, "negative_binomial_theta"),
            ("tweedie(1.5)", theta, "negative_binomial_theta"),
            ("gamma-log", phi, "beta_phi"),
            ("negative-binomial", phi, "beta_phi"),
            ("poisson-log", power, "tweedie_power"),
            ("beta", power, "tweedie_power"),
        ] {
            let err = scalar_family_from_name(name, overrides)
                .expect_err(&format!("{name} + {option} must be refused"));
            assert!(err.contains(&format!("{option} applies only to")), "{name}: {err}");
        }
        // Each override still reaches the family that carries it.
        let (nb, _) = scalar_family_from_name("negative-binomial", theta).expect("nb + theta");
        assert!(matches!(
            nb.response,
            ResponseFamily::NegativeBinomial { theta, theta_fixed: true } if theta == 2.0
        ));
        let (beta, _) = scalar_family_from_name("beta", phi).expect("beta + phi");
        assert!(matches!(beta.response, ResponseFamily::Beta { phi } if phi == 7.5));
        let (tw, _) = scalar_family_from_name("tweedie", power).expect("tweedie + power");
        assert!(matches!(tw.response, ResponseFamily::Tweedie { p } if p == 1.6));
        // The CLI surface refuses the same pair through `resolve_family`.
        let y = array![0.0, 1.0, 3.0, 2.0];
        let err = resolve_family(
            Some("poisson"),
            Some(2.0),
            None,
            y.view(),
            ResponseColumnKind::Numeric,
            "y",
        )
        .expect_err("--family poisson --negative-binomial-theta must be refused");
        assert!(err.contains("negative_binomial_theta applies only to"), "{err}");
    }

    /// SPEC R25: one spelling per family. The other spellings are refused
    /// with an error that names the accepted one.
    #[test]
    fn removed_family_spellings_name_the_canonical_one() {
        for (raw, expected) in [
            ("tw(1.5)", "unknown family `tw`; use `tweedie`"),
            ("nb", "unknown family `nb`; use `negative-binomial`"),
            ("negbin", "unknown family `negbin`; use `negative-binomial`"),
            ("NegBin-Log", "unknown family `negbin-log`; use `negative-binomial-log`"),
            ("t", "unknown family `t`; use `student-t`"),
            ("student_t", "unknown family `student_t`; use `student-t`"),
            ("inverse.gaussian", "unknown family `inverse.gaussian`; use `inverse-gaussian`"),
            ("invgauss", "unknown family `invgauss`; use `inverse-gaussian`"),
            ("inverse_gaussian", "unknown family `inverse_gaussian`; use `inverse-gaussian`"),
            ("negative_binomial(log)", "unknown family `negative_binomial`; use `negative-binomial`"),
        ] {
            let err = scalar_family_from_name(raw, FamilyNuisanceOverrides::default())
                .expect_err(raw);
            assert!(err.contains(expected), "{raw}: {err}");
        }
        for canonical in ["negative-binomial", "student-t", "inverse-gaussian", "tweedie(1.5)"] {
            scalar_family_from_name(canonical, FamilyNuisanceOverrides::default())
                .unwrap_or_else(|err| panic!("`{canonical}` must resolve: {err}"));
        }
    }

    #[test]
    fn tweedie_bare_requires_explicit_power_instead_of_derivative_free_profiling() {
        let y = array![0.0, 1.2, 3.4];
        for family in ["tweedie", "tweedie(log)"] {
            let error = resolve_family(
                Some(family),
                None,
                None,
                y.view(),
                ResponseColumnKind::Numeric,
                "y",
            )
            .expect_err("a bare Tweedie family must not trigger derivative-free profiling");
            assert!(error.contains("requires an explicit variance power"), "{error}");
        }
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
        assert!(require_legal_link(&ResponseFamily::Binomial, LinkFunction::LogLog).is_ok(),
            "binomial + loglog must be a legal pairing"
        );
        assert!(require_legal_link(&ResponseFamily::Binomial, LinkFunction::Cauchit).is_ok(),
            "binomial + cauchit must be a legal pairing"
        );
        // The other three canonical binomial links stay legal (no regression),
        // and a non-binomial family still rejects these two links.
        assert!(require_legal_link(&ResponseFamily::Binomial, LinkFunction::CLogLog).is_ok());
        assert!(require_legal_link(&ResponseFamily::Gaussian, LinkFunction::LogLog).is_err());
        assert!(require_legal_link(&ResponseFamily::Gaussian, LinkFunction::Cauchit).is_err());

        // End-to-end resolver path (`family(link)`) must now accept
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

    /// pyGAM audit families.md F11: `link="log"` with no family once picked
    /// Poisson when every `y` was a non-negative integer and Gamma otherwise, so
    /// a positive cost column rounded to whole dollars got a Poisson variance
    /// function from a data coincidence. A link several families admit does not
    /// determine the family; the caller names one, and the error lists the
    /// families the legality table admits for that link.
    #[test]
    fn a_link_several_families_admit_requires_the_family() {
        use gam_terms::inference::formula_dsl::{LinkChoice, LinkMode};
        let integer_valued = array![1.0, 3.0, 7.0, 2.0, 12.0];
        for (link, admitting) in [
            (
                LinkFunction::Log,
                "gaussian|binomial|poisson|tweedie|negative-binomial|gamma|inverse-gaussian",
            ),
            (
                LinkFunction::Inverse,
                "gaussian|poisson|gamma|inverse-gaussian",
            ),
            (LinkFunction::Sqrt, "gaussian|poisson|gamma|inverse-gaussian"),
        ] {
            let choice = LinkChoice {
                mode: LinkMode::Strict,
                link,
                mixture_components: None,
            };
            let error = resolve_family(
                None,
                None,
                Some(&choice),
                integer_valued.view(),
                ResponseColumnKind::Numeric,
                "y",
            )
            .expect_err("a link alone must not choose between variance functions");
            assert!(
                error.contains(&format!("name one with family=: {admitting}")),
                "{} error must list the admitting families: {error}",
                link.name()
            );
            for family in admitting.split('|') {
                // A Tweedie family is named with its variance power.
                let requested = if family == "tweedie" {
                    "tweedie(1.5)"
                } else {
                    family
                };
                let spec = resolve_family(
                    Some(requested),
                    None,
                    Some(&choice),
                    integer_valued.view(),
                    ResponseColumnKind::Numeric,
                    "y",
                )
                .unwrap_or_else(|err| panic!("{requested} + {link:?}: {err}"));
                assert_eq!(spec.response.name(), family);
                assert_eq!(spec.link.link_function(), link);
            }
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
