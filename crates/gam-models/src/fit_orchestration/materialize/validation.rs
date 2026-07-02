use super::*;
use gam_terms::inference::formula_dsl::LinkMode;

pub(crate) fn reject_marginal_slope_controls_for_transformation_normal(
    config: &FitConfig,
) -> Result<(), WorkflowError> {
    let family_requests_marginal_slope = config.family.as_deref().is_some_and(|family| {
        let canonical = family.to_ascii_lowercase().replace('_', "-");
        canonical == "bernoulli-marginal-slope" || canonical == "binary-marginal-slope"
    });
    if family_requests_marginal_slope
        || config.logslope_formula.is_some()
        || config.z_column.is_some()
        || config.ctn_stage1.is_some()
    {
        return Err(WorkflowError::InvalidConfig {
            reason: "transformation_normal cannot be combined with marginal-slope family controls"
                .to_string(),
        });
    }
    Ok(())
}

/// Reject `timewiggle(...)` / `survmodel(...)` in a formula whose response is
/// not `Surv(...)`.
///
/// These two DSL controls only have meaning under the survival likelihood: a
/// `timewiggle(...)` term parameterizes the time-varying baseline-hazard /
/// log-cumulative-hazard surface, and `survmodel(...)` selects the survival
/// likelihood mode. Both are read exclusively by `materialize_survival`. When
/// the main formula has no `Surv(...)` response, leaving them unguarded means
/// the term is parsed and option-validated and then dropped on the floor —
/// the contract violation reported in #371. We error instead, with the same
/// "only supported in the main survival formula" phrasing the auxiliary-formula
/// path already uses.
pub(crate) fn reject_survival_only_terms_for_nonsurvival(
    parsed: &ParsedFormula,
) -> Result<(), WorkflowError> {
    if parsed.timewiggle.is_some() {
        return Err(WorkflowError::InvalidConfig {
            reason: "timewiggle(...) is only supported in the main survival formula \
                     (a formula with a Surv(...) response); it is meaningless for a \
                     non-survival response and would otherwise be silently ignored"
                .to_string(),
        });
    }
    if parsed.survivalspec.is_some() {
        return Err(WorkflowError::InvalidConfig {
            reason: "survmodel(...) is only supported in the main survival formula \
                     (a formula with a Surv(...) response); it is meaningless for a \
                     non-survival response and would otherwise be silently ignored"
                .to_string(),
        });
    }
    Ok(())
}

/// Reject a non-default `config.survival_likelihood` when the response is not
/// `Surv(...)`.
///
/// `survival_likelihood` selects the survival likelihood mode
/// (`"location-scale"`, `"weibull"`, `"marginal-slope"`, `"latent"`,
/// `"latent-binary"`, …) and is read *exclusively* inside `materialize_survival`.
/// When the main formula has no `Surv(...)` response the survival materializer is
/// never reached, so a survival-only knob like
/// `survival_likelihood="weibull"` is parsed, validated, and then dropped on the
/// floor — the request silently degrades to an ordinary Gaussian GAM (#1767),
/// the same silent-no-op contract violation as the survival-only *terms* guarded
/// by [`reject_survival_only_terms_for_nonsurvival`].
///
/// The direct Rust API default is `"location-scale"` (lognormal AFT), while
/// CLI/config-layer callers may still pass their documented `"transformation"`
/// default explicitly. Both defaults are indistinguishable from "unset" at this
/// validation seam, so they are allowed through here; every other,
/// explicitly-requested mode is a configuration error and is rejected with the
/// same phrasing the survival-only-term path uses.
pub(crate) fn reject_survival_likelihood_for_nonsurvival(
    config: &FitConfig,
) -> Result<(), WorkflowError> {
    let mode = config.survival_likelihood.trim();
    // The library and CLI/config layers have different survival defaults; both are
    // indistinguishable from "unset" by the time a non-survival formula reaches
    // this seam, so neither should poison ordinary GAM materialization.
    if mode.eq_ignore_ascii_case("transformation") || mode.eq_ignore_ascii_case("location-scale") {
        return Ok(());
    }
    Err(WorkflowError::InvalidConfig {
        reason: format!(
            "survival_likelihood=\"{mode}\" is only supported in the main survival formula \
             (a formula with a Surv(...) response); it selects a survival likelihood mode that \
             is read exclusively by the survival fit path, so for a non-survival response it is \
             meaningless and would otherwise be silently ignored (the requested survival model \
             would degrade to an ordinary GAM). Wrap the response in Surv(...) or drop the \
             survival_likelihood configuration."
        ),
    })
}

/// Reject an *explicitly requested* `linkwiggle(...)` term when the resolved
/// response family is not binomial.
///
/// `linkwiggle(...)` adds a spline-flexible correction to the *link* function
/// (logit / probit / cloglog), which only carries meaning for a binomial mean
/// model — the standard and location-scale materializers wire `wiggle` into the
/// fit only inside their `family.is_binomial()` arm. For a Gaussian / Gamma /
/// Poisson / etc. response the term is built and then dropped on the floor,
/// the same silent-no-op contract violation as #371. We error here.
///
/// This guards only the *explicit* formula term (`parsed.linkwiggle`). The
/// implicit wiggle auto-derived from a `Flexible` link choice is checked by
/// [`reject_flexible_link_for_nonbinomial`].
pub(super) fn reject_explicit_linkwiggle_for_nonbinomial(
    parsed: &ParsedFormula,
    family: &LikelihoodSpec,
) -> Result<(), WorkflowError> {
    if parsed.linkwiggle.is_some() && !family.is_binomial() {
        return Err(WorkflowError::InvalidConfig {
            reason: "linkwiggle(...) corrects the link function of a binomial mean model \
                     and is only supported for a binomial response; it is meaningless for \
                     the resolved non-binomial family and would otherwise be silently ignored"
                .to_string(),
        });
    }
    Ok(())
}

pub(super) fn effective_link_choice_for_materialize(
    parsed: &ParsedFormula,
    config: &FitConfig,
) -> Result<Option<LinkChoice>, WorkflowError> {
    if let Some(linkspec) = parsed.linkspec.as_ref() {
        if linkspec.mixture_rho.is_some()
            || linkspec.sas_init.is_some()
            || linkspec.beta_logistic_init.is_some()
        {
            return Err(WorkflowError::InvalidConfig {
                reason: "link(...) initialization options are not supported by the materialized fit path; pass only link(type=...) in the formula"
                    .to_string(),
            });
        }
        return parse_link_choice(Some(&linkspec.link), false).map_err(WorkflowError::from);
    }
    parse_link_choice(config.link.as_deref(), config.flexible_link).map_err(WorkflowError::from)
}

/// Reject a `flexible(...)` link choice (the implicit link wiggle) when the
/// resolved response family is not binomial.
///
/// `flexible(base)` adds a jointly-fit anchored spline offset to the base link.
/// The whole offset engine ([`crate::gamlss::gaussian::BinomialMeanWiggleFamily`]
/// and the location-scale wiggle solver) is specialised to the binomial mean
/// likelihood: it differentiates the binomial neg-log-likelihood through the
/// warped link to fourth order under a monotone-spline constraint. For a
/// Gaussian / Poisson / Gamma / etc. response there is no implemented mean-wiggle
/// solver, so the standard and location-scale materializers used to build the
/// implicit wiggle and then drop it on the floor: a silent no-op of a
/// documented link (`flexible(identity)` on Gaussian, `flexible(log)` on
/// Poisson/Gamma fit bit-identically to the plain base link), gam#1275. Rather
/// than silently discard a requested-and-documented link configuration we error
/// loudly here, exactly as [`reject_explicit_linkwiggle_for_nonbinomial`] does
/// for the explicit term. Wiring a genuine non-binomial mean-wiggle is tracked
/// as a separate feature.
pub(super) fn reject_flexible_link_for_nonbinomial(
    link_choice: Option<&LinkChoice>,
    family: &LikelihoodSpec,
) -> Result<(), WorkflowError> {
    let requested_flexible =
        link_choice.is_some_and(|choice| matches!(choice.mode, LinkMode::Flexible));
    if requested_flexible && !family.is_binomial() {
        return Err(WorkflowError::InvalidConfig {
            reason: format!(
                "flexible(...) links (the jointly-fit anchored spline link offset) are \
                 implemented only for a binomial response; the resolved family is {} (a \
                 non-binomial family), for which the link offset has no solver and would \
                 otherwise be silently discarded. Use the plain base link, or fit a binomial \
                 response.",
                family.pretty_name()
            ),
        });
    }
    Ok(())
}

/// Detect whether a response column is binary (0/1 only).
pub fn is_binary_response(y: ArrayView1<'_, f64>) -> bool {
    if y.is_empty() {
        return false;
    }
    y.iter()
        .all(|v| (*v - 0.0).abs() < 1e-12 || (*v - 1.0).abs() < 1e-12)
}

/// Verify that the dataset has at least as many rows as the smooth terms in
/// `spec` need for their bases to be well-posed.
///
/// Each [`SmoothBasisSpec`] owns its own `min_sample_rows` lower bound — the
/// B-spline knot count, the *penalized* tensor-product floor (the sum of the
/// per-marginal column counts, not their Kronecker product, because a `te()`
/// is regularized and its effective dof is a small fraction of the column
/// count), the PCA matrix width — so this helper is a thin sum-and-compare:
/// the workflow has no per-basis-kind knowledge. Adding a new smooth kind
/// extends the basis `match` in `min_sample_rows`, not this gate.
///
/// Catches the README-quickstart failure mode (#309) where `n=4` against
/// `y ~ s(x)` would otherwise surface as an opaque `cached inner beta has
/// length 8` message from the inner-state seeding hook.
pub(super) fn check_smooth_capacity(
    spec: &gam_terms::smooth::TermCollectionSpec,
    n_rows: usize,
    response_name: &str,
) -> Result<(), WorkflowError> {
    // Intercept + 1 dof for the smoothing-parameter optimizer.
    let mut required: usize = 2;
    let mut per_term: Vec<(String, usize)> = Vec::new();
    for term in &spec.smooth_terms {
        let need = term.basis.min_sample_rows();
        required = required.saturating_add(need);
        per_term.push((term.name.clone(), need));
    }
    if per_term.is_empty() || n_rows >= required {
        return Ok(());
    }
    let breakdown = per_term
        .iter()
        .map(|(name, k)| format!("{name}≥{k}"))
        .collect::<Vec<_>>()
        .join(", ");
    Err(WorkflowError::InvalidConfig {
        reason: format!(
            "not enough observations to fit the requested formula: dataset has n={n_rows} \
             rows but the smooth terms on response '{response_name}' need at least \
             {required} rows total ({breakdown}, plus intercept + smoothing-parameter dof) \
             before REML estimation is well-posed. \
             Fix: add more training rows, replace `s(x)` with a linear term, or pass a \
             smaller basis via `s(x, k=3)`."
        ),
    })
}
