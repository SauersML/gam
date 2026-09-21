use super::*;
use gam_terms::inference::formula_dsl::LinkMode;

pub(crate) fn requests_bernoulli_marginal_slope(config: &FitConfig) -> bool {
    let family_requests_marginal_slope = config.family.as_deref().is_some_and(|family| {
        let canonical = family.to_ascii_lowercase().replace('_', "-");
        canonical == "bernoulli-marginal-slope" || canonical == "binary-marginal-slope"
    });
    family_requests_marginal_slope
        || config.slope_formula.is_some()
        || config.z_column.is_some()
        || config.ctn_stage1.is_some()
}

pub(crate) fn reject_marginal_slope_controls_for_transformation_normal(
    config: &FitConfig,
) -> Result<(), WorkflowError> {
    if requests_bernoulli_marginal_slope(config) {
        return Err(WorkflowError::TransformationNormalConflict {
            conflict: TransformationNormalConflict::MarginalSlopeControls,
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

/// Reject an explicitly-requested survival-only `FitConfig` knob when the
/// response is not `Surv(...)`.
///
/// Two knobs qualify, and they qualify for the same reason: both are read
/// exclusively inside `materialize_survival`, and both are `Option`-typed, so
/// "the caller asked for this" is carried by the type instead of guessed from a
/// value.
///
/// * `survival_likelihood` selects the likelihood mode.
/// * `survival_time_anchor` names the baseline time-basis centering anchor
///   (#2631). It reached this struct as part of collapsing the anchor rule to one
///   place; the CLI had always refused `--survival-time-anchor` without a
///   `Surv(...)` response, and the engine must refuse it identically or the two
///   front ends disagree again — this time about which configurations are legal.
///

/// `survival_likelihood` selects the survival likelihood mode
/// (`"transformation"`, `"location-scale"`, `"weibull"`, `"marginal-slope"`,
/// `"latent"`, `"latent-binary"`, …) and is read *exclusively* inside
/// `materialize_survival`. When the main formula has no `Surv(...)` response the
/// survival materializer is never reached, so a survival-only knob like
/// `survival_likelihood="weibull"` is parsed, validated, and then dropped on the
/// floor — the request silently degrades to an ordinary Gaussian GAM (#1767),
/// the same silent-no-op contract violation as the survival-only *terms* guarded
/// by [`reject_survival_only_terms_for_nonsurvival`].
///
/// `survival_likelihood` is now `Option<String>` defaulting to `None` at every
/// entrance (#2301): the single canonical default (`"transformation"`) is
/// resolved at the `Surv(...)` seam, not stored. So `None` is unambiguously
/// "unset" (allowed through), and ANY `Some(mode)` is an explicit request that
/// must be rejected on a non-survival response — the type carries the intent, so
/// this seam no longer has to guess default-vs-explicit from a string value.
pub(crate) fn reject_survival_only_config_for_nonsurvival(
    config: &FitConfig,
) -> Result<(), WorkflowError> {
    if let Some(anchor) = config.survival_time_anchor {
        return Err(WorkflowError::InvalidConfig {
            reason: format!(
                "survival_time_anchor={anchor} is only supported in the main survival formula \
                 (a formula with a Surv(...) response); it centers the baseline time basis, which \
                 exists only on the survival fit path, so for a non-survival response it is \
                 meaningless and would otherwise be silently ignored. Wrap the response in \
                 Surv(...) or drop the survival_time_anchor configuration."
            ),
        });
    }
    // The baseline, follow-up time basis and time-varying block settings are
    // read only by `materialize_survival`. On a non-survival response they would
    // be dropped and an ordinary GAM fitted. The CLI refused them, but a
    // `gamfit.fit` call or a Rust caller did not, so the refusal lives here,
    // where every front end arrives.
    let defaults = FitConfig::default();
    let survival_only_settings: Vec<&str> = [
        ("baseline_scale", config.baseline_scale.is_some()),
        ("baseline_shape", config.baseline_shape.is_some()),
        ("baseline_rate", config.baseline_rate.is_some()),
        ("baseline_makeham", config.baseline_makeham.is_some()),
        (
            "baseline_target",
            !config
                .baseline_target
                .trim()
                .eq_ignore_ascii_case(&defaults.baseline_target),
        ),
        (
            "time_basis",
            !config.time_basis.trim().eq_ignore_ascii_case(&defaults.time_basis),
        ),
        ("time_degree", config.time_degree != defaults.time_degree),
        (
            "time_num_internal_knots",
            config.time_num_internal_knots != defaults.time_num_internal_knots,
        ),
        (
            "survival_distribution",
            !config
                .survival_distribution
                .trim()
                .eq_ignore_ascii_case(&defaults.survival_distribution),
        ),
        ("threshold_time_k", config.threshold_time_k.is_some()),
        ("sigma_time_k", config.sigma_time_k.is_some()),
        ("slope_time_k", config.slope_time_k.is_some()),
    ]
    .into_iter()
    .filter_map(|(name, set)| set.then_some(name))
    .collect();
    if !survival_only_settings.is_empty() {
        return Err(WorkflowError::InvalidConfig {
            reason: format!(
                "{} {} read only by the survival fit path and require a Surv(...) response; for a \
                 non-survival response they would otherwise be silently ignored. Wrap the \
                 response in Surv(...) or drop them.",
                survival_only_settings.join(", "),
                if survival_only_settings.len() == 1 { "is" } else { "are" },
            ),
        });
    }
    // `survival_likelihood` is `None` by default across every entrance (#2301):
    // the sole canonical default is resolved to `"transformation"` at the
    // `Surv(...)` seam, not stored here. So `None` is genuinely "unset" and must
    // not poison ordinary GAM materialization, while ANY explicit `Some(mode)`
    // on a non-survival response is a survival knob that only
    // `materialize_survival` reads — it would be silently dropped, degrading the
    // fit to an ordinary GAM (#1767). Reject it, exactly as the survival-only
    // formula terms are rejected. Carrying intent in the `Option` is what lets
    // this seam distinguish default from explicit without guessing.
    let Some(mode) = config.survival_likelihood.as_deref() else {
        return Ok(());
    };
    let mode = mode.trim();
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
    if let Some(linkspec) = parsed.linkspec.as_ref()
        && (linkspec.mixture_rho.is_some()
            || linkspec.sas_init.is_some()
            || linkspec.beta_logistic_init.is_some())
    {
        return Err(WorkflowError::InvalidConfig {
            reason: "link(...) initialization options are not supported by the materialized fit path; pass only link(type=...) in the formula"
                .to_string(),
        });
    }
    resolve_link_spellings(
        parsed.linkspec.as_ref(),
        config.link.as_deref(),
        config.flexible_link,
    )
}

/// Resolve a fit's link from every spelling the request carries: the formula's
/// `link(...)`, the `link` argument and the `flexible_link` flag (gamfit's
/// `link=` and `flexible_link=`). All of them are read. `flexible_link` flexes
/// whichever base link is named (plain `flexible(probit)` when none is), a
/// `flexible(...)` in either place makes the choice flexible, and a `link`
/// argument whose base link differs from the formula's is refused by name
/// rather than dropped. This is the rule `resolve_marginal_slope_link` applies
/// to the probit-only marginal-slope families.
pub(super) fn resolve_link_spellings(
    linkspec: Option<&gam_terms::inference::formula_dsl::LinkFormulaSpec>,
    link_argument: Option<&str>,
    flexible_link: bool,
) -> Result<Option<LinkChoice>, WorkflowError> {
    let formula_choice = match linkspec {
        Some(linkspec) => parse_link_choice(Some(&linkspec.link), flexible_link)?,
        None => None,
    };
    let argument_choice = match (link_argument, &formula_choice) {
        (None, Some(_)) => None,
        (link, _) => parse_link_choice(link, flexible_link)?,
    };
    let (Some(formula), Some(argument), Some(linkspec), Some(link)) =
        (&formula_choice, &argument_choice, linkspec, link_argument)
    else {
        return Ok(formula_choice.or(argument_choice));
    };
    if formula.link != argument.link || formula.mixture_components != argument.mixture_components
    {
        return Err(WorkflowError::InvalidConfig {
            reason: format!(
                "the formula's link(type={}) and the link=\"{}\" argument name different \
                 links; name the link once, either in the formula or as link=",
                linkspec.link.trim(),
                link.trim()
            ),
        });
    }
    let flexible = [formula, argument]
        .into_iter()
        .any(|choice| matches!(choice.mode, LinkMode::Flexible));
    Ok(Some(LinkChoice {
        mode: if flexible {
            LinkMode::Flexible
        } else {
            LinkMode::Strict
        },
        ..formula.clone()
    }))
}

/// Refuse every link spelling a request carries, for a model with no link to
/// choose (gam#3298). A request names its link in three places: the formula's
/// `link(...)`, the `link` argument and the `flexible_link` flag (gamfit's
/// `link=` and `flexible_link=`). A model whose likelihood has no link, or
/// fixes its own, fits bit-identically with or without any of them, so each is
/// refused by name rather than accepted and never read.
pub(crate) fn refuse_link_spellings(
    linkspec: Option<&gam_terms::inference::formula_dsl::LinkFormulaSpec>,
    config: &FitConfig,
    model: &str,
) -> Result<(), WorkflowError> {
    let spelling = if let Some(linkspec) = linkspec {
        format!("the formula's link(type={})", linkspec.link.trim())
    } else if let Some(link) = config.link.as_deref() {
        format!("link=\"{}\"", link.trim())
    } else if config.flexible_link {
        "flexible_link=True".to_string()
    } else {
        return Ok(());
    };
    Err(WorkflowError::InvalidConfig {
        reason: format!(
            "{model} has no link to choose, so {spelling} would be accepted and never read; \
             remove it"
        ),
    })
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

/// Refuse a fit whose positive-weight rows cannot exceed the unpenalized
/// dimension the formula already fixes, before the family is inferred or any
/// basis is built.
///
/// REML/LAML estimate the smoothing parameters from the `n − M_p` residual
/// contrasts the unpenalized directions cannot absorb, so an identified fit
/// needs `n > M_p` (the full gate, once the penalties exist, is
/// `reject_prefit_unidentifiable_unpenalized_space` in gam-solve). The formula
/// alone already fixes part of `M_p`: the model intercept is one unpenalized
/// direction unless the formula drops it. Parametric and smooth terms may or
/// may not add more depending on their penalties, so they are left to that
/// later gate. Checking this lower bound first makes a one-row fit report the
/// row count rather than whatever the one row happens to trip next — an
/// auto-inferred binomial family calling `y = [1]` degenerate, or a smooth
/// calling its single covariate value constant.
pub(super) fn reject_too_few_rows_for_formula(
    parsed: &ParsedFormula,
    weights: ArrayView1<'_, f64>,
) -> Result<(), WorkflowError> {
    let has_intercept = !parsed
        .terms
        .iter()
        .any(|term| matches!(term, ParsedTerm::NoIntercept));
    let unpenalized_lower_bound = usize::from(has_intercept);
    let n_observations = weights.iter().filter(|&&weight| weight > 0.0).count();
    if n_observations > unpenalized_lower_bound {
        return Ok(());
    }
    let rows = if n_observations == 1 { "row" } else { "rows" };
    let directions = if has_intercept {
        "the intercept is an unpenalized coefficient direction (M_p >= 1)"
    } else {
        "a model needs at least one observation"
    };
    Err(WorkflowError::InvalidData {
        column: parsed.response.clone(),
        problem: format!(
            "has {n_observations} positive-weight {rows}: too few rows to fit this model. \
             REML estimates the smoothing parameters from the n - M_p residual contrasts the \
             unpenalized directions cannot absorb, and {directions}, so the fit needs more than \
             {unpenalized_lower_bound} positive-weight row(s). Add observations."
        ),
    })
}

/// Refuse precision hyperpriors and coefficient groups on a model class whose
/// fit request has no place for them.
///
/// `FitConfig::penalty_block_gamma_priors` (the frontends' `precision_hyperpriors`)
/// is realized only by the standard fit and by the survival
/// transformation/Weibull fit. `FitConfig::coefficient_groups` is realized only
/// by the standard fit. Every other materializer builds a request without these
/// fields, so without this check the fit would run with the default flat
/// smoothing-parameter prior while the caller believes the requested prior was
/// used.
pub(super) fn reject_unrealized_precision_priors(
    config: &FitConfig,
    model: &str,
    realizes_penalty_block_priors: bool,
) -> Result<(), WorkflowError> {
    if !realizes_penalty_block_priors && !config.penalty_block_gamma_priors.is_empty() {
        return Err(WorkflowError::InvalidConfig {
            reason: format!(
                "precision_hyperpriors is not supported for {model}: only standard and \
                 survival transformation/weibull fits realize penalty-block Gamma priors, \
                 so this fit would ignore them"
            ),
        });
    }
    if !config.coefficient_groups.is_empty() {
        return Err(WorkflowError::InvalidConfig {
            reason: format!(
                "coefficient_groups is not supported for {model}: only standard fits \
                 realize coefficient groups, so this fit would ignore them"
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
    // Exact membership: a value near 0 or 1 is not an outcome.
    y.iter().all(|&v| v == 0.0 || v == 1.0)
}

/// Judge the response column against the family's support and degeneracy
/// rules over the rows that enter the likelihood (positive prior weight).
///
/// Both rules are owned by [`ResponseFamily`] so the formula materializers
/// and the external-design path share one definition; this adapter only
/// attaches the column name and routes the violation to
/// [`WorkflowError::InvalidData`], the data-error class, because the fault is
/// in the supplied values rather than in the model configuration.
pub(super) fn validate_response_against_family(
    family: &LikelihoodSpec,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    response: &str,
) -> Result<(), WorkflowError> {
    family
        .response
        .validate_response_support(y, weights)
        .map_err(|violation| WorkflowError::InvalidData {
            column: response.to_string(),
            problem: violation.problem(),
        })?;
    family
        .response
        .validate_response_degeneracy(y, weights)
        .map_err(|degeneracy| WorkflowError::InvalidData {
            column: response.to_string(),
            problem: degeneracy.problem(),
        })
}

#[cfg(test)]
mod binary_response_tests {
    use super::is_binary_response;
    use ndarray::{Array1, array};

    #[test]
    fn only_exact_outcomes_make_a_binary_response() {
        assert!(is_binary_response(array![0.0, 1.0, 1.0, -0.0].view()));
        // A value one part in 1e13 from an outcome is not an outcome.
        assert!(!is_binary_response(array![0.0, 1.0 - 1.0e-13, 1.0].view()));
        assert!(!is_binary_response(array![1.0e-13, 1.0].view()));
        assert!(!is_binary_response(array![0.0, f64::NAN].view()));
        assert!(!is_binary_response(Array1::<f64>::zeros(0).view()));
    }
}

#[cfg(test)]
mod link_spelling_tests {
    use super::refuse_link_spellings;
    use crate::fit_orchestration::{FitConfig, WorkflowError};
    use gam_terms::inference::formula_dsl::LinkFormulaSpec;

    fn refusal(linkspec: Option<&LinkFormulaSpec>, config: &FitConfig) -> Option<String> {
        match refuse_link_spellings(linkspec, config, "this model") {
            Ok(()) => None,
            Err(WorkflowError::InvalidConfig { reason }) => Some(reason),
            Err(other) => panic!("a link spelling is a configuration refusal, got {other:?}"),
        }
    }

    #[test]
    fn every_link_spelling_is_refused_by_name_3298() {
        assert_eq!(refusal(None, &FitConfig::default()), None);

        let formula_link = LinkFormulaSpec {
            link: " probit ".to_string(),
            mixture_rho: None,
            sas_init: None,
            beta_logistic_init: None,
        };
        let reason = refusal(Some(&formula_link), &FitConfig::default()).unwrap();
        assert!(reason.contains("the formula's link(type=probit)"), "{reason}");
        assert!(reason.starts_with("this model has no link to choose"), "{reason}");

        let argument_link = FitConfig {
            link: Some("logit".to_string()),
            ..FitConfig::default()
        };
        let reason = refusal(None, &argument_link).unwrap();
        assert!(reason.contains("link=\"logit\""), "{reason}");

        let flexible = FitConfig {
            flexible_link: true,
            ..FitConfig::default()
        };
        let reason = refusal(None, &flexible).unwrap();
        assert!(reason.contains("flexible_link=True"), "{reason}");
    }
}
