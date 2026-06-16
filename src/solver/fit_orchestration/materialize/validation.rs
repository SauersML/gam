use super::*;

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
/// This guards only the *explicit* formula term (`parsed.linkwiggle`), not the
/// implicit wiggle auto-derived from a `Flexible` link choice: a flexible link
/// requested against a non-binomial family is a separate, already-handled link
/// concern, and silently declining to add a binomial-only correction there is
/// the intended behavior rather than a dropped user-authored term.
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
    spec: &crate::terms::smooth::TermCollectionSpec,
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
