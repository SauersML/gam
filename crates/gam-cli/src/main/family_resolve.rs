use super::*;

pub(crate) fn family_arg_canonical_name(arg: FamilyArg) -> Option<&'static str> {
    match arg {
        FamilyArg::Auto => None,
        FamilyArg::Gaussian => Some("gaussian"),
        FamilyArg::BinomialLogit => Some("binomial-logit"),
        FamilyArg::BinomialProbit => Some("binomial-probit"),
        FamilyArg::BinomialCloglog => Some("binomial-cloglog"),
        FamilyArg::LatentCloglogBinomial => Some("latent-cloglog-binomial"),
        FamilyArg::PoissonLog => Some("poisson"),
        FamilyArg::NegativeBinomial => Some("negative-binomial"),
        FamilyArg::GammaLog => Some("gamma"),
        FamilyArg::Tweedie => Some("tweedie"),
        FamilyArg::Beta => Some("beta"),
        FamilyArg::RoystonParmar => Some("royston-parmar"),
        FamilyArg::TransformationNormal => Some("transformation-normal"),
        FamilyArg::Expectile => Some("expectile"),
        // Dispatched by `run_fit` before the canonical family resolver (the
        // multinomial artifact is a softmax multi-output model, not a scalar
        // GLM), so this canonical name is used only for display/config echoing.
        FamilyArg::Multinomial => Some("multinomial"),
    }
}

/// CLI adapter over the canonical family resolver.
///
/// The fit-routing contract — explicit family vs link-implied family, the
/// SAS/Beta-Logistic links, negative-binomial `theta`, and response
/// auto-inference — lives once in `gam::resolve_family`. The CLI keeps only the
/// surface-specific concerns: translating the typed `FamilyArg` into the
/// canonical name and enforcing the CLI flag rule that
/// `--negative-binomial-theta` is meaningful exclusively with
/// `--family negative-binomial`.
///
/// The user's `link(sas_init=...)` / `link(beta_logistic_init=...)` state is
/// not threaded through this resolver: family resolution produces the
/// link-only placeholder, and the standard fit picks up the actual initial
/// state from `FitOptions.sas_link` (see `effective_sas_link_for_family` in
/// `src/solver/estimate.rs`), which overrides the family-embedded link. Keeping
/// the resolver link-state-free leaves a single, narrow family-routing contract
/// shared verbatim with the workflow and PyFFI surfaces.
#[cfg(test)]
pub(crate) fn resolve_family(
    arg: FamilyArg,
    negative_binomial_theta: Option<f64>,
    link_choice: Option<LinkChoice>,
    y: ArrayView1<'_, f64>,
    y_kind: ResponseColumnKind,
    response_name: &str,
) -> Result<LikelihoodSpec, String> {
    if negative_binomial_theta.is_some() && !matches!(arg, FamilyArg::NegativeBinomial) {
        return Err("--negative-binomial-theta requires --family negative-binomial".to_string());
    }
    gam::families::fit_orchestration::resolve_family(
        family_arg_canonical_name(arg),
        negative_binomial_theta,
        link_choice.as_ref(),
        y,
        y_kind,
        response_name,
    )
}

pub(crate) fn binomial_mean_linkwiggle_supports_family(
    family: &LikelihoodSpec,
    link_choice: Option<&LinkChoice>,
) -> bool {
    let standard_binomial = family.is_binomial()
        && matches!(
            &family.link,
            InverseLink::Standard(StandardLink::Logit)
                | InverseLink::Standard(StandardLink::Probit)
                | InverseLink::Standard(StandardLink::CLogLog)
        );
    standard_binomial
        && !link_choice.is_some_and(|choice| matches!(choice.mode, LinkMode::Flexible))
}
