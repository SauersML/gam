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
