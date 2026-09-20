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
        FamilyArg::InverseGaussian => Some("inverse-gaussian"),
        FamilyArg::Tweedie => Some("tweedie"),
        FamilyArg::Beta => Some("beta"),
        FamilyArg::StudentT => Some("student-t"),
        FamilyArg::RoystonParmar => Some("royston-parmar"),
        FamilyArg::Expectile => Some("expectile"),
        // Dispatched by `run_fit` before the canonical family resolver (the
        // multinomial artifact is a softmax multi-output model, not a scalar
        // GLM), so this canonical name is used only for display/config echoing.
        FamilyArg::Multinomial => Some("multinomial"),
    }
}

