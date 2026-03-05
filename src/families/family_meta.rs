use crate::types::{LikelihoodFamily, LinkFunction};

#[inline]
pub fn family_to_string(f: LikelihoodFamily) -> &'static str {
    match f {
        LikelihoodFamily::GaussianIdentity => "gaussian",
        LikelihoodFamily::BinomialLogit => "binomial-logit",
        LikelihoodFamily::BinomialProbit => "binomial-probit",
        LikelihoodFamily::BinomialCLogLog => "binomial-cloglog",
        LikelihoodFamily::BinomialSas => "binomial-sas",
        LikelihoodFamily::BinomialBetaLogistic => "binomial-beta-logistic",
        LikelihoodFamily::BinomialMixture => "binomial-blended-inverse-link",
        LikelihoodFamily::RoystonParmar => "royston-parmar",
    }
}

#[inline]
pub fn family_to_link(f: LikelihoodFamily) -> LinkFunction {
    match f {
        LikelihoodFamily::GaussianIdentity => LinkFunction::Identity,
        LikelihoodFamily::BinomialLogit => LinkFunction::Logit,
        LikelihoodFamily::BinomialProbit => LinkFunction::Probit,
        LikelihoodFamily::BinomialCLogLog => LinkFunction::CLogLog,
        LikelihoodFamily::BinomialSas => LinkFunction::Sas,
        LikelihoodFamily::BinomialBetaLogistic => LinkFunction::BetaLogistic,
        LikelihoodFamily::BinomialMixture => LinkFunction::Logit,
        LikelihoodFamily::RoystonParmar => LinkFunction::Identity,
    }
}

#[inline]
pub fn is_binomial_family(f: LikelihoodFamily) -> bool {
    matches!(
        f,
        LikelihoodFamily::BinomialLogit
            | LikelihoodFamily::BinomialProbit
            | LikelihoodFamily::BinomialCLogLog
            | LikelihoodFamily::BinomialSas
            | LikelihoodFamily::BinomialBetaLogistic
            | LikelihoodFamily::BinomialMixture
    )
}

#[inline]
pub fn pretty_family_name(f: LikelihoodFamily) -> &'static str {
    match f {
        LikelihoodFamily::GaussianIdentity => "Gaussian Identity",
        LikelihoodFamily::BinomialLogit => "Binomial Logit",
        LikelihoodFamily::BinomialProbit => "Binomial Probit",
        LikelihoodFamily::BinomialCLogLog => "Binomial CLogLog",
        LikelihoodFamily::BinomialSas => "Binomial SAS",
        LikelihoodFamily::BinomialBetaLogistic => "Binomial Beta-Logistic",
        LikelihoodFamily::BinomialMixture => "Binomial Blended Inverse-Link",
        LikelihoodFamily::RoystonParmar => "Royston Parmar",
    }
}
