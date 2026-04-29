use crate::types::{InverseLink, LikelihoodFamily, LinkFunction};

#[inline]
pub fn family_to_string(f: LikelihoodFamily) -> &'static str {
    f.name()
}

#[inline]
pub fn family_to_link(f: LikelihoodFamily) -> LinkFunction {
    f.link_function()
}

#[inline]
pub fn is_binomial_family(f: LikelihoodFamily) -> bool {
    f.is_binomial()
}

#[inline]
pub fn pretty_familyname(f: LikelihoodFamily) -> &'static str {
    f.pretty_name()
}

#[inline]
pub fn inverse_link_to_binomial_family(link: &InverseLink) -> LikelihoodFamily {
    match link {
        InverseLink::Standard(LinkFunction::Log) => LikelihoodFamily::PoissonLog,
        InverseLink::Standard(LinkFunction::Logit) => LikelihoodFamily::BinomialLogit,
        InverseLink::Standard(LinkFunction::Probit) => LikelihoodFamily::BinomialProbit,
        InverseLink::Standard(LinkFunction::CLogLog) => LikelihoodFamily::BinomialCLogLog,
        InverseLink::Standard(LinkFunction::Sas) | InverseLink::Sas(_) => {
            LikelihoodFamily::BinomialSas
        }
        InverseLink::Standard(LinkFunction::BetaLogistic) | InverseLink::BetaLogistic(_) => {
            LikelihoodFamily::BinomialBetaLogistic
        }
        InverseLink::LatentCLogLog(_) => LikelihoodFamily::BinomialLatentCLogLog,
        InverseLink::Mixture(_) => LikelihoodFamily::BinomialMixture,
        InverseLink::Standard(LinkFunction::Identity) => LikelihoodFamily::BinomialLogit,
    }
}
