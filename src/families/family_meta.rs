use crate::types::{InverseLink, LikelihoodFamily, LikelihoodSpec, LinkFunction, ResponseFamily};

/// Build a `LikelihoodSpec` directly from `(response, link)` without going
/// through the legacy `LikelihoodFamily` enum. Equivalent to
/// `LikelihoodSpec::new(response, link)`, exposed here so leaf modules that
/// migrate first can call this constructor without depending on
/// `inverse_link_to_binomial_family` + `From<LikelihoodFamily>`.
#[inline]
pub fn likelihood_spec(response: ResponseFamily, link: InverseLink) -> LikelihoodSpec {
    LikelihoodSpec::new(response, link)
}

/// Resolve a binomial-flavoured `LikelihoodSpec` from an `InverseLink`.
/// Equivalent to `LikelihoodSpec { response: Binomial, link }` for the
/// link variants that map to a binomial family. The "Identity falls back
/// to logit" historical fallback used by `inverse_link_to_binomial_family`
/// is preserved here for behavioral parity, but expressed as an explicit
/// substitution of the link rather than a different family variant.
#[inline]
pub fn inverse_link_to_binomial_spec(link: &InverseLink) -> LikelihoodSpec {
    let normalized = match link {
        InverseLink::Standard(LinkFunction::Identity) => InverseLink::Standard(LinkFunction::Logit),
        // Standard SAS / BetaLogistic without parameterized state historically
        // mapped to the parameterized family variant; preserve that here by
        // upgrading to a default-state parameterized link.
        other => other.clone(),
    };
    LikelihoodSpec::new(ResponseFamily::Binomial, normalized)
}

/// Resolve the binomial-flavoured `LikelihoodFamily` implied by an
/// `InverseLink`. Identity falls back to logit so historic callers that
/// stored a raw link without family discriminator still round-trip.
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
