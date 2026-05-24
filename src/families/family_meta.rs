use crate::types::{InverseLink, LikelihoodSpec, LinkFunction, ResponseFamily};

/// Error returned when an `InverseLink` cannot be paired with a particular
/// response family because the link is structurally unsupported for that
/// family. Carries the link name so call sites can produce a useful message
/// without losing the offending variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsupportedLinkError {
    pub family: &'static str,
    pub link_name: String,
}

impl UnsupportedLinkError {
    #[inline]
    pub fn new(family: &'static str, link: &InverseLink) -> Self {
        Self {
            family,
            link_name: inverse_link_diagnostic_name(link),
        }
    }
}

impl std::fmt::Display for UnsupportedLinkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "inverse link `{}` is not supported by the {} response family",
            self.link_name, self.family
        )
    }
}

impl std::error::Error for UnsupportedLinkError {}

#[inline]
fn inverse_link_diagnostic_name(link: &InverseLink) -> String {
    match link {
        InverseLink::Standard(lf) => lf.name().to_string(),
        InverseLink::LatentCLogLog(_) => "latent-cloglog".to_string(),
        InverseLink::Sas(_) => "sas".to_string(),
        InverseLink::BetaLogistic(_) => "beta-logistic".to_string(),
        InverseLink::Mixture(_) => "mixture".to_string(),
    }
}

/// Build a `LikelihoodSpec` directly from `(response, link)` without going
/// through the legacy `LikelihoodFamily` enum. Equivalent to
/// `LikelihoodSpec::new(response, link)`, exposed here so leaf modules that
/// migrate first can call this constructor without depending on a separate
/// `From<LikelihoodFamily>` bridge.
#[inline]
pub const fn likelihood_spec(response: ResponseFamily, link: InverseLink) -> LikelihoodSpec {
    LikelihoodSpec::new(response, link)
}

/// Resolve a binomial-flavoured `LikelihoodSpec` from an `InverseLink`.
///
/// The match is exhaustive over `LinkFunction` so that every future addition
/// to the link enum forces the author to declare whether it pairs with the
/// binomial family. Variants that are structurally not binomial (e.g.
/// `LinkFunction::Log`, which is the Poisson/Gamma log link, and
/// `LinkFunction::Identity`, which has no canonical binomial meaning)
/// return `UnsupportedLinkError` rather than being silently coerced.
#[inline]
pub fn inverse_link_to_binomial_spec(
    link: &InverseLink,
) -> Result<LikelihoodSpec, UnsupportedLinkError> {
    match link {
        InverseLink::Standard(LinkFunction::Logit)
        | InverseLink::Standard(LinkFunction::Probit)
        | InverseLink::Standard(LinkFunction::CLogLog)
        | InverseLink::Standard(LinkFunction::Sas)
        | InverseLink::Standard(LinkFunction::BetaLogistic) => {
            Ok(LikelihoodSpec::new(ResponseFamily::Binomial, link.clone()))
        }
        InverseLink::LatentCLogLog(_)
        | InverseLink::Sas(_)
        | InverseLink::BetaLogistic(_)
        | InverseLink::Mixture(_) => {
            Ok(LikelihoodSpec::new(ResponseFamily::Binomial, link.clone()))
        }
        InverseLink::Standard(LinkFunction::Log)
        | InverseLink::Standard(LinkFunction::Identity) => {
            Err(UnsupportedLinkError::new("binomial", link))
        }
    }
}
