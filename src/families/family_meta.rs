use crate::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};

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
    /// Construct an `UnsupportedLinkError` tagged with the response-family
    /// name (`"binomial"`, `"gaussian"`, ...) and a printable name for the
    /// offending `InverseLink` variant (extracted via the module-private
    /// `inverse_link_diagnostic_name`). No allocation beyond the link name.
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

/// Build a `LikelihoodSpec` directly from `(response, link)`.
#[inline]
pub const fn likelihood_spec(response: ResponseFamily, link: InverseLink) -> LikelihoodSpec {
    LikelihoodSpec::new(response, link)
}

/// Resolve a binomial-flavoured `LikelihoodSpec` from an `InverseLink`.
///
/// `StandardLink::Logit | Probit | CLogLog` and the state-bearing
/// `LatentCLogLog / Sas / BetaLogistic / Mixture` variants are accepted as
/// binomial-compatible. `StandardLink::Log | Identity` have no canonical
/// binomial meaning and return `UnsupportedLinkError`. Since
/// `InverseLink::Standard` carries `StandardLink` (not `LinkFunction`), the
/// previously-required `Standard(LinkFunction::Sas | BetaLogistic)` arm is
/// structurally impossible and has been removed.
#[inline]
pub fn inverse_link_to_binomial_spec(
    link: &InverseLink,
) -> Result<LikelihoodSpec, UnsupportedLinkError> {
    match link {
        InverseLink::Standard(StandardLink::Logit)
        | InverseLink::Standard(StandardLink::Probit)
        | InverseLink::Standard(StandardLink::CLogLog) => {
            Ok(LikelihoodSpec::new(ResponseFamily::Binomial, link.clone()))
        }
        InverseLink::LatentCLogLog(_)
        | InverseLink::Sas(_)
        | InverseLink::BetaLogistic(_)
        | InverseLink::Mixture(_) => {
            Ok(LikelihoodSpec::new(ResponseFamily::Binomial, link.clone()))
        }
        InverseLink::Standard(StandardLink::Log)
        | InverseLink::Standard(StandardLink::Identity) => {
            Err(UnsupportedLinkError::new("binomial", link))
        }
    }
}
