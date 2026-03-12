use crate::types::{LikelihoodFamily, LinkFunction};

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
