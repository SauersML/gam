//! One spelling per behavior (SPEC R25).
//!
//! The formula surface accepts exactly one spelling for each smooth type,
//! option key, option value and term function. The tables below list the
//! other spellings that once meant the same thing, each paired with its
//! canonical spelling. They are not accepted: parsing refuses them with an
//! error that names the canonical spelling, so a user who types one learns
//! the single supported form instead of getting a silent alias.

use std::collections::BTreeMap;

/// `(refused spelling, canonical spelling)`.
pub(crate) type RemovedSpelling = (&'static str, &'static str);

/// `bs=` smooth-type values (also tensor-margin `bs=` vector entries).
pub(crate) const SMOOTH_TYPES: &[RemovedSpelling] = &[
    ("tp", "tps"),
    ("gp", "matern"),
    ("cs", "cr"),
    ("cc", "cyclic"),
    ("cp", "cyclic"),
    ("cyclic-ps", "cyclic"),
    ("periodic", "cyclic"),
    ("curvature", "curv"),
    ("constant_curvature", "curv"),
    ("mkappa", "curv"),
    ("measurejet", "mjs"),
    ("measure_jet", "mjs"),
    ("web", "mjs"),
];

/// Option keys of smooth terms.
pub(crate) const SMOOTH_OPTION_KEYS: &[RemovedSpelling] = &[
    ("type", "bs"),
    ("basis_dim", "k"),
    ("basis-dim", "k"),
    ("basisdim", "k"),
    ("m", "penalty_order"),
    ("kernel", "method"),
];

/// `method=` values of the sphere smooth.
pub(crate) const SPHERE_METHODS: &[RemovedSpelling] = &[
    ("wahba", "sobolev"),
    ("wahba_sobolev", "sobolev"),
    ("wahba-sobolev", "sobolev"),
    ("spherical_harmonic", "harmonic"),
    ("spherical-harmonic", "harmonic"),
];

/// `bc=`/`boundary=` tokens.
pub(crate) const BOUNDARY_TOKENS: &[RemovedSpelling] = &[("cyclic", "periodic"), ("cc", "periodic")];

/// Formula term functions.
pub(crate) const TERM_FUNCTIONS: &[RemovedSpelling] = &[
    ("constrain", "linear"),
    ("constraint", "linear"),
    ("box", "linear"),
    ("nonnegative_coef", "nonnegative"),
    ("nonpositive_coef", "nonpositive"),
    ("re", "group"),
    ("tensor", "te"),
    ("interaction", "te"),
    ("periodic", "cyclic"),
    ("cc", "cyclic"),
    ("cp", "cyclic"),
    ("measurejet", "mjs"),
    ("measure_jet", "mjs"),
    ("web", "mjs"),
    ("curvature", "curv"),
    ("constant_curvature", "curv"),
    ("mkappa", "curv"),
];

/// `smooths[...]` descriptor `kind=` values.
pub(crate) const DESCRIPTOR_KINDS: &[RemovedSpelling] = &[
    ("curvature", "curv"),
    ("constant_curvature", "curv"),
    ("mkappa", "curv"),
    ("measurejet", "mjs"),
    ("measure_jet", "mjs"),
    ("web", "mjs"),
];

/// Option keys of the parametric term functions (`linear()`, `bounded()`).
pub(crate) const TERM_OPTION_KEYS: &[RemovedSpelling] =
    &[("lower", "min"), ("upper", "max"), ("pull", "prior")];

/// `bounded(..., prior=)` values.
pub(crate) const BOUNDED_PRIORS: &[RemovedSpelling] = &[
    ("log-jacobian", "uniform"),
    ("log_jacobian", "uniform"),
    ("jacobian", "uniform"),
];

/// The canonical spelling of a refused `spelling`, if `table` lists it.
pub(crate) fn canonical_for(table: &[RemovedSpelling], spelling: &str) -> Option<&'static str> {
    table
        .iter()
        .find(|(removed, _)| *removed == spelling)
        .map(|(_, canonical)| *canonical)
}

/// The error for a refused smooth-type value.
pub(crate) fn smooth_type_error(spelling: &str, canonical: &str) -> String {
    format!("unknown smooth type `{spelling}`; use `{canonical}`")
}

/// Strip surrounding whitespace and one layer of quotes from an option value.
fn bare_value(raw: &str) -> String {
    raw.trim()
        .trim_matches(|c| c == '"' || c == '\'')
        .trim()
        .to_ascii_lowercase()
}

/// The entries of a scalar or vector option value (`cr`, `c('cr','ps')`,
/// `[cr, ps]`, `(cr, ps)`), lowercased and unquoted.
fn value_entries(raw: &str) -> Vec<String> {
    let trimmed = raw.trim();
    let inner = if let Some(rest) = trimmed
        .strip_prefix("c(")
        .or_else(|| trimmed.strip_prefix("C("))
        .and_then(|rest| rest.strip_suffix(')'))
    {
        rest
    } else if let Some(rest) = trimmed.strip_prefix('[').and_then(|rest| rest.strip_suffix(']')) {
        rest
    } else if let Some(rest) = trimmed.strip_prefix('(').and_then(|rest| rest.strip_suffix(')')) {
        rest
    } else {
        trimmed
    };
    inner
        .split(',')
        .map(bare_value)
        .filter(|entry| !entry.is_empty())
        .collect()
}

/// Refuse every removed spelling in a smooth term's option map: removed option
/// keys, removed `bs=` smooth types (scalar or per-margin vector), removed
/// sphere `method=` values and removed `bc=`/`boundary=` tokens.
///
/// `term` is the function the user wrote (`s`, `te`, `sphere`, ...), used only
/// in the message.
pub(crate) fn reject_removed_smooth_spellings(
    term: &str,
    options: &BTreeMap<String, String>,
) -> Result<(), String> {
    for key in options.keys() {
        if let Some(canonical) = canonical_for(SMOOTH_OPTION_KEYS, key) {
            return Err(format!("unknown option `{key}` in {term}(); use `{canonical}`"));
        }
    }
    if let Some(raw) = options.get("bs") {
        for entry in value_entries(raw) {
            if let Some(canonical) = canonical_for(SMOOTH_TYPES, &entry) {
                return Err(smooth_type_error(&entry, canonical));
            }
        }
    }
    if let Some(raw) = options.get("method") {
        let value = bare_value(raw);
        if let Some(canonical) = canonical_for(SPHERE_METHODS, &value) {
            return Err(format!(
                "unknown sphere method `{value}` in {term}(); use `{canonical}`"
            ));
        }
    }
    for key in ["bc", "boundary"] {
        if let Some(raw) = options.get(key) {
            for entry in value_entries(raw) {
                if let Some(canonical) = canonical_for(BOUNDARY_TOKENS, &entry) {
                    return Err(format!(
                        "unknown boundary token `{entry}` in {term}({key}=...); use `{canonical}`"
                    ));
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn every_table_maps_to_a_spelling_it_does_not_also_refuse() {
        for table in [
            SMOOTH_TYPES,
            SMOOTH_OPTION_KEYS,
            SPHERE_METHODS,
            BOUNDARY_TOKENS,
            TERM_FUNCTIONS,
            DESCRIPTOR_KINDS,
            TERM_OPTION_KEYS,
            BOUNDED_PRIORS,
        ] {
            for (removed, canonical) in table {
                assert_ne!(removed, canonical);
                assert!(
                    canonical_for(table, canonical).is_none(),
                    "canonical `{canonical}` is itself refused"
                );
            }
        }
    }

    #[test]
    fn removed_smooth_spellings_name_the_canonical_one() {
        let cases: &[(&[(&str, &str)], &str)] = &[
            (&[("bs", "tp")], "unknown smooth type `tp`; use `tps`"),
            (&[("bs", "'gp'")], "unknown smooth type `gp`; use `matern`"),
            (&[("bs", "c('cc','ps')")], "unknown smooth type `cc`; use `cyclic`"),
            (&[("bs", "[ps, cs]")], "unknown smooth type `cs`; use `cr`"),
            (&[("basis_dim", "10")], "unknown option `basis_dim` in s(); use `k`"),
            (&[("type", "ps")], "unknown option `type` in s(); use `bs`"),
            (&[("m", "2")], "unknown option `m` in s(); use `penalty_order`"),
            (&[("kernel", "sobolev")], "unknown option `kernel` in s(); use `method`"),
            (&[("method", "wahba")], "unknown sphere method `wahba` in s(); use `sobolev`"),
            (
                &[("bc", "[cyclic, open]")],
                "unknown boundary token `cyclic` in s(bc=...); use `periodic`",
            ),
        ];
        for (pairs, expected) in cases {
            let err = reject_removed_smooth_spellings("s", &opts(pairs)).unwrap_err();
            assert_eq!(&err, expected);
        }
    }

    #[test]
    fn canonical_smooth_spellings_pass() {
        for pairs in [
            &[("bs", "tps"), ("k", "10")][..],
            &[("bs", "c('cyclic','ps')"), ("penalty_order", "2")][..],
            &[("method", "sobolev"), ("bc", "[periodic, open]")][..],
        ] {
            assert_eq!(reject_removed_smooth_spellings("s", &opts(pairs)), Ok(()));
        }
    }
}
