//! Smooth-structure advisories, in the crate that owns the structure.
//!
//! Three things about a resolved `TermCollectionSpec` change how its terms must
//! be READ, and none of them is visible in the fitted numbers:
//!
//! * several one-dimensional spatial smooths of the same family, where the user
//!   almost certainly meant one multi-dimensional smooth over those features;
//! * a feature carried by both a smooth and an explicit linear term, which makes
//!   the fit residualize the smooth against that column so the smooth reports
//!   only the nonlinear remainder;
//! * nested or duplicate smooths, where automatic hierarchical ownership gives
//!   the shared subspace to the higher-priority term and residualizes the other.
//!
//! Each of those silently changes what a term MEANS. A user who is not told will
//! read a residualized smooth as the whole effect.
//!
//! This lived in `gam-cli` and had no counterpart anywhere else, so a user
//! fitting `s(x1, bs=tps) + s(x2, bs=tps)` from Python got two unrelated 1-D
//! smooths and no word about it, while the identical CLI invocation said to
//! write `thinplate(x1, x2)`. SPEC line 10 requires the CLI, the Python library
//! and the Rust library to be unified with a single source of truth; an
//! advisory that exists on one surface only is a parity gap by that rule
//! (issue #2470).
//!
//! The engine was already here — `analyze_smooth_ownership` and
//! [`smooth_term_feature_cols`] are this module's neighbours. Only the messages
//! were stranded. Rendering stays with each surface: the CLI writes them to
//! stderr with its own prefix, and nothing here knows about a terminal.

use std::collections::BTreeMap;

use super::structure_analysis::{
    SmoothStructureAnalysis, analyze_smooth_ownership, smooth_term_feature_cols,
};
use super::{SmoothBasisSpec, SmoothTermSpec, TermCollectionSpec};

/// The spatial-basis family a smooth belongs to, and the feature columns it
/// spans — or `None` for a basis with no multi-dimensional form to advise about.
///
/// Wrapper bases are transparent: a `by=` variable, a factor sum-to-zero
/// constraint or a by-smooth all delegate to the smooth they wrap, because the
/// advice is about the INNER basis's dimensionality either way.
pub(crate) fn spatial_basis_warning_family_and_cols(
    term: &SmoothTermSpec,
) -> Option<(&'static str, &[usize])> {
    spatial_basis_warning_family_and_cols_basis(&term.basis)
}

/// [`spatial_basis_warning_family_and_cols`] against a basis spec directly.
pub(crate) fn spatial_basis_warning_family_and_cols_basis(
    basis: &SmoothBasisSpec,
) -> Option<(&'static str, &[usize])> {
    match basis {
        SmoothBasisSpec::ByVariable { inner, .. }
        | SmoothBasisSpec::FactorSumToZero { inner, .. } => {
            spatial_basis_warning_family_and_cols_basis(inner)
        }
        SmoothBasisSpec::BySmooth { smooth, .. } => {
            spatial_basis_warning_family_and_cols_basis(smooth)
        }
        SmoothBasisSpec::ThinPlate { feature_cols, .. } => Some(("thinplate/tps", feature_cols)),
        SmoothBasisSpec::Sphere { feature_cols, .. } => Some(("sphere/sos", feature_cols)),
        SmoothBasisSpec::ConstantCurvature { feature_cols, .. } => {
            Some(("constant_curvature", feature_cols))
        }
        SmoothBasisSpec::Matern { feature_cols, .. } => Some(("matern", feature_cols)),
        SmoothBasisSpec::MeasureJet { feature_cols, .. } => Some(("measurejet", feature_cols)),
        SmoothBasisSpec::Duchon { feature_cols, .. } => Some(("duchon", feature_cols)),
        // Bases with no multi-dimensional isotropic form: there is no "you
        // probably meant one smooth over both" to suggest.
        SmoothBasisSpec::BSpline1D { .. }
        | SmoothBasisSpec::Pca { .. }
        | SmoothBasisSpec::TensorBSpline { .. }
        | SmoothBasisSpec::FactorSmooth { .. } => None,
    }
}

/// Two or more separate ONE-dimensional smooths of the same isotropic spatial
/// family, which is almost always a mis-specified multi-dimensional smooth.
pub(crate) fn collect_spatial_smooth_usage_warnings(
    spec: &TermCollectionSpec,
    headers: &[String],
    label: &str,
) -> Vec<String> {
    let mut grouped: BTreeMap<&'static str, Vec<String>> = BTreeMap::new();
    for term in &spec.smooth_terms {
        let Some((family, feature_cols)) = spatial_basis_warning_family_and_cols(term) else {
            continue;
        };
        // Only SEPARATE one-dimensional smooths are suspicious. A term that
        // already spans several features is the thing this advice asks for.
        if feature_cols.len() != 1 {
            continue;
        }
        let col = feature_cols[0];
        let feature_name = headers
            .get(col)
            .cloned()
            .unwrap_or_else(|| format!("#{col}"));
        grouped.entry(family).or_default().push(feature_name);
    }

    grouped
        .into_iter()
        .filter_map(|(family, cols)| {
            if cols.len() < 2 {
                return None;
            }
            // `spatial_basis_warning_family_and_cols` returns one of SIX family
            // strings, and only these four get advice: `constant_curvature` and
            // `measurejet` are detected (they are isotropic spatial bases) but
            // deliberately fall through to `None`, because there is no
            // multivariate spelling to recommend for them. Returning `None`
            // rather than emitting a generic message is the existing behaviour
            // and is preserved here deliberately -- this module is a MOVE.
            let example = match family {
                "thinplate/tps" => format!("thinplate({})", cols.join(", ")),
                "matern" => format!("matern({})", cols.join(", ")),
                "duchon" => format!("duchon({})", cols.join(", ")),
                "sphere/sos" => format!("sphere({})", cols.join(", ")),
                _ => return None,
            };
            let bad_example = match family {
                "thinplate/tps" => cols
                    .iter()
                    .map(|col| format!("s({col}, bs=tps)"))
                    .collect::<Vec<_>>()
                    .join(" + "),
                "matern" => cols
                    .iter()
                    .map(|col| format!("s({col}, bs=matern)"))
                    .collect::<Vec<_>>()
                    .join(" + "),
                "duchon" => cols
                    .iter()
                    .map(|col| format!("s({col}, bs=duchon)"))
                    .collect::<Vec<_>>()
                    .join(" + "),
                "sphere/sos" => cols
                    .iter()
                    .map(|col| format!("s({col}, bs=sphere)"))
                    .collect::<Vec<_>>()
                    .join(" + "),
                _ => return None,
            };
            Some(format!(
                "{label}: detected {} separate 1D {family} spatial smooths over [{}]. These build unrelated additive 1D smooths, not one shared spatial manifold. TIP: if you intended one spatial surface, replace `{bad_example}` with one multivariate term such as `{example}`.",
                cols.len(),
                cols.join(", "),
            ))
        })
        .collect()
}

/// A feature carried by both a smooth and an explicit linear term, which makes
/// the fit residualize the smooth against that column.
pub(crate) fn collect_linear_smooth_overlap_warnings(
    spec: &TermCollectionSpec,
    headers: &[String],
    label: &str,
) -> Vec<String> {
    let linear_by_col = spec
        .linear_terms
        .iter()
        .map(|term| (term.feature_col, term.name.as_str()))
        .collect::<BTreeMap<_, _>>();
    let mut warnings = Vec::new();
    for smooth in &spec.smooth_terms {
        let overlaps = smooth_term_feature_cols(smooth)
            .into_iter()
            .filter_map(|col| {
                linear_by_col.get(&col).map(|linear_name| {
                    let feature_name = headers
                        .get(col)
                        .cloned()
                        .unwrap_or_else(|| format!("#{col}"));
                    (feature_name, (*linear_name).to_string())
                })
            })
            .collect::<Vec<_>>();
        if overlaps.is_empty() {
            continue;
        }
        let overlap_features = overlaps
            .iter()
            .map(|(feature_name, _)| feature_name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        let linear_terms = overlaps
            .iter()
            .map(|(_, linear_name)| format!("linear({linear_name})"))
            .collect::<Vec<_>>()
            .join(" + ");
        warnings.push(format!(
            "{label}: feature(s) [{overlap_features}] appear both in smooth term `{}` and explicit linear term(s) `{linear_terms}`. The fit now residualizes the smooth against the intercept and those overlapping linear columns, so the smooth contributes only the nonlinear remainder on those variables. This changes the term decomposition and interpretation.",
            smooth.name
        ));
    }
    warnings
}

/// Nested or duplicate smooths, where automatic hierarchical ownership gives the
/// shared realized subspace to the higher-priority term.
pub(crate) fn collect_hierarchical_smooth_overlap_warnings(
    spec: &TermCollectionSpec,
    headers: &[String],
    label: &str,
) -> Vec<String> {
    let feature_label = |col: usize| {
        headers
            .get(col)
            .cloned()
            .unwrap_or_else(|| format!("#{col}"))
    };
    let join_feature_labels = |cols: &[usize]| {
        cols.iter()
            .map(|&col| feature_label(col))
            .collect::<Vec<_>>()
            .join(", ")
    };

    // The ownership decision is READ from the analysis that the design build
    // also consumes, never re-derived here: an advisory that disagreed with the
    // assembly it describes would be worse than no advisory at all.
    let SmoothStructureAnalysis {
        ownership_order,
        term_feature_cols,
        term_owners,
        ..
    } = analyze_smooth_ownership(&spec.smooth_terms);

    let mut warnings = Vec::new();
    for &target_idx in &ownership_order {
        let owners = &term_owners[target_idx];
        if owners.is_empty() {
            continue;
        }
        let target = &spec.smooth_terms[target_idx];
        let target_features = join_feature_labels(&term_feature_cols[target_idx]);
        let owner_descriptions = owners
            .iter()
            .map(|&owner_idx| {
                format!(
                    "`{}` over [{}]",
                    spec.smooth_terms[owner_idx].name,
                    join_feature_labels(&term_feature_cols[owner_idx]),
                )
            })
            .collect::<Vec<_>>()
            .join(", ");

        warnings.push(format!(
            "{label}: smooth term `{}` over [{target_features}] overlaps nested or duplicate smooth term(s) {}. The fit uses automatic hierarchical ownership: those higher-priority smooth term(s) keep any shared realized subspace, and `{}` is residualized against that overlap before fitting.",
            target.name, owner_descriptions, target.name,
        ));
    }
    warnings
}

/// Every smooth-structure advisory for a resolved spec, in a stable order.
///
/// `headers` names the feature columns; a column past the end of `headers` is
/// reported as `#<index>` rather than dropped, because a warning that silently
/// omits a term is worse than one with an ugly name in it. `label` is the stage
/// the caller is reporting from and is prefixed to every message.
pub fn collect_smooth_structure_warnings(
    spec: &TermCollectionSpec,
    headers: &[String],
    label: &str,
) -> Vec<String> {
    let mut warnings = collect_spatial_smooth_usage_warnings(spec, headers, label);
    warnings.extend(collect_linear_smooth_overlap_warnings(spec, headers, label));
    warnings.extend(collect_hierarchical_smooth_overlap_warnings(
        spec, headers, label,
    ));
    warnings
}
