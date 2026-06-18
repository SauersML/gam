use crate::basis::{
    BSplineIdentifiability, CenterStrategy, ConstantCurvatureIdentifiability,
    MaternIdentifiability, MeasureJetIdentifiability, SpatialIdentifiability,
    SphericalSplineIdentifiability,
};

use super::{ByVarKind, SmoothBasisSpec, SmoothTermSpec, TensorBSplineIdentifiability};

use std::collections::BTreeSet;

fn smooth_basis_feature_cols(basis: &SmoothBasisSpec) -> Vec<usize> {
    match basis {
        SmoothBasisSpec::ByVariable { inner, by_col, .. }
        | SmoothBasisSpec::FactorSumToZero { inner, by_col, .. } => {
            let mut cols = smooth_basis_feature_cols(inner);
            cols.push(*by_col);
            cols.sort_unstable();
            cols.dedup();
            cols
        }
        SmoothBasisSpec::BySmooth { smooth, .. } => smooth_basis_feature_cols(smooth),
        SmoothBasisSpec::BSpline1D { feature_col, .. } => vec![*feature_col],
        SmoothBasisSpec::ThinPlate { feature_cols, .. }
        | SmoothBasisSpec::Sphere { feature_cols, .. }
        | SmoothBasisSpec::ConstantCurvature { feature_cols, .. }
        | SmoothBasisSpec::Matern { feature_cols, .. }
        | SmoothBasisSpec::MeasureJet { feature_cols, .. }
        | SmoothBasisSpec::Duchon { feature_cols, .. }
        | SmoothBasisSpec::Pca { feature_cols, .. }
        | SmoothBasisSpec::TensorBSpline { feature_cols, .. } => feature_cols.clone(),
        SmoothBasisSpec::FactorSmooth { spec } => {
            let mut cols = spec.continuous_cols.clone();
            cols.push(spec.group_col);
            cols.sort_unstable();
            cols.dedup();
            cols
        }
    }
}

pub fn smooth_term_feature_cols(term: &SmoothTermSpec) -> Vec<usize> {
    smooth_basis_feature_cols(&term.basis)
}

fn smooth_basis_family_rank(term: &SmoothTermSpec) -> u8 {
    match &term.basis {
        SmoothBasisSpec::ByVariable { inner, .. }
        | SmoothBasisSpec::FactorSumToZero { inner, .. } => {
            smooth_basis_family_rank(&SmoothTermSpec {
                name: term.name.clone(),
                basis: (**inner).clone(),
                shape: term.shape,
                joint_null_rotation: None,
            })
        }
        SmoothBasisSpec::BSpline1D { .. } => 0,
        SmoothBasisSpec::TensorBSpline { .. } => 1,
        SmoothBasisSpec::ThinPlate { .. } => 2,
        SmoothBasisSpec::Sphere { .. } => 3,
        SmoothBasisSpec::Matern { .. } => 4,
        SmoothBasisSpec::Duchon { .. } => 5,
        SmoothBasisSpec::Pca { .. } => 6,
        SmoothBasisSpec::ConstantCurvature { .. } => 8,
        SmoothBasisSpec::MeasureJet { .. } => 9,
        SmoothBasisSpec::BySmooth { smooth, .. } => smooth_basis_family_rank(&SmoothTermSpec {
            name: term.name.clone(),
            basis: (**smooth).clone(),
            shape: term.shape,
            joint_null_rotation: None,
        }),
        SmoothBasisSpec::FactorSmooth { .. } => 7,
    }
}

pub(super) fn smooth_has_frozen_identifiability(term: &SmoothTermSpec) -> bool {
    match &term.basis {
        SmoothBasisSpec::ByVariable { inner, .. }
        | SmoothBasisSpec::FactorSumToZero { inner, .. } => {
            smooth_has_frozen_identifiability(&SmoothTermSpec {
                name: term.name.clone(),
                basis: (**inner).clone(),
                shape: term.shape,
                joint_null_rotation: None,
            })
        }
        SmoothBasisSpec::BSpline1D { spec, .. } => {
            matches!(
                spec.identifiability,
                BSplineIdentifiability::FrozenTransform { .. }
            )
        }
        SmoothBasisSpec::ThinPlate { spec, .. } => matches!(
            spec.identifiability,
            SpatialIdentifiability::FrozenTransform { .. }
        ),
        SmoothBasisSpec::Sphere { spec, .. } => {
            matches!(spec.center_strategy, CenterStrategy::UserProvided(_))
                || matches!(
                    spec.identifiability,
                    SphericalSplineIdentifiability::FrozenTransform { .. }
                )
        }
        SmoothBasisSpec::ConstantCurvature { spec, .. } => {
            matches!(spec.center_strategy, CenterStrategy::UserProvided(_))
                || matches!(
                    spec.identifiability,
                    ConstantCurvatureIdentifiability::FrozenTransform { .. }
                )
        }
        SmoothBasisSpec::MeasureJet { spec, .. } => {
            matches!(spec.center_strategy, CenterStrategy::UserProvided(_))
                || matches!(
                    spec.identifiability,
                    MeasureJetIdentifiability::FrozenTransform { .. }
                )
        }
        SmoothBasisSpec::Matern { spec, .. } => matches!(
            spec.identifiability,
            MaternIdentifiability::FrozenTransform { .. }
        ),
        SmoothBasisSpec::BySmooth { by_kind, .. } => match by_kind {
            ByVarKind::Factor { frozen_levels, .. } => frozen_levels.is_some(),
            ByVarKind::Numeric { .. } => true,
        },
        SmoothBasisSpec::FactorSmooth { spec } => spec.group_frozen_levels.is_some(),
        SmoothBasisSpec::Duchon { spec, .. } => matches!(
            spec.identifiability,
            SpatialIdentifiability::FrozenTransform { .. }
        ),
        SmoothBasisSpec::Pca {
            centered,
            center_mean,
            pca_basis_path,
            ..
        } => !*centered || center_mean.is_some() || pca_basis_path.is_some(),
        SmoothBasisSpec::TensorBSpline { spec, .. } => matches!(
            spec.identifiability,
            TensorBSplineIdentifiability::FrozenTransform { .. }
        ),
    }
}

fn compare_smooth_ownership_priority(
    lhs_idx: usize,
    lhs: &SmoothTermSpec,
    rhs_idx: usize,
    rhs: &SmoothTermSpec,
) -> std::cmp::Ordering {
    let lhs_cols = smooth_term_feature_cols(lhs);
    let rhs_cols = smooth_term_feature_cols(rhs);
    lhs_cols
        .len()
        .cmp(&rhs_cols.len())
        .then_with(|| lhs_cols.cmp(&rhs_cols))
        .then_with(|| smooth_basis_family_rank(lhs).cmp(&smooth_basis_family_rank(rhs)))
        .then_with(|| lhs.name.cmp(&rhs.name))
        .then(lhs_idx.cmp(&rhs_idx))
}

/// The `(by_col, level_bits)` row-gate of a factor-`by=` level smooth
/// (`s(x, by=fac)`, treatment-contrast level), or `None` for any other smooth
/// (including numeric-`by` scaling, which is NOT row-gated).
///
/// A level-gated smooth's design is zero on every row outside its level, so its
/// columns are NOT in the column span of an un-gated (full-support) smooth on
/// the same covariate. Ownership/orthogonalization must therefore skip it
/// (otherwise the per-group deviation is residualized away to zero — #1276).
fn factor_by_level_gate_of(term: &SmoothTermSpec) -> Option<(usize, u64)> {
    match &term.basis {
        SmoothBasisSpec::ByVariable {
            by_col,
            by: crate::terms::smooth::ByVariableSpec::Level { value_bits, .. },
            ..
        } => Some((*by_col, *value_bits)),
        _ => None,
    }
}

fn smooth_is_owned_by_prior_term(owner: &SmoothTermSpec, target: &SmoothTermSpec) -> bool {
    // A factor-`by=` level smooth is row-gated (zero off its level), so its
    // columns lie outside the span of any owner that is not gated to the SAME
    // (by_col, level): the un-gated population smooth `s(x)` does not span the
    // group deviation `s(x, by=g==level)`. Residualizing the gated deviation
    // against the population smooth collapses it to zero (#1276). Identifiability
    // of the deviation comes from its own factor-level gate + penalty, handled
    // by `factor_by_level_gate` in design construction — not from ownership.
    if let Some(target_gate) = factor_by_level_gate_of(target) {
        if factor_by_level_gate_of(owner) != Some(target_gate) {
            return false;
        }
    }
    let owner_features = smooth_term_feature_cols(owner)
        .into_iter()
        .collect::<BTreeSet<_>>();
    let target_features = smooth_term_feature_cols(target)
        .into_iter()
        .collect::<BTreeSet<_>>();
    owner_features.is_subset(&target_features)
}

/// Static (spec-only) description of the hierarchical smooth-ownership decomposition.
///
/// This is the single source of truth for the deterministic ownership policy that
/// `apply_global_smooth_identifiability` uses during the fit: the processing order of
/// smooth terms, the feature columns each term spans, the candidate lower-order owners of
/// each term (nested/duplicate feature sets), and the basis-family rank used as a
/// tie-breaker. The fit engine consumes this structure and additionally applies a numerical
/// cross-residual overlap test on the realized design columns; the CLI structure-warning
/// path consumes the same structure for diagnostic messages, so both paths agree on which
/// smooths own which subspaces.
pub struct SmoothStructureAnalysis {
    /// Smooth-term indices sorted into ownership-processing order (lowest priority first):
    /// lower-order / narrower smooths come first and own their subspaces.
    pub ownership_order: Vec<usize>,
    /// `term_feature_cols[idx]` are the sorted, deduplicated feature columns that smooth term
    /// `idx` spans (indexed by the original smooth-term index, not by `ownership_order`).
    pub term_feature_cols: Vec<Vec<usize>>,
    /// `term_owners[idx]` are the indices of prior (in `ownership_order`) smooth terms whose
    /// feature set is a subset of term `idx`'s feature set, i.e. candidate owners of `idx`.
    /// The list is given in ownership-processing order.
    pub term_owners: Vec<Vec<usize>>,
}

/// Compute the static hierarchical smooth-ownership decomposition from the smooth-term specs.
///
/// `smoothspecs` is the same slice that `apply_global_smooth_identifiability` receives.
pub fn analyze_smooth_ownership(smoothspecs: &[SmoothTermSpec]) -> SmoothStructureAnalysis {
    let term_feature_cols: Vec<Vec<usize>> =
        smoothspecs.iter().map(smooth_term_feature_cols).collect();

    let mut ownership_order: Vec<usize> = (0..smoothspecs.len()).collect();
    ownership_order.sort_by(|&lhs, &rhs| {
        compare_smooth_ownership_priority(lhs, &smoothspecs[lhs], rhs, &smoothspecs[rhs])
    });

    let mut term_owners = vec![Vec::<usize>::new(); smoothspecs.len()];
    for (pos, &target_idx) in ownership_order.iter().enumerate() {
        let target = &smoothspecs[target_idx];
        term_owners[target_idx] = ownership_order[..pos]
            .iter()
            .copied()
            .filter(|&owner_idx| smooth_is_owned_by_prior_term(&smoothspecs[owner_idx], target))
            .collect();
    }

    SmoothStructureAnalysis {
        ownership_order,
        term_feature_cols,
        term_owners,
    }
}
