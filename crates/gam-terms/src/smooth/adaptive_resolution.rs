//! The resolution of a formula-default smooth basis that nobody chose.
//!
//! A formula default (`s(x)`, `s(x, bs="cyclic")`, `fs(x, g)`, an auto-sized
//! radial or sphere smooth) is built at its provisioned size, adequate without
//! growth, because most routes never refine it. The standard formula workflow
//! does: it starts every basis it grows at the penalized-resolution pilot
//! ([`starting_resolution`]) and refines it from the converged fit's own
//! evidence (#1689, #3078, #3149). This module is the one place that knows,
//! per basis family, what that resolution is, where the loop starts it, how
//! one level of nested refinement changes it, how far the data can identify
//! it, how many raw coefficients it realizes, and how to write it back into
//! the spec. An explicit user size carries no adaptive provenance and is never
//! touched.

use ndarray::{ArrayView1, ArrayView2, Axis};
use serde::{Deserialize, Serialize};

use super::{ByVarKind, ByVariableSpec, FactorSmoothFlavour, SmoothBasisSpec};
use crate::basis::{
    BSplineKnotSpec, CenterStrategy, DuchonNullspaceOrder, OneDimensionalBoundary,
    SPHERICAL_HARMONIC_MAX_DEGREE, SphereMethod, center_strategy_is_auto,
    center_strategy_with_num_centers, count_unique_coordinate_rows,
    default_spherical_harmonic_degree, duchon_nullspace_dimension, penalized_resolution_rank,
    refined_harmonic_degree, refined_internal_knots, refined_num_centers, refined_periodic_basis,
    starting_num_centers, thin_plate_polynomial_basis_dimension,
};
use crate::term_builder::{
    factor_smooth_pilot_internal_knots, pilot_cyclic_basis_dim, pilot_duchon_center_count,
    pilot_internal_knots, pilot_univariate_spline_basis_dim,
};

/// The resolution coordinate of one adaptive smooth basis.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub enum AdaptiveResolution {
    /// Requested center count of an auto-sized radial basis (thin-plate,
    /// Duchon, constant-curvature, measure-jet, Wahba sphere).
    Centers(usize),
    /// Internal-knot count of an open formula-default B-spline, or of the
    /// shared marginal of a factor smooth.
    InternalKnots(usize),
    /// Basis dimension of a formula-default cyclic smooth.
    PeriodicBasis(usize),
    /// Maximum degree of a formula-default spherical-harmonic basis.
    HarmonicDegree(usize),
}

impl AdaptiveResolution {
    fn value(&self) -> usize {
        match self {
            Self::Centers(v)
            | Self::InternalKnots(v)
            | Self::PeriodicBasis(v)
            | Self::HarmonicDegree(v) => *v,
        }
    }

    fn with_value(&self, value: usize) -> Self {
        match self {
            Self::Centers(_) => Self::Centers(value),
            Self::InternalKnots(_) => Self::InternalKnots(value),
            Self::PeriodicBasis(_) => Self::PeriodicBasis(value),
            Self::HarmonicDegree(_) => Self::HarmonicDegree(value),
        }
    }

    fn same_kind(&self, other: &Self) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }

    /// `self` bounded above by `bound`, but never below `floor`: a proposal
    /// is limited by what the data can identify, and a basis that already
    /// converged is never shrunk. A `bound` of another kind leaves `self`
    /// unchanged.
    pub fn clamped(&self, bound: &Self, floor: &Self) -> Self {
        if !self.same_kind(bound) || !self.same_kind(floor) {
            return self.clone();
        }
        self.with_value(self.value().min(bound.value()).max(floor.value()))
    }

    /// The number of unit steps from `self` up to `target`. Zero when
    /// `target` does not exceed `self`.
    pub fn steps_to(&self, target: &Self) -> usize {
        if !self.same_kind(target) {
            return 0;
        }
        target.value().saturating_sub(self.value())
    }

    /// The point `step` of [`Self::steps_to`]`(target)` on the path from
    /// `self` to `target`: step 0 is `self` and the last step is `target`.
    pub fn toward(&self, target: &Self, step: usize) -> Self {
        let step = step.min(self.steps_to(target));
        self.with_value(self.value() + step)
    }

    /// Whether `self` exceeds `other`.
    pub fn exceeds(&self, other: &Self) -> bool {
        self.same_kind(other) && self.value() > other.value()
    }
}

impl std::fmt::Display for AdaptiveResolution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Centers(c) => write!(f, "{c} centers"),
            Self::InternalKnots(k) => write!(f, "{k} internal knots"),
            Self::PeriodicBasis(b) => write!(f, "cyclic basis dimension {b}"),
            Self::HarmonicDegree(l) => write!(f, "harmonic degree {l}"),
        }
    }
}

fn radial_center_strategy(basis: &SmoothBasisSpec) -> Option<(&CenterStrategy, &[usize])> {
    use SmoothBasisSpec as B;
    match basis {
        B::ByVariable { inner, .. } | B::FactorSumToZero { inner, .. } => {
            radial_center_strategy(inner)
        }
        B::BySmooth { smooth, .. } => radial_center_strategy(smooth),
        B::ThinPlate {
            feature_cols, spec, ..
        } => Some((&spec.center_strategy, feature_cols.as_slice())),
        B::Duchon {
            feature_cols, spec, ..
        } => Some((&spec.center_strategy, feature_cols.as_slice())),
        B::ConstantCurvature { feature_cols, spec } => {
            Some((&spec.center_strategy, feature_cols.as_slice()))
        }
        B::MeasureJet {
            feature_cols, spec, ..
        } => Some((&spec.center_strategy, feature_cols.as_slice())),
        B::Sphere { feature_cols, spec } if matches!(spec.method, SphereMethod::Wahba) => {
            Some((&spec.center_strategy, feature_cols.as_slice()))
        }
        // Matérn's learned range changes both its basin and its realized
        // kernel rank as centers move; it has no validated saturation
        // contract, so the resolution loop does not claim it.
        _ => None,
    }
    .filter(|(strategy, cols)| !cols.is_empty() && center_strategy_is_auto(strategy))
}

fn radial_center_strategy_mut(
    basis: &mut SmoothBasisSpec,
) -> Option<(&mut CenterStrategy, Vec<usize>)> {
    use SmoothBasisSpec as B;
    match basis {
        B::ByVariable { inner, .. } | B::FactorSumToZero { inner, .. } => {
            radial_center_strategy_mut(inner)
        }
        B::BySmooth { smooth, .. } => radial_center_strategy_mut(smooth),
        B::ThinPlate {
            feature_cols, spec, ..
        } => Some((&mut spec.center_strategy, feature_cols.clone())),
        B::Duchon {
            feature_cols, spec, ..
        } => Some((&mut spec.center_strategy, feature_cols.clone())),
        B::ConstantCurvature { feature_cols, spec } => {
            Some((&mut spec.center_strategy, feature_cols.clone()))
        }
        B::MeasureJet {
            feature_cols, spec, ..
        } => Some((&mut spec.center_strategy, feature_cols.clone())),
        B::Sphere { feature_cols, spec } if matches!(spec.method, SphereMethod::Wahba) => {
            Some((&mut spec.center_strategy, feature_cols.clone()))
        }
        _ => None,
    }
}

/// The current resolution of `basis` when nobody chose it, else `None`.
///
/// Radial and Wahba-sphere smooths are recognised through the row-gating
/// wrappers (`by=`, factor sum-to-zero), which do not change what their
/// centers resolve. Every other family is adaptive only ungated: a row-gated
/// smooth is supported by its gate's rows alone, so the covariate's distinct
/// values do not bound its basis, and it keeps its starting resolution.
pub fn adaptive_resolution_of(basis: &SmoothBasisSpec) -> Option<AdaptiveResolution> {
    use SmoothBasisSpec as B;
    if let Some((strategy, cols)) = radial_center_strategy(basis) {
        return Some(AdaptiveResolution::Centers(
            strategy.planned_num_centers(cols.len()),
        ));
    }
    match basis {
        B::BSpline1D { spec, .. } => match (&spec.knotspec, &spec.boundary) {
            (
                BSplineKnotSpec::Automatic {
                    num_internal_knots: knots,
                    adaptive: true,
                    ..
                },
                OneDimensionalBoundary::Open,
            ) => Some(AdaptiveResolution::InternalKnots(*knots)),
            (
                BSplineKnotSpec::PeriodicUniform {
                    num_basis,
                    adaptive: true,
                    ..
                },
                OneDimensionalBoundary::Cyclic { .. },
            ) => Some(AdaptiveResolution::PeriodicBasis(*num_basis)),
            _ => None,
        },
        B::FactorSmooth { spec }
            if spec.adaptive
                && spec.continuous_cols.len() == 1
                && !matches!(spec.flavour, FactorSmoothFlavour::Re) =>
        {
            match &spec.marginal.knotspec {
                BSplineKnotSpec::Generate {
                    num_internal_knots, ..
                }
                | BSplineKnotSpec::Automatic {
                    num_internal_knots, ..
                } => Some(AdaptiveResolution::InternalKnots(*num_internal_knots)),
                _ => None,
            }
        }
        B::Sphere { spec, .. }
            if spec.adaptive_degree && matches!(spec.method, SphereMethod::Harmonic) =>
        {
            spec.max_degree.map(AdaptiveResolution::HarmonicDegree)
        }
        _ => None,
    }
}

/// The rows of `data` the basis of `basis` is identified on: a factor-by
/// level's block sees only the rows of its level, a factor-by smooth that
/// shares one spec across its levels is identified on its smallest level, and
/// every other smooth sees every row. This is the `n` a loop's starting
/// resolution is sized from, so a level of a `by=` factor is not handed a
/// basis its own rows cannot support (#1561: sized from the pooled rows,
/// `s(x, bs='tp', by=group)` at 100 rows a level got an ill-conditioned block
/// no λ could recover; #3179).
pub fn smooth_identification_rows(basis: &SmoothBasisSpec, data: ArrayView2<'_, f64>) -> usize {
    let level_rows = |by_col: usize| -> Vec<(u64, usize)> {
        let mut counts: Vec<(u64, usize)> = Vec::new();
        if let Some(values) = column(data, by_col) {
            for &value in values.iter().filter(|value| value.is_finite()) {
                let bits = gam_data::canonical_level_bits(value);
                match counts.iter_mut().find(|(level, _)| *level == bits) {
                    Some((_, count)) => *count += 1,
                    None => counts.push((bits, 1)),
                }
            }
        }
        counts
    };
    match basis {
        SmoothBasisSpec::ByVariable {
            by_col,
            by: ByVariableSpec::Level { value_bits, .. },
            ..
        } => {
            let level = gam_data::canonical_level_bits(f64::from_bits(*value_bits));
            level_rows(*by_col)
                .into_iter()
                .find(|(bits, _)| *bits == level)
                .map_or(0, |(_, count)| count)
        }
        SmoothBasisSpec::BySmooth {
            by_kind: ByVarKind::Factor { feature_col, .. },
            ..
        } => level_rows(*feature_col)
            .into_iter()
            .map(|(_, count)| count)
            .min()
            .unwrap_or(0),
        _ => data.nrows(),
    }
}

/// The radial basis inside the row-gating wrappers (`by=`, factor
/// sum-to-zero), which do not change what its centers resolve.
fn radial_inner(basis: &SmoothBasisSpec) -> &SmoothBasisSpec {
    use SmoothBasisSpec as B;
    match basis {
        B::ByVariable { inner, .. } | B::FactorSumToZero { inner, .. } => radial_inner(inner),
        B::BySmooth { smooth, .. } => radial_inner(smooth),
        other => other,
    }
}

/// The resolution the standard formula workflow starts a basis it grows at:
/// the penalized-resolution pilot, the smallest basis whose penalized span
/// holds every direction an optimally smoothed fit on the basis's rows keeps
/// ([`smooth_identification_rows`]), held to what the data can identify
/// ([`adaptive_resolution_support`]). `None` for a basis the loop does not
/// grow ([`adaptive_resolution_of`]), which keeps its provisioned formula
/// default: the pilot is a start the loop refines, not a final basis (#3149).
///
/// * open B-spline: the order-`m` null space plus `⌈n^{1/(2m+1)}⌉` directions;
/// * cyclic basis: the constant plus the same rank, at least `degree + 1`;
/// * factor-smooth marginal: the least-populated group's pilot, held below the
///   smallest group's covariate resolution;
/// * radial centers: the polynomial null space plus the penalized resolution
///   rank of the minimal embedding order (a 1-D Duchon also at least the pilot
///   `s(x)`, #1867); a 1-D thin-plate, curvature or measure-jet smooth starts
///   at its formula default;
/// * Wahba sphere centers: the constant plus the order-`m` rank on the 2-D
///   sphere;
/// * harmonic degree: the least degree whose span holds the order-`m` rank.
pub fn starting_resolution(
    basis: &SmoothBasisSpec,
    data: ArrayView2<'_, f64>,
) -> Option<AdaptiveResolution> {
    use SmoothBasisSpec as B;
    let current = adaptive_resolution_of(basis)?;
    let n = smooth_identification_rows(basis, data);
    let start = match (&current, basis) {
        (AdaptiveResolution::Centers(planned), _) => {
            let centers = match radial_inner(basis) {
                B::Duchon {
                    feature_cols, spec, ..
                } => {
                    let d = feature_cols.len();
                    let polynomial_cols = match spec.nullspace_order {
                        DuchonNullspaceOrder::Zero => 1,
                        DuchonNullspaceOrder::Linear => d + 1,
                        DuchonNullspaceOrder::Degree(degree) => {
                            duchon_nullspace_dimension(d, degree)
                        }
                    };
                    let univariate_floor = match feature_cols.as_slice() {
                        [col] => pilot_univariate_spline_basis_dim(column(data, *col)?, n),
                        _ => 0,
                    };
                    pilot_duchon_center_count(n, d, polynomial_cols, univariate_floor)
                }
                B::ThinPlate { feature_cols, .. } if feature_cols.len() > 1 => {
                    let d = feature_cols.len();
                    starting_num_centers(n, d, thin_plate_polynomial_basis_dimension(d))
                }
                B::ConstantCurvature { feature_cols, .. } if feature_cols.len() > 1 => {
                    starting_num_centers(n, feature_cols.len(), 1)
                }
                B::MeasureJet { feature_cols, .. } if feature_cols.len() > 1 => {
                    starting_num_centers(n, feature_cols.len(), 1)
                }
                B::Sphere { spec, .. } => 1usize
                    .saturating_add(penalized_resolution_rank(n, 2, spec.penalty_order.max(1)))
                    .min(n)
                    .max(1),
                _ => *planned,
            };
            AdaptiveResolution::Centers(centers)
        }
        (AdaptiveResolution::InternalKnots(_), B::BSpline1D { spec, .. }) => {
            AdaptiveResolution::InternalKnots(pilot_internal_knots(
                n,
                spec.degree,
                spec.penalty_order.min(spec.degree).max(1),
            ))
        }
        (AdaptiveResolution::InternalKnots(_), B::FactorSmooth { spec }) => {
            AdaptiveResolution::InternalKnots(factor_smooth_pilot_internal_knots(
                column(data, spec.continuous_cols[0])?,
                column(data, spec.group_col)?,
                spec.marginal.degree,
                spec.marginal.penalty_order,
            ))
        }
        (AdaptiveResolution::PeriodicBasis(_), B::BSpline1D { spec, .. }) => {
            AdaptiveResolution::PeriodicBasis(pilot_cyclic_basis_dim(
                n,
                spec.degree,
                spec.penalty_order.min(spec.degree).max(1),
            ))
        }
        (AdaptiveResolution::HarmonicDegree(_), B::Sphere { spec, .. }) => {
            AdaptiveResolution::HarmonicDegree(default_spherical_harmonic_degree(
                n,
                spec.penalty_order,
            ))
        }
        _ => return None,
    };
    let support = adaptive_resolution_support(basis, data)?;
    Some(start.with_value(start.value().min(support.value())))
}

/// One level of uniform nested refinement of `current` for `basis`: every
/// knot interval split once, every center cell given one new center, the
/// harmonic span paired with one new direction per existing one.
pub fn refined_adaptive_resolution(current: &AdaptiveResolution) -> AdaptiveResolution {
    match current {
        AdaptiveResolution::Centers(c) => AdaptiveResolution::Centers(refined_num_centers(*c)),
        AdaptiveResolution::InternalKnots(k) => {
            AdaptiveResolution::InternalKnots(refined_internal_knots(*k))
        }
        AdaptiveResolution::PeriodicBasis(b) => {
            AdaptiveResolution::PeriodicBasis(refined_periodic_basis(*b))
        }
        AdaptiveResolution::HarmonicDegree(l) => {
            AdaptiveResolution::HarmonicDegree(refined_harmonic_degree(*l))
        }
    }
}

fn distinct_finite(column: ArrayView1<'_, f64>) -> usize {
    let mut values: Vec<f64> = column.iter().copied().filter(|v| v.is_finite()).collect();
    values.sort_by(f64::total_cmp);
    values.dedup();
    values.len()
}

fn column(data: ArrayView2<'_, f64>, col: usize) -> Option<ArrayView1<'_, f64>> {
    (col < data.ncols()).then(move || data.index_axis_move(Axis(1), col))
}

/// The largest resolution the data can identify for `basis`: a function
/// observed at `u` distinct covariate values has at most `u` identifiable
/// values, so each family's interpolating limit is its support bound.
///
/// * open B-spline (and the shared factor-smooth marginal): `K + d + 1 ≤ u`;
/// * cyclic basis: dimension `≤ u`;
/// * radial centers: at most one per distinct coordinate row;
/// * harmonic degree: the non-constant span `L(L + 2)` leaves the constant
///   direction to the intercept, `L(L + 2) ≤ u − 1`, within the engine's
///   largest supported degree.
///
/// `None` when `basis` is not adaptive or references a column `data` lacks.
pub fn adaptive_resolution_support(
    basis: &SmoothBasisSpec,
    data: ArrayView2<'_, f64>,
) -> Option<AdaptiveResolution> {
    use SmoothBasisSpec as B;
    let current = adaptive_resolution_of(basis)?;
    if let Some((_, cols)) = radial_center_strategy(basis) {
        if cols.iter().any(|&c| c >= data.ncols()) {
            return None;
        }
        return Some(AdaptiveResolution::Centers(count_unique_coordinate_rows(
            data, cols,
        )));
    }
    match (basis, &current) {
        (B::BSpline1D { feature_col, spec }, AdaptiveResolution::InternalKnots(_)) => {
            let u = distinct_finite(column(data, *feature_col)?);
            Some(AdaptiveResolution::InternalKnots(
                u.saturating_sub(spec.degree + 1),
            ))
        }
        (B::BSpline1D { feature_col, .. }, AdaptiveResolution::PeriodicBasis(_)) => Some(
            AdaptiveResolution::PeriodicBasis(distinct_finite(column(data, *feature_col)?)),
        ),
        (B::FactorSmooth { spec }, AdaptiveResolution::InternalKnots(_)) => {
            // The groups share one marginal basis, so its identifiable span is
            // the pooled covariate's; each group's curve is penalized toward
            // the shared fit, never identified by its own rows alone.
            let u = distinct_finite(column(data, spec.continuous_cols[0])?);
            Some(AdaptiveResolution::InternalKnots(
                u.saturating_sub(spec.marginal.degree + 1),
            ))
        }
        (B::Sphere { feature_cols, .. }, AdaptiveResolution::HarmonicDegree(_)) => {
            if feature_cols.iter().any(|&c| c >= data.ncols()) {
                return None;
            }
            let directions = count_unique_coordinate_rows(data, feature_cols).saturating_sub(1);
            let degree = (1..=SPHERICAL_HARMONIC_MAX_DEGREE)
                .take_while(|&l| l * (l + 2) <= directions)
                .last()
                .unwrap_or(0);
            Some(AdaptiveResolution::HarmonicDegree(degree))
        }
        _ => None,
    }
}

/// Raw coefficient width `basis` realizes at `resolution`, before any
/// identifiability constraint. Only differences between two resolutions of
/// the same term are meaningful: they are the coefficients a refinement adds.
pub fn adaptive_resolution_width(
    basis: &SmoothBasisSpec,
    data: ArrayView2<'_, f64>,
    resolution: &AdaptiveResolution,
) -> usize {
    use SmoothBasisSpec as B;
    match (basis, resolution) {
        (_, AdaptiveResolution::Centers(c)) => *c,
        (_, AdaptiveResolution::PeriodicBasis(b)) => *b,
        (_, AdaptiveResolution::HarmonicDegree(l)) => l.saturating_mul(l + 2),
        (B::FactorSmooth { spec }, AdaptiveResolution::InternalKnots(k)) => {
            let levels = match &spec.group_frozen_levels {
                Some(levels) => levels.len(),
                None => column(data, spec.group_col).map_or(1, distinct_finite),
            };
            (k + spec.marginal.degree + 1).saturating_mul(levels.max(1))
        }
        (B::BSpline1D { spec, .. }, AdaptiveResolution::InternalKnots(k)) => k + spec.degree + 1,
        (_, AdaptiveResolution::InternalKnots(k)) => *k,
    }
}

/// Write `resolution` into `basis`, keeping its adaptive provenance so every
/// refit stays owned by the same loop.
pub fn apply_adaptive_resolution(
    basis: &mut SmoothBasisSpec,
    resolution: &AdaptiveResolution,
) -> Result<(), String> {
    use SmoothBasisSpec as B;
    let current = adaptive_resolution_of(basis).ok_or_else(|| {
        "adaptive resolution requested for a basis nobody left unsized".to_string()
    })?;
    if !current.same_kind(resolution) {
        return Err(format!(
            "adaptive resolution {resolution:?} does not match the basis resolution {current:?}"
        ));
    }
    if let AdaptiveResolution::Centers(centers) = resolution {
        let (strategy, cols) = radial_center_strategy_mut(basis)
            .ok_or_else(|| "adaptive center count requested for a non-radial basis".to_string())?;
        *strategy = center_strategy_with_num_centers(strategy, *centers, cols.len())
            .map_err(|error| error.to_string())?;
        return Ok(());
    }
    let unsupported = |basis: &str| {
        format!("adaptive resolution {resolution:?} cannot be written into a {basis} basis")
    };
    match (basis, resolution) {
        (B::BSpline1D { spec, .. }, AdaptiveResolution::InternalKnots(k)) => {
            match &mut spec.knotspec {
                BSplineKnotSpec::Automatic {
                    num_internal_knots, ..
                } => *num_internal_knots = *k,
                other => return Err(unsupported(&format!("B-spline {other:?}"))),
            }
        }
        (B::BSpline1D { spec, .. }, AdaptiveResolution::PeriodicBasis(b)) => {
            match &mut spec.knotspec {
                BSplineKnotSpec::PeriodicUniform { num_basis, .. } => *num_basis = *b,
                other => return Err(unsupported(&format!("B-spline {other:?}"))),
            }
        }
        (B::FactorSmooth { spec }, AdaptiveResolution::InternalKnots(k)) => {
            match &mut spec.marginal.knotspec {
                BSplineKnotSpec::Generate {
                    num_internal_knots, ..
                } => *num_internal_knots = *k,
                BSplineKnotSpec::Automatic {
                    num_internal_knots, ..
                } => *num_internal_knots = *k,
                other => return Err(unsupported(&format!("factor-smooth {other:?}"))),
            }
        }
        (B::Sphere { spec, .. }, AdaptiveResolution::HarmonicDegree(l)) => {
            spec.max_degree = Some(*l);
        }
        (other, _) => return Err(unsupported(&format!("{other:?}"))),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::AdaptiveResolution as R;

    #[test]
    fn path_reaches_the_target_and_never_retreats() {
        let from = R::InternalKnots(5);
        let to = R::InternalKnots(9);
        assert_eq!(from.steps_to(&to), 4);
        assert_eq!(from.toward(&to, 0), from);
        assert_eq!(from.toward(&to, 1), R::InternalKnots(6));
        assert_eq!(from.toward(&to, 7), to);
        assert_eq!(to.steps_to(&from), 0);
    }

    #[test]
    fn clamp_bounds_by_support_but_never_shrinks() {
        let current = R::Centers(5);
        assert_eq!(
            R::Centers(9).clamped(&R::Centers(7), &current),
            R::Centers(7)
        );
        assert_eq!(R::Centers(9).clamped(&R::Centers(3), &current), current);
        assert!(!R::Centers(4).exceeds(&R::Centers(4)));
        assert!(R::Centers(5).exceeds(&R::Centers(4)));
        assert!(!R::Centers(5).exceeds(&R::InternalKnots(4)));
    }
}
