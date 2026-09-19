//! The resolution of a formula-default smooth basis that nobody chose.
//!
//! A formula default (`s(x)`, `s(x, bs="cc")`, `fs(x, g)`, `te(x, z)`, an
//! auto-sized radial or sphere smooth) starts at the penalized-resolution
//! pilot and is refined by the standard formula workflow from the converged
//! fit's own evidence (#1689, #3078). This module is the one place that knows,
//! per basis family, what that resolution is, how one level of nested
//! refinement changes it, how far the data can identify it, how many raw
//! coefficients it realizes, and how to write it back into the spec. An
//! explicit user size carries no adaptive provenance and is never touched.

use ndarray::{ArrayView1, ArrayView2, Axis};
use serde::{Deserialize, Serialize};

use super::{FactorSmoothFlavour, SmoothBasisSpec};
use crate::basis::{
    BSplineKnotSpec, CenterStrategy, OneDimensionalBoundary, SPHERICAL_HARMONIC_MAX_DEGREE,
    SphereMethod, center_strategy_is_auto, center_strategy_with_num_centers,
    count_unique_coordinate_rows, refined_harmonic_degree, refined_internal_knots,
    refined_num_centers, refined_periodic_basis, select_cr_knots,
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
    /// Per-margin basis dimensions of a formula-default tensor product, in
    /// the tensor's (canonical) margin order.
    MarginDims(Vec<usize>),
    /// Maximum degree of a formula-default spherical-harmonic basis.
    HarmonicDegree(usize),
}

impl AdaptiveResolution {
    fn components(&self) -> Vec<usize> {
        match self {
            Self::Centers(v)
            | Self::InternalKnots(v)
            | Self::PeriodicBasis(v)
            | Self::HarmonicDegree(v) => vec![*v],
            Self::MarginDims(dims) => dims.clone(),
        }
    }

    fn with_components(&self, values: Vec<usize>) -> Self {
        let scalar = || values.first().copied().unwrap_or(0);
        match self {
            Self::Centers(_) => Self::Centers(scalar()),
            Self::InternalKnots(_) => Self::InternalKnots(scalar()),
            Self::PeriodicBasis(_) => Self::PeriodicBasis(scalar()),
            Self::HarmonicDegree(_) => Self::HarmonicDegree(scalar()),
            Self::MarginDims(_) => Self::MarginDims(values),
        }
    }

    fn same_shape(&self, other: &Self) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
            && self.components().len() == other.components().len()
    }

    /// `self` bounded above by `bound` component-wise, but never below
    /// `floor`: a proposal is limited by what the data can identify, and a
    /// basis that already converged is never shrunk. A `bound` of another
    /// shape leaves `self` unchanged.
    pub fn clamped(&self, bound: &Self, floor: &Self) -> Self {
        if !self.same_shape(bound) || !self.same_shape(floor) {
            return self.clone();
        }
        let values = self
            .components()
            .into_iter()
            .zip(bound.components())
            .zip(floor.components())
            .map(|((value, bound), floor)| value.min(bound).max(floor))
            .collect();
        self.with_components(values)
    }

    /// The number of monotone steps between `self` and `target`: the largest
    /// component increase. Zero when `target` does not exceed `self`.
    pub fn steps_to(&self, target: &Self) -> usize {
        if !self.same_shape(target) {
            return 0;
        }
        self.components()
            .into_iter()
            .zip(target.components())
            .map(|(from, to)| to.saturating_sub(from))
            .max()
            .unwrap_or(0)
    }

    /// The point `step` of [`Self::steps_to`]`(target)` on the monotone path
    /// from `self` to `target`: every component advances by
    /// `⌈step · (to − from) / steps⌉`, so step 0 is `self`, the last step is
    /// `target`, and each component is non-decreasing along the path.
    pub fn toward(&self, target: &Self, step: usize) -> Self {
        let steps = self.steps_to(target);
        if steps == 0 {
            return self.clone();
        }
        let step = step.min(steps);
        let values = self
            .components()
            .into_iter()
            .zip(target.components())
            .map(|(from, to)| {
                let gap = to.saturating_sub(from);
                from + (step * gap).div_ceil(steps)
            })
            .collect();
        self.with_components(values)
    }

    /// Whether any component of `self` exceeds the matching one of `other`.
    pub fn exceeds(&self, other: &Self) -> bool {
        self.same_shape(other)
            && self
                .components()
                .into_iter()
                .zip(other.components())
                .any(|(a, b)| a > b)
    }
}

impl std::fmt::Display for AdaptiveResolution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Centers(c) => write!(f, "{c} centers"),
            Self::InternalKnots(k) => write!(f, "{k} internal knots"),
            Self::PeriodicBasis(b) => write!(f, "cyclic basis dimension {b}"),
            Self::MarginDims(dims) => {
                let dims: Vec<String> = dims.iter().map(usize::to_string).collect();
                write!(f, "tensor margin dimensions {}", dims.join("x"))
            }
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
        B::ConstantCurvature { feature_cols, spec } => Some((&spec.center_strategy, feature_cols.as_slice())),
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

/// Basis dimension of one tensor margin, and the knots of an NCR margin.
fn margin_dim(margin: &crate::basis::BSplineBasisSpec) -> Option<usize> {
    match &margin.knotspec {
        BSplineKnotSpec::NaturalCubicRegression { knots } => Some(knots.len()),
        BSplineKnotSpec::Generate {
            num_internal_knots, ..
        } => Some(num_internal_knots + margin.degree + 1),
        BSplineKnotSpec::PeriodicUniform { num_basis, .. } => Some(*num_basis),
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
                    num_internal_knots: Some(knots),
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
                    num_internal_knots: Some(num_internal_knots),
                    ..
                } => Some(AdaptiveResolution::InternalKnots(*num_internal_knots)),
                _ => None,
            }
        }
        B::TensorBSpline { feature_cols, spec }
            if spec.adaptive && spec.marginalspecs.len() == feature_cols.len() =>
        {
            spec.marginalspecs
                .iter()
                .map(margin_dim)
                .collect::<Option<Vec<_>>>()
                .map(AdaptiveResolution::MarginDims)
        }
        B::Sphere { spec, .. }
            if spec.adaptive_degree && matches!(spec.method, SphereMethod::Harmonic) =>
        {
            spec.max_degree.map(AdaptiveResolution::HarmonicDegree)
        }
        _ => None,
    }
}

/// One level of uniform nested refinement of `current` for `basis`: every
/// knot interval split once, every center cell given one new center, the
/// harmonic span paired with one new direction per existing one.
pub fn refined_adaptive_resolution(
    basis: &SmoothBasisSpec,
    current: &AdaptiveResolution,
) -> AdaptiveResolution {
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
        AdaptiveResolution::MarginDims(dims) => {
            let SmoothBasisSpec::TensorBSpline { spec, .. } = basis else {
                return current.clone();
            };
            AdaptiveResolution::MarginDims(
                dims.iter()
                    .zip(&spec.marginalspecs)
                    .map(|(&dim, margin)| match &margin.knotspec {
                        // `k` value-knots bound `k − 1` intervals; splitting
                        // each once gives `2k − 1` knots.
                        BSplineKnotSpec::NaturalCubicRegression { .. } => {
                            dim.saturating_mul(2).saturating_sub(1)
                        }
                        BSplineKnotSpec::PeriodicUniform { .. } => refined_periodic_basis(dim),
                        _ => {
                            let order = margin.degree + 1;
                            refined_internal_knots(dim.saturating_sub(order)) + order
                        }
                    })
                    .collect(),
            )
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
/// * cyclic basis and every tensor margin: dimension `≤ u` on its axis;
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
        (B::TensorBSpline { feature_cols, .. }, AdaptiveResolution::MarginDims(_)) => feature_cols
            .iter()
            .map(|&c| column(data, c).map(distinct_finite))
            .collect::<Option<Vec<_>>>()
            .map(AdaptiveResolution::MarginDims),
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
        (_, AdaptiveResolution::MarginDims(dims)) => {
            dims.iter().fold(1usize, |acc, &d| acc.saturating_mul(d))
        }
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
    data: ArrayView2<'_, f64>,
    resolution: &AdaptiveResolution,
) -> Result<(), String> {
    use SmoothBasisSpec as B;
    let current = adaptive_resolution_of(basis)
        .ok_or_else(|| "adaptive resolution requested for a basis nobody left unsized".to_string())?;
    if !current.same_shape(resolution) {
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
        (B::BSpline1D { spec, .. }, AdaptiveResolution::InternalKnots(k)) => match &mut spec.knotspec {
            BSplineKnotSpec::Automatic {
                num_internal_knots, ..
            } => *num_internal_knots = Some(*k),
            other => return Err(unsupported(&format!("B-spline {other:?}"))),
        },
        (B::BSpline1D { spec, .. }, AdaptiveResolution::PeriodicBasis(b)) => match &mut spec.knotspec {
            BSplineKnotSpec::PeriodicUniform { num_basis, .. } => *num_basis = *b,
            other => return Err(unsupported(&format!("B-spline {other:?}"))),
        },
        (B::FactorSmooth { spec }, AdaptiveResolution::InternalKnots(k)) => {
            match &mut spec.marginal.knotspec {
                BSplineKnotSpec::Generate {
                    num_internal_knots, ..
                } => *num_internal_knots = *k,
                BSplineKnotSpec::Automatic {
                    num_internal_knots, ..
                } => *num_internal_knots = Some(*k),
                other => return Err(unsupported(&format!("factor-smooth {other:?}"))),
            }
        }
        (B::TensorBSpline { feature_cols, spec }, AdaptiveResolution::MarginDims(dims)) => {
            for ((margin, &dim), &col) in spec.marginalspecs.iter_mut().zip(dims).zip(feature_cols.iter()) {
                match &mut margin.knotspec {
                    BSplineKnotSpec::NaturalCubicRegression { knots } => {
                        let values = column(data, col).ok_or_else(|| {
                            format!("tensor margin column {col} is missing from the data")
                        })?;
                        *knots = select_cr_knots(values, dim).map_err(|e| e.to_string())?;
                    }
                    BSplineKnotSpec::Generate {
                        num_internal_knots, ..
                    } => {
                        *num_internal_knots = dim.checked_sub(margin.degree + 1).ok_or_else(|| {
                            format!(
                                "tensor margin dimension {dim} is below its degree-{} order",
                                margin.degree
                            )
                        })?;
                    }
                    BSplineKnotSpec::PeriodicUniform { num_basis, .. } => *num_basis = dim,
                    other => return Err(unsupported(&format!("tensor margin {other:?}"))),
                }
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
    fn monotone_path_reaches_the_target_and_never_retreats() {
        let from = R::MarginDims(vec![5, 5]);
        let to = R::MarginDims(vec![9, 7]);
        assert_eq!(from.steps_to(&to), 4);
        assert_eq!(from.toward(&to, 0), from);
        assert_eq!(from.toward(&to, 1), R::MarginDims(vec![6, 6]));
        assert_eq!(from.toward(&to, 4), to);
    }

    #[test]
    fn clamp_bounds_by_support_but_never_shrinks() {
        let proposal = R::MarginDims(vec![9, 9]);
        let support = R::MarginDims(vec![3, 20]);
        let current = R::MarginDims(vec![5, 5]);
        assert_eq!(proposal.clamped(&support, &current), R::MarginDims(vec![5, 9]));
        assert!(!R::Centers(4).exceeds(&R::Centers(4)));
        assert!(R::Centers(5).exceeds(&R::Centers(4)));
    }
}
