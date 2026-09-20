//! The resolution of a formula-default smooth basis that nobody chose.
//!
//! A formula default (`s(x)`, `s(x, bs="cyclic")`, `fs(x, g)`, an auto-sized
//! radial or sphere smooth) starts at the penalized-resolution
//! pilot and is refined by the standard formula workflow from the converged
//! fit's own evidence (#1689, #3078). This module is the one place that knows,
//! per basis family, what that resolution is, how one level of nested
//! refinement changes it, how far the data can identify it, how many raw
//! coefficients it realizes, and how to write it back into the spec. An
//! explicit user size carries no adaptive provenance and is never touched.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use serde::{Deserialize, Serialize};

use super::{FactorSmoothFlavour, SmoothBasisSpec};
use crate::basis::{
    BSplineKnotSpec, CenterStrategy, OneDimensionalBoundary, SPHERICAL_HARMONIC_MAX_DEGREE,
    SphereMethod, center_strategy_is_auto, center_strategy_spectral_basis,
    center_strategy_with_num_centers, count_unique_coordinate_rows, realized_center_strategy,
    refined_harmonic_degree, refined_internal_knots, refined_num_centers, refined_periodic_basis,
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
    radial_center_strategy_any(basis)
        .filter(|(strategy, cols)| !cols.is_empty() && center_strategy_is_auto(strategy))
}

/// The center strategy of a radial basis whether or not anybody sized it: a
/// frozen (fitted) spec carries `UserProvided` centers, which the nesting
/// check reads.
fn radial_center_strategy_any(basis: &SmoothBasisSpec) -> Option<(&CenterStrategy, &[usize])> {
    use SmoothBasisSpec as B;
    match basis {
        B::ByVariable { inner, .. } | B::FactorSumToZero { inner, .. } => {
            radial_center_strategy_any(inner)
        }
        B::BySmooth { smooth, .. } => radial_center_strategy_any(smooth),
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

/// Whether one level of [`refined_adaptive_resolution`] of `basis` realizes a
/// basis whose span CONTAINS the current one (#3331).
///
/// The growth loop compares two fits' REML/LAML evidence. That comparison is a
/// Bayes factor between nested smoothing priors only when the coarse basis is a
/// subspace of the fine one with the same unpenalized null space and penalty
/// order; between non-nested bases the difference in evidence is the
/// difference of two unrelated prior normalizers and says nothing about
/// resolution. So a term is eligible only when its refinement is nested by
/// construction:
///
/// * open B-spline, factor-smooth marginal: `K → 2K + 1` internal knots keeps
///   every old knot (quantile placement `j/(K+1)` and uniform spacing `h/2`
///   are both exact in floating point at the refined level);
/// * cyclic basis: `b → 2b` on the same period halves the uniform spacing;
/// * harmonic degree: the span of degrees `≤ L` is structural;
/// * radial centers: only farthest-point centers without a learned spectral
///   subspace, whose greedy maximin order is (up to capped tie orbits, which
///   [`realized_basis_nests`] rejects after the fact) a prefix of the refined
///   selection. Equal-mass, k-means, grid and spectral plans re-place every
///   center and never nest.
pub fn adaptive_refinement_can_nest(basis: &SmoothBasisSpec) -> bool {
    match adaptive_resolution_of(basis) {
        Some(AdaptiveResolution::Centers(_)) => radial_center_strategy(basis).is_some_and(
            |(strategy, _)| {
                center_strategy_spectral_basis(strategy).is_none()
                    && matches!(
                        realized_center_strategy(strategy),
                        CenterStrategy::FarthestPoint { .. }
                    )
            },
        ),
        Some(_) => true,
        None => false,
    }
}

/// `a ⊆ b` as multisets of sorted knot vectors, by exact equality.
fn knots_nest(coarse: &Array1<f64>, fine: &Array1<f64>) -> bool {
    let mut fine_iter = fine.iter().copied();
    'coarse: for &c in coarse.iter() {
        for f in fine_iter.by_ref() {
            if f == c {
                continue 'coarse;
            }
            if f > c {
                return false;
            }
        }
        return false;
    }
    true
}

fn knotspec_nests(coarse: &BSplineKnotSpec, fine: &BSplineKnotSpec) -> bool {
    match (coarse, fine) {
        (BSplineKnotSpec::Provided(a), BSplineKnotSpec::Provided(b)) => knots_nest(a, b),
        (
            BSplineKnotSpec::PeriodicUniform {
                data_range: range_a,
                num_basis: b1,
                ..
            },
            BSplineKnotSpec::PeriodicUniform {
                data_range: range_b,
                num_basis: b2,
                ..
            },
        ) => range_a == range_b && *b1 > 0 && b2 % b1 == 0,
        _ => false,
    }
}

fn bspline_nests(coarse: &crate::basis::BSplineBasisSpec, fine: &crate::basis::BSplineBasisSpec) -> bool {
    coarse.degree == fine.degree
        && coarse.penalty_order == fine.penalty_order
        && coarse.double_penalty == fine.double_penalty
        && std::mem::discriminant(&coarse.boundary) == std::mem::discriminant(&fine.boundary)
        && knotspec_nests(&coarse.knotspec, &fine.knotspec)
}

fn rows_nest(coarse: &Array2<f64>, fine: &Array2<f64>) -> bool {
    coarse.ncols() == fine.ncols()
        && coarse
            .rows()
            .into_iter()
            .all(|row| fine.rows().into_iter().any(|other| other == row))
}

/// Whether the FROZEN (fitted) spec `fine` realizes a basis whose span contains
/// the frozen spec `coarse`'s, with the same degree, penalty order and null
/// space (#3331). Both arguments come from
/// [`super::freeze_term_collection_from_design`], so they carry the knots and
/// centers the fits actually used, not the requests that produced them. The
/// check is exact: a refinement whose realized knots or centers drifted is not
/// nested and its evidence is never compared against the coarse fit's.
pub fn realized_basis_nests(coarse: &SmoothBasisSpec, fine: &SmoothBasisSpec) -> bool {
    use SmoothBasisSpec as B;
    match (coarse, fine) {
        (B::ByVariable { inner: a, .. }, B::ByVariable { inner: b, .. })
        | (B::FactorSumToZero { inner: a, .. }, B::FactorSumToZero { inner: b, .. }) => {
            realized_basis_nests(a, b)
        }
        (B::BySmooth { smooth: a, .. }, B::BySmooth { smooth: b, .. }) => {
            realized_basis_nests(a, b)
        }
        (
            B::BSpline1D {
                feature_col: col_a,
                spec: a,
            },
            B::BSpline1D {
                feature_col: col_b,
                spec: b,
            },
        ) => col_a == col_b && bspline_nests(a, b),
        (B::FactorSmooth { spec: a }, B::FactorSmooth { spec: b }) => {
            a.continuous_cols == b.continuous_cols
                && a.group_col == b.group_col
                && std::mem::discriminant(&a.flavour) == std::mem::discriminant(&b.flavour)
                && bspline_nests(&a.marginal, &b.marginal)
        }
        (B::Sphere { spec: a, .. }, B::Sphere { spec: b, .. })
            if matches!(a.method, SphereMethod::Harmonic)
                && matches!(b.method, SphereMethod::Harmonic) =>
        {
            a.penalty_order == b.penalty_order
                && a.double_penalty == b.double_penalty
                && matches!((a.max_degree, b.max_degree), (Some(la), Some(lb)) if lb >= la)
        }
        _ => match (
            radial_center_strategy_any(coarse),
            radial_center_strategy_any(fine),
        ) {
            (
                Some((CenterStrategy::UserProvided(a), cols_a)),
                Some((CenterStrategy::UserProvided(b), cols_b)),
            ) => {
                std::mem::discriminant(coarse) == std::mem::discriminant(fine)
                    && cols_a == cols_b
                    && rows_nest(a, b)
            }
            _ => false,
        },
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
    use super::{knots_nest, knotspec_nests};
    use crate::basis::BSplineKnotSpec;
    use ndarray::Array1;

    /// Quantile knots at level `k` sit at `j/(k+1)`; the refined level
    /// `2k+1` places `2j/(2k+2)`, which rounds to the same float, so the
    /// realized coarse knot vector is a sub-multiset of the refined one.
    #[test]
    fn quantile_and_uniform_knot_chains_nest_exactly() {
        let lo = -1.3_f64;
        let hi = 2.7_f64;
        for k in [1usize, 3, 4, 7, 10, 19] {
            let fine_k = crate::basis::refined_internal_knots(k);
            let quantile = |m: usize| -> Array1<f64> {
                (1..=m).map(|j| j as f64 / (m + 1) as f64).collect()
            };
            assert!(knots_nest(&quantile(k), &quantile(fine_k)), "quantile k={k}");
            let uniform = |m: usize| -> Array1<f64> {
                let h = (hi - lo) / (m + 1) as f64;
                let mut v: Vec<f64> = vec![lo; 4];
                v.extend((1..=m).map(|j| lo + j as f64 * h));
                v.extend([hi; 4]);
                Array1::from(v)
            };
            assert!(knots_nest(&uniform(k), &uniform(fine_k)), "uniform k={k}");
            // A different, non-refined level does not nest.
            assert!(!knots_nest(&uniform(k + 1), &uniform(fine_k)), "k+1={}", k + 1);
        }
        // Boundary multiplicity counts: a coarse vector with more repeats
        // than the fine one is not contained.
        let a = Array1::from(vec![0.0, 0.0, 0.0, 1.0]);
        let b = Array1::from(vec![0.0, 0.0, 0.5, 1.0]);
        assert!(!knots_nest(&a, &b));
    }

    #[test]
    fn periodic_chain_nests_only_on_the_same_period() {
        let spec = |range: (f64, f64), b: usize| BSplineKnotSpec::PeriodicUniform {
            data_range: range,
            num_basis: b,
            adaptive: false,
        };
        assert!(knotspec_nests(&spec((0.0, 1.0), 6), &spec((0.0, 1.0), 12)));
        assert!(!knotspec_nests(&spec((0.0, 1.0), 6), &spec((0.0, 1.0), 9)));
        assert!(!knotspec_nests(&spec((0.0, 1.0), 6), &spec((0.0, 1.1), 12)));
    }

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
