//! First-class multi-chart manifold atoms (#1890).
//!
//! The numerical SAE representation stores one decoder block and one routing
//! column per chart.  A registered [`ManifoldChartAtlas`] quotients that storage
//! into one *semantic* atom without changing the represented function:
//!
//! ```text
//! sum_{c in A} a_c gamma_c
//!   = (sum_c a_c) sum_c [a_c / (sum_j a_j)] gamma_c.
//! ```
//!
//! The bracketed weights are a partition of unity on the atlas support and the
//! prefactor is the atlas activation.  Thus registration is image-exact, keeps
//! the independently valid local coordinates at a pole or half-twist, and does
//! not duplicate reconstruction logic.  Ordinary orientation-preserving
//! over-tiles can still be physically fused; seams that must survive are stored
//! here with their exact unit-speed affine transition.

use ndarray::{Array1, ArrayView1};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

use super::SaeManifoldTerm;

/// Geometric role of an atlas overlap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtlasSeamKind {
    /// An ordinary overlap between two regular local charts.
    Regular,
    /// A pole overlap: both charts are required because one coordinate frame is
    /// singular at the other's pole.  The transition remains an isometry on the
    /// regular overlap; the tag prevents a consumer from flattening the cover.
    Pole,
}

/// Exact affine isometry between two unit-speed one-dimensional charts.
///
/// The convention is `t_to = sign * t_from + offset (mod period)`.  Unit speed
/// makes the only possible linear part `sign in {+1,-1}`; no fitted slope or
/// numerical tolerance is stored.  `-1` records an orientation-reversing
/// overlap.  Non-orientability is a property of the transition *cocycle*, not a
/// single negative edge; see [`ManifoldChartAtlas::orientability`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnitSpeedChartTransition {
    pub from_chart: usize,
    pub to_chart: usize,
    pub sign: i8,
    pub offset: f64,
    pub period: f64,
    pub seam_kind: AtlasSeamKind,
}

impl UnitSpeedChartTransition {
    #[must_use = "transition validation errors must be handled"]
    pub fn new(
        from_chart: usize,
        to_chart: usize,
        sign: i8,
        offset: f64,
        period: f64,
        seam_kind: AtlasSeamKind,
    ) -> Result<Self, String> {
        if from_chart == to_chart {
            return Err("unit-speed chart transition cannot be a self-edge".to_string());
        }
        if !matches!(sign, -1 | 1) {
            return Err(format!(
                "unit-speed chart transition sign must be +1 or -1, got {sign}"
            ));
        }
        if !(period.is_finite() && period > 0.0) {
            return Err(format!(
                "unit-speed chart transition period must be finite and positive, got {period}"
            ));
        }
        if !offset.is_finite() {
            return Err(format!(
                "unit-speed chart transition offset must be finite, got {offset}"
            ));
        }
        Ok(Self {
            from_chart,
            to_chart,
            sign,
            offset: offset.rem_euclid(period),
            period,
            seam_kind,
        })
    }

    /// Apply the exact unit-speed transition.
    #[must_use]
    pub fn apply(&self, coordinate: f64) -> f64 {
        (self.sign as f64 * coordinate + self.offset).rem_euclid(self.period)
    }

    /// Exact inverse isometry, with the chart endpoints swapped.
    #[must_use]
    pub fn inverse(&self) -> Self {
        let sign = self.sign;
        let offset = (-(sign as f64) * self.offset).rem_euclid(self.period);
        Self {
            from_chart: self.to_chart,
            to_chart: self.from_chart,
            sign,
            offset,
            period: self.period,
            seam_kind: self.seam_kind,
        }
    }

    /// Compose `self: A -> B` followed by `next: B -> C`.
    #[must_use = "transition composition errors must be handled"]
    pub fn compose(&self, next: &Self) -> Result<Self, String> {
        if self.to_chart != next.from_chart {
            return Err(format!(
                "cannot compose chart transitions {}->{} and {}->{}",
                self.from_chart, self.to_chart, next.from_chart, next.to_chart
            ));
        }
        if self.period.to_bits() != next.period.to_bits() {
            return Err(format!(
                "cannot compose chart transitions with periods {} and {}",
                self.period, next.period
            ));
        }
        let sign = self.sign * next.sign;
        let offset = (next.sign as f64 * self.offset + next.offset).rem_euclid(self.period);
        Self::new(
            self.from_chart,
            next.to_chart,
            sign,
            offset,
            self.period,
            if matches!(self.seam_kind, AtlasSeamKind::Pole)
                || matches!(next.seam_kind, AtlasSeamKind::Pole)
            {
                AtlasSeamKind::Pole
            } else {
                AtlasSeamKind::Regular
            },
        )
    }

    pub(crate) fn remap(&mut self, old_to_new: &[Option<usize>]) -> Result<(), String> {
        self.from_chart = old_to_new
            .get(self.from_chart)
            .and_then(|x| *x)
            .ok_or_else(|| {
                "cannot remove an atlas chart while its seam is registered".to_string()
            })?;
        self.to_chart = old_to_new
            .get(self.to_chart)
            .and_then(|x| *x)
            .ok_or_else(|| {
                "cannot remove an atlas chart while its seam is registered".to_string()
            })?;
        Ok(())
    }
}

/// Orientability of the signed transition cocycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtlasOrientability {
    /// All transition signs admit a consistent choice of local chart
    /// orientations.  Negative tree edges are allowed: only cycle holonomy is
    /// invariant under changing a chart's local orientation.
    Orientable,
    /// Some cycle has negative sign holonomy (the Möbius obstruction).
    NonOrientable,
}

/// Several local charts representing one manifold atom.
#[derive(Clone, Debug, PartialEq)]
pub struct ManifoldChartAtlas {
    charts: Vec<usize>,
    transitions: Vec<UnitSpeedChartTransition>,
}

impl ManifoldChartAtlas {
    #[must_use = "atlas validation errors must be handled"]
    pub fn from_transition(transition: UnitSpeedChartTransition) -> Result<Self, String> {
        let charts = vec![
            transition.from_chart.min(transition.to_chart),
            transition.from_chart.max(transition.to_chart),
        ];
        let atlas = Self {
            charts,
            transitions: vec![transition],
        };
        atlas.validate()?;
        Ok(atlas)
    }

    #[must_use]
    pub fn charts(&self) -> &[usize] {
        &self.charts
    }

    #[must_use]
    pub fn transitions(&self) -> &[UnitSpeedChartTransition] {
        &self.transitions
    }

    #[must_use]
    pub fn contains_chart(&self, chart: usize) -> bool {
        self.charts.binary_search(&chart).is_ok()
    }

    pub(crate) fn add_transition(
        &mut self,
        transition: UnitSpeedChartTransition,
    ) -> Result<(), String> {
        self.charts.push(transition.from_chart);
        self.charts.push(transition.to_chart);
        self.charts.sort_unstable();
        self.charts.dedup();
        self.transitions.push(transition);
        self.canonicalize_transitions();
        self.validate()
    }

    pub(crate) fn replace_directed_transition(
        &mut self,
        transition: UnitSpeedChartTransition,
    ) -> Result<bool, String> {
        let Some(existing) = self.transitions.iter_mut().find(|existing| {
            existing.from_chart == transition.from_chart
                && existing.to_chart == transition.to_chart
                && existing.seam_kind == transition.seam_kind
        }) else {
            return Ok(false);
        };
        *existing = transition;
        self.canonicalize_transitions();
        self.validate()?;
        Ok(true)
    }

    pub(crate) fn merge_with_transition(
        &mut self,
        mut other: Self,
        transition: UnitSpeedChartTransition,
    ) -> Result<(), String> {
        self.charts.append(&mut other.charts);
        self.charts.sort_unstable();
        self.charts.dedup();
        self.transitions.append(&mut other.transitions);
        // The new edge is what connects the formerly separate atlas
        // components. Validate only after it is present; validating the plain
        // union first would (correctly but prematurely) reject it as
        // disconnected and make atlas growth impossible.
        self.transitions.push(transition);
        self.canonicalize_transitions();
        self.validate()
    }

    fn canonicalize_transitions(&mut self) {
        self.transitions.sort_by(|a, b| {
            a.from_chart
                .cmp(&b.from_chart)
                .then(a.to_chart.cmp(&b.to_chart))
                .then(a.sign.cmp(&b.sign))
                .then(a.offset.total_cmp(&b.offset))
                .then((a.seam_kind as u8).cmp(&(b.seam_kind as u8)))
        });
    }

    fn validate(&self) -> Result<(), String> {
        if self.charts.len() < 2 {
            return Err("a chart atlas requires at least two charts".to_string());
        }
        if self.charts.windows(2).any(|w| w[0] >= w[1]) {
            return Err("atlas chart indices must be strictly increasing".to_string());
        }
        if self.transitions.is_empty() {
            return Err("a chart atlas requires at least one transition".to_string());
        }
        for transition in &self.transitions {
            if !self.contains_chart(transition.from_chart)
                || !self.contains_chart(transition.to_chart)
            {
                return Err("atlas transition endpoint is not an atlas chart".to_string());
            }
        }
        // A semantic atom must be one connected cover, not a bag of unrelated
        // charts.  Connectivity is purely combinatorial and deterministic.
        let mut reached = BTreeSet::from([self.charts[0]]);
        let mut queue = VecDeque::from([self.charts[0]]);
        while let Some(chart) = queue.pop_front() {
            for transition in &self.transitions {
                let next = if transition.from_chart == chart {
                    Some(transition.to_chart)
                } else if transition.to_chart == chart {
                    Some(transition.from_chart)
                } else {
                    None
                };
                if let Some(next) = next {
                    if reached.insert(next) {
                        queue.push_back(next);
                    }
                }
            }
        }
        if reached.len() != self.charts.len() {
            return Err("atlas transition graph is disconnected".to_string());
        }
        Ok(())
    }

    /// Read orientability from the sign cocycle.  Local orientations are
    /// propagated across the transition graph; a contradictory revisit is
    /// exactly a negative-holonomy cycle and therefore the Möbius obstruction.
    #[must_use]
    pub fn orientability(&self) -> AtlasOrientability {
        let mut orientation = BTreeMap::new();
        let root = self.charts[0];
        orientation.insert(root, 1_i8);
        let mut queue = VecDeque::from([root]);
        while let Some(chart) = queue.pop_front() {
            let here = orientation[&chart];
            for transition in &self.transitions {
                let (next, sign) = if transition.from_chart == chart {
                    (transition.to_chart, transition.sign)
                } else if transition.to_chart == chart {
                    (transition.from_chart, transition.sign)
                } else {
                    continue;
                };
                let required = here * sign;
                match orientation.get(&next) {
                    Some(&existing) if existing != required => {
                        return AtlasOrientability::NonOrientable;
                    }
                    Some(_) => {}
                    None => {
                        orientation.insert(next, required);
                        queue.push_back(next);
                    }
                }
            }
        }
        AtlasOrientability::Orientable
    }

    /// Factor the chart gates on one row into `(atlas activation, partition of
    /// unity)`.  The returned weights follow [`Self::charts`] and sum exactly to
    /// one whenever the atlas is active.  This is an algebraic refactoring of
    /// the existing reconstruction, not an approximate router.
    #[must_use = "partition errors must be handled"]
    pub fn partition_of_unity(
        &self,
        row_assignments: ArrayView1<'_, f64>,
    ) -> Result<(f64, Array1<f64>), String> {
        if let Some(&bad) = self
            .charts
            .iter()
            .find(|&&chart| chart >= row_assignments.len())
        {
            return Err(format!(
                "atlas chart {bad} is outside assignment row width {}",
                row_assignments.len()
            ));
        }
        let activation: f64 = self
            .charts
            .iter()
            .map(|&chart| row_assignments[chart])
            .sum();
        if !(activation.is_finite() && activation >= 0.0) {
            return Err(format!(
                "atlas activation must be finite and nonnegative, got {activation}"
            ));
        }
        let mut weights = Array1::<f64>::zeros(self.charts.len());
        if activation > 0.0 {
            for (slot, &chart) in self.charts.iter().enumerate() {
                let weight = row_assignments[chart] / activation;
                if !(weight.is_finite() && weight >= 0.0) {
                    return Err(format!(
                        "atlas partition weight for chart {chart} must be finite and nonnegative, got {weight}"
                    ));
                }
                weights[slot] = weight;
            }
        }
        Ok((activation, weights))
    }

    pub(crate) fn remap(&mut self, old_to_new: &[Option<usize>]) -> Result<(), String> {
        let mut charts = Vec::with_capacity(self.charts.len());
        for &chart in &self.charts {
            charts.push(
                old_to_new
                    .get(chart)
                    .and_then(|x| *x)
                    .ok_or_else(|| format!("cannot remove registered atlas chart {chart}"))?,
            );
        }
        charts.sort_unstable();
        charts.dedup();
        self.charts = charts;
        for transition in &mut self.transitions {
            transition.remap(old_to_new)?;
        }
        self.canonicalize_transitions();
        self.validate()
    }

    pub(crate) fn shift_indices(&mut self, offset: usize) {
        for chart in &mut self.charts {
            *chart += offset;
        }
        for transition in &mut self.transitions {
            transition.from_chart += offset;
            transition.to_chart += offset;
        }
    }
}

impl SaeManifoldTerm {
    /// Registered multi-chart semantic atoms in canonical chart-index order.
    #[must_use]
    pub fn chart_atlases(&self) -> &[ManifoldChartAtlas] {
        &self.chart_atlases
    }

    /// Number of semantic atoms after quotienting local charts that belong to
    /// one registered atlas.  Raw decoder blocks remain available through
    /// [`SaeManifoldTerm::k_atoms`]; this count is the topology-aware dictionary
    /// size used for reporting a multi-chart atom as one object.
    #[must_use]
    pub fn semantic_atom_count(&self) -> usize {
        self.k_atoms()
            - self
                .chart_atlases
                .iter()
                .map(|atlas| atlas.charts.len() - 1)
                .sum::<usize>()
    }

    /// Whether two numerical chart blocks have already been quotiented into the
    /// same semantic atlas atom.
    #[must_use]
    pub fn charts_share_atlas(&self, a: usize, b: usize) -> bool {
        self.chart_atlases
            .iter()
            .any(|atlas| atlas.contains_chart(a) && atlas.contains_chart(b))
    }

    /// Register an exact transition, creating or joining atlas components as
    /// needed.  Atlas membership is disjoint by construction: merging two
    /// existing components consumes both and emits one connected component.
    #[must_use = "atlas registration errors must be handled"]
    pub fn register_chart_transition(
        &mut self,
        transition: UnitSpeedChartTransition,
    ) -> Result<(), String> {
        let k = self.k_atoms();
        if transition.from_chart >= k || transition.to_chart >= k {
            return Err(format!(
                "chart transition {}->{} is outside dictionary width K={k}",
                transition.from_chart, transition.to_chart
            ));
        }
        let left = self
            .chart_atlases
            .iter()
            .position(|atlas| atlas.contains_chart(transition.from_chart));
        let right = self
            .chart_atlases
            .iter()
            .position(|atlas| atlas.contains_chart(transition.to_chart));
        match (left, right) {
            (None, None) => self
                .chart_atlases
                .push(ManifoldChartAtlas::from_transition(transition)?),
            (Some(index), None) | (None, Some(index)) => {
                self.chart_atlases[index].add_transition(transition)?;
            }
            (Some(left), Some(right)) if left == right => {
                self.chart_atlases[left].add_transition(transition)?;
            }
            (Some(left), Some(right)) => {
                let (keep, take) = if left < right {
                    (left, right)
                } else {
                    (right, left)
                };
                let other = self.chart_atlases.remove(take);
                self.chart_atlases[keep].merge_with_transition(other, transition)?;
            }
        }
        self.chart_atlases
            .sort_by_key(|atlas| atlas.charts().first().copied().unwrap_or(usize::MAX));
        Ok(())
    }

    /// Replace the fitted map on an already-registered directed seam.  Used
    /// after a polish refit so the persisted transition describes the terminal
    /// charts rather than their pre-refit warm start.
    pub(crate) fn refresh_chart_transition(
        &mut self,
        transition: UnitSpeedChartTransition,
    ) -> Result<(), String> {
        for atlas in &mut self.chart_atlases {
            if atlas.replace_directed_transition(transition)? {
                return Ok(());
            }
        }
        Err(format!(
            "cannot refresh unregistered chart transition {}->{}",
            transition.from_chart, transition.to_chart
        ))
    }

    /// Atlas activation and partition weights for one assignment row.
    #[must_use = "partition errors must be handled"]
    pub fn atlas_partition_of_unity(
        &self,
        atlas_index: usize,
        row_assignments: ArrayView1<'_, f64>,
    ) -> Result<(f64, Array1<f64>), String> {
        self.chart_atlases
            .get(atlas_index)
            .ok_or_else(|| {
                format!(
                    "atlas index {atlas_index} is outside {} registered atlases",
                    self.chart_atlases.len()
                )
            })?
            .partition_of_unity(row_assignments)
    }

    pub(crate) fn remap_chart_atlases(
        &mut self,
        old_to_new: &[Option<usize>],
    ) -> Result<(), String> {
        for atlas in &mut self.chart_atlases {
            atlas.remap(old_to_new)?;
        }
        self.chart_atlases
            .sort_by_key(|atlas| atlas.charts().first().copied().unwrap_or(usize::MAX));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn tr(from: usize, to: usize, sign: i8, offset: f64) -> UnitSpeedChartTransition {
        UnitSpeedChartTransition::new(from, to, sign, offset, 1.0, AtlasSeamKind::Regular).unwrap()
    }

    #[test]
    fn exact_transition_inverse_and_composition_are_unit_speed() {
        let ab = tr(0, 1, -1, 0.25);
        let ba = ab.inverse();
        let identity = ab.compose(&ba).unwrap();
        assert_eq!(identity.sign, 1);
        assert_eq!(identity.offset.to_bits(), 0.0_f64.to_bits());
        for coordinate in [0.0, 0.125, 0.75, 1.25] {
            assert!(
                (ba.apply(ab.apply(coordinate)) - coordinate.rem_euclid(1.0)).abs()
                    < 8.0 * f64::EPSILON
            );
        }
    }

    #[test]
    fn sign_holonomy_distinguishes_coordinate_reversal_from_mobius() {
        // One negative edge is removable by flipping one chart orientation.
        let mut interval_cover = ManifoldChartAtlas::from_transition(tr(0, 1, -1, 0.0)).unwrap();
        assert_eq!(
            interval_cover.orientability(),
            AtlasOrientability::Orientable
        );

        interval_cover.add_transition(tr(1, 2, 1, 0.0)).unwrap();
        interval_cover.add_transition(tr(2, 0, 1, 0.0)).unwrap();
        assert_eq!(
            interval_cover.orientability(),
            AtlasOrientability::NonOrientable,
            "the cycle sign product is -1: the atlas has Möbius holonomy"
        );
    }

    #[test]
    fn chart_gates_factor_exactly_into_activation_times_partition() {
        let atlas = ManifoldChartAtlas::from_transition(tr(0, 2, 1, 0.2)).unwrap();
        let assignments = array![0.2, 0.5, 0.3];
        let (activation, weights) = atlas.partition_of_unity(assignments.view()).unwrap();
        assert_eq!(activation, 0.5);
        assert!((weights.sum() - 1.0).abs() < f64::EPSILON);
        assert_eq!(activation * weights[0], assignments[0]);
        assert_eq!(activation * weights[1], assignments[2]);
    }

    #[test]
    fn bridge_transition_joins_two_connected_atlas_components() {
        let mut left = ManifoldChartAtlas::from_transition(tr(0, 1, 1, 0.1)).unwrap();
        let right = ManifoldChartAtlas::from_transition(tr(2, 3, -1, 0.2)).unwrap();
        left.merge_with_transition(right, tr(1, 2, 1, 0.3)).unwrap();
        assert_eq!(left.charts(), &[0, 1, 2, 3]);
        assert_eq!(left.transitions().len(), 3);
        assert_eq!(left.orientability(), AtlasOrientability::Orientable);
    }
}
