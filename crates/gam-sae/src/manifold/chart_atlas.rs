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
        // Composition is a closed algebraic operation on isometries: composing a
        // transition with its inverse yields the IDENTITY self-map (`from == to`),
        // which `new` rejects as a registration self-edge.  Build the result
        // directly so `compose` stays total on validated inputs.
        Ok(Self {
            from_chart: self.from_chart,
            to_chart: next.to_chart,
            sign,
            offset,
            period: self.period,
            seam_kind: if matches!(self.seam_kind, AtlasSeamKind::Pole)
                || matches!(next.seam_kind, AtlasSeamKind::Pole)
            {
                AtlasSeamKind::Pole
            } else {
                AtlasSeamKind::Regular
            },
        })
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

/// Provenance of an orthogonal sphere-chart transition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SphereTransitionProvenance {
    /// The map is derived analytically from the chart definitions and may
    /// contribute an exact orientation cocycle sign.
    Analytic,
    /// The map is a polar factor fitted from decoder frames. Orthogonality of
    /// the stored matrix does not turn its estimated sign into an exact fact.
    Fitted,
}

/// Ambient isometry between two `latent_dim = 2` sphere charts (#1890 pole
/// seams), with analytic-versus-fitted provenance retained in the type.
///
/// Two lat/lon charts covering ONE ambient sphere with their poles in each
/// other's interior are related by an ambient rotation `R ∈ O(3)` acting on the
/// intrinsic unit vector `u = [x, y, z]`: `u_to = R · u_from`.  A sphere pole
/// seam is *intrinsically* two-dimensional — the transition is a full `3×3`
/// orthogonal matrix, not the `±1` sign a one-dimensional
/// [`UnitSpeedChartTransition`] carries — so a pole seam CANNOT be described by
/// a one-dimensional affine map, and is stored as this distinct kind rather than
/// a `Pole`-tagged 1-D transition (which would assert a map the overlap does not
/// have). For an analytically derived seam, orientability is read from `det R`:
/// `+1` preserves orientation and `-1` reverses it, exactly the role `sign`
/// plays for the 1-D transition. A fitted polar factor retains its determinant
/// only as geometry and never enters the exact sign cocycle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereChartTransition {
    from_chart: usize,
    to_chart: usize,
    /// Ambient rotation `R ∈ O(3)`, row-major, mapping `from_chart`'s intrinsic
    /// unit vector to `to_chart`'s: `u_to = R · u_from`.
    rotation: [[f64; 3]; 3],
    seam_kind: AtlasSeamKind,
    provenance: SphereTransitionProvenance,
}

impl SphereChartTransition {
    #[must_use]
    pub fn from_chart(&self) -> usize {
        self.from_chart
    }

    #[must_use]
    pub fn to_chart(&self) -> usize {
        self.to_chart
    }

    #[must_use]
    pub fn rotation(&self) -> &[[f64; 3]; 3] {
        &self.rotation
    }

    #[must_use]
    pub fn seam_kind(&self) -> AtlasSeamKind {
        self.seam_kind
    }

    #[must_use]
    pub fn provenance(&self) -> SphereTransitionProvenance {
        self.provenance
    }

    #[must_use = "analytic transition validation errors must be handled"]
    pub fn new_analytic(
        from_chart: usize,
        to_chart: usize,
        rotation: [[f64; 3]; 3],
        seam_kind: AtlasSeamKind,
    ) -> Result<Self, String> {
        Self::validate(
            from_chart,
            to_chart,
            rotation,
            seam_kind,
            SphereTransitionProvenance::Analytic,
        )
    }

    #[must_use = "fitted transition validation errors must be handled"]
    pub fn new_fitted(
        from_chart: usize,
        to_chart: usize,
        rotation: [[f64; 3]; 3],
        seam_kind: AtlasSeamKind,
    ) -> Result<Self, String> {
        Self::validate(
            from_chart,
            to_chart,
            rotation,
            seam_kind,
            SphereTransitionProvenance::Fitted,
        )
    }

    fn validate(
        from_chart: usize,
        to_chart: usize,
        rotation: [[f64; 3]; 3],
        seam_kind: AtlasSeamKind,
        provenance: SphereTransitionProvenance,
    ) -> Result<Self, String> {
        if from_chart == to_chart {
            return Err("sphere chart transition cannot be a self-edge".to_string());
        }
        if rotation.iter().flatten().any(|x| !x.is_finite()) {
            return Err("sphere chart transition rotation must be finite".to_string());
        }
        // `RᵀR = I`: accept precisely the backward error of a three-term dot
        // product at this matrix's scale. There is no statistical or fitted
        // tolerance here: a non-orthogonal fitted map belongs in the noisy
        // holonomy certificate, not this exact-transition type.
        let frame_scale: f64 = rotation.iter().flatten().map(|value| value * value).sum();
        let backward_error = f64::EPSILON * 3.0 * frame_scale.max(1.0);
        for i in 0..3 {
            for j in 0..3 {
                let dot: f64 = (0..3).map(|k| rotation[k][i] * rotation[k][j]).sum();
                let target = if i == j { 1.0 } else { 0.0 };
                if (dot - target).abs() > backward_error {
                    return Err(format!(
                        "sphere chart transition rotation is not orthonormal: (RᵀR)[{i},{j}] = {dot}, machine backward-error bound={backward_error}"
                    ));
                }
            }
        }
        // A square orthogonal matrix is nonsingular and has determinant ±1;
        // re-testing that theorem with an independent numeric band would only
        // reintroduce a second, potentially contradictory threshold.
        Ok(Self {
            from_chart,
            to_chart,
            rotation,
            seam_kind,
            provenance,
        })
    }

    fn determinant_of(r: &[[f64; 3]; 3]) -> f64 {
        r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
            - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
    }

    /// Signed determinant of the ambient rotation.
    #[must_use]
    pub fn determinant(&self) -> f64 {
        Self::determinant_of(&self.rotation)
    }

    /// Exact orientation contribution when and only when the seam is analytic.
    /// A fitted polar factor deliberately returns `None` even though its
    /// numerical determinant is ±1.
    #[must_use]
    pub fn analytic_sign(&self) -> Option<i8> {
        matches!(self.provenance, SphereTransitionProvenance::Analytic)
            .then(|| if self.determinant() >= 0.0 { 1 } else { -1 })
    }

    /// Apply the validated orthogonal ambient map to a unit vector.
    #[must_use]
    pub fn apply(&self, u: [f64; 3]) -> [f64; 3] {
        let mut out = [0.0; 3];
        for (i, row) in self.rotation.iter().enumerate() {
            out[i] = row[0] * u[0] + row[1] * u[1] + row[2] * u[2];
        }
        out
    }

    /// Inverse isometry (`R⁻¹ = Rᵀ` for an orthogonal `R`), endpoints
    /// swapped.
    #[must_use]
    pub fn inverse(&self) -> Self {
        let mut transpose = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                transpose[i][j] = self.rotation[j][i];
            }
        }
        Self {
            from_chart: self.to_chart,
            to_chart: self.from_chart,
            rotation: transpose,
            seam_kind: self.seam_kind,
            provenance: self.provenance,
        }
    }

    /// Compose `self: A -> B` with `next: B -> C` (matrix product `R_next · R_self`).
    #[must_use = "transition composition errors must be handled"]
    pub fn compose(&self, next: &Self) -> Result<Self, String> {
        if self.to_chart != next.from_chart {
            return Err(format!(
                "cannot compose sphere chart transitions {}->{} and {}->{}",
                self.from_chart, self.to_chart, next.from_chart, next.to_chart
            ));
        }
        let mut product = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                product[i][j] = (0..3)
                    .map(|k| next.rotation[i][k] * self.rotation[k][j])
                    .sum();
            }
        }
        // As with the 1-D transition, composing with the inverse yields the
        // identity self-map; build directly so `compose` stays total (the
        // self-edge check only guards registration).
        Ok(Self {
            from_chart: self.from_chart,
            to_chart: next.to_chart,
            rotation: product,
            seam_kind: if matches!(self.seam_kind, AtlasSeamKind::Pole)
                || matches!(next.seam_kind, AtlasSeamKind::Pole)
            {
                AtlasSeamKind::Pole
            } else {
                AtlasSeamKind::Regular
            },
            provenance: if matches!(self.provenance, SphereTransitionProvenance::Analytic)
                && matches!(next.provenance, SphereTransitionProvenance::Analytic)
            {
                SphereTransitionProvenance::Analytic
            } else {
                SphereTransitionProvenance::Fitted
            },
        })
    }

    pub(crate) fn remap(&mut self, old_to_new: &[Option<usize>]) -> Result<(), String> {
        self.from_chart = old_to_new
            .get(self.from_chart)
            .and_then(|x| *x)
            .ok_or_else(|| {
                "cannot remove an atlas chart while its sphere seam is registered".to_string()
            })?;
        self.to_chart = old_to_new
            .get(self.to_chart)
            .and_then(|x| *x)
            .ok_or_else(|| {
                "cannot remove an atlas chart while its sphere seam is registered".to_string()
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
///
/// Overlaps come in two geometric kinds. Analytic instances share one exact
/// sign cocycle; fitted sphere polar maps remain unsigned until a statistical
/// certificate resolves them. The one-dimensional unit-speed affine
/// [`UnitSpeedChartTransition`] (circles, Möbius half-twists) and the
/// two-dimensional ambient-rotation
/// [`SphereChartTransition`] (sphere pole seams).  Both are stored so a single
/// atlas can mix them; connectivity and orientability read the union of their
/// signed edges.
#[derive(Clone, Debug, PartialEq)]
pub struct ManifoldChartAtlas {
    charts: Vec<usize>,
    transitions: Vec<UnitSpeedChartTransition>,
    sphere_transitions: Vec<SphereChartTransition>,
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
            sphere_transitions: Vec::new(),
        };
        atlas.validate()?;
        Ok(atlas)
    }

    #[must_use = "atlas validation errors must be handled"]
    pub fn from_sphere_transition(transition: SphereChartTransition) -> Result<Self, String> {
        let charts = vec![
            transition.from_chart.min(transition.to_chart),
            transition.from_chart.max(transition.to_chart),
        ];
        let atlas = Self {
            charts,
            transitions: Vec::new(),
            sphere_transitions: vec![transition],
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

    /// The registered two-dimensional sphere pole seams, in canonical order.
    #[must_use]
    pub fn sphere_transitions(&self) -> &[SphereChartTransition] {
        &self.sphere_transitions
    }

    /// Every transition with an analytic sign as `(from, to, sign)`. Fitted
    /// sphere edges are intentionally absent from this exact cocycle.
    fn signed_edges(&self) -> impl Iterator<Item = (usize, usize, i8)> + '_ {
        self.transitions
            .iter()
            .map(|t| (t.from_chart, t.to_chart, t.sign))
            .chain(self.sphere_transitions.iter().filter_map(|t| {
                t.analytic_sign()
                    .map(|sign| (t.from_chart, t.to_chart, sign))
            }))
    }

    fn transition_edges(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.transitions
            .iter()
            .map(|transition| (transition.from_chart, transition.to_chart))
            .chain(
                self.sphere_transitions
                    .iter()
                    .map(|transition| (transition.from_chart, transition.to_chart)),
            )
    }

    fn transition_count(&self) -> usize {
        self.transitions.len() + self.sphere_transitions.len()
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

    pub(crate) fn add_sphere_transition(
        &mut self,
        transition: SphereChartTransition,
    ) -> Result<(), String> {
        self.charts.push(transition.from_chart);
        self.charts.push(transition.to_chart);
        self.charts.sort_unstable();
        self.charts.dedup();
        self.sphere_transitions.push(transition);
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

    pub(crate) fn replace_directed_sphere_transition(
        &mut self,
        transition: SphereChartTransition,
    ) -> Result<bool, String> {
        let Some(existing) = self.sphere_transitions.iter_mut().find(|existing| {
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
        self.sphere_transitions
            .append(&mut other.sphere_transitions);
        // The new edge is what connects the formerly separate atlas
        // components. Validate only after it is present; validating the plain
        // union first would (correctly but prematurely) reject it as
        // disconnected and make atlas growth impossible.
        self.transitions.push(transition);
        self.canonicalize_transitions();
        self.validate()
    }

    pub(crate) fn merge_with_sphere_transition(
        &mut self,
        mut other: Self,
        transition: SphereChartTransition,
    ) -> Result<(), String> {
        self.charts.append(&mut other.charts);
        self.charts.sort_unstable();
        self.charts.dedup();
        self.transitions.append(&mut other.transitions);
        self.sphere_transitions
            .append(&mut other.sphere_transitions);
        self.sphere_transitions.push(transition);
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
        self.sphere_transitions.sort_by(|a, b| {
            a.from_chart
                .cmp(&b.from_chart)
                .then(a.to_chart.cmp(&b.to_chart))
                .then((a.provenance as u8).cmp(&(b.provenance as u8)))
                .then(a.determinant().total_cmp(&b.determinant()))
                .then((a.seam_kind as u8).cmp(&(b.seam_kind as u8)))
                .then_with(|| {
                    for (x, y) in a.rotation.iter().flatten().zip(b.rotation.iter().flatten()) {
                        let ord = x.total_cmp(y);
                        if ord != std::cmp::Ordering::Equal {
                            return ord;
                        }
                    }
                    std::cmp::Ordering::Equal
                })
        });
    }

    fn validate(&self) -> Result<(), String> {
        if self.charts.len() < 2 {
            return Err("a chart atlas requires at least two charts".to_string());
        }
        if self.charts.windows(2).any(|w| w[0] >= w[1]) {
            return Err("atlas chart indices must be strictly increasing".to_string());
        }
        if self.transition_count() == 0 {
            return Err("a chart atlas requires at least one transition".to_string());
        }
        for (from, to) in self.transition_edges() {
            if !self.contains_chart(from) || !self.contains_chart(to) {
                return Err("atlas transition endpoint is not an atlas chart".to_string());
            }
        }
        // A semantic atom must be one connected cover, not a bag of unrelated
        // charts.  Connectivity is purely combinatorial and deterministic.
        let mut reached = BTreeSet::from([self.charts[0]]);
        let mut queue = VecDeque::from([self.charts[0]]);
        while let Some(chart) = queue.pop_front() {
            for (from, to) in self.transition_edges() {
                let next = if from == chart {
                    Some(to)
                } else if to == chart {
                    Some(from)
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
    /// Fitted sphere signs remain unknown: after contracting every consistent
    /// analytic signed component, an unknown-edge forest is harmless (all its
    /// signs can be gauged away), while an unknown cycle is unresolved.
    #[must_use]
    pub fn orientability(&self) -> Option<AtlasOrientability> {
        let mut orientation = BTreeMap::new();
        let mut analytic_component = BTreeMap::new();
        let mut component_count = 0usize;
        for &root in &self.charts {
            if orientation.contains_key(&root) {
                continue;
            }
            let component = component_count;
            component_count += 1;
            orientation.insert(root, 1_i8);
            analytic_component.insert(root, component);
            let mut queue = VecDeque::from([root]);
            while let Some(chart) = queue.pop_front() {
                let here = orientation[&chart];
                for (from, to, edge_sign) in self.signed_edges() {
                    let (next, sign) = if from == chart {
                        (to, edge_sign)
                    } else if to == chart {
                        (from, edge_sign)
                    } else {
                        continue;
                    };
                    let required = here * sign;
                    match orientation.get(&next) {
                        Some(&existing) if existing != required => {
                            return Some(AtlasOrientability::NonOrientable);
                        }
                        Some(_) => {}
                        None => {
                            orientation.insert(next, required);
                            analytic_component.insert(next, component);
                            queue.push_back(next);
                        }
                    }
                }
            }
        }

        let mut parents: Vec<_> = (0..component_count).collect();
        for transition in self
            .sphere_transitions
            .iter()
            .filter(|transition| transition.analytic_sign().is_none())
        {
            let left = disjoint_set_root(&mut parents, analytic_component[&transition.from_chart]);
            let right = disjoint_set_root(&mut parents, analytic_component[&transition.to_chart]);
            if left == right {
                return None;
            }
            parents[right] = left;
        }
        Some(AtlasOrientability::Orientable)
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
        for transition in &mut self.sphere_transitions {
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
        for transition in &mut self.sphere_transitions {
            transition.from_chart += offset;
            transition.to_chart += offset;
        }
    }
}

fn disjoint_set_root(parents: &mut [usize], node: usize) -> usize {
    let mut root = node;
    while parents[root] != root {
        root = parents[root];
    }
    let mut cursor = node;
    while parents[cursor] != cursor {
        let next = parents[cursor];
        parents[cursor] = root;
        cursor = next;
    }
    root
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

    /// Register an exact sphere pole seam (an ambient rotation between two
    /// `latent_dim = 2` charts), creating or joining atlas components exactly as
    /// [`Self::register_chart_transition`] does for the one-dimensional kind.
    #[must_use = "atlas registration errors must be handled"]
    pub fn register_sphere_chart_transition(
        &mut self,
        transition: SphereChartTransition,
    ) -> Result<(), String> {
        let k = self.k_atoms();
        if transition.from_chart >= k || transition.to_chart >= k {
            return Err(format!(
                "sphere chart transition {}->{} is outside dictionary width K={k}",
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
                .push(ManifoldChartAtlas::from_sphere_transition(transition)?),
            (Some(index), None) | (None, Some(index)) => {
                self.chart_atlases[index].add_sphere_transition(transition)?;
            }
            (Some(left), Some(right)) if left == right => {
                self.chart_atlases[left].add_sphere_transition(transition)?;
            }
            (Some(left), Some(right)) => {
                let (keep, take) = if left < right {
                    (left, right)
                } else {
                    (right, left)
                };
                let other = self.chart_atlases.remove(take);
                self.chart_atlases[keep].merge_with_sphere_transition(other, transition)?;
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

    /// Replace the fitted rotation on an already-registered sphere seam.
    pub(crate) fn refresh_sphere_chart_transition(
        &mut self,
        transition: SphereChartTransition,
    ) -> Result<(), String> {
        for atlas in &mut self.chart_atlases {
            if atlas.replace_directed_sphere_transition(transition)? {
                return Ok(());
            }
        }
        Err(format!(
            "cannot refresh unregistered sphere chart transition {}->{}",
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
            Some(AtlasOrientability::Orientable)
        );

        interval_cover.add_transition(tr(1, 2, 1, 0.0)).unwrap();
        interval_cover.add_transition(tr(2, 0, 1, 0.0)).unwrap();
        assert_eq!(
            interval_cover.orientability(),
            Some(AtlasOrientability::NonOrientable),
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
        assert_eq!(left.orientability(), Some(AtlasOrientability::Orientable));
    }

    #[test]
    fn sphere_transition_exact_type_rejects_fitted_near_orthogonality() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let reflection = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert_eq!(
            SphereChartTransition::new_analytic(0, 1, identity, AtlasSeamKind::Pole)
                .unwrap()
                .analytic_sign(),
            Some(1)
        );
        assert_eq!(
            SphereChartTransition::new_analytic(0, 1, reflection, AtlasSeamKind::Pole)
                .unwrap()
                .analytic_sign(),
            Some(-1)
        );

        let mut fitted_but_not_exact = identity;
        fitted_but_not_exact[0][0] += 1.0e-8;
        assert!(
            SphereChartTransition::new_fitted(0, 1, fitted_but_not_exact, AtlasSeamKind::Pole,)
                .is_err(),
            "a fitted approximate rotation must use the statistical certificate path"
        );
    }

    #[test]
    fn fitted_sphere_transition_cannot_enter_the_analytic_sign_cocycle() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let fitted =
            SphereChartTransition::new_fitted(0, 1, identity, AtlasSeamKind::Pole).unwrap();
        assert_eq!(fitted.analytic_sign(), None);
        assert_eq!(fitted.provenance(), SphereTransitionProvenance::Fitted);
        let atlas = ManifoldChartAtlas::from_sphere_transition(fitted).unwrap();
        assert_eq!(
            atlas.orientability(),
            Some(AtlasOrientability::Orientable),
            "one unknown bridge is a forest edge and cannot create holonomy"
        );

        let analytic =
            SphereChartTransition::new_analytic(0, 1, identity, AtlasSeamKind::Pole).unwrap();
        assert_eq!(analytic.analytic_sign(), Some(1));
        assert_eq!(analytic.provenance(), SphereTransitionProvenance::Analytic);
        let atlas = ManifoldChartAtlas::from_sphere_transition(analytic).unwrap();
        assert_eq!(atlas.orientability(), Some(AtlasOrientability::Orientable));
    }

    #[test]
    fn fitted_sphere_cycle_is_unresolved_but_cannot_erase_a_known_negative_cycle() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let fitted = |from, to| {
            SphereChartTransition::new_fitted(from, to, identity, AtlasSeamKind::Pole).unwrap()
        };
        let mut unknown_cycle = ManifoldChartAtlas::from_sphere_transition(fitted(0, 1)).unwrap();
        unknown_cycle.add_sphere_transition(fitted(1, 2)).unwrap();
        unknown_cycle.add_sphere_transition(fitted(2, 0)).unwrap();
        assert_eq!(unknown_cycle.orientability(), None);

        let mut known_negative = ManifoldChartAtlas::from_transition(tr(0, 1, -1, 0.0)).unwrap();
        known_negative.add_transition(tr(1, 2, 1, 0.0)).unwrap();
        known_negative.add_transition(tr(2, 0, 1, 0.0)).unwrap();
        known_negative.add_sphere_transition(fitted(2, 3)).unwrap();
        assert_eq!(
            known_negative.orientability(),
            Some(AtlasOrientability::NonOrientable)
        );
    }
}
