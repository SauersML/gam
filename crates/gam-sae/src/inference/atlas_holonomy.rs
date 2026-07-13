//! Finite-sample, gauge-invariant atlas holonomy certificates (#2311).
//!
//! A fitted transition matrix is an algebraic object; it is not, by itself, a
//! statistical proof that its sign or cycle angle is correct.  This module
//! keeps those two facts separate with a closed provenance sum:
//!
//! - [`AtlasHolonomyCertificate::ExactAnalytic`] carries transitions whose
//!   signs were established analytically, with zero sampling error;
//! - [`AtlasHolonomyCertificate::GaussianPcaPlugin`] carries the projected
//!   spiked-Gaussian PCA model, shared-patch covariance, explicit subspace tail
//!   and every refusal that can prevent a noisy transition from being promoted.
//!
//! The Gaussian path never uses an ambient-dimension approximation.  Each edge
//! is first represented in the numerical span of its two retained local PCA
//! frames.  The cycle variance is then formed from projector derivatives: all
//! incident-edge gradients for one patch are added *before* contracting with
//! that patch's covariance.  Consequently an arbitrary `O(2)` change of local
//! PCA gauge cancels exactly, and adjacent edges do not double-count their
//! shared patch.

use crate::manifold::{AtlasOrientability, SphereChartTransition};
use gam_linalg::faer_ndarray::{FaerEigh, FaerSvd};
use ndarray::{Array1, Array2, ArrayView2, s};
use statrs::distribution::{ContinuousCDF, Normal};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

const INTRINSIC_DIMENSION: usize = 2;

/// Familywise error level requested by the caller.
///
/// There is deliberately no default: topology promotion must state the error
/// probability it is willing to spend.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AtlasFamilywiseLevel {
    alpha: f64,
}

impl AtlasFamilywiseLevel {
    #[must_use = "confidence-level validation errors must be handled"]
    pub fn new(alpha: f64) -> Result<Self, String> {
        if !(alpha.is_finite() && alpha > 0.0 && alpha < 1.0) {
            return Err(format!(
                "atlas familywise alpha must be finite and strictly between zero and one, got {alpha}"
            ));
        }
        Ok(Self { alpha })
    }

    #[must_use]
    pub fn alpha(self) -> f64 {
        self.alpha
    }
}

/// Why a statistical atlas claim was not allowed to promote.
#[derive(Clone, Debug, PartialEq)]
pub enum AtlasStatisticalRefusal {
    PilotProjectionUncertified {
        chart: usize,
    },
    PopulationSpectrumUncertified {
        chart: usize,
    },
    GaussianLinearizationIsPlugin {
        cycle_index: usize,
    },
    DegenerateFirstOrderLimitUnresolved {
        cycle_index: usize,
        bilinear_quadratic_bias_diagnostic: f64,
        bilinear_quadratic_variance_diagnostic: f64,
    },
    PopulationCrossGramMarginUncertified {
        edge: AtlasHolonomyEdgeId,
    },
    SingularProjectedCrossGram {
        edge: AtlasHolonomyEdgeId,
        smallest_singular_value: f64,
        numerical_rank_threshold: f64,
    },
    PatchTailCrossesEigengap {
        edge: AtlasHolonomyEdgeId,
        chart: usize,
        covariance_error_bound: f64,
        eigengap_lower: f64,
    },
    OrientationFlipBoundExceedsLevel {
        flip_probability_bound: f64,
        allocated_alpha: f64,
    },
    ImproperCycleHolonomy {
        cycle_index: usize,
    },
    PolarLinearizationUnresolved {
        cycle_index: usize,
        edge: AtlasHolonomyEdgeId,
        cross_gram_error_bound: f64,
        population_smallest_singular_value_lower_bound: f64,
    },
    CycleAngleBranchCutCrossed {
        cycle_index: usize,
        absolute_angle: f64,
        uncertainty_radius: f64,
    },
    GaussBonnetRoundingMarginExhausted {
        residual_to_integer_curvature: f64,
        total_remainder_bound: f64,
    },
    GaussBonnetErrorBoundExceedsLevel {
        misround_probability_bound: f64,
        allocated_alpha: f64,
    },
    GaussBonnetGaussianLinearizationIsPlugin,
    GaussBonnetFirstOrderLimitDegenerate {
        first_order_variance: f64,
    },
}

/// Exact row identity for the independent pilot/inference split of one patch.
///
/// Counts alone cannot establish cross-fitting: the concrete row sets are
/// retained so the constructor can prove that a patch never fits its pilot
/// projection on a row later used for inference, and the joint covariance
/// model can prove whether inference errors from different patches are
/// independent or require an explicit cross block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GaussianPatchRowSplit {
    pilot_rows: GaussianRowSet,
    inference_rows: GaussianRowSet,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum GaussianRowSet {
    Explicit(Vec<usize>),
    Contiguous { start: usize, len: usize },
}

impl GaussianRowSet {
    fn len(&self) -> usize {
        match self {
            Self::Explicit(rows) => rows.len(),
            Self::Contiguous { len, .. } => *len,
        }
    }

    fn contains(&self, row: usize) -> bool {
        match self {
            Self::Explicit(rows) => rows.binary_search(&row).is_ok(),
            Self::Contiguous { start, len } => row >= *start && row - *start < *len,
        }
    }

    fn intersects(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Explicit(left), right) => left.iter().any(|row| right.contains(*row)),
            (left, Self::Explicit(right)) => right.iter().any(|row| left.contains(*row)),
            (
                Self::Contiguous {
                    start: left_start,
                    len: left_len,
                },
                Self::Contiguous {
                    start: right_start,
                    len: right_len,
                },
            ) => {
                let left_end = left_start.saturating_add(*left_len);
                let right_end = right_start.saturating_add(*right_len);
                *left_start < right_end && *right_start < left_end
            }
        }
    }

    fn materialize(&self) -> Vec<usize> {
        match self {
            Self::Explicit(rows) => rows.clone(),
            Self::Contiguous { start, len } => (*start..start.saturating_add(*len)).collect(),
        }
    }
}

impl GaussianPatchRowSplit {
    #[must_use = "Gaussian patch row-split validation errors must be handled"]
    pub fn new(mut pilot_rows: Vec<usize>, mut inference_rows: Vec<usize>) -> Result<Self, String> {
        pilot_rows.sort_unstable();
        inference_rows.sort_unstable();
        if pilot_rows.is_empty() || inference_rows.is_empty() {
            return Err(
                "Gaussian PCA pilot and inference row sets must both be non-empty".to_string(),
            );
        }
        if pilot_rows.windows(2).any(|rows| rows[0] == rows[1])
            || inference_rows.windows(2).any(|rows| rows[0] == rows[1])
        {
            return Err(
                "Gaussian PCA pilot and inference row sets must not contain duplicates".to_string(),
            );
        }
        if pilot_rows
            .iter()
            .any(|row| inference_rows.binary_search(row).is_ok())
        {
            return Err("Gaussian PCA pilot and inference row sets must be disjoint".to_string());
        }
        Ok(Self {
            pilot_rows: GaussianRowSet::Explicit(pilot_rows),
            inference_rows: GaussianRowSet::Explicit(inference_rows),
        })
    }

    /// Compact constructor for synthetic/analytic row populations.
    #[must_use = "Gaussian patch row-split validation errors must be handled"]
    pub fn from_disjoint_ranges(
        pilot_start: usize,
        pilot_len: usize,
        inference_start: usize,
        inference_len: usize,
    ) -> Result<Self, String> {
        if pilot_len == 0 || inference_len == 0 {
            return Err("Gaussian PCA pilot and inference ranges must be non-empty".to_string());
        }
        let pilot_rows = GaussianRowSet::Contiguous {
            start: pilot_start,
            len: pilot_len,
        };
        let inference_rows = GaussianRowSet::Contiguous {
            start: inference_start,
            len: inference_len,
        };
        if pilot_rows.intersects(&inference_rows) {
            return Err("Gaussian PCA pilot and inference ranges must be disjoint".to_string());
        }
        Ok(Self {
            pilot_rows,
            inference_rows,
        })
    }

    #[must_use]
    pub fn pilot_row_count(&self) -> usize {
        self.pilot_rows.len()
    }

    #[must_use]
    pub fn inference_row_count(&self) -> usize {
        self.inference_rows.len()
    }
}

/// Why the pilot projection may replace the ambient space in a certificate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PilotProjectionProvenance {
    /// A fixed retained frame with a deterministic proof that it contains the
    /// population tangent exactly. Full ambient retention is one such proof.
    /// This branch spends no probability budget and admits no leakage field.
    ExactAnalyticCapture,
    /// A frame fitted on the independent pilot rows, without a population
    /// capture theorem. It remains useful computationally but cannot sign a
    /// topology claim about the original ambient tangent.
    IndependentPilotEstimate,
}

impl PilotProjectionProvenance {
    fn is_certified(self) -> bool {
        matches!(self, Self::ExactAnalyticCapture)
    }
}

/// An authoritative statistical decision.  A refused decision carries the
/// exact mathematical condition which failed; consumers must never reinterpret
/// a refusal as a negative result.
#[derive(Clone, Debug, PartialEq)]
pub enum AtlasStatisticalDecision<T> {
    Certified {
        value: T,
        error_probability_bound: f64,
    },
    Refused {
        reasons: Vec<AtlasStatisticalRefusal>,
    },
}

impl<T> AtlasStatisticalDecision<T> {
    #[must_use]
    pub fn certified_value(&self) -> Option<&T> {
        match self {
            Self::Certified { value, .. } => Some(value),
            Self::Refused { .. } => None,
        }
    }

    #[must_use]
    pub fn error_probability_bound(&self) -> Option<f64> {
        match self {
            Self::Certified {
                error_probability_bound,
                ..
            } => Some(*error_probability_bound),
            Self::Refused { .. } => None,
        }
    }

    #[must_use]
    pub fn refusals(&self) -> &[AtlasStatisticalRefusal] {
        match self {
            Self::Certified { .. } => &[],
            Self::Refused { reasons } => reasons,
        }
    }
}

/// Canonical identity of one connected overlap component between two charts.
///
/// A chart pair is not an edge identity: disconnected overlap components are
/// distinct transition domains and can form a two-edge holonomy cycle.  The
/// endpoint order is canonical so this value can be compared directly across
/// geometry, statistical, nerve, and persistence layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct AtlasHolonomyEdgeId {
    a: usize,
    b: usize,
    overlap: usize,
}

impl AtlasHolonomyEdgeId {
    #[must_use = "atlas edge identity validation errors must be handled"]
    pub fn new(a: usize, b: usize, overlap: usize) -> Result<Self, String> {
        if a == b {
            return Err("an atlas transition cannot be a self-edge".to_string());
        }
        Ok(Self {
            a: a.min(b),
            b: a.max(b),
            overlap,
        })
    }

    #[must_use]
    pub fn a(self) -> usize {
        self.a
    }

    #[must_use]
    pub fn b(self) -> usize {
        self.b
    }

    #[must_use]
    pub fn overlap(self) -> usize {
        self.overlap
    }
}

/// Direction in which a canonical overlap edge is traversed by a cycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtlasHolonomyEdgeDirection {
    AToB,
    BToA,
}

/// One validated, directed step of a holonomy cycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AtlasHolonomyCycleStep {
    edge: AtlasHolonomyEdgeId,
    direction: AtlasHolonomyEdgeDirection,
}

impl AtlasHolonomyCycleStep {
    fn from_traversal(edge: AtlasHolonomyEdgeId, forward: bool) -> Self {
        Self {
            edge,
            direction: if forward {
                AtlasHolonomyEdgeDirection::AToB
            } else {
                AtlasHolonomyEdgeDirection::BToA
            },
        }
    }

    #[must_use]
    pub fn edge(self) -> AtlasHolonomyEdgeId {
        self.edge
    }

    #[must_use]
    pub fn direction(self) -> AtlasHolonomyEdgeDirection {
        self.direction
    }

    #[must_use]
    pub fn from(self) -> usize {
        match self.direction {
            AtlasHolonomyEdgeDirection::AToB => self.edge.a,
            AtlasHolonomyEdgeDirection::BToA => self.edge.b,
        }
    }

    #[must_use]
    pub fn to(self) -> usize {
        match self.direction {
            AtlasHolonomyEdgeDirection::AToB => self.edge.b,
            AtlasHolonomyEdgeDirection::BToA => self.edge.a,
        }
    }
}

/// One canonical undirected transition sign.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AtlasSignedEdge {
    a: usize,
    b: usize,
    /// Identity of a connected overlap component between the same chart pair.
    /// Parallel components are distinct cocycle edges and can themselves close
    /// a two-edge fundamental cycle.
    overlap: usize,
    sign: i8,
}

impl AtlasSignedEdge {
    #[must_use = "signed-edge validation errors must be handled"]
    pub fn new_analytic(a: usize, b: usize, overlap: usize, sign: i8) -> Result<Self, String> {
        let identity = AtlasHolonomyEdgeId::new(a, b, overlap)?;
        if !matches!(sign, -1 | 1) {
            return Err(format!(
                "an atlas transition sign must be +1 or -1, got {sign}"
            ));
        }
        Ok(Self {
            a: identity.a,
            b: identity.b,
            overlap: identity.overlap,
            sign,
        })
    }

    /// Extract an exact sign only from an analytically derived sphere seam.
    /// Fitted polar factors are rejected even when their stored matrix is
    /// perfectly orthogonal.
    #[must_use = "analytic sphere transition provenance must be handled"]
    pub fn from_analytic_sphere_transition(
        transition: &SphereChartTransition,
        overlap: usize,
    ) -> Result<Self, String> {
        let sign = transition.analytic_sign().ok_or_else(|| {
            "a fitted sphere transition cannot enter an exact analytic holonomy certificate"
                .to_string()
        })?;
        Self::new_analytic(
            transition.from_chart(),
            transition.to_chart(),
            overlap,
            sign,
        )
    }

    #[must_use]
    pub fn identity(self) -> AtlasHolonomyEdgeId {
        AtlasHolonomyEdgeId {
            a: self.a,
            b: self.b,
            overlap: self.overlap,
        }
    }

    #[must_use]
    pub fn sign(self) -> i8 {
        self.sign
    }
}

/// An exact signed cocycle, with no sampling model and therefore no alpha.
#[derive(Clone, Debug, PartialEq)]
pub struct ExactAnalyticHolonomyCertificate {
    chart_count: usize,
    edges: Vec<AtlasSignedEdge>,
    orientability: AtlasOrientability,
}

impl ExactAnalyticHolonomyCertificate {
    #[must_use = "exact holonomy validation errors must be handled"]
    pub fn new(chart_count: usize, mut edges: Vec<AtlasSignedEdge>) -> Result<Self, String> {
        edges.sort_by_key(|edge| (edge.a, edge.b, edge.overlap));
        for (position, edge) in edges.iter().enumerate() {
            if edge.b >= chart_count {
                return Err(format!(
                    "exact atlas edge ({}, {}, overlap {}) is outside the {chart_count}-chart atlas",
                    edge.a, edge.b, edge.overlap
                ));
            }
            if position > 0
                && (
                    edges[position - 1].a,
                    edges[position - 1].b,
                    edges[position - 1].overlap,
                ) == (edge.a, edge.b, edge.overlap)
            {
                return Err(format!(
                    "duplicate exact atlas edge ({}, {}, overlap {})",
                    edge.a, edge.b, edge.overlap
                ));
            }
        }
        let orientability = orientability_from_edges(chart_count, &edges);
        Ok(Self {
            chart_count,
            edges,
            orientability,
        })
    }

    #[must_use]
    pub fn chart_count(&self) -> usize {
        self.chart_count
    }

    #[must_use]
    pub fn edges(&self) -> &[AtlasSignedEdge] {
        &self.edges
    }

    #[must_use]
    pub fn orientability(&self) -> AtlasOrientability {
        self.orientability
    }
}

/// Population bounds used only by the non-asymptotic orientation tail.
///
/// `signal_variance_upper + noise_variance_upper` bounds the largest
/// eigenvalue of the projected population covariance; `eigengap_lower` bounds
/// the separation between the retained tangent spike and projected noise.  The
/// values are bounds, not estimates, so the orientation probability remains a
/// finite-sample statement even though cycle power uses a plug-in covariance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaussianPcaPopulationBounds {
    pub noise_variance_upper: f64,
    pub signal_variance_upper: f64,
    pub eigengap_lower: f64,
}

impl GaussianPcaPopulationBounds {
    #[must_use = "population-bound validation errors must be handled"]
    pub fn new(
        noise_variance_upper: f64,
        signal_variance_upper: f64,
        eigengap_lower: f64,
    ) -> Result<Self, String> {
        if !(noise_variance_upper.is_finite() && noise_variance_upper >= 0.0) {
            return Err(format!(
                "noise-variance upper bound must be finite and nonnegative, got {noise_variance_upper}"
            ));
        }
        if !(signal_variance_upper.is_finite() && signal_variance_upper > 0.0) {
            return Err(format!(
                "signal-variance upper bound must be finite and positive, got {signal_variance_upper}"
            ));
        }
        if !(eigengap_lower.is_finite() && eigengap_lower > 0.0) {
            return Err(format!(
                "PCA eigengap lower bound must be finite and positive, got {eigengap_lower}"
            ));
        }
        let spectral_radius_upper = noise_variance_upper + signal_variance_upper;
        if !spectral_radius_upper.is_finite() {
            return Err(format!(
                "projected population spectral-radius upper bound must be finite, got noise={noise_variance_upper}, signal={signal_variance_upper}"
            ));
        }
        if eigengap_lower > spectral_radius_upper {
            return Err(format!(
                "PCA eigengap lower bound {eigengap_lower} cannot exceed the projected population spectral-radius upper bound {spectral_radius_upper}"
            ));
        }
        Ok(Self {
            noise_variance_upper,
            signal_variance_upper,
            eigengap_lower,
        })
    }

    fn spectral_radius_upper(self) -> f64 {
        self.noise_variance_upper + self.signal_variance_upper
    }
}

/// Statistical authority for the spectrum used by one projected PCA patch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GaussianPcaSpectrumProvenance {
    /// Population bounds supplied by an independent analytic or concentration
    /// argument. Only this branch can drive the finite-sample orientation tail.
    CertifiedPopulation(GaussianPcaPopulationBounds),
    /// Spectrum estimated from the inference covariance. This drives plug-in
    /// power diagnostics but is never silently promoted to a probability bound.
    PlugInEstimate {
        noise_variance: f64,
        signal_variance: f64,
        eigengap: f64,
    },
}

impl GaussianPcaSpectrumProvenance {
    fn validate(self) -> Result<Self, String> {
        match self {
            Self::CertifiedPopulation(bounds) => {
                GaussianPcaPopulationBounds::new(
                    bounds.noise_variance_upper,
                    bounds.signal_variance_upper,
                    bounds.eigengap_lower,
                )?;
            }
            Self::PlugInEstimate {
                noise_variance,
                signal_variance,
                eigengap,
            } => {
                if !(noise_variance.is_finite() && noise_variance >= 0.0) {
                    return Err(format!(
                        "plug-in PCA noise variance must be finite and nonnegative, got {noise_variance}"
                    ));
                }
                if !(signal_variance.is_finite()
                    && signal_variance > 0.0
                    && eigengap.is_finite()
                    && eigengap > 0.0)
                {
                    return Err(format!(
                        "plug-in PCA signal variance and eigengap must be finite and positive, got signal={signal_variance}, gap={eigengap}"
                    ));
                }
            }
        }
        Ok(self)
    }

    fn certified_bounds(self) -> Option<GaussianPcaPopulationBounds> {
        match self {
            Self::CertifiedPopulation(bounds) => Some(bounds),
            Self::PlugInEstimate { .. } => None,
        }
    }
}

/// How the inference-split patch mean was handled.  This determines the exact
/// covariance degrees of freedom rather than pretending that `n` and `n - 1`
/// are interchangeable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GaussianPatchCentering {
    KnownOrIndependentMean,
    MeanEstimatedOnInferenceRows,
}

impl GaussianPatchCentering {
    fn covariance_degrees_of_freedom(self, inference_rows: usize) -> Option<usize> {
        match self {
            Self::KnownOrIndependentMean => Some(inference_rows),
            Self::MeanEstimatedOnInferenceRows => inference_rows.checked_sub(1),
        }
    }

    fn rows_for_degrees_of_freedom(self, degrees_of_freedom: usize) -> Option<usize> {
        match self {
            Self::KnownOrIndependentMean => Some(degrees_of_freedom),
            Self::MeanEstimatedOnInferenceRows => degrees_of_freedom.checked_add(1),
        }
    }
}

/// One locally projected PCA patch in the isotropic-spike model.
///
/// `projection_frame` is a fixed ambient `P × r` frame fitted on a disjoint
/// projection split.  `tangent_coordinates` is the inference-split local PCA
/// frame inside that fixed `r`-space.  This split is what makes replacing `P`
/// by the pairwise projected rank a finite-sample operation rather than a
/// data-dependent dimension shortcut.  The caller chooses `r`; there is no
/// hidden `q`.
#[derive(Clone, Debug, PartialEq)]
pub struct GaussianPcaPatch {
    chart: usize,
    row_split: GaussianPatchRowSplit,
    pilot_projection: PilotProjectionProvenance,
    centering: GaussianPatchCentering,
    /// Exact Wishart degrees of freedom implied by `inference_rows` and
    /// `centering`, validated once at construction and carried into every
    /// variance, tail, and occupancy prescription.
    covariance_degrees_of_freedom: usize,
    projection_frame: Array2<f64>,
    tangent_coordinates: Array2<f64>,
    noise_variance_estimate: f64,
    signal_variance_estimate: f64,
    spectrum_provenance: GaussianPcaSpectrumProvenance,
}

impl GaussianPcaPatch {
    #[must_use = "Gaussian PCA patch validation errors must be handled"]
    pub fn new(
        chart: usize,
        row_split: GaussianPatchRowSplit,
        pilot_projection: PilotProjectionProvenance,
        centering: GaussianPatchCentering,
        projection_frame: Array2<f64>,
        tangent_coordinates: Array2<f64>,
        noise_variance_estimate: f64,
        signal_variance_estimate: f64,
        spectrum_provenance: GaussianPcaSpectrumProvenance,
    ) -> Result<Self, String> {
        let (ambient, retained) = projection_frame.dim();
        let spectrum_provenance = spectrum_provenance.validate()?;
        let inference_rows = row_split.inference_rows.len();
        let covariance_dof = centering
            .covariance_degrees_of_freedom(inference_rows)
            .filter(|&value| value > 0)
            .ok_or_else(|| {
                format!(
                    "Gaussian PCA patch {chart} inference split has no covariance degrees of freedom"
                )
            })?;
        if ambient == 0 || retained < INTRINSIC_DIMENSION || retained > ambient {
            return Err(format!(
                "Gaussian PCA patch {chart} frame shape ({ambient}, {retained}) must satisfy ambient >= retained >= {INTRINSIC_DIMENSION}"
            ));
        }
        if tangent_coordinates.dim() != (retained, INTRINSIC_DIMENSION) {
            return Err(format!(
                "Gaussian PCA patch {chart} tangent coordinates have shape {:?}, expected ({retained}, {INTRINSIC_DIMENSION})",
                tangent_coordinates.dim()
            ));
        }
        if projection_frame.iter().any(|value| !value.is_finite())
            || tangent_coordinates.iter().any(|value| !value.is_finite())
        {
            return Err(format!(
                "Gaussian PCA patch {chart} projection and tangent frames must be finite"
            ));
        }
        if !(noise_variance_estimate.is_finite() && noise_variance_estimate >= 0.0) {
            return Err(format!(
                "Gaussian PCA patch {chart} noise estimate must be finite and nonnegative, got {noise_variance_estimate}"
            ));
        }
        if !(signal_variance_estimate.is_finite() && signal_variance_estimate > 0.0) {
            return Err(format!(
                "Gaussian PCA patch {chart} signal estimate must be finite and positive, got {signal_variance_estimate}"
            ));
        }
        let gram = projection_frame.t().dot(&projection_frame);
        let frame_scale: f64 = projection_frame.iter().map(|value| value * value).sum();
        let backward_error = f64::EPSILON * ambient.max(retained) as f64 * frame_scale.max(1.0);
        for i in 0..retained {
            for j in 0..retained {
                let target = if i == j { 1.0 } else { 0.0 };
                if (gram[[i, j]] - target).abs() > backward_error {
                    return Err(format!(
                        "Gaussian PCA patch {chart} retained frame is not orthonormal at ({i}, {j}): residual={}, machine backward-error bound={backward_error}",
                        gram[[i, j]] - target
                    ));
                }
            }
        }
        let tangent_gram = tangent_coordinates.t().dot(&tangent_coordinates);
        let tangent_scale: f64 = tangent_coordinates.iter().map(|value| value * value).sum();
        let tangent_backward_error =
            f64::EPSILON * retained.max(INTRINSIC_DIMENSION) as f64 * tangent_scale.max(1.0);
        for i in 0..INTRINSIC_DIMENSION {
            for j in 0..INTRINSIC_DIMENSION {
                let target = if i == j { 1.0 } else { 0.0 };
                if (tangent_gram[[i, j]] - target).abs() > tangent_backward_error {
                    return Err(format!(
                        "Gaussian PCA patch {chart} tangent coordinates are not orthonormal at ({i}, {j})"
                    ));
                }
            }
        }
        Ok(Self {
            chart,
            row_split,
            pilot_projection,
            centering,
            covariance_degrees_of_freedom: covariance_dof,
            projection_frame,
            tangent_coordinates,
            noise_variance_estimate,
            signal_variance_estimate,
            spectrum_provenance,
        })
    }

    #[must_use]
    pub fn ambient_dimension(&self) -> usize {
        self.projection_frame.nrows()
    }

    #[must_use]
    pub fn retained_dimension(&self) -> usize {
        self.projection_frame.ncols()
    }

    #[must_use]
    pub fn covariance_degrees_of_freedom(&self) -> usize {
        self.covariance_degrees_of_freedom
    }

    #[must_use]
    pub fn projector_variance_scale(&self) -> f64 {
        let noise = self.noise_variance_estimate;
        let signal = self.signal_variance_estimate;
        noise * (signal + noise) / (signal * signal * self.covariance_degrees_of_freedom() as f64)
    }

    fn tangent_frame(&self) -> Array2<f64> {
        self.projection_frame.dot(&self.tangent_coordinates)
    }

    fn audit_summary(&self) -> GaussianPcaPatchSummary {
        GaussianPcaPatchSummary {
            chart: self.chart,
            projection_fit_rows: self.row_split.pilot_rows.len(),
            inference_rows: self.row_split.inference_rows.len(),
            centering: self.centering,
            covariance_degrees_of_freedom: self.covariance_degrees_of_freedom,
            ambient_dimension: self.ambient_dimension(),
            retained_dimension: self.retained_dimension(),
            noise_variance_estimate: self.noise_variance_estimate,
            signal_variance_estimate: self.signal_variance_estimate,
            pilot_projection: self.pilot_projection,
            spectrum_provenance: self.spectrum_provenance,
        }
    }

    /// Fit a projected tangent on rows disjoint from the pilot projection.
    ///
    /// The resulting spectrum and horizontal covariance scale are explicitly
    /// plug-in quantities. A full-ambient retained frame has zero projection
    /// leakage by algebra; a reduced pilot frame remains uncertified until a
    /// caller supplies an independent capture theorem.
    #[must_use = "cross-fitted Gaussian PCA construction errors must be handled"]
    pub fn fit_cross_fitted_plugin(
        chart: usize,
        row_split: GaussianPatchRowSplit,
        data: ArrayView2<'_, f64>,
        retained_dimension: usize,
    ) -> Result<Self, String> {
        let ambient = data.ncols();
        if ambient < INTRINSIC_DIMENSION || retained_dimension < INTRINSIC_DIMENSION + 1 {
            return Err(format!(
                "cross-fitted patch {chart} requires ambient >= {INTRINSIC_DIMENSION} and retained dimension >= {}",
                INTRINSIC_DIMENSION + 1
            ));
        }
        if retained_dimension > ambient {
            return Err(format!(
                "cross-fitted patch {chart} retained dimension {retained_dimension} exceeds ambient {ambient}"
            ));
        }
        let pilot_rows = row_split.pilot_rows.materialize();
        let inference_rows = row_split.inference_rows.materialize();
        if pilot_rows.len() < 2 || inference_rows.len() <= INTRINSIC_DIMENSION {
            return Err(format!(
                "cross-fitted patch {chart} needs at least two pilot rows and more than {INTRINSIC_DIMENSION} inference rows"
            ));
        }
        if pilot_rows
            .iter()
            .chain(&inference_rows)
            .any(|&row| row >= data.nrows())
        {
            return Err(format!(
                "cross-fitted patch {chart} row identity exceeds data height {}",
                data.nrows()
            ));
        }
        let pilot_covariance = selected_covariance(data.view(), &pilot_rows, None)?;
        let (_, pilot_vectors) = pilot_covariance
            .eigh(faer::Side::Lower)
            .map_err(|error| format!("cross-fitted patch {chart} pilot PCA failed: {error}"))?;
        let mut projection_frame = Array2::<f64>::zeros((ambient, retained_dimension));
        for column in 0..retained_dimension {
            let source = ambient - 1 - column;
            projection_frame
                .column_mut(column)
                .assign(&pilot_vectors.column(source));
        }
        let inference_covariance =
            selected_covariance(data.view(), &inference_rows, Some(&projection_frame))?;
        let (inference_values, inference_vectors) = inference_covariance
            .eigh(faer::Side::Lower)
            .map_err(|error| format!("cross-fitted patch {chart} inference PCA failed: {error}"))?;
        let mut tangent_coordinates =
            Array2::<f64>::zeros((retained_dimension, INTRINSIC_DIMENSION));
        for column in 0..INTRINSIC_DIMENSION {
            let source = retained_dimension - 1 - column;
            tangent_coordinates
                .column_mut(column)
                .assign(&inference_vectors.column(source));
        }
        let noise_count = retained_dimension - INTRINSIC_DIMENSION;
        let noise_variance = inference_values
            .slice(s![0..noise_count])
            .iter()
            .copied()
            .sum::<f64>()
            / noise_count as f64;
        let weakest_tangent = inference_values[noise_count];
        let strongest_noise = inference_values[noise_count - 1];
        let signal_variance = weakest_tangent - noise_variance;
        let eigengap = weakest_tangent - strongest_noise;
        let spectrum_provenance = GaussianPcaSpectrumProvenance::PlugInEstimate {
            noise_variance,
            signal_variance,
            eigengap,
        }
        .validate()?;
        let pilot_projection = if retained_dimension == ambient {
            PilotProjectionProvenance::ExactAnalyticCapture
        } else {
            PilotProjectionProvenance::IndependentPilotEstimate
        };
        Self::new(
            chart,
            row_split,
            pilot_projection,
            GaussianPatchCentering::MeanEstimatedOnInferenceRows,
            projection_frame,
            tangent_coordinates,
            noise_variance,
            signal_variance,
            spectrum_provenance,
        )
    }
}

fn selected_covariance(
    data: ArrayView2<'_, f64>,
    rows: &[usize],
    projection: Option<&Array2<f64>>,
) -> Result<Array2<f64>, String> {
    let dimension = projection.map_or(data.ncols(), Array2::ncols);
    let mut mean = Array1::<f64>::zeros(dimension);
    for &row in rows {
        if let Some(frame) = projection {
            mean += &frame.t().dot(&data.row(row));
        } else {
            mean += &data.row(row);
        }
    }
    mean /= rows.len() as f64;
    let mut covariance = Array2::<f64>::zeros((dimension, dimension));
    for &row in rows {
        let centered = if let Some(frame) = projection {
            frame.t().dot(&data.row(row)) - &mean
        } else {
            data.row(row).to_owned() - &mean
        };
        for left in 0..dimension {
            for right in 0..=left {
                covariance[[left, right]] += centered[left] * centered[right];
            }
        }
    }
    let degrees_of_freedom = rows
        .len()
        .checked_sub(1)
        .ok_or_else(|| "sample covariance requires at least two rows".to_string())?
        as f64;
    for left in 0..dimension {
        for right in 0..=left {
            let value = covariance[[left, right]] / degrees_of_freedom;
            covariance[[left, right]] = value;
            covariance[[right, left]] = value;
        }
    }
    Ok(covariance)
}

/// Scalar provenance retained for every patch used by a noisy certificate.
///
/// These are exactly the sample-split, dimension, variance, and population
/// inputs that determine projector variance and the finite-sample tail.  Raw
/// frames remain edge-construction inputs rather than being duplicated into the
/// certificate, while every scalar entering a reported probability remains
/// independently auditable.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaussianPcaPatchSummary {
    pub chart: usize,
    pub projection_fit_rows: usize,
    pub inference_rows: usize,
    pub centering: GaussianPatchCentering,
    pub covariance_degrees_of_freedom: usize,
    pub ambient_dimension: usize,
    pub retained_dimension: usize,
    pub noise_variance_estimate: f64,
    pub signal_variance_estimate: f64,
    pub pilot_projection: PilotProjectionProvenance,
    pub spectrum_provenance: GaussianPcaSpectrumProvenance,
}

impl GaussianPcaPatchSummary {
    #[must_use]
    pub fn projector_variance_scale(self) -> f64 {
        let noise = self.noise_variance_estimate;
        let signal = self.signal_variance_estimate;
        noise * (signal + noise) / (signal * signal * self.covariance_degrees_of_freedom as f64)
    }
}

/// Whether the joint horizontal-error covariance is an exact Gaussian law or
/// an asymptotic plug-in approximation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GaussianPcaCovarianceAuthority {
    CertifiedGaussianLinearization,
    AsymptoticPlugIn,
}

/// Provenance of the off-diagonal patch covariance blocks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CrossPatchCovarianceProvenance {
    /// Every patch's inference rows are pairwise disjoint, so its cross blocks
    /// are structurally zero. The constructor verifies this from row identity.
    DisjointInferenceRows,
    /// Shared inference rows are permitted and every cross block is carried by
    /// the supplied joint covariance rather than silently treated as zero.
    ExplicitJointCovariance,
}

/// Joint covariance of the horizontal tangent-frame errors in each patch's
/// retained pilot coordinates.
///
/// Patch `j` contributes `r_j * d` row-major coordinates for an error matrix
/// `E_j` with `Delta_j = W_j E_j`. Keeping the operator in these small
/// coordinates makes cross-patch covariance exact without allocating an
/// ambient `(P d V)^2` matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct GaussianPcaErrorModel {
    authority: GaussianPcaCovarianceAuthority,
    cross_patch_provenance: CrossPatchCovarianceProvenance,
    offsets: Vec<usize>,
    covariance: Array2<f64>,
}

impl GaussianPcaErrorModel {
    fn coordinate_offsets(patches: &[GaussianPcaPatch]) -> Vec<usize> {
        let mut offsets = Vec::with_capacity(patches.len() + 1);
        offsets.push(0);
        for patch in patches {
            offsets.push(
                offsets.last().copied().unwrap_or(0)
                    + patch.retained_dimension() * INTRINSIC_DIMENSION,
            );
        }
        offsets
    }

    /// Construct the block-diagonal plug-in operator implied by pairwise-
    /// disjoint inference rows and fitted isotropic-spike patch scalars.
    /// Estimated noise/signal values can never select certified authority.
    #[must_use = "Gaussian PCA error-model validation errors must be handled"]
    pub fn independent(patches: &[GaussianPcaPatch]) -> Result<Self, String> {
        for left in 0..patches.len() {
            for right in (left + 1)..patches.len() {
                if patches[left]
                    .row_split
                    .inference_rows
                    .intersects(&patches[right].row_split.inference_rows)
                {
                    return Err(format!(
                        "Gaussian PCA patches {} and {} share inference rows; an explicit joint covariance is required",
                        patches[left].chart, patches[right].chart
                    ));
                }
            }
        }
        let offsets = Self::coordinate_offsets(patches);
        let dimension = offsets.last().copied().unwrap_or(0);
        let mut covariance = Array2::<f64>::zeros((dimension, dimension));
        for (patch_index, patch) in patches.iter().enumerate() {
            let retained = patch.retained_dimension();
            let normal = identity_square(retained)
                - patch
                    .tangent_coordinates
                    .dot(&patch.tangent_coordinates.t());
            let scale = patch.projector_variance_scale();
            let offset = offsets[patch_index];
            for row_left in 0..retained {
                for row_right in 0..retained {
                    for tangent in 0..INTRINSIC_DIMENSION {
                        let left = offset + row_left * INTRINSIC_DIMENSION + tangent;
                        let right = offset + row_right * INTRINSIC_DIMENSION + tangent;
                        covariance[[left, right]] = scale * normal[[row_left, row_right]];
                    }
                }
            }
        }
        Self::validate_joint(
            patches,
            GaussianPcaCovarianceAuthority::AsymptoticPlugIn,
            CrossPatchCovarianceProvenance::DisjointInferenceRows,
            covariance,
        )
    }

    /// Construct an authoritative Gaussian linearized-error law from an exact
    /// caller-supplied joint covariance, including every shared-row cross block.
    #[must_use = "Gaussian PCA error-model validation errors must be handled"]
    pub fn certified_joint(
        patches: &[GaussianPcaPatch],
        cross_patch_provenance: CrossPatchCovarianceProvenance,
        covariance: Array2<f64>,
    ) -> Result<Self, String> {
        Self::validate_joint(
            patches,
            GaussianPcaCovarianceAuthority::CertifiedGaussianLinearization,
            cross_patch_provenance,
            covariance,
        )
    }

    /// Construct an explicitly supplied asymptotic plug-in joint covariance.
    #[must_use = "Gaussian PCA error-model validation errors must be handled"]
    pub fn plugin_joint(
        patches: &[GaussianPcaPatch],
        cross_patch_provenance: CrossPatchCovarianceProvenance,
        covariance: Array2<f64>,
    ) -> Result<Self, String> {
        Self::validate_joint(
            patches,
            GaussianPcaCovarianceAuthority::AsymptoticPlugIn,
            cross_patch_provenance,
            covariance,
        )
    }

    fn validate_joint(
        patches: &[GaussianPcaPatch],
        authority: GaussianPcaCovarianceAuthority,
        cross_patch_provenance: CrossPatchCovarianceProvenance,
        covariance: Array2<f64>,
    ) -> Result<Self, String> {
        let offsets = Self::coordinate_offsets(patches);
        let dimension = offsets.last().copied().unwrap_or(0);
        if covariance.dim() != (dimension, dimension) {
            return Err(format!(
                "joint Gaussian PCA covariance has shape {:?}, expected ({dimension}, {dimension})",
                covariance.dim()
            ));
        }
        if covariance.iter().any(|value| !value.is_finite()) {
            return Err("joint Gaussian PCA covariance must be finite".to_string());
        }
        let scale = covariance
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max)
            .max(f64::MIN_POSITIVE);
        let backward_error = f64::EPSILON * dimension.max(1) as f64 * scale;
        let mut symmetric = covariance;
        for row in 0..dimension {
            for column in row..dimension {
                if (symmetric[[row, column]] - symmetric[[column, row]]).abs() > backward_error {
                    return Err(format!(
                        "joint Gaussian PCA covariance is not symmetric at ({row}, {column})"
                    ));
                }
                let value = (symmetric[[row, column]] + symmetric[[column, row]]) / 2.0;
                symmetric[[row, column]] = value;
                symmetric[[column, row]] = value;
            }
        }
        if dimension > 0 {
            let (eigenvalues, _) = symmetric.eigh(faer::Side::Lower).map_err(|error| {
                format!("joint Gaussian PCA covariance eigendecomposition failed: {error}")
            })?;
            if eigenvalues.iter().any(|&value| value < -backward_error) {
                return Err(
                    "joint Gaussian PCA covariance must be positive semidefinite".to_string(),
                );
            }
        }
        if matches!(
            cross_patch_provenance,
            CrossPatchCovarianceProvenance::DisjointInferenceRows
        ) {
            for left_patch in 0..patches.len() {
                for right_patch in (left_patch + 1)..patches.len() {
                    if patches[left_patch]
                        .row_split
                        .inference_rows
                        .intersects(&patches[right_patch].row_split.inference_rows)
                    {
                        return Err(format!(
                            "disjoint covariance provenance contradicts shared inference rows in patches {left_patch} and {right_patch}"
                        ));
                    }
                    for row in offsets[left_patch]..offsets[left_patch + 1] {
                        for column in offsets[right_patch]..offsets[right_patch + 1] {
                            if symmetric[[row, column]].abs() > backward_error {
                                return Err(format!(
                                    "disjoint covariance provenance has a nonzero cross block for patches {left_patch} and {right_patch}"
                                ));
                            }
                        }
                    }
                }
            }
        }
        Ok(Self {
            authority,
            cross_patch_provenance,
            offsets,
            covariance: symmetric,
        })
    }

    #[must_use]
    pub fn authority(&self) -> GaussianPcaCovarianceAuthority {
        self.authority
    }

    #[must_use]
    pub fn cross_patch_provenance(&self) -> &CrossPatchCovarianceProvenance {
        &self.cross_patch_provenance
    }
}

/// One edge requested from the Gaussian-PCA atlas.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PopulationCrossGramProvenance {
    /// No population separation proof is available. The observed cross-Gram
    /// remains useful geometry, but cannot set its own finite-sample threshold.
    EstimatedOnly,
    /// Deterministic lower bound on the smallest singular value of the true
    /// population cross-Gram for this overlap component.
    CertifiedSmallestSingularValue { lower_bound: f64 },
}

impl PopulationCrossGramProvenance {
    fn validate(self) -> Result<Self, String> {
        if let Self::CertifiedSmallestSingularValue { lower_bound } = self
            && !(lower_bound.is_finite() && lower_bound > 0.0 && lower_bound <= 1.0)
        {
            return Err(format!(
                "population cross-Gram singular-value lower bound must be finite in (0, 1], got {lower_bound}"
            ));
        }
        Ok(self)
    }

    fn certified_lower_bound(self) -> Option<f64> {
        match self {
            Self::EstimatedOnly => None,
            Self::CertifiedSmallestSingularValue { lower_bound } => Some(lower_bound),
        }
    }
}

/// One edge requested from the Gaussian-PCA atlas.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectedAtlasEdgeSpec {
    a: usize,
    b: usize,
    overlap: usize,
    population_cross_gram: PopulationCrossGramProvenance,
    /// Deterministic continuum/discretization error for this edge, in radians.
    /// This is supplied by the geometry builder; it is never inferred from a
    /// grid or hidden angular threshold.
    geometric_remainder_bound: f64,
}

impl ProjectedAtlasEdgeSpec {
    #[must_use = "projected-edge validation errors must be handled"]
    pub fn new(
        a: usize,
        b: usize,
        overlap: usize,
        population_cross_gram: PopulationCrossGramProvenance,
        geometric_remainder_bound: f64,
    ) -> Result<Self, String> {
        let identity = AtlasHolonomyEdgeId::new(a, b, overlap)?;
        let population_cross_gram = population_cross_gram.validate()?;
        if !(geometric_remainder_bound.is_finite() && geometric_remainder_bound >= 0.0) {
            return Err(format!(
                "edge geometric remainder must be finite and nonnegative, got {geometric_remainder_bound}"
            ));
        }
        Ok(Self {
            a: identity.a,
            b: identity.b,
            overlap: identity.overlap,
            population_cross_gram,
            geometric_remainder_bound,
        })
    }
}

/// Public relative geometry of one projected two-patch edge.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectedAtlasEdgeGeometry {
    pub a: usize,
    pub b: usize,
    pub overlap: usize,
    /// Numerical rank of `span(W_a, W_b)`.  This replaces ambient `P` in all
    /// second-order variance and concentration terms.
    pub projected_dimension: usize,
    pub principal_angle_cosines: [f64; INTRINSIC_DIMENSION],
    pub orientation_margin: f64,
    pub population_cross_gram: PopulationCrossGramProvenance,
    pub estimated_sign: Option<i8>,
    pub transition_a_to_b: Option<[[f64; INTRINSIC_DIMENSION]; INTRINSIC_DIMENSION]>,
    pub geometric_remainder_bound: f64,
}

impl ProjectedAtlasEdgeGeometry {
    #[must_use]
    pub fn identity(&self) -> AtlasHolonomyEdgeId {
        AtlasHolonomyEdgeId {
            a: self.a,
            b: self.b,
            overlap: self.overlap,
        }
    }
}

/// Pilot-side requirement for one patch's projection frame.
///
/// This is deliberately not a row-count scalar. More pilot rows cannot turn a
/// fitted reduced frame into a population-capture theorem.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtlasPilotOccupancyPrescription {
    /// The retained frame has an analytic zero-leakage proof (including full
    /// ambient retention), so topology certification imposes no pilot sample
    /// size requirement.
    ExactCaptureNoSamplingRequirement,
    /// The retained frame is only an independent pilot estimate. A population
    /// capture theorem is required before any finite row count is meaningful.
    PopulationCaptureTheoremRequired,
}

/// Inference-side occupancy required for the requested orientation error.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AtlasInferenceOccupancyPrescription {
    /// Closed-form Wishart occupancy after every population tail input has
    /// independently supplied authority.
    Required {
        rows: usize,
        covariance_degrees_of_freedom: usize,
        projected_dimension: usize,
        aligned_frame_error_budget: f64,
    },
    /// The requested probability is mathematically meaningful, but its row
    /// requirement exceeds the integer range accepted by this implementation.
    /// A magic maximum row count would falsely present this as attainable.
    RequiredRowsExceedRepresentableRange {
        projected_dimension: usize,
        aligned_frame_error_budget: f64,
    },
    /// A population spectrum or cross-Gram margin is absent. More inference
    /// rows alone cannot repair the missing authority.
    PopulationTailInputsRequired,
}

/// Separate pilot and inference requirements for one incident atlas patch.
#[derive(Clone, Debug, PartialEq)]
pub struct AtlasPatchSamplePrescription {
    pub chart: usize,
    pub current_pilot_rows: usize,
    pub pilot: AtlasPilotOccupancyPrescription,
    pub current_inference_rows: usize,
    pub current_covariance_degrees_of_freedom: usize,
    pub inference: AtlasInferenceOccupancyPrescription,
}

/// Outcome of the two-sided trivial-holonomy test.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtlasCycleConclusion {
    NonTrivialHolonomy,
    NotRejected,
}

/// Limiting law identified for one cycle before any hypothesis decision.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AtlasCycleAsymptoticRegime {
    /// The nonzero first derivative contracts an authoritative joint Gaussian
    /// error law. The nonlinear contribution is handled by a separate
    /// high-probability remainder, never folded into a z standard error.
    FirstOrderGaussian {
        variance: f64,
        authority: GaussianPcaCovarianceAuthority,
    },
    /// The first derivative cancels at numerical backward-error scale. The
    /// displayed bilinear moments cover only the explicit endpoint product;
    /// they are diagnostics, not the full Hessian or a claimed limit law.
    FirstOrderDegenerate {
        bilinear_quadratic_bias_diagnostic: f64,
        bilinear_quadratic_variance_diagnostic: f64,
    },
}

/// One fundamental-cycle readout with every uncertainty term exposed.
#[derive(Clone, Debug, PartialEq)]
pub struct AtlasCycleHolonomy {
    pub cycle_index: usize,
    /// Ordered, directed transitions.  Edge identity includes the connected
    /// overlap component, so parallel transitions remain distinguishable.
    steps: Vec<AtlasHolonomyCycleStep>,
    pub absolute_angle: Option<f64>,
    pub asymptotic_regime: Option<AtlasCycleAsymptoticRegime>,
    pub first_order_variance: Option<f64>,
    pub naive_edgewise_first_order_variance: Option<f64>,
    pub covariance_aggregation_adjustment: Option<f64>,
    pub bilinear_quadratic_bias_diagnostic: Option<f64>,
    pub bilinear_quadratic_variance_diagnostic: Option<f64>,
    pub standard_error: Option<f64>,
    pub polar_linearization_remainder_bound: Option<f64>,
    pub geometric_remainder_bound: f64,
    pub gaussian_error_budget: f64,
    pub subspace_tail_probability_bound: f64,
    pub decision: AtlasStatisticalDecision<AtlasCycleConclusion>,
}

impl AtlasCycleHolonomy {
    #[must_use]
    pub fn steps(&self) -> &[AtlasHolonomyCycleStep] {
        &self.steps
    }

    /// Derive the closed chart walk from the single authoritative step list.
    #[must_use]
    pub fn closed_chart_walk(&self) -> Vec<usize> {
        let Some(first) = self.steps.first().copied() else {
            return Vec::new();
        };
        let mut charts = Vec::with_capacity(self.steps.len() + 1);
        charts.push(first.from());
        charts.extend(self.steps.iter().copied().map(AtlasHolonomyCycleStep::to));
        charts
    }
}

/// A covariance source shared by one or more Gauss--Bonnet angle terms.
#[derive(Clone, Debug, PartialEq)]
pub struct GaussBonnetNoiseSource {
    source: usize,
    covariance: Array2<f64>,
}

impl GaussBonnetNoiseSource {
    #[must_use = "Gauss-Bonnet covariance validation errors must be handled"]
    pub fn new(source: usize, covariance: Array2<f64>) -> Result<Self, String> {
        let (rows, cols) = covariance.dim();
        if rows == 0 || rows != cols {
            return Err(format!(
                "Gauss-Bonnet covariance source {source} must be non-empty and square, got ({rows}, {cols})"
            ));
        }
        if covariance.iter().any(|value| !value.is_finite()) {
            return Err(format!(
                "Gauss-Bonnet covariance source {source} must be finite"
            ));
        }
        let scale = covariance
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max)
            .max(f64::MIN_POSITIVE);
        let backward_error = f64::EPSILON * rows as f64 * scale;
        let mut symmetric = covariance.clone();
        for i in 0..rows {
            for j in i..rows {
                if (covariance[[i, j]] - covariance[[j, i]]).abs() > backward_error {
                    return Err(format!(
                        "Gauss-Bonnet covariance source {source} is not symmetric at ({i}, {j}) within machine backward error"
                    ));
                }
                let value = (covariance[[i, j]] + covariance[[j, i]]) / 2.0;
                symmetric[[i, j]] = value;
                symmetric[[j, i]] = value;
            }
        }
        let (eigenvalues, _) = symmetric.eigh(faer::Side::Lower).map_err(|error| {
            format!("Gauss-Bonnet covariance eigendecomposition failed: {error}")
        })?;
        if eigenvalues.iter().any(|&value| value < -backward_error) {
            return Err(format!(
                "Gauss-Bonnet covariance source {source} is not positive semidefinite"
            ));
        }
        Ok(Self {
            source,
            covariance: symmetric,
        })
    }
}

/// Gradient of one measured angle with respect to one shared noise source.
#[derive(Clone, Debug, PartialEq)]
pub struct GaussBonnetSourceGradient {
    source: usize,
    gradient: Array1<f64>,
}

impl GaussBonnetSourceGradient {
    #[must_use = "Gauss-Bonnet gradient validation errors must be handled"]
    pub fn new(source: usize, gradient: Array1<f64>) -> Result<Self, String> {
        if gradient.is_empty() || gradient.iter().any(|value| !value.is_finite()) {
            return Err(format!(
                "Gauss-Bonnet gradient for source {source} must be non-empty and finite"
            ));
        }
        Ok(Self { source, gradient })
    }
}

/// One signed curvature/angle-defect contribution.
#[derive(Clone, Debug, PartialEq)]
pub struct GaussBonnetContribution {
    curvature_estimate: f64,
    polar_linearization_remainder_bound: f64,
    geometric_remainder_bound: f64,
    source_gradients: Vec<GaussBonnetSourceGradient>,
}

impl GaussBonnetContribution {
    #[must_use = "Gauss-Bonnet contribution validation errors must be handled"]
    pub fn new(
        curvature_estimate: f64,
        polar_linearization_remainder_bound: f64,
        geometric_remainder_bound: f64,
        source_gradients: Vec<GaussBonnetSourceGradient>,
    ) -> Result<Self, String> {
        if !curvature_estimate.is_finite() {
            return Err(format!(
                "Gauss-Bonnet curvature estimate must be finite, got {curvature_estimate}"
            ));
        }
        for (name, value) in [
            ("polar linearization", polar_linearization_remainder_bound),
            ("geometric", geometric_remainder_bound),
        ] {
            if !(value.is_finite() && value >= 0.0) {
                return Err(format!(
                    "Gauss-Bonnet {name} remainder must be finite and nonnegative, got {value}"
                ));
            }
        }
        Ok(Self {
            curvature_estimate,
            polar_linearization_remainder_bound,
            geometric_remainder_bound,
            source_gradients,
        })
    }
}

/// Inputs for the integer Gauss--Bonnet confidence calculation.
#[derive(Clone, Debug, PartialEq)]
pub struct GaussBonnetInput {
    covariance_authority: GaussBonnetCovarianceAuthority,
    sources: Vec<GaussBonnetNoiseSource>,
    contributions: Vec<GaussBonnetContribution>,
}

/// Whether the propagated Gauss--Bonnet covariance supports a finite-sample
/// Gaussian probability or is only an asymptotic plug-in diagnostic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GaussBonnetCovarianceAuthority {
    /// Every source vector is jointly Gaussian and distinct source IDs are
    /// certified independent. Correlated quantities must occupy one shared
    /// vector source so their covariance is never discarded between IDs.
    CertifiedIndependentGaussianSources,
    AsymptoticPlugIn,
}

impl GaussBonnetInput {
    #[must_use = "certified Gauss-Bonnet input validation errors must be handled"]
    pub fn certified_independent_gaussian(
        sources: Vec<GaussBonnetNoiseSource>,
        contributions: Vec<GaussBonnetContribution>,
    ) -> Result<Self, String> {
        Self::validate(
            GaussBonnetCovarianceAuthority::CertifiedIndependentGaussianSources,
            sources,
            contributions,
        )
    }

    #[must_use = "plug-in Gauss-Bonnet input validation errors must be handled"]
    pub fn asymptotic_plugin(
        sources: Vec<GaussBonnetNoiseSource>,
        contributions: Vec<GaussBonnetContribution>,
    ) -> Result<Self, String> {
        Self::validate(
            GaussBonnetCovarianceAuthority::AsymptoticPlugIn,
            sources,
            contributions,
        )
    }

    fn validate(
        covariance_authority: GaussBonnetCovarianceAuthority,
        mut sources: Vec<GaussBonnetNoiseSource>,
        contributions: Vec<GaussBonnetContribution>,
    ) -> Result<Self, String> {
        sources.sort_by_key(|source| source.source);
        if sources
            .windows(2)
            .any(|pair| pair[0].source == pair[1].source)
        {
            return Err("Gauss-Bonnet covariance source identifiers must be unique".to_string());
        }
        let dimensions: BTreeMap<usize, usize> = sources
            .iter()
            .map(|source| (source.source, source.covariance.nrows()))
            .collect();
        for contribution in &contributions {
            for gradient in &contribution.source_gradients {
                let dimension = dimensions.get(&gradient.source).ok_or_else(|| {
                    format!(
                        "Gauss-Bonnet gradient names absent covariance source {}",
                        gradient.source
                    )
                })?;
                if gradient.gradient.len() != *dimension {
                    return Err(format!(
                        "Gauss-Bonnet gradient for source {} has length {} but covariance dimension is {}",
                        gradient.source,
                        gradient.gradient.len(),
                        dimension
                    ));
                }
            }
        }
        Ok(Self {
            covariance_authority,
            sources,
            contributions,
        })
    }
}

/// Integer Euler characteristic certified by a Gauss--Bonnet rounding cell.
///
/// Keeping this distinct from an arbitrary integer prevents topology-promotion
/// consumers from accidentally treating an unqualified rounded scalar as a
/// finite-sample Euler-characteristic claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AtlasEulerCharacteristic(i64);

impl AtlasEulerCharacteristic {
    #[must_use]
    pub fn value(self) -> i64 {
        self.0
    }
}

/// Integer Gauss--Bonnet readout and its shared-source covariance audit.
#[derive(Clone, Debug, PartialEq)]
pub struct GaussBonnetConfidence {
    pub covariance_authority: GaussBonnetCovarianceAuthority,
    pub total_curvature_estimate: f64,
    pub nearest_integer_candidate: AtlasEulerCharacteristic,
    pub residual_to_integer_curvature: f64,
    pub first_order_variance: f64,
    pub naive_contribution_variance: f64,
    pub shared_source_covariance_adjustment: f64,
    pub standard_error: Option<f64>,
    pub polar_linearization_remainder_bound: f64,
    pub geometric_remainder_bound: f64,
    /// Distance from this observed estimate to its nearest integer-cell
    /// boundary after deterministic remainders.
    pub observed_rounding_margin: f64,
    /// Uniform half-cell margin governing the frequentist probability that
    /// rounding selects the wrong integer, independent of the realized
    /// residual inside the selected cell.
    pub stochastic_rounding_margin: f64,
    pub misround_probability_bound: Option<f64>,
    pub decision: AtlasStatisticalDecision<AtlasEulerCharacteristic>,
}

/// Complete noisy-PCA analysis, including refused decisions.
#[derive(Clone, Debug, PartialEq)]
pub struct GaussianPcaHolonomyAnalysis {
    familywise_level: AtlasFamilywiseLevel,
    chart_count: usize,
    patch_summaries: Vec<GaussianPcaPatchSummary>,
    error_model: GaussianPcaErrorModel,
    edges: Vec<ProjectedAtlasEdgeGeometry>,
    orientation: AtlasStatisticalDecision<AtlasOrientability>,
    orientation_flip_probability_bound: Option<f64>,
    sample_prescription: Vec<AtlasPatchSamplePrescription>,
    cycles: Vec<AtlasCycleHolonomy>,
    gauss_bonnet: Option<GaussBonnetConfidence>,
}

impl GaussianPcaHolonomyAnalysis {
    #[must_use]
    pub fn familywise_level(&self) -> AtlasFamilywiseLevel {
        self.familywise_level
    }

    #[must_use]
    pub fn chart_count(&self) -> usize {
        self.chart_count
    }

    #[must_use]
    pub fn patch_summaries(&self) -> &[GaussianPcaPatchSummary] {
        &self.patch_summaries
    }

    #[must_use]
    pub fn error_model(&self) -> &GaussianPcaErrorModel {
        &self.error_model
    }

    #[must_use]
    pub fn edges(&self) -> &[ProjectedAtlasEdgeGeometry] {
        &self.edges
    }

    #[must_use]
    pub fn orientation(&self) -> &AtlasStatisticalDecision<AtlasOrientability> {
        &self.orientation
    }

    #[must_use]
    pub fn orientation_flip_probability_bound(&self) -> Option<f64> {
        self.orientation_flip_probability_bound
    }

    #[must_use]
    pub fn sample_prescription(&self) -> &[AtlasPatchSamplePrescription] {
        &self.sample_prescription
    }

    #[must_use]
    pub fn cycles(&self) -> &[AtlasCycleHolonomy] {
        &self.cycles
    }

    #[must_use]
    pub fn gauss_bonnet(&self) -> Option<&GaussBonnetConfidence> {
        self.gauss_bonnet.as_ref()
    }

    /// Integer topology claim only when the Gauss--Bonnet rounding cell meets
    /// its allocated familywise error probability.
    #[must_use]
    pub fn certified_euler_characteristic(&self) -> Option<AtlasEulerCharacteristic> {
        self.gauss_bonnet
            .as_ref()
            .and_then(|confidence| confidence.decision.certified_value())
            .copied()
    }
}

/// Closed provenance sum for every production atlas holonomy result.
#[derive(Clone, Debug, PartialEq)]
pub enum AtlasHolonomyCertificate {
    ExactAnalytic(ExactAnalyticHolonomyCertificate),
    GaussianPcaPlugin(GaussianPcaHolonomyAnalysis),
}

impl AtlasHolonomyCertificate {
    #[must_use]
    pub fn chart_count(&self) -> usize {
        match self {
            Self::ExactAnalytic(certificate) => certificate.chart_count(),
            Self::GaussianPcaPlugin(analysis) => analysis.chart_count,
        }
    }

    /// Complete canonical edge inventory, including overlap component.
    ///
    /// This remains available when a noisy orientation decision is refused.
    /// Consumers can therefore validate that a refusal belongs to exactly the
    /// atlas they are reporting without converting the refusal into a
    /// promotable signed cocycle.
    #[must_use]
    pub fn edge_inventory(&self) -> Vec<AtlasHolonomyEdgeId> {
        match self {
            Self::ExactAnalytic(certificate) => certificate
                .edges()
                .iter()
                .copied()
                .map(AtlasSignedEdge::identity)
                .collect(),
            Self::GaussianPcaPlugin(analysis) => analysis
                .edges
                .iter()
                .map(ProjectedAtlasEdgeGeometry::identity)
                .collect(),
        }
    }

    #[must_use]
    pub fn certified_orientability(&self) -> Option<AtlasOrientability> {
        match self {
            Self::ExactAnalytic(certificate) => Some(certificate.orientability()),
            Self::GaussianPcaPlugin(analysis) => analysis.orientation.certified_value().copied(),
        }
    }

    /// A noisy PCA certificate can additionally sign an integer
    /// Gauss--Bonnet claim. Exact transition cocycles make no curvature claim;
    /// their consumer must use a separate exact good-cover proof.
    #[must_use]
    pub fn certified_euler_characteristic(&self) -> Option<AtlasEulerCharacteristic> {
        match self {
            Self::ExactAnalytic(_) => None,
            Self::GaussianPcaPlugin(analysis) => analysis.certified_euler_characteristic(),
        }
    }

    #[must_use]
    pub fn provenance_label(&self) -> &'static str {
        match self {
            Self::ExactAnalytic(_) => "exact_analytic",
            Self::GaussianPcaPlugin(_) => "gaussian_pca_plugin",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, array};

    fn projection_frame(angle: f64, padded_ambient: usize) -> Array2<f64> {
        let mut frame = Array2::<f64>::zeros((3 + padded_ambient, 3));
        let (cosine, sine) = (angle.cos(), angle.sin());
        frame[[0, 0]] = cosine;
        frame[[2, 0]] = sine;
        frame[[1, 1]] = 1.0;
        frame[[0, 2]] = -sine;
        frame[[2, 2]] = cosine;
        frame
    }

    fn tangent_gauge(angle: f64, reflected: bool) -> Array2<f64> {
        let (cosine, sine) = (angle.cos(), angle.sin());
        let reflection = if reflected { -1.0 } else { 1.0 };
        arr2(&[
            [cosine, -reflection * sine],
            [sine, reflection * cosine],
            [0.0, 0.0],
        ])
    }

    fn patch(
        chart: usize,
        plane_angle: f64,
        gauge_angle: f64,
        reflected: bool,
        inference_rows: usize,
        padded_ambient: usize,
    ) -> GaussianPcaPatch {
        let base = chart * 4_000_000;
        GaussianPcaPatch::new(
            chart,
            GaussianPatchRowSplit::from_disjoint_ranges(
                base,
                1_000,
                base + 1_000_000,
                inference_rows,
            )
            .unwrap(),
            PilotProjectionProvenance::ExactAnalyticCapture,
            GaussianPatchCentering::MeanEstimatedOnInferenceRows,
            projection_frame(plane_angle, padded_ambient),
            tangent_gauge(gauge_angle, reflected),
            0.01,
            1.0,
            GaussianPcaSpectrumProvenance::CertifiedPopulation(
                GaussianPcaPopulationBounds::new(0.01, 2.0, 1.0).unwrap(),
            ),
        )
        .unwrap()
    }

    fn certified_analysis(
        patches: Vec<GaussianPcaPatch>,
        edges: Vec<ProjectedAtlasEdgeSpec>,
        level: AtlasFamilywiseLevel,
        gauss_bonnet: Option<GaussBonnetInput>,
    ) -> GaussianPcaHolonomyAnalysis {
        let offsets = GaussianPcaErrorModel::coordinate_offsets(&patches);
        let dimension = offsets.last().copied().unwrap_or(0);
        let mut covariance = Array2::<f64>::zeros((dimension, dimension));
        for (patch_index, patch) in patches.iter().enumerate() {
            let retained = patch.retained_dimension();
            let normal = identity_square(retained)
                - patch
                    .tangent_coordinates
                    .dot(&patch.tangent_coordinates.t());
            // Exact DGP parameters of the test fixtures, not fitted patch
            // scalars: sigma^2=0.01 and lambda=1.
            let scale = 0.01 * 1.01 / patch.covariance_degrees_of_freedom() as f64;
            for row_left in 0..retained {
                for row_right in 0..retained {
                    for tangent in 0..INTRINSIC_DIMENSION {
                        covariance[[
                            offsets[patch_index] + row_left * INTRINSIC_DIMENSION + tangent,
                            offsets[patch_index] + row_right * INTRINSIC_DIMENSION + tangent,
                        ]] = scale * normal[[row_left, row_right]];
                    }
                }
            }
        }
        let error_model = GaussianPcaErrorModel::certified_joint(
            &patches,
            CrossPatchCovarianceProvenance::DisjointInferenceRows,
            covariance,
        )
        .unwrap();
        GaussianPcaHolonomyAnalysis::certify(patches, edges, error_model, level, gauss_bonnet)
            .unwrap()
    }

    fn certified_edge(
        a: usize,
        b: usize,
        overlap: usize,
        geometric_remainder_bound: f64,
    ) -> ProjectedAtlasEdgeSpec {
        ProjectedAtlasEdgeSpec::new(
            a,
            b,
            overlap,
            PopulationCrossGramProvenance::CertifiedSmallestSingularValue { lower_bound: 0.5 },
            geometric_remainder_bound,
        )
        .unwrap()
    }

    fn triangle_edges() -> Vec<ProjectedAtlasEdgeSpec> {
        vec![
            certified_edge(0, 1, 0, 0.0),
            certified_edge(1, 2, 0, 0.0),
            certified_edge(2, 0, 0, 0.0),
        ]
    }

    fn triangle_analysis(
        gauges: [(f64, bool); 3],
        padded_ambient: usize,
    ) -> GaussianPcaHolonomyAnalysis {
        certified_analysis(
            vec![
                patch(0, 0.0, gauges[0].0, gauges[0].1, 1_000_000, padded_ambient),
                patch(1, 0.2, gauges[1].0, gauges[1].1, 1_000_000, padded_ambient),
                patch(
                    2,
                    -0.15,
                    gauges[2].0,
                    gauges[2].1,
                    1_000_000,
                    padded_ambient,
                ),
            ],
            triangle_edges(),
            AtlasFamilywiseLevel::new(0.05).unwrap(),
            None,
        )
    }

    fn assert_near(left: f64, right: f64) {
        let scale = left.abs().max(right.abs()).max(1.0);
        assert!((left - right).abs() <= f64::EPSILON.sqrt() * scale);
    }

    fn required_inference_prescription(
        prescription: &AtlasPatchSamplePrescription,
    ) -> (usize, usize, usize, f64) {
        match prescription.inference {
            AtlasInferenceOccupancyPrescription::Required {
                rows,
                covariance_degrees_of_freedom,
                projected_dimension,
                aligned_frame_error_budget,
            } => (
                rows,
                covariance_degrees_of_freedom,
                projected_dimension,
                aligned_frame_error_budget,
            ),
            AtlasInferenceOccupancyPrescription::RequiredRowsExceedRepresentableRange {
                ..
            } => panic!("test fixture has a representable inference occupancy"),
            AtlasInferenceOccupancyPrescription::PopulationTailInputsRequired => {
                panic!("test fixture supplies every population tail input")
            }
        }
    }

    /// Test-only independent SplitMix64 + Box--Muller stream. The production
    /// certificate contains no RNG; this deliberately avoids sharing any
    /// numerical path with its variance and tail calculations.
    struct DeterministicGaussian {
        state: u64,
        spare: Option<f64>,
    }

    impl DeterministicGaussian {
        fn new(seed: u64) -> Self {
            Self {
                state: seed,
                spare: None,
            }
        }

        fn uniform_open(&mut self) -> f64 {
            self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
            let mut value = self.state;
            value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
            value ^= value >> 31;
            ((value >> 11) as f64 + 0.5) / ((1_u64 << 53) as f64)
        }

        fn normal(&mut self) -> f64 {
            if let Some(spare) = self.spare.take() {
                return spare;
            }
            let radius = (-2.0 * self.uniform_open().ln()).sqrt();
            let angle = std::f64::consts::TAU * self.uniform_open();
            self.spare = Some(radius * angle.sin());
            radius * angle.cos()
        }
    }

    fn identity_projection(ambient: usize) -> Array2<f64> {
        let mut identity = Array2::<f64>::zeros((ambient, ambient));
        for diagonal in 0..ambient {
            identity[[diagonal, diagonal]] = 1.0;
        }
        identity
    }

    fn plane_tangent(angle: f64, ambient: usize) -> Array2<f64> {
        let mut tangent = Array2::<f64>::zeros((ambient, INTRINSIC_DIMENSION));
        tangent[[0, 0]] = angle.cos();
        tangent[[2, 0]] = angle.sin();
        tangent[[1, 1]] = 1.0;
        tangent
    }

    fn orthonormalize_two_columns(mut frame: Array2<f64>) -> Array2<f64> {
        let first_norm = frame.column(0).dot(&frame.column(0)).sqrt();
        for row in 0..frame.nrows() {
            frame[[row, 0]] /= first_norm;
        }
        let projection = frame.column(0).dot(&frame.column(1));
        for row in 0..frame.nrows() {
            frame[[row, 1]] -= projection * frame[[row, 0]];
        }
        let second_norm = frame.column(1).dot(&frame.column(1)).sqrt();
        for row in 0..frame.nrows() {
            frame[[row, 1]] /= second_norm;
        }
        frame
    }

    fn perturb_tangent(
        tangent: &Array2<f64>,
        standard_deviation: f64,
        gaussian: &mut DeterministicGaussian,
    ) -> Array2<f64> {
        let mut perturbed = tangent.clone();
        for value in &mut perturbed {
            *value += standard_deviation * gaussian.normal();
        }
        orthonormalize_two_columns(perturbed)
    }

    fn sample_spiked_pca_tangent(
        population_tangent: &Array2<f64>,
        rows: usize,
        noise_standard_deviation: f64,
        gaussian: &mut DeterministicGaussian,
    ) -> Array2<f64> {
        let ambient = population_tangent.nrows();
        let mut data = Array2::<f64>::zeros((rows, ambient));
        for row in 0..rows {
            let scores = [gaussian.normal(), gaussian.normal()];
            for ambient_coordinate in 0..ambient {
                data[[row, ambient_coordinate]] = population_tangent[[ambient_coordinate, 0]]
                    * scores[0]
                    + population_tangent[[ambient_coordinate, 1]] * scores[1]
                    + noise_standard_deviation * gaussian.normal();
            }
        }
        let row_ids: Vec<_> = (0..rows).collect();
        let covariance = selected_covariance(data.view(), &row_ids, None).unwrap();
        let (_, eigenvectors) = covariance.eigh(faer::Side::Lower).unwrap();
        let mut tangent = Array2::<f64>::zeros((ambient, INTRINSIC_DIMENSION));
        for column in 0..INTRINSIC_DIMENSION {
            tangent
                .column_mut(column)
                .assign(&eigenvectors.column(ambient - 1 - column));
        }
        tangent
    }

    fn align_tangent_to_population(fitted: &Array2<f64>, population: &Array2<f64>) -> Array2<f64> {
        let cross = fitted.t().dot(population);
        let (left, _, right_t) = cross.svd(true, true).unwrap();
        fitted.dot(&left.unwrap().dot(&right_t.unwrap()))
    }

    fn projected_patch_from_tangent(
        chart: usize,
        inference_rows: usize,
        tangent: Array2<f64>,
    ) -> GaussianPcaPatch {
        let ambient = tangent.nrows();
        let base = chart * 4_000_000;
        GaussianPcaPatch::new(
            chart,
            GaussianPatchRowSplit::from_disjoint_ranges(
                base,
                inference_rows,
                base + 1_000_000,
                inference_rows,
            )
            .unwrap(),
            PilotProjectionProvenance::ExactAnalyticCapture,
            GaussianPatchCentering::MeanEstimatedOnInferenceRows,
            identity_projection(ambient),
            tangent,
            0.01,
            1.0,
            GaussianPcaSpectrumProvenance::CertifiedPopulation(
                GaussianPcaPopulationBounds::new(0.01, 2.0, 1.0).unwrap(),
            ),
        )
        .unwrap()
    }

    fn polar_edge_angle(from: &Array2<f64>, to: &Array2<f64>) -> f64 {
        let cross = to.t().dot(from);
        (cross[[1, 0]] - cross[[0, 1]]).atan2(cross[[0, 0]] + cross[[1, 1]])
    }

    fn fitted_cycle_angle(frames: &[Array2<f64>]) -> f64 {
        let mut holonomy = identity_2();
        for (from, to) in [(0, 1), (1, 2), (2, 0)] {
            let cross = frames[to].t().dot(&frames[from]);
            let (left, _, right_t) = cross.svd(true, true).unwrap();
            let transition = left.unwrap().dot(&right_t.unwrap());
            holonomy = transition.dot(&holonomy);
        }
        (holonomy[[1, 0]] - holonomy[[0, 1]]).atan2(holonomy[[0, 0]] + holonomy[[1, 1]])
    }

    fn wrap_signed_angle(angle: f64) -> f64 {
        (angle + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI
    }

    #[test]
    fn authoritative_edge_constructors_enforce_canonical_identity() {
        assert!(GaussianPcaPopulationBounds::new(f64::MAX, f64::MAX, 1.0).is_err());
        assert!(GaussianPcaPopulationBounds::new(0.0, 1.0, 2.0).is_err());
        assert!(AtlasHolonomyEdgeId::new(1, 1, 0).is_err());
        assert_eq!(
            AtlasHolonomyEdgeId::new(4, 2, 9).unwrap(),
            AtlasHolonomyEdgeId::new(2, 4, 9).unwrap()
        );
        assert!(AtlasSignedEdge::new_analytic(0, 1, 0, 0).is_err());
        assert!(
            ProjectedAtlasEdgeSpec::new(
                0,
                1,
                0,
                PopulationCrossGramProvenance::EstimatedOnly,
                f64::NAN,
            )
            .is_err()
        );
        assert!(
            ExactAnalyticHolonomyCertificate::new(
                2,
                vec![
                    AtlasSignedEdge::new_analytic(0, 1, 3, 1).unwrap(),
                    AtlasSignedEdge::new_analytic(0, 1, 3, -1).unwrap(),
                ],
            )
            .is_err()
        );
        assert!(
            ExactAnalyticHolonomyCertificate::new(
                2,
                vec![AtlasSignedEdge::new_analytic(0, 2, 0, 1).unwrap()],
            )
            .is_err()
        );
    }

    #[test]
    fn fitted_sphere_seam_cannot_construct_an_exact_analytic_edge() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let fitted =
            SphereChartTransition::new_fitted(0, 1, identity, crate::manifold::AtlasSeamKind::Pole)
                .unwrap();
        assert!(AtlasSignedEdge::from_analytic_sphere_transition(&fitted, 0).is_err());

        let analytic = SphereChartTransition::new_analytic(
            0,
            1,
            identity,
            crate::manifold::AtlasSeamKind::Pole,
        )
        .unwrap();
        assert_eq!(
            AtlasSignedEdge::from_analytic_sphere_transition(&analytic, 0)
                .unwrap()
                .sign(),
            1
        );
    }

    #[test]
    fn exact_parallel_overlap_components_form_their_own_orientation_cycle() {
        let certificate = ExactAnalyticHolonomyCertificate::new(
            2,
            vec![
                AtlasSignedEdge::new_analytic(0, 1, 0, 1).unwrap(),
                AtlasSignedEdge::new_analytic(0, 1, 1, -1).unwrap(),
            ],
        )
        .unwrap();
        assert_eq!(
            certificate.orientability(),
            AtlasOrientability::NonOrientable
        );
        let authoritative = AtlasHolonomyCertificate::ExactAnalytic(certificate);
        assert_eq!(
            authoritative.edge_inventory(),
            vec![
                AtlasHolonomyEdgeId::new(0, 1, 0).unwrap(),
                AtlasHolonomyEdgeId::new(0, 1, 1).unwrap(),
            ]
        );
    }

    #[test]
    fn gaussian_certificate_retains_auditable_patch_inputs() {
        let analysis = triangle_analysis([(0.0, false); 3], 0);
        assert_eq!(analysis.patch_summaries().len(), 3);
        let patch = analysis.patch_summaries()[0];
        assert_eq!(patch.chart, 0);
        assert_eq!(patch.projection_fit_rows, 1_000);
        assert_eq!(patch.inference_rows, 1_000_000);
        assert_eq!(patch.covariance_degrees_of_freedom, 999_999);
        assert_eq!(patch.ambient_dimension, 3);
        assert_eq!(patch.retained_dimension, 3);
        assert_eq!(
            patch.centering,
            GaussianPatchCentering::MeanEstimatedOnInferenceRows
        );
        assert_near(patch.projector_variance_scale(), 0.01 * 1.01 / 999_999.0);
    }

    #[test]
    fn shared_inference_rows_require_an_explicit_joint_covariance() {
        let make_patch = |chart, pilot_rows| {
            GaussianPcaPatch::new(
                chart,
                GaussianPatchRowSplit::new(pilot_rows, (100..200).collect()).unwrap(),
                PilotProjectionProvenance::ExactAnalyticCapture,
                GaussianPatchCentering::MeanEstimatedOnInferenceRows,
                identity_projection(3),
                arr2(&[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
                0.01,
                1.0,
                GaussianPcaSpectrumProvenance::CertifiedPopulation(
                    GaussianPcaPopulationBounds::new(0.01, 2.0, 1.0).unwrap(),
                ),
            )
            .unwrap()
        };
        let patches = vec![
            make_patch(0, (0..50).collect()),
            make_patch(1, (50..100).collect()),
        ];

        assert!(GaussianPcaErrorModel::independent(&patches).is_err());
        let dimension = GaussianPcaErrorModel::coordinate_offsets(&patches)
            .last()
            .copied()
            .unwrap();
        let joint = Array2::<f64>::zeros((dimension, dimension));
        let mut tiny_indefinite = joint.clone();
        tiny_indefinite[[0, 0]] = -1.0e-30;
        assert!(
            GaussianPcaErrorModel::certified_joint(
                &patches,
                CrossPatchCovarianceProvenance::ExplicitJointCovariance,
                tiny_indefinite,
            )
            .is_err(),
            "covariance validation must be relative to its own scale"
        );
        assert!(
            GaussianPcaErrorModel::certified_joint(
                &patches,
                CrossPatchCovarianceProvenance::DisjointInferenceRows,
                joint.clone(),
            )
            .is_err()
        );
        let plugin_model = GaussianPcaErrorModel::plugin_joint(
            &patches,
            CrossPatchCovarianceProvenance::ExplicitJointCovariance,
            joint.clone(),
        )
        .unwrap();
        assert_eq!(
            plugin_model.authority(),
            GaussianPcaCovarianceAuthority::AsymptoticPlugIn
        );
        let model = GaussianPcaErrorModel::certified_joint(
            &patches,
            CrossPatchCovarianceProvenance::ExplicitJointCovariance,
            joint,
        )
        .unwrap();
        assert_eq!(
            model.cross_patch_provenance(),
            &CrossPatchCovarianceProvenance::ExplicitJointCovariance
        );
        assert_eq!(
            model.authority(),
            GaussianPcaCovarianceAuthority::CertifiedGaussianLinearization
        );
    }

    #[test]
    fn fitted_patch_scalars_can_only_build_an_asymptotic_plugin_covariance() {
        let patches = vec![
            patch(0, 0.0, 0.0, false, 1_000, 0),
            patch(1, 0.2, 0.0, false, 1_000, 0),
        ];
        let model = GaussianPcaErrorModel::independent(&patches).unwrap();
        assert_eq!(
            model.authority(),
            GaussianPcaCovarianceAuthority::AsymptoticPlugIn
        );
    }

    #[test]
    fn nonnested_pilot_frames_use_the_retained_normal_cross_operator() {
        let frame_a = arr2(&[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ]);
        let frame_b = arr2(&[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]);
        let tangent = arr2(&[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]);
        let make_patch = |chart, frame| {
            GaussianPcaPatch::new(
                chart,
                GaussianPatchRowSplit::from_disjoint_ranges(
                    chart * 1_000,
                    100,
                    chart * 1_000 + 500,
                    100,
                )
                .unwrap(),
                PilotProjectionProvenance::ExactAnalyticCapture,
                GaussianPatchCentering::MeanEstimatedOnInferenceRows,
                frame,
                tangent.clone(),
                0.01,
                1.0,
                GaussianPcaSpectrumProvenance::CertifiedPopulation(
                    GaussianPcaPopulationBounds::new(0.01, 2.0, 1.0).unwrap(),
                ),
            )
            .unwrap()
        };
        let patches = vec![make_patch(0, frame_a), make_patch(1, frame_b)];
        let edge = build_projected_edge(&patches, certified_edge(0, 1, 0, 0.0)).unwrap();
        let normal = identity_square(3) - tangent.dot(&tangent.t());
        let normal_cross_operator = normal.dot(&edge.projection_cross_gram_ba).dot(&normal);

        assert_eq!(edge.public.projected_dimension, 4);
        assert_near(frobenius_squared(normal_cross_operator.view()), 0.0);
        assert_near(
            frobenius_squared(edge.patch_gradient_a.unwrap().view()),
            0.0,
        );
        assert_near(
            frobenius_squared(edge.patch_gradient_b.unwrap().view()),
            0.0,
        );
    }

    #[test]
    fn zero_tilt_parallel_cycle_preserves_overlap_identity_and_refuses_normal_law() {
        let analysis = certified_analysis(
            vec![
                patch(0, 0.0, 0.0, false, 1_000_000, 0),
                patch(1, 0.0, 0.0, false, 1_000_000, 0),
            ],
            vec![certified_edge(0, 1, 7, 0.0), certified_edge(0, 1, 3, 0.0)],
            AtlasFamilywiseLevel::new(0.05).unwrap(),
            None,
        );
        assert_eq!(analysis.cycles().len(), 1);
        assert_eq!(analysis.cycles()[0].closed_chart_walk(), vec![0, 1, 0]);
        assert_eq!(
            analysis.cycles()[0]
                .steps()
                .iter()
                .copied()
                .map(AtlasHolonomyCycleStep::edge)
                .collect::<Vec<_>>(),
            vec![
                AtlasHolonomyEdgeId::new(0, 1, 3).unwrap(),
                AtlasHolonomyEdgeId::new(0, 1, 7).unwrap(),
            ]
        );
        assert_eq!(
            analysis.cycles()[0]
                .steps()
                .iter()
                .copied()
                .map(AtlasHolonomyCycleStep::direction)
                .collect::<Vec<_>>(),
            vec![
                AtlasHolonomyEdgeDirection::AToB,
                AtlasHolonomyEdgeDirection::BToA,
            ]
        );
        let cycle = &analysis.cycles()[0];
        assert_near(cycle.first_order_variance.unwrap(), 0.0);
        assert_near(cycle.bilinear_quadratic_variance_diagnostic.unwrap(), 0.0);
        assert!(matches!(
            cycle.asymptotic_regime,
            Some(AtlasCycleAsymptoticRegime::FirstOrderDegenerate { .. })
        ));
        assert!(matches!(
            cycle.decision.refusals(),
            [AtlasStatisticalRefusal::DegenerateFirstOrderLimitUnresolved { .. }]
        ));
    }

    #[test]
    fn singular_parallel_overlap_refusals_name_each_edge_identity() {
        let analysis = certified_analysis(
            vec![
                patch(0, 0.0, 0.0, false, 1_000_000, 0),
                patch(1, std::f64::consts::FRAC_PI_2, 0.0, false, 1_000_000, 0),
            ],
            vec![certified_edge(0, 1, 7, 0.0), certified_edge(0, 1, 3, 0.0)],
            AtlasFamilywiseLevel::new(0.05).unwrap(),
            None,
        );
        let refused_edges: BTreeSet<_> = analysis
            .orientation()
            .refusals()
            .iter()
            .filter_map(|reason| match reason {
                AtlasStatisticalRefusal::SingularProjectedCrossGram { edge, .. } => Some(*edge),
                _ => None,
            })
            .collect();
        assert_eq!(
            refused_edges,
            BTreeSet::from([
                AtlasHolonomyEdgeId::new(0, 1, 3).unwrap(),
                AtlasHolonomyEdgeId::new(0, 1, 7).unwrap(),
            ])
        );
    }

    #[test]
    fn projector_cycle_is_invariant_to_o2_patch_gauges() {
        let reference = triangle_analysis([(0.0, false); 3], 0);
        let regauged = triangle_analysis([(0.37, true), (-0.91, false), (1.23, true)], 0);
        assert_eq!(
            reference.orientation().certified_value(),
            regauged.orientation().certified_value()
        );
        assert_eq!(reference.cycles().len(), 1);
        assert_eq!(regauged.cycles().len(), 1);
        let left = &reference.cycles()[0];
        let right = &regauged.cycles()[0];
        assert_near(left.absolute_angle.unwrap(), right.absolute_angle.unwrap());
        assert_near(
            left.first_order_variance.unwrap(),
            right.first_order_variance.unwrap(),
        );
        assert_near(
            left.bilinear_quadratic_variance_diagnostic.unwrap(),
            right.bilinear_quadratic_variance_diagnostic.unwrap(),
        );
    }

    #[test]
    fn shared_patch_covariance_is_aggregated_before_the_quadratic_form() {
        let analysis = triangle_analysis([(0.0, false); 3], 0);
        let cycle = &analysis.cycles()[0];
        let correct = cycle.first_order_variance.unwrap();
        let naive = cycle.naive_edgewise_first_order_variance.unwrap();
        let adjustment = cycle.covariance_aggregation_adjustment.unwrap();
        assert_near(correct - naive, adjustment);
        assert!(adjustment.abs() > f64::EPSILON);
    }

    #[test]
    fn unrelated_uncertified_patch_does_not_refuse_another_cycle() {
        let mut unrelated = patch(3, 0.0, 0.0, false, 1_000_000, 0);
        unrelated.pilot_projection = PilotProjectionProvenance::IndependentPilotEstimate;
        unrelated.spectrum_provenance = GaussianPcaSpectrumProvenance::PlugInEstimate {
            noise_variance: 0.01,
            signal_variance: 1.0,
            eigengap: 1.0,
        };
        let analysis = certified_analysis(
            vec![
                patch(0, 0.0, 0.0, false, 1_000_000, 0),
                patch(1, 0.2, 0.0, false, 1_000_000, 0),
                patch(2, -0.15, 0.0, false, 1_000_000, 0),
                unrelated,
            ],
            triangle_edges(),
            AtlasFamilywiseLevel::new(0.05).unwrap(),
            None,
        );

        assert_eq!(analysis.cycles().len(), 1);
        assert!(
            analysis.cycles()[0]
                .decision
                .refusals()
                .iter()
                .all(|reason| !matches!(
                    reason,
                    AtlasStatisticalRefusal::PilotProjectionUncertified { chart: 3 }
                        | AtlasStatisticalRefusal::PopulationSpectrumUncertified { chart: 3 }
                ))
        );
    }

    #[test]
    fn independent_projector_simulation_calibrates_cycle_plugin_variance() {
        const REPLICATES: usize = 2_048;
        let ambient = 4;
        let inference_rows = 100_000;
        let true_tangents = [
            plane_tangent(0.0, ambient),
            plane_tangent(0.2, ambient),
            plane_tangent(-0.15, ambient),
        ];
        let base = certified_analysis(
            true_tangents
                .iter()
                .cloned()
                .enumerate()
                .map(|(chart, tangent)| {
                    projected_patch_from_tangent(chart, inference_rows, tangent)
                })
                .collect(),
            triangle_edges(),
            AtlasFamilywiseLevel::new(0.05).unwrap(),
            None,
        );
        let cycle = &base.cycles()[0];
        let plugin_standard_error = cycle.standard_error.unwrap();
        assert!(plugin_standard_error > 0.0);
        let perturbation_sd = base.patch_summaries()[0].projector_variance_scale().sqrt();
        let gaussian_boundary =
            cycle_rejection_boundary(plugin_standard_error, 0.0, 0.0, cycle.gaussian_error_budget)
                .unwrap();

        let mut gaussian = DeterministicGaussian::new(0x2311_0001);
        let mut sum = 0.0;
        let mut sum_squares = 0.0;
        let mut rejections = 0usize;
        for _ in 0..REPLICATES {
            let fitted: Vec<_> = true_tangents
                .iter()
                .map(|tangent| perturb_tangent(tangent, perturbation_sd, &mut gaussian))
                .collect();
            let angle = wrap_signed_angle(
                polar_edge_angle(&fitted[0], &fitted[1])
                    + polar_edge_angle(&fitted[1], &fitted[2])
                    + polar_edge_angle(&fitted[2], &fitted[0]),
            );
            sum += angle;
            sum_squares += angle * angle;
            rejections += usize::from(angle.abs() > gaussian_boundary);
        }
        let mean = sum / REPLICATES as f64;
        let empirical_sd = (sum_squares / REPLICATES as f64 - mean * mean)
            .max(0.0)
            .sqrt();
        assert!(
            (empirical_sd / plugin_standard_error - 1.0).abs() <= 0.15,
            "plugin sd={plugin_standard_error:.6e}, independent Monte-Carlo sd={empirical_sd:.6e}"
        );
        let rejection_rate = rejections as f64 / REPLICATES as f64;
        let nominal = cycle.gaussian_error_budget;
        let binomial_standard_error = (nominal * (1.0 - nominal) / REPLICATES as f64).sqrt();
        eprintln!(
            "ATLAS_CALIBRATION linearized_projectors replicates={REPLICATES} plugin_sd={plugin_standard_error:.9e} empirical_sd={empirical_sd:.9e} nominal={nominal:.6} rejection_rate={rejection_rate:.6}"
        );
        assert!(
            (rejection_rate - nominal).abs() <= 5.0 * binomial_standard_error,
            "nominal={nominal:.6}, empirical rejection={rejection_rate:.6}"
        );
    }

    #[test]
    fn actual_spiked_rows_sample_covariance_and_pca_calibrate_cycle_plugin_variance() {
        const REPLICATES: usize = 384;
        const ROWS_PER_PATCH: usize = 512;
        let true_tangents = [
            plane_tangent(0.0, 3),
            plane_tangent(0.2, 3),
            plane_tangent(-0.15, 3),
        ];
        let patches: Vec<_> = true_tangents
            .iter()
            .cloned()
            .enumerate()
            .map(|(chart, tangent)| projected_patch_from_tangent(chart, ROWS_PER_PATCH, tangent))
            .collect();
        let error_model = GaussianPcaErrorModel::independent(&patches).unwrap();
        let analysis = GaussianPcaHolonomyAnalysis::certify(
            patches,
            triangle_edges(),
            error_model,
            AtlasFamilywiseLevel::new(0.05).unwrap(),
            None,
        )
        .unwrap();
        let plugin_standard_error = analysis.cycles()[0].standard_error.unwrap();
        let rejection_boundary =
            cycle_rejection_boundary(plugin_standard_error, 0.0, 0.0, 0.05).unwrap();
        assert!(
            analysis.cycles()[0]
                .decision
                .refusals()
                .iter()
                .any(|reason| {
                    matches!(
                        reason,
                        AtlasStatisticalRefusal::GaussianLinearizationIsPlugin { .. }
                    )
                })
        );

        let mut gaussian = DeterministicGaussian::new(0x2311_0005);
        let mut sum = 0.0;
        let mut sum_squares = 0.0;
        let mut rejections = 0usize;
        for _ in 0..REPLICATES {
            let fitted: Vec<_> = true_tangents
                .iter()
                .map(|tangent| {
                    sample_spiked_pca_tangent(tangent, ROWS_PER_PATCH, 0.1, &mut gaussian)
                })
                .collect();
            let angle = fitted_cycle_angle(&fitted);
            sum += angle;
            sum_squares += angle * angle;
            rejections += usize::from(angle.abs() > rejection_boundary);
        }
        let mean = sum / REPLICATES as f64;
        let empirical_sd = (sum_squares / REPLICATES as f64 - mean * mean)
            .max(0.0)
            .sqrt();
        assert!(
            (empirical_sd / plugin_standard_error - 1.0).abs() <= 0.20,
            "plugin sd={plugin_standard_error:.6e}, actual-row PCA sd={empirical_sd:.6e}"
        );
        let rejection_rate = rejections as f64 / REPLICATES as f64;
        let nominal = 0.05;
        let binomial_standard_error = (nominal * (1.0 - nominal) / REPLICATES as f64).sqrt();
        eprintln!(
            "ATLAS_CALIBRATION actual_row_pca replicates={REPLICATES} rows_per_patch={ROWS_PER_PATCH} plugin_sd={plugin_standard_error:.9e} empirical_sd={empirical_sd:.9e} nominal={nominal:.6} rejection_rate={rejection_rate:.6}"
        );
        assert!(
            (rejection_rate - nominal).abs() <= 5.0 * binomial_standard_error,
            "nominal={nominal:.6}, actual-row PCA rejection={rejection_rate:.6}"
        );
    }

    #[test]
    fn two_sided_cycle_test_has_nominal_size_and_closed_form_power() {
        const REPLICATES: usize = 4_096;
        let alpha = 0.05;
        let standard_error = 0.2;
        let boundary = cycle_rejection_boundary(standard_error, 0.0, 0.0, alpha).unwrap();
        let critical = boundary / standard_error;
        let noncentrality = 3.0;
        let normal = Normal::new(0.0, 1.0).unwrap();
        let expected_power =
            1.0 - normal.cdf(critical - noncentrality) + normal.cdf(-critical - noncentrality);
        let mut gaussian = DeterministicGaussian::new(0x2311_0002);
        let mut null_rejections = 0usize;
        let mut alternative_rejections = 0usize;
        for _ in 0..REPLICATES {
            null_rejections += usize::from((standard_error * gaussian.normal()).abs() > boundary);
            alternative_rejections += usize::from(
                (standard_error * (noncentrality + gaussian.normal())).abs() > boundary,
            );
        }
        let null_rate = null_rejections as f64 / REPLICATES as f64;
        let power = alternative_rejections as f64 / REPLICATES as f64;
        let null_mc_se = (alpha * (1.0 - alpha) / REPLICATES as f64).sqrt();
        let power_mc_se = (expected_power * (1.0 - expected_power) / REPLICATES as f64).sqrt();
        eprintln!(
            "ATLAS_CALIBRATION gaussian_oracle replicates={REPLICATES} nominal={alpha:.6} null_rate={null_rate:.6} expected_power={expected_power:.6} observed_power={power:.6}"
        );
        assert!((null_rate - alpha).abs() <= 5.0 * null_mc_se);
        assert!((power - expected_power).abs() <= 5.0 * power_mc_se);
        assert!(power > null_rate);
    }

    #[test]
    fn observed_orientation_flips_do_not_exceed_finite_sample_bound() {
        const REPLICATES: usize = 2_048;
        let inference_rows = 1_000_000;
        let true_a = plane_tangent(0.0, 3);
        let true_b = plane_tangent(0.2, 3);
        let analysis = certified_analysis(
            vec![
                projected_patch_from_tangent(0, inference_rows, true_a.clone()),
                projected_patch_from_tangent(1, inference_rows, true_b.clone()),
            ],
            vec![certified_edge(0, 1, 0, 0.0)],
            AtlasFamilywiseLevel::new(0.05).unwrap(),
            None,
        );
        assert!(analysis.orientation().certified_value().is_some());
        let bound = analysis.orientation_flip_probability_bound().unwrap();
        let perturbation_sd = analysis.patch_summaries()[0]
            .projector_variance_scale()
            .sqrt();
        let mut gaussian = DeterministicGaussian::new(0x2311_0003);
        let mut flips = 0usize;
        for _ in 0..REPLICATES {
            let fitted_a = perturb_tangent(&true_a, perturbation_sd, &mut gaussian);
            let fitted_b = perturb_tangent(&true_b, perturbation_sd, &mut gaussian);
            flips += usize::from(determinant_2(fitted_b.t().dot(&fitted_a).view()) < 0.0);
        }
        let observed = flips as f64 / REPLICATES as f64;
        eprintln!(
            "ATLAS_CALIBRATION linearized_orientation replicates={REPLICATES} flips={flips} observed_rate={observed:.9e} bound={bound:.9e}"
        );
        assert!(
            observed <= bound + 1.0 / REPLICATES as f64,
            "observed flip rate {observed:.6e} exceeded finite-sample bound {bound:.6e}"
        );
    }

    #[test]
    fn near_margin_wishart_pca_orientation_sweep_respects_the_bound() {
        const REPLICATES: usize = 256;
        let population_a = plane_tangent(0.0, 3);
        let separation = 1.45_f64;
        let population_b = plane_tangent(separation, 3);
        let mut gaussian = DeterministicGaussian::new(0x2311_0006);
        let mut smallest_row_flips = 0usize;

        for rows in [4, 16, 64] {
            let make_patch = |chart, tangent| {
                GaussianPcaPatch::new(
                    chart,
                    GaussianPatchRowSplit::from_disjoint_ranges(
                        chart * 10_000,
                        rows,
                        chart * 10_000 + 5_000,
                        rows,
                    )
                    .unwrap(),
                    PilotProjectionProvenance::ExactAnalyticCapture,
                    GaussianPatchCentering::MeanEstimatedOnInferenceRows,
                    identity_projection(3),
                    tangent,
                    4.0,
                    1.0,
                    GaussianPcaSpectrumProvenance::CertifiedPopulation(
                        GaussianPcaPopulationBounds::new(4.0, 5.0, 1.0).unwrap(),
                    ),
                )
                .unwrap()
            };
            let patches = vec![
                make_patch(0, population_a.clone()),
                make_patch(1, population_b.clone()),
            ];
            let error_model = GaussianPcaErrorModel::independent(&patches).unwrap();
            let edge = ProjectedAtlasEdgeSpec::new(
                0,
                1,
                0,
                PopulationCrossGramProvenance::CertifiedSmallestSingularValue {
                    lower_bound: separation.cos(),
                },
                0.0,
            )
            .unwrap();
            let analysis = GaussianPcaHolonomyAnalysis::certify(
                patches,
                vec![edge],
                error_model,
                AtlasFamilywiseLevel::new(0.05).unwrap(),
                None,
            )
            .unwrap();
            let bound = analysis.orientation_flip_probability_bound().unwrap();
            let mut flips = 0usize;
            for _ in 0..REPLICATES {
                let fitted_a = sample_spiked_pca_tangent(&population_a, rows, 2.0, &mut gaussian);
                let fitted_b = sample_spiked_pca_tangent(&population_b, rows, 2.0, &mut gaussian);
                let aligned_a = align_tangent_to_population(&fitted_a, &population_a);
                let aligned_b = align_tangent_to_population(&fitted_b, &population_b);
                flips += usize::from(determinant_2(aligned_b.t().dot(&aligned_a).view()) < 0.0);
            }
            if rows == 4 {
                smallest_row_flips = flips;
            }
            let observed = flips as f64 / REPLICATES as f64;
            eprintln!(
                "ATLAS_CALIBRATION wishart_orientation rows={rows} replicates={REPLICATES} flips={flips} observed_rate={observed:.9e} bound={bound:.9e}"
            );
            assert!(
                observed <= bound + 1.0 / REPLICATES as f64,
                "rows={rows}, observed Wishart-PCA flips={observed:.6}, bound={bound:.6}"
            );
        }
        assert!(
            smallest_row_flips > 0,
            "near-margin low-occupancy sweep must actually enter the flip regime"
        );
    }

    #[test]
    fn observed_cross_gram_margin_cannot_set_its_own_flip_bound() {
        let patches = vec![
            patch(0, 0.0, 0.0, false, 1_000_000, 0),
            patch(1, 0.2, 0.0, false, 1_000_000, 0),
        ];
        let error_model = GaussianPcaErrorModel::independent(&patches).unwrap();
        let analysis = GaussianPcaHolonomyAnalysis::certify(
            patches,
            vec![
                ProjectedAtlasEdgeSpec::new(
                    0,
                    1,
                    0,
                    PopulationCrossGramProvenance::EstimatedOnly,
                    0.0,
                )
                .unwrap(),
            ],
            error_model,
            AtlasFamilywiseLevel::new(0.05).unwrap(),
            None,
        )
        .unwrap();

        assert_eq!(analysis.orientation_flip_probability_bound(), None);
        assert!(matches!(
            analysis.orientation().refusals(),
            [AtlasStatisticalRefusal::PopulationCrossGramMarginUncertified { .. }]
        ));
        assert!(analysis.sample_prescription().iter().all(|entry| matches!(
            entry.inference,
            AtlasInferenceOccupancyPrescription::PopulationTailInputsRequired
        )));
    }

    #[test]
    fn tiny_certified_margin_has_typed_unrepresentable_occupancy() {
        let analysis = certified_analysis(
            vec![
                patch(0, 0.0, 0.0, false, 1_000_000, 0),
                patch(1, 0.2, 0.0, false, 1_000_000, 0),
            ],
            vec![
                ProjectedAtlasEdgeSpec::new(
                    0,
                    1,
                    0,
                    PopulationCrossGramProvenance::CertifiedSmallestSingularValue {
                        lower_bound: f64::MIN_POSITIVE,
                    },
                    0.0,
                )
                .unwrap(),
            ],
            AtlasFamilywiseLevel::new(0.05).unwrap(),
            None,
        );

        assert!(analysis.sample_prescription().iter().all(|prescription| {
            matches!(
                prescription.inference,
                AtlasInferenceOccupancyPrescription::RequiredRowsExceedRepresentableRange {
                    projected_dimension: 3,
                    aligned_frame_error_budget,
                } if aligned_frame_error_budget.is_finite()
                    && aligned_frame_error_budget > 0.0
            )
        }));
    }

    #[test]
    fn ambient_zero_padding_does_not_enter_projected_variance_or_tail() {
        let base = triangle_analysis([(0.0, false); 3], 0);
        let padded = triangle_analysis([(0.0, false); 3], 97);
        let base_ranks: Vec<_> = base
            .edges()
            .iter()
            .map(|edge| edge.projected_dimension)
            .collect();
        let padded_ranks: Vec<_> = padded
            .edges()
            .iter()
            .map(|edge| edge.projected_dimension)
            .collect();
        assert_eq!(base_ranks, padded_ranks);
        eprintln!(
            "ATLAS_PROJECTION ambient_base=3 ambient_padded=100 projected_ranks={base_ranks:?} base_variance={:.9e} padded_variance={:.9e} base_tail={:.9e} padded_tail={:.9e}",
            base.cycles()[0].first_order_variance.unwrap(),
            padded.cycles()[0].first_order_variance.unwrap(),
            base.orientation_flip_probability_bound().unwrap(),
            padded.orientation_flip_probability_bound().unwrap(),
        );
        assert_near(
            base.cycles()[0].first_order_variance.unwrap(),
            padded.cycles()[0].first_order_variance.unwrap(),
        );
        assert_near(
            base.orientation_flip_probability_bound().unwrap(),
            padded.orientation_flip_probability_bound().unwrap(),
        );
        assert_eq!(
            base.cycles()[0].decision.certified_value(),
            padded.cycles()[0].decision.certified_value()
        );
        assert_near(
            base.cycles()[0]
                .bilinear_quadratic_variance_diagnostic
                .unwrap(),
            padded.cycles()[0]
                .bilinear_quadratic_variance_diagnostic
                .unwrap(),
        );
    }

    #[test]
    fn orientation_bound_decreases_with_rows_and_prescription_is_closed_form() {
        let build = |rows| {
            certified_analysis(
                vec![
                    patch(0, 0.0, 0.0, false, rows, 0),
                    patch(1, 0.2, 0.0, false, rows, 0),
                ],
                vec![certified_edge(0, 1, 0, 0.0)],
                AtlasFamilywiseLevel::new(0.05).unwrap(),
                None,
            )
        };
        let small = build(16);
        let large = build(1_000_000);
        assert!(
            large.orientation_flip_probability_bound().unwrap()
                < small.orientation_flip_probability_bound().unwrap()
        );
        let small_prescription = &small.sample_prescription()[0];
        let large_prescription = &large.sample_prescription()[0];
        let small_required = required_inference_prescription(small_prescription);
        let large_required = required_inference_prescription(large_prescription);
        assert_eq!(
            small_required.0, large_required.0,
            "the requested inference occupancy is a property of the target error level"
        );
        assert!(small_prescription.current_inference_rows < small_required.0);
        assert_eq!(small_prescription.current_pilot_rows, 1_000);
        assert_eq!(
            small_prescription.pilot,
            AtlasPilotOccupancyPrescription::ExactCaptureNoSamplingRequirement,
        );
        assert!(large.orientation().certified_value().is_some());
    }

    #[test]
    fn pilot_capture_and_inference_occupancy_are_separate_requirements() {
        let mut estimated_pilot = patch(0, 0.0, 0.0, false, 1_000_000, 0);
        estimated_pilot.pilot_projection = PilotProjectionProvenance::IndependentPilotEstimate;
        let analysis = certified_analysis(
            vec![estimated_pilot, patch(1, 0.2, 0.0, false, 1_000_000, 0)],
            vec![certified_edge(0, 1, 0, 0.0)],
            AtlasFamilywiseLevel::new(0.05).unwrap(),
            None,
        );

        let estimated = analysis
            .sample_prescription()
            .iter()
            .find(|entry| entry.chart == 0)
            .unwrap();
        assert_eq!(estimated.current_pilot_rows, 1_000);
        assert_eq!(
            estimated.pilot,
            AtlasPilotOccupancyPrescription::PopulationCaptureTheoremRequired
        );
        assert!(matches!(
            estimated.inference,
            AtlasInferenceOccupancyPrescription::Required { .. }
        ));
        assert!(matches!(
            analysis.orientation().refusals(),
            [AtlasStatisticalRefusal::PilotProjectionUncertified { chart: 0 }]
        ));
    }

    #[test]
    fn orientation_occupancy_counts_distinct_projected_edge_incidences() {
        let level = AtlasFamilywiseLevel::new(0.05).unwrap();
        let single = certified_analysis(
            vec![
                patch(0, 0.0, 0.0, false, 1_000_000, 0),
                patch(1, 0.2, 0.0, false, 1_000_000, 0),
            ],
            vec![certified_edge(0, 1, 0, 0.0)],
            level,
            None,
        );
        let two_incident_edges = certified_analysis(
            vec![
                patch(0, 0.0, 0.0, false, 1_000_000, 0),
                patch(1, 0.2, 0.0, false, 1_000_000, 0),
                patch(2, -0.2, 0.0, false, 1_000_000, 0),
            ],
            vec![certified_edge(0, 1, 0, 0.0), certified_edge(0, 2, 0, 0.0)],
            level,
            None,
        );
        let single_center = single
            .sample_prescription()
            .iter()
            .find(|entry| entry.chart == 0)
            .unwrap();
        let two_edge_center = two_incident_edges
            .sample_prescription()
            .iter()
            .find(|entry| entry.chart == 0)
            .unwrap();
        let single_required = required_inference_prescription(single_center);
        let two_edge_required = required_inference_prescription(two_edge_center);
        assert!(
            two_edge_required.1 > single_required.1,
            "the union bound must spend error probability on both non-nested edge projections"
        );

        let edge = &single.edges()[0];
        let frame_budget = single_required.3;
        assert_near(
            2.0 * frame_budget + frame_budget * frame_budget,
            edge.population_cross_gram.certified_lower_bound().unwrap(),
        );
    }

    fn cancellation_gauss_bonnet(remainder: f64) -> GaussBonnetInput {
        GaussBonnetInput::certified_independent_gaussian(
            vec![GaussBonnetNoiseSource::new(7, arr2(&[[1.0]])).unwrap()],
            vec![
                GaussBonnetContribution::new(
                    std::f64::consts::PI,
                    remainder,
                    0.0,
                    vec![GaussBonnetSourceGradient::new(7, array![1.0]).unwrap()],
                )
                .unwrap(),
                GaussBonnetContribution::new(
                    std::f64::consts::PI,
                    0.0,
                    0.0,
                    vec![GaussBonnetSourceGradient::new(7, array![-1.0]).unwrap()],
                )
                .unwrap(),
            ],
        )
        .unwrap()
    }

    fn nondegenerate_gauss_bonnet(
        remainder: f64,
        authority: GaussBonnetCovarianceAuthority,
    ) -> GaussBonnetInput {
        let sources = vec![GaussBonnetNoiseSource::new(7, arr2(&[[1.0]])).unwrap()];
        let contributions = vec![
            GaussBonnetContribution::new(
                std::f64::consts::TAU,
                remainder,
                0.0,
                vec![GaussBonnetSourceGradient::new(7, array![1.0]).unwrap()],
            )
            .unwrap(),
        ];
        match authority {
            GaussBonnetCovarianceAuthority::CertifiedIndependentGaussianSources => {
                GaussBonnetInput::certified_independent_gaussian(sources, contributions).unwrap()
            }
            GaussBonnetCovarianceAuthority::AsymptoticPlugIn => {
                GaussBonnetInput::asymptotic_plugin(sources, contributions).unwrap()
            }
        }
    }

    #[test]
    fn gauss_bonnet_cancellation_is_aggregated_and_refuses_a_degenerate_normal_law() {
        let confidence = gauss_bonnet_confidence(&cancellation_gauss_bonnet(0.0), 0.05).unwrap();
        assert_eq!(confidence.first_order_variance, 0.0);
        assert_eq!(confidence.naive_contribution_variance, 2.0);
        assert_eq!(confidence.shared_source_covariance_adjustment, -2.0);
        assert_eq!(confidence.standard_error, None);
        assert_eq!(confidence.misround_probability_bound, None);
        assert!(matches!(
            confidence.decision.refusals(),
            [AtlasStatisticalRefusal::GaussBonnetFirstOrderLimitDegenerate { .. }]
        ));
    }

    #[test]
    fn gauss_bonnet_refuses_when_remainder_consumes_rounding_cell() {
        let confidence = gauss_bonnet_confidence(
            &nondegenerate_gauss_bonnet(
                std::f64::consts::PI,
                GaussBonnetCovarianceAuthority::CertifiedIndependentGaussianSources,
            ),
            0.05,
        )
        .unwrap();
        assert!(confidence.decision.certified_value().is_none());
        assert!(matches!(
            confidence.decision.refusals(),
            [AtlasStatisticalRefusal::GaussBonnetRoundingMarginExhausted { .. }]
        ));
    }

    #[test]
    fn gauss_bonnet_plugin_covariance_cannot_mint_a_frequentist_certificate() {
        let confidence = gauss_bonnet_confidence(
            &nondegenerate_gauss_bonnet(0.0, GaussBonnetCovarianceAuthority::AsymptoticPlugIn),
            0.05,
        )
        .unwrap();
        assert_eq!(confidence.misround_probability_bound, None);
        assert!(matches!(
            confidence.decision.refusals(),
            [AtlasStatisticalRefusal::GaussBonnetGaussianLinearizationIsPlugin]
        ));
    }

    #[test]
    fn gauss_bonnet_degeneracy_is_relative_to_the_covariance_scale() {
        let variance = 1.0e-30;
        assert!(
            GaussBonnetNoiseSource::new(0, arr2(&[[-variance]])).is_err(),
            "a tiny negative variance must not hide below an absolute tolerance"
        );
        let input = GaussBonnetInput::certified_independent_gaussian(
            vec![GaussBonnetNoiseSource::new(0, arr2(&[[variance]])).unwrap()],
            vec![
                GaussBonnetContribution::new(
                    std::f64::consts::TAU,
                    0.0,
                    0.0,
                    vec![GaussBonnetSourceGradient::new(0, array![1.0]).unwrap()],
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let confidence = gauss_bonnet_confidence(&input, 0.05).unwrap();

        assert_eq!(confidence.first_order_variance, variance);
        assert_eq!(confidence.standard_error, Some(variance.sqrt()));
        assert!(confidence.decision.certified_value().is_some());
        assert!(confidence.decision.refusals().is_empty());
    }

    #[test]
    fn integer_euler_confidence_handles_shared_gaussian_curvature_and_remainder() {
        const REPLICATES: usize = 4_096;
        let requested_alpha = 0.05;
        let target_misround_probability = 0.04;
        let deterministic_remainder = 0.2;
        let normal = Normal::new(0.0, 1.0).unwrap();
        let critical = normal.inverse_cdf(1.0 - target_misround_probability / 2.0);
        let standard_error = (std::f64::consts::PI - deterministic_remainder) / critical;
        let source_standard_deviation = standard_error / 1.5;
        let true_chi = AtlasEulerCharacteristic(2);
        let template = GaussBonnetInput::certified_independent_gaussian(
            vec![
                GaussBonnetNoiseSource::new(
                    0,
                    arr2(&[[source_standard_deviation * source_standard_deviation]]),
                )
                .unwrap(),
            ],
            vec![
                GaussBonnetContribution::new(
                    std::f64::consts::PI * true_chi.value() as f64,
                    deterministic_remainder / 2.0,
                    0.0,
                    vec![GaussBonnetSourceGradient::new(0, array![1.0]).unwrap()],
                )
                .unwrap(),
                GaussBonnetContribution::new(
                    std::f64::consts::PI * true_chi.value() as f64,
                    deterministic_remainder / 2.0,
                    0.0,
                    vec![GaussBonnetSourceGradient::new(0, array![0.5]).unwrap()],
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let template_confidence = gauss_bonnet_confidence(&template, requested_alpha).unwrap();
        assert!(template_confidence.shared_source_covariance_adjustment > 0.0);
        assert_near(
            template_confidence.misround_probability_bound.unwrap(),
            target_misround_probability,
        );
        let mut gaussian = DeterministicGaussian::new(0x2311_0004);
        let mut correctly_rounded = 0usize;
        for _ in 0..REPLICATES {
            let mut input = template.clone();
            input.contributions[0].curvature_estimate = std::f64::consts::TAU
                * true_chi.value() as f64
                - input.contributions[1].curvature_estimate
                + deterministic_remainder
                + standard_error * gaussian.normal();
            let confidence = gauss_bonnet_confidence(&input, requested_alpha).unwrap();
            assert!(confidence.shared_source_covariance_adjustment > 0.0);
            if confidence.decision.certified_value().is_some() {
                assert_eq!(
                    confidence.decision.error_probability_bound(),
                    confidence.misround_probability_bound
                );
                assert!(confidence.misround_probability_bound.unwrap() <= requested_alpha);
            } else {
                assert!(matches!(
                    confidence.decision.refusals(),
                    [AtlasStatisticalRefusal::GaussBonnetRoundingMarginExhausted { .. }]
                ));
            }
            correctly_rounded += usize::from(confidence.nearest_integer_candidate == true_chi);
        }
        let wrong = REPLICATES - correctly_rounded;
        let observed = wrong as f64 / REPLICATES as f64;
        let binomial_standard_error =
            (target_misround_probability * (1.0 - target_misround_probability) / REPLICATES as f64)
                .sqrt();
        eprintln!(
            "ATLAS_CALIBRATION gauss_bonnet replicates={REPLICATES} target_bound={target_misround_probability:.6} observed_misround={observed:.6} shared_covariance_adjustment={:.9e} deterministic_remainder={deterministic_remainder:.6}",
            template_confidence.shared_source_covariance_adjustment
        );
        assert!(
            observed <= target_misround_probability + 5.0 * binomial_standard_error,
            "declared upper bound={target_misround_probability:.6}, observed={observed:.6}"
        );
    }
}

#[derive(Clone, Debug)]
struct EdgeWork {
    public: ProjectedAtlasEdgeGeometry,
    transition: Option<Array2<f64>>,
    smallest_singular_value: f64,
    numerical_rank_threshold: f64,
    angle_gradient: Option<Array2<f64>>,
    patch_gradient_a: Option<Array2<f64>>,
    patch_gradient_b: Option<Array2<f64>>,
    projection_cross_gram_ba: Array2<f64>,
}

#[derive(Clone, Debug)]
struct FundamentalCycle {
    steps: Vec<(usize, bool)>,
}

fn orientability_from_edges(chart_count: usize, edges: &[AtlasSignedEdge]) -> AtlasOrientability {
    let mut adjacency = vec![Vec::<(usize, i8)>::new(); chart_count];
    for edge in edges {
        adjacency[edge.a].push((edge.b, edge.sign));
        adjacency[edge.b].push((edge.a, edge.sign));
    }
    let mut orientations = vec![None; chart_count];
    for root in 0..chart_count {
        if orientations[root].is_some() {
            continue;
        }
        orientations[root] = Some(1_i8);
        let mut queue = VecDeque::from([(root, 1_i8)]);
        while let Some((chart, here)) = queue.pop_front() {
            for &(next, sign) in &adjacency[chart] {
                let required = here * sign;
                match orientations[next] {
                    Some(existing) if existing != required => {
                        return AtlasOrientability::NonOrientable;
                    }
                    Some(_) => {}
                    None => {
                        orientations[next] = Some(required);
                        queue.push_back((next, required));
                    }
                }
            }
        }
    }
    AtlasOrientability::Orientable
}

fn determinant_2(matrix: ArrayView2<'_, f64>) -> f64 {
    matrix[[0, 0]] * matrix[[1, 1]] - matrix[[0, 1]] * matrix[[1, 0]]
}

fn identity_2() -> Array2<f64> {
    let mut identity = Array2::<f64>::zeros((INTRINSIC_DIMENSION, INTRINSIC_DIMENSION));
    for diagonal in 0..INTRINSIC_DIMENSION {
        identity[[diagonal, diagonal]] = 1.0;
    }
    identity
}

fn identity_square(dimension: usize) -> Array2<f64> {
    let mut identity = Array2::<f64>::zeros((dimension, dimension));
    for diagonal in 0..dimension {
        identity[[diagonal, diagonal]] = 1.0;
    }
    identity
}

fn rotation_generator() -> Array2<f64> {
    let mut generator = Array2::<f64>::zeros((INTRINSIC_DIMENSION, INTRINSIC_DIMENSION));
    generator[[0, 1]] = -1.0;
    generator[[1, 0]] = 1.0;
    generator
}

fn frobenius_inner(left: ArrayView2<'_, f64>, right: ArrayView2<'_, f64>) -> f64 {
    left.iter().zip(right.iter()).map(|(&x, &y)| x * y).sum()
}

fn frobenius_squared(matrix: ArrayView2<'_, f64>) -> f64 {
    matrix.iter().map(|value| value * value).sum()
}

fn project_retained_normal(patch: &GaussianPcaPatch, local: &Array2<f64>) -> Array2<f64> {
    local
        - &patch
            .tangent_coordinates
            .dot(&patch.tangent_coordinates.t().dot(local))
}

fn build_projected_edge(
    patches: &[GaussianPcaPatch],
    spec: ProjectedAtlasEdgeSpec,
) -> Result<EdgeWork, String> {
    let patch_a = &patches[spec.a];
    let patch_b = &patches[spec.b];
    let projection_cross_gram_ba = patch_b.projection_frame.t().dot(&patch_a.projection_frame);
    let retained_a = patch_a.retained_dimension();
    let retained_b = patch_b.retained_dimension();
    // The intersection dimension is the multiplicity of singular value one
    // in W_b^T W_a, hence dim(span(W_a,W_b)) = r_a+r_b-dim(intersection).
    // This retained-coordinate SVD removes the ambient P-by-(r_a+r_b) SVD.
    let (_, projection_cosines, _) =
        projection_cross_gram_ba
            .svd(false, false)
            .map_err(|error| {
                format!(
                    "edge ({}, {}, overlap {}) retained projection cross-Gram SVD failed: {error}",
                    spec.a, spec.b, spec.overlap
                )
            })?;
    let intersection_backward_error =
        f64::EPSILON * patch_a.ambient_dimension().max(retained_a + retained_b) as f64;
    let intersection_dimension = projection_cosines
        .iter()
        .filter(|&&cosine| (1.0 - cosine).abs() <= intersection_backward_error)
        .count();
    let projected_dimension = retained_a + retained_b - intersection_dimension;
    if projected_dimension < INTRINSIC_DIMENSION {
        return Err(format!(
            "edge ({}, {}, overlap {}) projected union rank {projected_dimension} is below intrinsic dimension {INTRINSIC_DIMENSION}",
            spec.a, spec.b, spec.overlap
        ));
    }
    // Coordinates are mapped from chart a to chart b, hence M = U_b^T U_a.
    let cross = patch_b
        .tangent_coordinates
        .t()
        .dot(&projection_cross_gram_ba.dot(&patch_a.tangent_coordinates));
    let (left, singular, right_t) = cross.svd(true, true).map_err(|error| {
        format!(
            "edge ({}, {}, overlap {}) cross-Gram SVD failed: {error}",
            spec.a, spec.b, spec.overlap
        )
    })?;
    let left = left.ok_or_else(|| {
        format!(
            "edge ({}, {}, overlap {}) cross-Gram SVD omitted requested left vectors",
            spec.a, spec.b, spec.overlap
        )
    })?;
    let right_t = right_t.ok_or_else(|| {
        format!(
            "edge ({}, {}, overlap {}) cross-Gram SVD omitted requested right vectors",
            spec.a, spec.b, spec.overlap
        )
    })?;
    if singular.len() != INTRINSIC_DIMENSION {
        return Err(format!(
            "edge ({}, {}, overlap {}) cross-Gram has {} singular values, expected {INTRINSIC_DIMENSION}",
            spec.a,
            spec.b,
            spec.overlap,
            singular.len()
        ));
    }
    let largest_singular_value = singular[0];
    let smallest_singular_value = singular[INTRINSIC_DIMENSION - 1];
    let numerical_rank_threshold =
        f64::EPSILON * INTRINSIC_DIMENSION as f64 * largest_singular_value.max(1.0);
    let orientation_margin = determinant_2(cross.view()).abs();
    let principal_angle_cosines = [singular[0].clamp(0.0, 1.0), singular[1].clamp(0.0, 1.0)];
    if smallest_singular_value <= numerical_rank_threshold {
        return Ok(EdgeWork {
            public: ProjectedAtlasEdgeGeometry {
                a: spec.a,
                b: spec.b,
                overlap: spec.overlap,
                projected_dimension,
                principal_angle_cosines,
                orientation_margin,
                population_cross_gram: spec.population_cross_gram,
                estimated_sign: None,
                transition_a_to_b: None,
                geometric_remainder_bound: spec.geometric_remainder_bound,
            },
            transition: None,
            smallest_singular_value,
            numerical_rank_threshold,
            angle_gradient: None,
            patch_gradient_a: None,
            patch_gradient_b: None,
            projection_cross_gram_ba,
        });
    }
    let transition = left.dot(&right_t);
    let sign = if determinant_2(cross.view()) > 0.0 {
        1
    } else {
        -1
    };
    let trace_h: f64 = singular.iter().sum();
    let angle_gradient = transition.dot(&rotation_generator()) / trace_h;
    // dM = dU_b^T U_a + U_b^T dU_a.  Remove the vertical gauge
    // component before cycle aggregation, leaving a projector derivative.
    let raw_a = projection_cross_gram_ba
        .t()
        .dot(&patch_b.tangent_coordinates)
        .dot(&angle_gradient);
    let raw_b = projection_cross_gram_ba
        .dot(&patch_a.tangent_coordinates)
        .dot(&angle_gradient.t());
    let patch_gradient_a = project_retained_normal(patch_a, &raw_a);
    let patch_gradient_b = project_retained_normal(patch_b, &raw_b);
    let mut public_transition = [[0.0; INTRINSIC_DIMENSION]; INTRINSIC_DIMENSION];
    for i in 0..INTRINSIC_DIMENSION {
        for j in 0..INTRINSIC_DIMENSION {
            public_transition[i][j] = transition[[i, j]];
        }
    }
    Ok(EdgeWork {
        public: ProjectedAtlasEdgeGeometry {
            a: spec.a,
            b: spec.b,
            overlap: spec.overlap,
            projected_dimension,
            principal_angle_cosines,
            orientation_margin,
            population_cross_gram: spec.population_cross_gram,
            estimated_sign: Some(sign),
            transition_a_to_b: Some(public_transition),
            geometric_remainder_bound: spec.geometric_remainder_bound,
        },
        transition: Some(transition),
        smallest_singular_value,
        numerical_rank_threshold,
        angle_gradient: Some(angle_gradient),
        patch_gradient_a: Some(patch_gradient_a),
        patch_gradient_b: Some(patch_gradient_b),
        projection_cross_gram_ba,
    })
}

fn fundamental_cycles(
    chart_count: usize,
    edges: &[ProjectedAtlasEdgeSpec],
) -> Result<Vec<FundamentalCycle>, String> {
    let mut adjacency = vec![Vec::<(usize, usize)>::new(); chart_count];
    for (edge_index, edge) in edges.iter().enumerate() {
        adjacency[edge.a].push((edge.b, edge_index));
        adjacency[edge.b].push((edge.a, edge_index));
    }
    for neighbors in &mut adjacency {
        neighbors.sort_unstable();
    }
    let mut parent = vec![None::<usize>; chart_count];
    let mut parent_edge = vec![None::<usize>; chart_count];
    let mut reached = vec![false; chart_count];
    let mut tree_edges = BTreeSet::<usize>::new();
    for root in 0..chart_count {
        if reached[root] {
            continue;
        }
        reached[root] = true;
        let mut queue = VecDeque::from([root]);
        while let Some(chart) = queue.pop_front() {
            for &(next, edge_index) in &adjacency[chart] {
                if !reached[next] {
                    reached[next] = true;
                    parent[next] = Some(chart);
                    parent_edge[next] = Some(edge_index);
                    tree_edges.insert(edge_index);
                    queue.push_back(next);
                }
            }
        }
    }
    let mut cycles = Vec::new();
    for (chord_index, chord) in edges.iter().enumerate() {
        if tree_edges.contains(&chord_index) {
            continue;
        }
        let mut ancestors_a = BTreeMap::<usize, usize>::new();
        let mut path_a = Vec::new();
        let mut cursor = chord.a;
        loop {
            ancestors_a.insert(cursor, path_a.len());
            path_a.push(cursor);
            let Some(next) = parent[cursor] else {
                break;
            };
            cursor = next;
        }
        let mut path_b = Vec::new();
        cursor = chord.b;
        let lca = loop {
            if ancestors_a.contains_key(&cursor) {
                break cursor;
            }
            path_b.push(cursor);
            cursor = parent[cursor].ok_or_else(|| {
                format!(
                    "non-tree edge ({}, {}, overlap {}) joins different spanning-forest components",
                    chord.a, chord.b, chord.overlap
                )
            })?;
        };
        path_b.push(lca);
        let lca_position = ancestors_a[&lca];
        let mut walk = path_a[..=lca_position].to_vec();
        for &chart in path_b[..path_b.len() - 1].iter().rev() {
            walk.push(chart);
        }
        walk.push(chord.a);
        let mut steps = Vec::with_capacity(walk.len() - 1);
        let tree_step_count = walk.len().saturating_sub(2);
        for endpoints in walk.windows(2).take(tree_step_count) {
            let child = if parent[endpoints[0]] == Some(endpoints[1]) {
                endpoints[0]
            } else if parent[endpoints[1]] == Some(endpoints[0]) {
                endpoints[1]
            } else {
                return Err(format!(
                    "fundamental-cycle charts ({}, {}) are not joined by a spanning-tree edge",
                    endpoints[0], endpoints[1]
                ));
            };
            let edge_index = parent_edge[child].ok_or_else(|| {
                format!("fundamental-cycle chart {child} has no spanning-tree edge identity")
            })?;
            let forward = endpoints[0] == edges[edge_index].a;
            steps.push((edge_index, forward));
        }
        let chord_from = walk[walk.len() - 2];
        steps.push((chord_index, chord_from == chord.a));
        cycles.push(FundamentalCycle { steps });
    }
    Ok(cycles)
}

fn aligned_frame_error(projector_error: f64) -> f64 {
    let q = projector_error.clamp(0.0, 1.0);
    let cosine = (1.0 - q * q).sqrt();
    q * (2.0 / (1.0 + cosine)).sqrt()
}

fn projector_error_for_aligned_frame_error(frame_error: f64) -> f64 {
    let h = frame_error.clamp(0.0, 2.0_f64.sqrt());
    h * (1.0 - h * h / 4.0).sqrt()
}

#[derive(Clone, Copy, Debug)]
struct PatchTail {
    covariance_error: f64,
    projector_error: f64,
    aligned_frame_error: f64,
}

fn patch_tail(
    patch: &GaussianPcaPatch,
    projected_dimension: usize,
    tail_parameter: f64,
) -> Option<PatchTail> {
    let bounds = patch.spectrum_provenance.certified_bounds()?;
    let degrees_of_freedom = patch.covariance_degrees_of_freedom() as f64;
    let u = ((projected_dimension as f64).sqrt() + (2.0 * tail_parameter).sqrt())
        / degrees_of_freedom.sqrt();
    let covariance_error = bounds.spectral_radius_upper() * (2.0 * u + u * u);
    let projector_error = 2.0 * covariance_error / bounds.eigengap_lower;
    Some(PatchTail {
        covariance_error,
        projector_error,
        aligned_frame_error: aligned_frame_error(projector_error),
    })
}

/// Largest common aligned-frame error at the two endpoints which cannot reach
/// the singular-cross-Gram boundary. If both endpoint errors are at most `h`,
/// the cross-Gram perturbation is at most `2h + h²`. Solving
/// `2h + h² = m` uses a deterministic lower bound `m` on the population
/// cross-Gram's smallest singular value. An observed random margin cannot set
/// its own unconditional error threshold.
fn orientation_endpoint_frame_budget(edge: &EdgeWork) -> Option<f64> {
    let population_margin = edge.public.population_cross_gram.certified_lower_bound()?;
    Some(population_margin / ((1.0 + population_margin).sqrt() + 1.0))
}

fn covariance_budget_ratio(patch: &GaussianPcaPatch, frame_budget: f64) -> f64 {
    let Some(bounds) = patch.spectrum_provenance.certified_bounds() else {
        return 0.0;
    };
    let projector_budget = projector_error_for_aligned_frame_error(frame_budget);
    let covariance_budget = bounds.eigengap_lower * projector_budget / 2.0;
    covariance_budget / bounds.spectral_radius_upper()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RequiredCovarianceDegreesOfFreedom {
    Representable(usize),
    ExceedsRepresentableRange,
}

fn required_covariance_degrees_of_freedom(
    patch: &GaussianPcaPatch,
    projected_dimension: usize,
    frame_budget: f64,
    tail_parameter: f64,
) -> RequiredCovarianceDegreesOfFreedom {
    let normalized_budget = covariance_budget_ratio(patch, frame_budget);
    if !(normalized_budget.is_finite() && normalized_budget > 0.0) {
        return RequiredCovarianceDegreesOfFreedom::ExceedsRepresentableRange;
    }
    // Rationalizing `sqrt(1 + b) - 1` preserves tiny, valid budgets instead of
    // cancelling them to zero before representability is decided.
    let u_budget = normalized_budget / ((1.0 + normalized_budget).sqrt() + 1.0);
    if !(u_budget.is_finite() && u_budget > 0.0) {
        return RequiredCovarianceDegreesOfFreedom::ExceedsRepresentableRange;
    }
    let numerator = (projected_dimension as f64).sqrt() + (2.0 * tail_parameter).sqrt();
    if !(numerator.is_finite() && numerator > 0.0) {
        return RequiredCovarianceDegreesOfFreedom::ExceedsRepresentableRange;
    }
    let required = (numerator / u_budget).powi(2).ceil().max(1.0);
    if !required.is_finite() {
        return RequiredCovarianceDegreesOfFreedom::ExceedsRepresentableRange;
    }
    // Convert through a wider integer so the rounded `usize::MAX as f64`
    // boundary cannot saturate back to the same magic value.
    let required_wide = required as u128;
    match usize::try_from(required_wide) {
        Ok(required) if required > 0 => RequiredCovarianceDegreesOfFreedom::Representable(required),
        _ => RequiredCovarianceDegreesOfFreedom::ExceedsRepresentableRange,
    }
}

fn supported_tail_parameter(
    patch: &GaussianPcaPatch,
    projected_dimension: usize,
    frame_budget: f64,
) -> f64 {
    let normalized_budget = covariance_budget_ratio(patch, frame_budget);
    let u_budget = if normalized_budget.is_finite() && normalized_budget > 0.0 {
        normalized_budget / ((1.0 + normalized_budget).sqrt() + 1.0)
    } else {
        0.0
    };
    let available = (patch.covariance_degrees_of_freedom() as f64).sqrt() * u_budget
        - (projected_dimension as f64).sqrt();
    if available > 0.0 {
        available * available / 2.0
    } else {
        0.0
    }
}

#[derive(Clone, Copy, Debug)]
enum PatchOccupancyRequirement {
    Representable {
        covariance_degrees_of_freedom: usize,
        projected_dimension: usize,
        aligned_frame_error_budget: f64,
    },
    ExceedsRepresentableRange {
        projected_dimension: usize,
        aligned_frame_error_budget: f64,
    },
}

fn orientation_tail_and_prescription(
    patches: &[GaussianPcaPatch],
    edges: &[EdgeWork],
    allocated_alpha: f64,
) -> Result<
    (
        AtlasStatisticalDecision<AtlasOrientability>,
        Option<f64>,
        Vec<AtlasPatchSamplePrescription>,
    ),
    String,
> {
    if edges.is_empty() {
        return Ok((
            AtlasStatisticalDecision::Certified {
                value: AtlasOrientability::Orientable,
                error_probability_bound: 0.0,
            },
            Some(0.0),
            Vec::new(),
        ));
    }
    let mut reasons = Vec::new();
    let incident_charts: BTreeSet<_> = edges
        .iter()
        .flat_map(|edge| [edge.public.a, edge.public.b])
        .collect();
    let mut bound_inputs_certified = true;
    for &chart in &incident_charts {
        let patch = &patches[chart];
        if !patch.pilot_projection.is_certified() {
            bound_inputs_certified = false;
            reasons
                .push(AtlasStatisticalRefusal::PilotProjectionUncertified { chart: patch.chart });
        }
        if patch.spectrum_provenance.certified_bounds().is_none() {
            bound_inputs_certified = false;
            reasons.push(AtlasStatisticalRefusal::PopulationSpectrumUncertified {
                chart: patch.chart,
            });
        }
    }
    for edge in edges {
        if edge
            .public
            .population_cross_gram
            .certified_lower_bound()
            .is_none()
        {
            bound_inputs_certified = false;
            reasons.push(
                AtlasStatisticalRefusal::PopulationCrossGramMarginUncertified {
                    edge: edge.public.identity(),
                },
            );
        }
        if edge.public.estimated_sign.is_none() {
            reasons.push(AtlasStatisticalRefusal::SingularProjectedCrossGram {
                edge: edge.public.identity(),
                smallest_singular_value: edge.smallest_singular_value,
                numerical_rank_threshold: edge.numerical_rank_threshold,
            });
        }
    }
    // Each edge uses its own pairwise projection space. Distinct incident
    // projections are not generally nested, so the finite-sample union bound
    // must count edge endpoints, not merely distinct patches.
    let incidence_count = 2 * edges.len();
    let requested_tail_parameter = (2.0 * incidence_count as f64 / allocated_alpha).ln();
    let mut patch_requirements = vec![None::<PatchOccupancyRequirement>; patches.len()];
    let mut flip_probability_bound = 0.0_f64;
    for edge in edges {
        let Some(frame_budget) = orientation_endpoint_frame_budget(edge) else {
            continue;
        };
        for chart in [edge.public.a, edge.public.b] {
            let patch = &patches[chart];
            if patch.spectrum_provenance.certified_bounds().is_none() {
                continue;
            }
            let projected_dimension = edge.public.projected_dimension;
            let requirement = match required_covariance_degrees_of_freedom(
                patch,
                projected_dimension,
                frame_budget,
                requested_tail_parameter,
            ) {
                RequiredCovarianceDegreesOfFreedom::Representable(
                    covariance_degrees_of_freedom,
                ) => PatchOccupancyRequirement::Representable {
                    covariance_degrees_of_freedom,
                    projected_dimension,
                    aligned_frame_error_budget: frame_budget,
                },
                RequiredCovarianceDegreesOfFreedom::ExceedsRepresentableRange => {
                    PatchOccupancyRequirement::ExceedsRepresentableRange {
                        projected_dimension,
                        aligned_frame_error_budget: frame_budget,
                    }
                }
            };
            let replace = match (patch_requirements[chart], requirement) {
                (None, _) => true,
                (Some(PatchOccupancyRequirement::ExceedsRepresentableRange { .. }), _) => false,
                (_, PatchOccupancyRequirement::ExceedsRepresentableRange { .. }) => true,
                (
                    Some(PatchOccupancyRequirement::Representable {
                        covariance_degrees_of_freedom: current,
                        ..
                    }),
                    PatchOccupancyRequirement::Representable {
                        covariance_degrees_of_freedom: candidate,
                        ..
                    },
                ) => candidate > current,
            };
            if replace {
                patch_requirements[chart] = Some(requirement);
            }
            let supported = supported_tail_parameter(patch, projected_dimension, frame_budget);
            flip_probability_bound += (2.0 * (-supported).exp()).min(1.0);
        }
    }
    flip_probability_bound = flip_probability_bound.min(1.0);
    let mut prescriptions = Vec::with_capacity(incident_charts.len());
    for chart in incident_charts {
        let patch = &patches[chart];
        let pilot = if patch.pilot_projection.is_certified() {
            AtlasPilotOccupancyPrescription::ExactCaptureNoSamplingRequirement
        } else {
            AtlasPilotOccupancyPrescription::PopulationCaptureTheoremRequired
        };
        let all_incident_margins_certified = edges
            .iter()
            .filter(|edge| edge.public.a == chart || edge.public.b == chart)
            .all(|edge| {
                edge.public
                    .population_cross_gram
                    .certified_lower_bound()
                    .is_some()
            });
        let inference = if patch.spectrum_provenance.certified_bounds().is_some()
            && all_incident_margins_certified
        {
            let requirement = patch_requirements[chart]
                .ok_or_else(|| {
                    format!(
                        "atlas patch {chart} has certified incident tail inputs but no occupancy requirement"
                    )
                })?;
            match requirement {
                PatchOccupancyRequirement::Representable {
                    covariance_degrees_of_freedom,
                    projected_dimension,
                    aligned_frame_error_budget,
                } => match patch
                    .centering
                    .rows_for_degrees_of_freedom(covariance_degrees_of_freedom)
                {
                    Some(rows) => AtlasInferenceOccupancyPrescription::Required {
                        rows,
                        covariance_degrees_of_freedom,
                        projected_dimension,
                        aligned_frame_error_budget,
                    },
                    None => {
                        AtlasInferenceOccupancyPrescription::RequiredRowsExceedRepresentableRange {
                            projected_dimension,
                            aligned_frame_error_budget,
                        }
                    }
                },
                PatchOccupancyRequirement::ExceedsRepresentableRange {
                    projected_dimension,
                    aligned_frame_error_budget,
                } => AtlasInferenceOccupancyPrescription::RequiredRowsExceedRepresentableRange {
                    projected_dimension,
                    aligned_frame_error_budget,
                },
            }
        } else {
            AtlasInferenceOccupancyPrescription::PopulationTailInputsRequired
        };
        prescriptions.push(AtlasPatchSamplePrescription {
            chart,
            current_pilot_rows: patch.row_split.pilot_rows.len(),
            pilot,
            current_inference_rows: patch.row_split.inference_rows.len(),
            current_covariance_degrees_of_freedom: patch.covariance_degrees_of_freedom(),
            inference,
        });
    }
    if bound_inputs_certified && flip_probability_bound > allocated_alpha {
        reasons.push(AtlasStatisticalRefusal::OrientationFlipBoundExceedsLevel {
            flip_probability_bound,
            allocated_alpha,
        });
    }
    let decision = if reasons.is_empty() {
        let signs: Vec<AtlasSignedEdge> = edges
            .iter()
            .filter_map(|edge| {
                edge.public.estimated_sign.map(|sign| AtlasSignedEdge {
                    a: edge.public.a,
                    b: edge.public.b,
                    overlap: edge.public.overlap,
                    sign,
                })
            })
            .collect();
        AtlasStatisticalDecision::Certified {
            value: orientability_from_edges(patches.len(), &signs),
            error_probability_bound: flip_probability_bound,
        }
    } else {
        AtlasStatisticalDecision::Refused { reasons }
    };
    Ok((
        decision,
        bound_inputs_certified.then_some(flip_probability_bound),
        prescriptions,
    ))
}

fn edge_step_matrix(edge: &EdgeWork, forward: bool) -> Option<Array2<f64>> {
    let transition = edge.transition.as_ref()?;
    Some(if forward {
        transition.clone()
    } else {
        transition.t().to_owned()
    })
}

fn gaussian_two_sided_radius(standard_error: f64, error_probability: f64) -> Result<f64, String> {
    let normal = Normal::new(0.0, 1.0)
        .map_err(|error| format!("standard-normal construction failed: {error}"))?;
    Ok(normal.inverse_cdf(1.0 - error_probability / 2.0) * standard_error)
}

fn cycle_rejection_boundary(
    standard_error: f64,
    polar_linearization_remainder_bound: f64,
    geometric_remainder_bound: f64,
    gaussian_error_budget: f64,
) -> Result<f64, String> {
    Ok(
        gaussian_two_sided_radius(standard_error, gaussian_error_budget)?
            + polar_linearization_remainder_bound
            + geometric_remainder_bound,
    )
}

fn covariance_quadratic(covariance: &Array2<f64>, gradient: &Array1<f64>) -> f64 {
    gradient.dot(&covariance.dot(gradient))
}

fn write_patch_gradient(
    model: &GaussianPcaErrorModel,
    patch: usize,
    gradient: &Array2<f64>,
    scale: f64,
    target: &mut Array1<f64>,
) {
    let offset = model.offsets[patch];
    for row in 0..gradient.nrows() {
        for column in 0..INTRINSIC_DIMENSION {
            target[offset + row * INTRINSIC_DIMENSION + column] += scale * gradient[[row, column]];
        }
    }
}

fn quadratic_gaussian_moments(quadratic: &Array2<f64>, covariance: &Array2<f64>) -> (f64, f64) {
    let product = quadratic.dot(covariance);
    let bias: f64 = (0..product.nrows())
        .map(|index| product[[index, index]])
        .sum();
    let trace_square: f64 = (0..product.nrows())
        .flat_map(|row| (0..product.ncols()).map(move |column| (row, column)))
        .map(|(row, column)| product[[row, column]] * product[[column, row]])
        .sum();
    (bias, (2.0 * trace_square).max(0.0))
}

fn analyze_cycle(
    cycle_index: usize,
    cycle: &FundamentalCycle,
    patches: &[GaussianPcaPatch],
    edges: &[EdgeWork],
    error_model: &GaussianPcaErrorModel,
    allocated_alpha: f64,
) -> Result<AtlasCycleHolonomy, String> {
    let gaussian_error_budget = allocated_alpha / 2.0;
    let subspace_tail_probability_bound = allocated_alpha - gaussian_error_budget;
    let cycle_steps: Vec<_> = cycle
        .steps
        .iter()
        .map(|&(edge, forward)| {
            AtlasHolonomyCycleStep::from_traversal(edges[edge].public.identity(), forward)
        })
        .collect();
    let geometric_remainder_bound: f64 = cycle
        .steps
        .iter()
        .map(|&(edge, _)| edges[edge].public.geometric_remainder_bound)
        .sum();
    let empty_refusal = |reasons| AtlasCycleHolonomy {
        cycle_index,
        steps: cycle_steps.clone(),
        absolute_angle: None,
        asymptotic_regime: None,
        first_order_variance: None,
        naive_edgewise_first_order_variance: None,
        covariance_aggregation_adjustment: None,
        bilinear_quadratic_bias_diagnostic: None,
        bilinear_quadratic_variance_diagnostic: None,
        standard_error: None,
        polar_linearization_remainder_bound: None,
        geometric_remainder_bound,
        gaussian_error_budget,
        subspace_tail_probability_bound,
        decision: AtlasStatisticalDecision::Refused { reasons },
    };
    let mut reasons = Vec::new();
    for &(edge_index, _) in &cycle.steps {
        let edge = &edges[edge_index];
        if edge.transition.is_none() {
            reasons.push(AtlasStatisticalRefusal::SingularProjectedCrossGram {
                edge: edge.public.identity(),
                smallest_singular_value: edge.smallest_singular_value,
                numerical_rank_threshold: edge.numerical_rank_threshold,
            });
        }
    }
    if !reasons.is_empty() {
        return Ok(empty_refusal(reasons));
    }
    let step_matrices: Vec<Array2<f64>> = cycle
        .steps
        .iter()
        .map(|&(edge, forward)| {
            edge_step_matrix(&edges[edge], forward).ok_or_else(|| {
                format!(
                    "cycle {cycle_index} edge {:?} lost its validated polar transition",
                    edges[edge].public.identity()
                )
            })
        })
        .collect::<Result<_, _>>()?;
    let mut before = Vec::with_capacity(step_matrices.len());
    let mut product = identity_2();
    for transition in &step_matrices {
        before.push(product.clone());
        product = transition.dot(&product);
    }
    let holonomy = product;
    if determinant_2(holonomy.view()) < 0.0 {
        return Ok(empty_refusal(vec![
            AtlasStatisticalRefusal::ImproperCycleHolonomy { cycle_index },
        ]));
    }
    let mut after = vec![identity_2(); step_matrices.len()];
    let mut suffix = identity_2();
    for index in (0..step_matrices.len()).rev() {
        after[index] = suffix.clone();
        suffix = suffix.dot(&step_matrices[index]);
    }
    let generator = rotation_generator();
    let holonomy_gradient = holonomy.dot(&generator) / 2.0;
    let mut coefficients = Vec::with_capacity(step_matrices.len());
    for (position, &(edge_index, forward)) in cycle.steps.iter().enumerate() {
        let transition_gradient = after[position]
            .t()
            .dot(&holonomy_gradient)
            .dot(&before[position].t());
        let canonical = edges[edge_index]
            .transition
            .as_ref()
            .ok_or_else(|| format!("cycle {cycle_index} lost edge {}", edge_index))?;
        let canonical_tangent = canonical.dot(&generator);
        let step_tangent = if forward {
            canonical_tangent
        } else {
            canonical_tangent.t().to_owned()
        };
        coefficients.push(frobenius_inner(
            transition_gradient.view(),
            step_tangent.view(),
        ));
    }

    let dimension = error_model.covariance.nrows();
    let mut aggregate_gradient = Array1::<f64>::zeros(dimension);
    let mut naive_first_order_variance = 0.0_f64;
    let mut quadratic = Array2::<f64>::zeros((dimension, dimension));
    for (position, &(edge_index, _)) in cycle.steps.iter().enumerate() {
        let coefficient = coefficients[position];
        let edge = &edges[edge_index];
        let gradient_a = edge.patch_gradient_a.as_ref().ok_or_else(|| {
            format!("cycle {cycle_index} lost patch-a gradient for edge {edge_index}")
        })?;
        let gradient_b = edge.patch_gradient_b.as_ref().ok_or_else(|| {
            format!("cycle {cycle_index} lost patch-b gradient for edge {edge_index}")
        })?;
        let mut edge_gradient = Array1::<f64>::zeros(dimension);
        write_patch_gradient(
            error_model,
            edge.public.a,
            gradient_a,
            coefficient,
            &mut edge_gradient,
        );
        write_patch_gradient(
            error_model,
            edge.public.b,
            gradient_b,
            coefficient,
            &mut edge_gradient,
        );
        aggregate_gradient += &edge_gradient;
        naive_first_order_variance += covariance_quadratic(&error_model.covariance, &edge_gradient);

        let angle_gradient = edge.angle_gradient.as_ref().ok_or_else(|| {
            format!("cycle {cycle_index} lost angle gradient for edge {edge_index}")
        })?;
        let offset_a = error_model.offsets[edge.public.a];
        let offset_b = error_model.offsets[edge.public.b];
        for coordinate_b in 0..edge.projection_cross_gram_ba.nrows() {
            for coordinate_a in 0..edge.projection_cross_gram_ba.ncols() {
                let frame_inner = edge.projection_cross_gram_ba[[coordinate_b, coordinate_a]];
                for tangent_b in 0..INTRINSIC_DIMENSION {
                    for tangent_a in 0..INTRINSIC_DIMENSION {
                        let a_index = offset_a + coordinate_a * INTRINSIC_DIMENSION + tangent_a;
                        let b_index = offset_b + coordinate_b * INTRINSIC_DIMENSION + tangent_b;
                        let value =
                            coefficient * angle_gradient[[tangent_b, tangent_a]] * frame_inner
                                / 2.0;
                        quadratic[[a_index, b_index]] += value;
                        quadratic[[b_index, a_index]] += value;
                    }
                }
            }
        }
    }
    let first_order_variance =
        covariance_quadratic(&error_model.covariance, &aggregate_gradient).max(0.0);
    let covariance_aggregation_adjustment = first_order_variance - naive_first_order_variance;
    let (quadratic_bias, quadratic_variance) =
        quadratic_gaussian_moments(&quadratic, &error_model.covariance);
    let variance_scale = first_order_variance
        .abs()
        .max(naive_first_order_variance.abs())
        .max(f64::MIN_POSITIVE);
    let variance_backward_error = f64::EPSILON * dimension.max(1) as f64 * variance_scale;
    let degenerate = first_order_variance <= variance_backward_error;
    let asymptotic_regime = if degenerate {
        AtlasCycleAsymptoticRegime::FirstOrderDegenerate {
            bilinear_quadratic_bias_diagnostic: quadratic_bias,
            bilinear_quadratic_variance_diagnostic: quadratic_variance,
        }
    } else {
        AtlasCycleAsymptoticRegime::FirstOrderGaussian {
            variance: first_order_variance,
            authority: error_model.authority,
        }
    };
    let standard_error = (!degenerate).then(|| first_order_variance.sqrt());
    let absolute_angle = (holonomy[[1, 0]] - holonomy[[0, 1]])
        .atan2(holonomy[[0, 0]] + holonomy[[1, 1]])
        .abs();

    let incident_charts: BTreeSet<_> = cycle
        .steps
        .iter()
        .flat_map(|&(edge_index, _)| {
            let edge = &edges[edge_index].public;
            [edge.a, edge.b]
        })
        .collect();
    for chart in incident_charts {
        let patch = &patches[chart];
        if !patch.pilot_projection.is_certified() {
            reasons
                .push(AtlasStatisticalRefusal::PilotProjectionUncertified { chart: patch.chart });
        }
        if patch.spectrum_provenance.certified_bounds().is_none() {
            reasons.push(AtlasStatisticalRefusal::PopulationSpectrumUncertified {
                chart: patch.chart,
            });
        }
    }
    for &(edge_index, _) in &cycle.steps {
        let edge = &edges[edge_index];
        if edge
            .public
            .population_cross_gram
            .certified_lower_bound()
            .is_none()
        {
            reasons.push(
                AtlasStatisticalRefusal::PopulationCrossGramMarginUncertified {
                    edge: edge.public.identity(),
                },
            );
        }
    }
    if matches!(
        error_model.authority,
        GaussianPcaCovarianceAuthority::AsymptoticPlugIn
    ) {
        reasons.push(AtlasStatisticalRefusal::GaussianLinearizationIsPlugin { cycle_index });
    }
    if degenerate {
        reasons.push(
            AtlasStatisticalRefusal::DegenerateFirstOrderLimitUnresolved {
                cycle_index,
                bilinear_quadratic_bias_diagnostic: quadratic_bias,
                bilinear_quadratic_variance_diagnostic: quadratic_variance,
            },
        );
    }

    let endpoint_event_count = 2 * cycle.steps.len();
    let tail_parameter = (2.0 * endpoint_event_count as f64 / subspace_tail_probability_bound).ln();
    let mut polar_linearization_remainder_bound = 0.0_f64;
    for (position, &(edge_index, _)) in cycle.steps.iter().enumerate() {
        let edge = &edges[edge_index];
        let Some(tail_a) = patch_tail(
            &patches[edge.public.a],
            patches[edge.public.a].retained_dimension(),
            tail_parameter,
        ) else {
            continue;
        };
        let Some(tail_b) = patch_tail(
            &patches[edge.public.b],
            patches[edge.public.b].retained_dimension(),
            tail_parameter,
        ) else {
            continue;
        };
        let Some(population_margin) = edge.public.population_cross_gram.certified_lower_bound()
        else {
            continue;
        };
        for (chart, tail) in [(edge.public.a, tail_a), (edge.public.b, tail_b)] {
            let bounds = patches[chart]
                .spectrum_provenance
                .certified_bounds()
                .ok_or_else(|| format!("cycle {cycle_index} lost certified patch bounds"))?;
            if tail.covariance_error >= bounds.eigengap_lower / 2.0 || tail.projector_error >= 1.0 {
                reasons.push(AtlasStatisticalRefusal::PatchTailCrossesEigengap {
                    edge: edge.public.identity(),
                    chart,
                    covariance_error_bound: tail.covariance_error,
                    eigengap_lower: bounds.eigengap_lower,
                });
            }
        }
        let cross_error = tail_a.aligned_frame_error
            + tail_b.aligned_frame_error
            + tail_a.aligned_frame_error * tail_b.aligned_frame_error;
        if cross_error >= population_margin {
            reasons.push(AtlasStatisticalRefusal::PolarLinearizationUnresolved {
                cycle_index,
                edge: edge.public.identity(),
                cross_gram_error_bound: cross_error,
                population_smallest_singular_value_lower_bound: population_margin,
            });
            continue;
        }
        let polar_difference = 2.0 * cross_error / (2.0 * population_margin - cross_error);
        if polar_difference >= 2.0 {
            reasons.push(AtlasStatisticalRefusal::PolarLinearizationUnresolved {
                cycle_index,
                edge: edge.public.identity(),
                cross_gram_error_bound: cross_error,
                population_smallest_singular_value_lower_bound: population_margin,
            });
            continue;
        }
        let total_angle_change = 2.0 * (polar_difference / 2.0).asin();
        let gradient_norm = edge
            .angle_gradient
            .as_ref()
            .map(|gradient| frobenius_squared(gradient.view()).sqrt())
            .unwrap_or(0.0);
        let linear_change_bound = gradient_norm * (INTRINSIC_DIMENSION as f64).sqrt() * cross_error;
        polar_linearization_remainder_bound +=
            coefficients[position].abs() * (total_angle_change + linear_change_bound);
    }

    let analyzed = |decision| AtlasCycleHolonomy {
        cycle_index,
        steps: cycle_steps.clone(),
        absolute_angle: Some(absolute_angle),
        asymptotic_regime: Some(asymptotic_regime),
        first_order_variance: Some(first_order_variance),
        naive_edgewise_first_order_variance: Some(naive_first_order_variance),
        covariance_aggregation_adjustment: Some(covariance_aggregation_adjustment),
        bilinear_quadratic_bias_diagnostic: Some(quadratic_bias),
        bilinear_quadratic_variance_diagnostic: Some(quadratic_variance),
        standard_error,
        polar_linearization_remainder_bound: Some(polar_linearization_remainder_bound),
        geometric_remainder_bound,
        gaussian_error_budget,
        subspace_tail_probability_bound,
        decision,
    };
    if !reasons.is_empty() {
        return Ok(analyzed(AtlasStatisticalDecision::Refused { reasons }));
    }
    let rejection_boundary = cycle_rejection_boundary(
        standard_error
            .ok_or_else(|| format!("cycle {cycle_index} degenerate Gaussian law reached z-test"))?,
        polar_linearization_remainder_bound,
        geometric_remainder_bound,
        gaussian_error_budget,
    )?;
    if std::f64::consts::PI - absolute_angle <= rejection_boundary {
        return Ok(analyzed(AtlasStatisticalDecision::Refused {
            reasons: vec![AtlasStatisticalRefusal::CycleAngleBranchCutCrossed {
                cycle_index,
                absolute_angle,
                uncertainty_radius: rejection_boundary,
            }],
        }));
    }
    let conclusion = if absolute_angle > rejection_boundary {
        AtlasCycleConclusion::NonTrivialHolonomy
    } else {
        AtlasCycleConclusion::NotRejected
    };
    Ok(analyzed(AtlasStatisticalDecision::Certified {
        value: conclusion,
        error_probability_bound: gaussian_error_budget + subspace_tail_probability_bound,
    }))
}

fn gauss_bonnet_confidence(
    input: &GaussBonnetInput,
    allocated_alpha: f64,
) -> Result<GaussBonnetConfidence, String> {
    let total_curvature_estimate: f64 = input
        .contributions
        .iter()
        .map(|contribution| contribution.curvature_estimate)
        .sum();
    let polar_linearization_remainder_bound: f64 = input
        .contributions
        .iter()
        .map(|contribution| contribution.polar_linearization_remainder_bound)
        .sum();
    let geometric_remainder_bound: f64 = input
        .contributions
        .iter()
        .map(|contribution| contribution.geometric_remainder_bound)
        .sum();
    let source_map: BTreeMap<usize, &GaussBonnetNoiseSource> = input
        .sources
        .iter()
        .map(|source| (source.source, source))
        .collect();
    let mut total_gradients = BTreeMap::<usize, Array1<f64>>::new();
    let mut naive_contribution_variance = 0.0_f64;
    for contribution in &input.contributions {
        for gradient in &contribution.source_gradients {
            let source = source_map[&gradient.source];
            let covariance_gradient = source.covariance.dot(&gradient.gradient);
            naive_contribution_variance += gradient.gradient.dot(&covariance_gradient);
            total_gradients
                .entry(gradient.source)
                .and_modify(|total| *total += &gradient.gradient)
                .or_insert_with(|| gradient.gradient.clone());
        }
    }
    let mut first_order_variance = 0.0_f64;
    for (source_id, gradient) in &total_gradients {
        let covariance_gradient = source_map[source_id].covariance.dot(gradient);
        first_order_variance += gradient.dot(&covariance_gradient);
    }
    // Degeneracy is relative to the propagated quadratic form's own scale.
    // An absolute `max(1)` floor would incorrectly erase a perfectly
    // nondegenerate, exactly specified Gaussian law merely because its units
    // make the variance smaller than machine epsilon.
    let variance_scale = naive_contribution_variance
        .abs()
        .max(first_order_variance.abs())
        .max(f64::MIN_POSITIVE);
    let variance_backward_error =
        f64::EPSILON * total_gradients.len().max(1) as f64 * variance_scale;
    if first_order_variance < -variance_backward_error {
        return Err(format!(
            "Gauss-Bonnet propagated covariance produced negative variance {first_order_variance}"
        ));
    }
    if first_order_variance < 0.0 {
        first_order_variance = 0.0;
    }
    let degenerate = first_order_variance <= variance_backward_error;
    let standard_error = (!degenerate).then(|| first_order_variance.sqrt());
    let integer_f64 = (total_curvature_estimate / std::f64::consts::TAU).round();
    if !(integer_f64.is_finite()
        && integer_f64 >= i64::MIN as f64
        && integer_f64 <= i64::MAX as f64)
    {
        return Err("Gauss-Bonnet integer candidate is outside i64 range".to_string());
    }
    let nearest_integer_candidate = AtlasEulerCharacteristic(integer_f64 as i64);
    let residual_to_integer_curvature = (total_curvature_estimate
        - std::f64::consts::TAU * nearest_integer_candidate.value() as f64)
        .abs();
    let total_remainder = polar_linearization_remainder_bound + geometric_remainder_bound;
    let observed_rounding_margin =
        std::f64::consts::PI - residual_to_integer_curvature - total_remainder;
    // The wrong-integer event is `|noise| >= pi - deterministic_remainder`.
    // Its probability is uniform over the realized position inside the chosen
    // rounding cell. Subtracting that observed residual from the Gaussian tail
    // radius would turn a frequentist error rate into a data-dependent number
    // and could not be advertised as the misround probability.
    let stochastic_rounding_margin = std::f64::consts::PI - total_remainder;
    let mut authority_refusals = Vec::new();
    if matches!(
        input.covariance_authority,
        GaussBonnetCovarianceAuthority::AsymptoticPlugIn
    ) {
        authority_refusals.push(AtlasStatisticalRefusal::GaussBonnetGaussianLinearizationIsPlugin);
    }
    if degenerate {
        authority_refusals.push(
            AtlasStatisticalRefusal::GaussBonnetFirstOrderLimitDegenerate {
                first_order_variance,
            },
        );
    }
    let (misround_probability_bound, decision) = if !authority_refusals.is_empty() {
        (
            None,
            AtlasStatisticalDecision::Refused {
                reasons: authority_refusals,
            },
        )
    } else if observed_rounding_margin <= 0.0 || stochastic_rounding_margin <= 0.0 {
        (
            Some(1.0),
            AtlasStatisticalDecision::Refused {
                reasons: vec![
                    AtlasStatisticalRefusal::GaussBonnetRoundingMarginExhausted {
                        residual_to_integer_curvature,
                        total_remainder_bound: total_remainder,
                    },
                ],
            },
        )
    } else {
        let normal = Normal::new(0.0, 1.0)
            .map_err(|error| format!("standard-normal construction failed: {error}"))?;
        let probability = (2.0
            * (1.0
                - normal.cdf(
                    stochastic_rounding_margin
                        / standard_error.ok_or_else(|| {
                            "nondegenerate Gauss-Bonnet law lost its standard error".to_string()
                        })?,
                )))
        .clamp(0.0, 1.0);
        if probability <= allocated_alpha {
            (
                Some(probability),
                AtlasStatisticalDecision::Certified {
                    value: nearest_integer_candidate,
                    error_probability_bound: probability,
                },
            )
        } else {
            (
                Some(probability),
                AtlasStatisticalDecision::Refused {
                    reasons: vec![AtlasStatisticalRefusal::GaussBonnetErrorBoundExceedsLevel {
                        misround_probability_bound: probability,
                        allocated_alpha,
                    }],
                },
            )
        }
    };
    Ok(GaussBonnetConfidence {
        covariance_authority: input.covariance_authority,
        total_curvature_estimate,
        nearest_integer_candidate,
        residual_to_integer_curvature,
        first_order_variance,
        naive_contribution_variance,
        shared_source_covariance_adjustment: first_order_variance - naive_contribution_variance,
        standard_error,
        polar_linearization_remainder_bound,
        geometric_remainder_bound,
        observed_rounding_margin,
        stochastic_rounding_margin,
        misround_probability_bound,
        decision,
    })
}

impl GaussianPcaHolonomyAnalysis {
    /// Build the complete projected PCA holonomy analysis.
    #[must_use = "Gaussian PCA holonomy construction errors must be handled"]
    pub fn certify(
        patches: Vec<GaussianPcaPatch>,
        mut edge_specs: Vec<ProjectedAtlasEdgeSpec>,
        error_model: GaussianPcaErrorModel,
        familywise_level: AtlasFamilywiseLevel,
        gauss_bonnet_input: Option<GaussBonnetInput>,
    ) -> Result<Self, String> {
        for (expected, patch) in patches.iter().enumerate() {
            if patch.chart != expected {
                return Err(format!(
                    "Gaussian PCA patch indices must be contiguous: position {expected} contains chart {}",
                    patch.chart
                ));
            }
        }
        let expected_offsets = GaussianPcaErrorModel::coordinate_offsets(&patches);
        if error_model.offsets != expected_offsets {
            return Err(
                "Gaussian PCA error-model coordinates do not match the ordered patch frames"
                    .to_string(),
            );
        }
        let chart_count = patches.len();
        if let Some(first) = patches.first() {
            for patch in &patches {
                if patch.ambient_dimension() != first.ambient_dimension() {
                    return Err(format!(
                        "Gaussian PCA patch {} ambient dimension {} differs from {}",
                        patch.chart,
                        patch.ambient_dimension(),
                        first.ambient_dimension()
                    ));
                }
            }
        }
        let patch_summaries = patches
            .iter()
            .map(GaussianPcaPatch::audit_summary)
            .collect();
        edge_specs.sort_by_key(|edge| (edge.a, edge.b, edge.overlap));
        for (position, edge) in edge_specs.iter().enumerate() {
            if edge.b >= chart_count {
                return Err(format!(
                    "projected atlas edge ({}, {}, overlap {}) is outside the {chart_count}-chart atlas",
                    edge.a, edge.b, edge.overlap
                ));
            }
            if position > 0
                && (
                    edge_specs[position - 1].a,
                    edge_specs[position - 1].b,
                    edge_specs[position - 1].overlap,
                ) == (edge.a, edge.b, edge.overlap)
            {
                return Err(format!(
                    "duplicate projected atlas edge ({}, {}, overlap {})",
                    edge.a, edge.b, edge.overlap
                ));
            }
        }
        let edge_work: Vec<EdgeWork> = edge_specs
            .iter()
            .copied()
            .map(|edge| build_projected_edge(&patches, edge))
            .collect::<Result<_, _>>()?;
        let fundamental = fundamental_cycles(chart_count, &edge_specs)?;
        let simultaneous_claims = 1 + fundamental.len() + usize::from(gauss_bonnet_input.is_some());
        let allocated_alpha = familywise_level.alpha() / simultaneous_claims as f64;
        let (orientation, orientation_flip_probability_bound, sample_prescription) =
            orientation_tail_and_prescription(&patches, &edge_work, allocated_alpha)?;
        let cycles = fundamental
            .iter()
            .enumerate()
            .map(|(index, cycle)| {
                analyze_cycle(
                    index,
                    cycle,
                    &patches,
                    &edge_work,
                    &error_model,
                    allocated_alpha,
                )
            })
            .collect::<Result<_, _>>()?;
        let gauss_bonnet = gauss_bonnet_input
            .as_ref()
            .map(|input| gauss_bonnet_confidence(input, allocated_alpha))
            .transpose()?;
        Ok(Self {
            familywise_level,
            chart_count,
            patch_summaries,
            error_model,
            edges: edge_work.into_iter().map(|edge| edge.public).collect(),
            orientation,
            orientation_flip_probability_bound,
            sample_prescription,
            cycles,
            gauss_bonnet,
        })
    }
}

impl AtlasHolonomyCertificate {
    #[must_use = "Gaussian PCA holonomy construction errors must be handled"]
    pub fn gaussian_pca(
        patches: Vec<GaussianPcaPatch>,
        edge_specs: Vec<ProjectedAtlasEdgeSpec>,
        error_model: GaussianPcaErrorModel,
        familywise_level: AtlasFamilywiseLevel,
        gauss_bonnet_input: Option<GaussBonnetInput>,
    ) -> Result<Self, String> {
        Ok(Self::GaussianPcaPlugin(
            GaussianPcaHolonomyAnalysis::certify(
                patches,
                edge_specs,
                error_model,
                familywise_level,
                gauss_bonnet_input,
            )?,
        ))
    }
}
