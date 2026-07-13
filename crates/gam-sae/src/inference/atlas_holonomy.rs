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

use crate::manifold::AtlasOrientability;
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
    SingularProjectedCrossGram {
        a: usize,
        b: usize,
        smallest_singular_value: f64,
        numerical_rank_threshold: f64,
    },
    PatchTailCrossesEigengap {
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
        a: usize,
        b: usize,
        cross_gram_error_bound: f64,
        smallest_singular_value: f64,
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

/// One canonical undirected transition sign.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AtlasSignedEdge {
    pub a: usize,
    pub b: usize,
    /// Identity of a connected overlap component between the same chart pair.
    /// Parallel components are distinct cocycle edges and can themselves close
    /// a two-edge fundamental cycle.
    pub overlap: usize,
    pub sign: i8,
}

impl AtlasSignedEdge {
    #[must_use = "signed-edge validation errors must be handled"]
    pub fn new(a: usize, b: usize, overlap: usize, sign: i8) -> Result<Self, String> {
        if a == b {
            return Err("an atlas transition cannot be a self-edge".to_string());
        }
        if !matches!(sign, -1 | 1) {
            return Err(format!(
                "an atlas transition sign must be +1 or -1, got {sign}"
            ));
        }
        Ok(Self {
            a: a.min(b),
            b: a.max(b),
            overlap,
            sign,
        })
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
                    "exact atlas edge ({}, {}) is outside the {chart_count}-chart atlas",
                    edge.a, edge.b
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
    pub chart: usize,
    pub projection_fit_rows: usize,
    pub inference_rows: usize,
    pub centering: GaussianPatchCentering,
    pub projection_frame: Array2<f64>,
    pub tangent_coordinates: Array2<f64>,
    pub noise_variance_estimate: f64,
    pub signal_variance_estimate: f64,
    pub population_bounds: GaussianPcaPopulationBounds,
}

impl GaussianPcaPatch {
    #[must_use = "Gaussian PCA patch validation errors must be handled"]
    pub fn new(
        chart: usize,
        projection_fit_rows: usize,
        inference_rows: usize,
        centering: GaussianPatchCentering,
        projection_frame: Array2<f64>,
        tangent_coordinates: Array2<f64>,
        noise_variance_estimate: f64,
        signal_variance_estimate: f64,
        population_bounds: GaussianPcaPopulationBounds,
    ) -> Result<Self, String> {
        let (ambient, retained) = projection_frame.dim();
        if projection_fit_rows == 0 {
            return Err(format!(
                "Gaussian PCA patch {chart} projection split must contain at least one row"
            ));
        }
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
        let tangent_scale: f64 = tangent_coordinates
            .iter()
            .map(|value| value * value)
            .sum();
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
        let _ = covariance_dof;
        Ok(Self {
            chart,
            projection_fit_rows,
            inference_rows,
            centering,
            projection_frame,
            tangent_coordinates,
            noise_variance_estimate,
            signal_variance_estimate,
            population_bounds,
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
        self.centering
            .covariance_degrees_of_freedom(self.inference_rows)
            .unwrap_or(0)
    }

    #[must_use]
    pub fn projector_variance_scale(&self) -> f64 {
        let noise = self.noise_variance_estimate;
        let signal = self.signal_variance_estimate;
        noise * (signal + noise)
            / (signal * signal * self.covariance_degrees_of_freedom() as f64)
    }

    fn tangent_frame(&self) -> Array2<f64> {
        self.projection_frame.dot(&self.tangent_coordinates)
    }
}

/// One edge requested from the Gaussian-PCA atlas.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectedAtlasEdgeSpec {
    pub a: usize,
    pub b: usize,
    pub overlap: usize,
    /// Deterministic continuum/discretization error for this edge, in radians.
    /// This is supplied by the geometry builder; it is never inferred from a
    /// grid or hidden angular threshold.
    pub geometric_remainder_bound: f64,
}

impl ProjectedAtlasEdgeSpec {
    #[must_use = "projected-edge validation errors must be handled"]
    pub fn new(
        a: usize,
        b: usize,
        overlap: usize,
        geometric_remainder_bound: f64,
    ) -> Result<Self, String> {
        if a == b {
            return Err("a projected atlas edge cannot be a self-edge".to_string());
        }
        if !(geometric_remainder_bound.is_finite() && geometric_remainder_bound >= 0.0) {
            return Err(format!(
                "edge geometric remainder must be finite and nonnegative, got {geometric_remainder_bound}"
            ));
        }
        Ok(Self {
            a: a.min(b),
            b: a.max(b),
            overlap,
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
    pub estimated_sign: Option<i8>,
    pub transition_a_to_b: Option<[[f64; INTRINSIC_DIMENSION]; INTRINSIC_DIMENSION]>,
    pub geometric_remainder_bound: f64,
}

/// Required occupancy for one patch at the requested orientation error level.
#[derive(Clone, Debug, PartialEq)]
pub struct AtlasPatchSamplePrescription {
    pub chart: usize,
    pub current_rows: usize,
    pub required_rows: usize,
    pub current_covariance_degrees_of_freedom: usize,
    pub required_covariance_degrees_of_freedom: usize,
    pub projected_dimension: usize,
    pub aligned_frame_error_budget: f64,
}

/// Outcome of the two-sided trivial-holonomy test.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtlasCycleConclusion {
    NonTrivialHolonomy,
    NotRejected,
}

/// One fundamental-cycle readout with every uncertainty term exposed.
#[derive(Clone, Debug, PartialEq)]
pub struct AtlasCycleHolonomy {
    pub cycle_index: usize,
    /// Closed chart walk, so the first chart is repeated at the end.
    pub charts: Vec<usize>,
    pub absolute_angle: Option<f64>,
    pub first_order_variance: Option<f64>,
    pub naive_edgewise_first_order_variance: Option<f64>,
    pub shared_patch_covariance_adjustment: Option<f64>,
    pub second_order_variance: Option<f64>,
    pub standard_error: Option<f64>,
    pub polar_linearization_remainder_bound: Option<f64>,
    pub geometric_remainder_bound: f64,
    pub gaussian_error_budget: f64,
    pub subspace_tail_probability_bound: f64,
    pub decision: AtlasStatisticalDecision<AtlasCycleConclusion>,
}

/// A covariance source shared by one or more Gauss--Bonnet angle terms.
#[derive(Clone, Debug, PartialEq)]
pub struct GaussBonnetNoiseSource {
    pub source: usize,
    pub covariance: Array2<f64>,
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
            .max(1.0);
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
        let (eigenvalues, _) = symmetric
            .eigh(faer::Side::Lower)
            .map_err(|error| format!("Gauss-Bonnet covariance eigendecomposition failed: {error}"))?;
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
    pub source: usize,
    pub gradient: Array1<f64>,
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
    pub curvature_estimate: f64,
    pub polar_linearization_remainder_bound: f64,
    pub geometric_remainder_bound: f64,
    pub source_gradients: Vec<GaussBonnetSourceGradient>,
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
            (
                "polar linearization",
                polar_linearization_remainder_bound,
            ),
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
    pub sources: Vec<GaussBonnetNoiseSource>,
    pub contributions: Vec<GaussBonnetContribution>,
}

impl GaussBonnetInput {
    #[must_use = "Gauss-Bonnet input validation errors must be handled"]
    pub fn new(
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
            sources,
            contributions,
        })
    }
}

/// Integer Gauss--Bonnet readout and its shared-source covariance audit.
#[derive(Clone, Debug, PartialEq)]
pub struct GaussBonnetConfidence {
    pub total_curvature_estimate: f64,
    pub nearest_integer_candidate: i64,
    pub residual_to_integer_curvature: f64,
    pub first_order_variance: f64,
    pub naive_contribution_variance: f64,
    pub shared_source_covariance_adjustment: f64,
    pub standard_error: f64,
    pub polar_linearization_remainder_bound: f64,
    pub geometric_remainder_bound: f64,
    pub rounding_margin: f64,
    pub misround_probability_bound: f64,
    pub decision: AtlasStatisticalDecision<i64>,
}

/// Complete noisy-PCA analysis, including refused decisions.
#[derive(Clone, Debug, PartialEq)]
pub struct GaussianPcaHolonomyAnalysis {
    pub familywise_level: AtlasFamilywiseLevel,
    pub chart_count: usize,
    pub edges: Vec<ProjectedAtlasEdgeGeometry>,
    pub orientation: AtlasStatisticalDecision<AtlasOrientability>,
    pub orientation_flip_probability_bound: f64,
    pub sample_prescription: Vec<AtlasPatchSamplePrescription>,
    pub cycles: Vec<AtlasCycleHolonomy>,
    pub gauss_bonnet: Option<GaussBonnetConfidence>,
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

    /// Certified edge signs.  A noisy analysis whose flip bound was refused
    /// intentionally exposes no promotable cocycle.
    #[must_use]
    pub fn certified_edges(&self) -> Option<Vec<AtlasSignedEdge>> {
        match self {
            Self::ExactAnalytic(certificate) => Some(certificate.edges().to_vec()),
            Self::GaussianPcaPlugin(analysis) => {
                analysis.orientation.certified_value()?;
                analysis
                    .edges
                    .iter()
                    .map(|edge| {
                        Some(AtlasSignedEdge {
                            a: edge.a,
                            b: edge.b,
                            overlap: edge.overlap,
                            sign: edge.estimated_sign?,
                        })
                    })
                    .collect()
            }
        }
    }

    #[must_use]
    pub fn certified_orientability(&self) -> Option<AtlasOrientability> {
        match self {
            Self::ExactAnalytic(certificate) => Some(certificate.orientability()),
            Self::GaussianPcaPlugin(analysis) => {
                analysis.orientation.certified_value().copied()
            }
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

#[derive(Clone, Debug)]
struct EdgeWork {
    public: ProjectedAtlasEdgeGeometry,
    transition: Option<Array2<f64>>,
    smallest_singular_value: f64,
    numerical_rank_threshold: f64,
    angle_gradient: Option<Array2<f64>>,
    patch_gradient_a: Option<Array2<f64>>,
    patch_gradient_b: Option<Array2<f64>>,
}

#[derive(Clone, Debug)]
struct FundamentalCycle {
    charts: Vec<usize>,
    steps: Vec<(usize, bool)>,
}

fn orientability_from_edges(
    chart_count: usize,
    edges: &[AtlasSignedEdge],
) -> AtlasOrientability {
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

fn rotation_generator() -> Array2<f64> {
    let mut generator = Array2::<f64>::zeros((INTRINSIC_DIMENSION, INTRINSIC_DIMENSION));
    generator[[0, 1]] = -1.0;
    generator[[1, 0]] = 1.0;
    generator
}

fn frobenius_inner(left: ArrayView2<'_, f64>, right: ArrayView2<'_, f64>) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(&x, &y)| x * y)
        .sum()
}

fn frobenius_squared(matrix: ArrayView2<'_, f64>) -> f64 {
    matrix.iter().map(|value| value * value).sum()
}

fn canonical_edge_key(a: usize, b: usize) -> (usize, usize) {
    (a.min(b), a.max(b))
}

fn project_normal(tangent: ArrayView2<'_, f64>, value: &Array2<f64>) -> Array2<f64> {
    value - &tangent.dot(&tangent.t().dot(value))
}

fn build_projected_edge(
    patches: &[GaussianPcaPatch],
    spec: ProjectedAtlasEdgeSpec,
) -> Result<EdgeWork, String> {
    let patch_a = &patches[spec.a];
    let patch_b = &patches[spec.b];
    let ambient = patch_a.ambient_dimension();
    let retained_a = patch_a.retained_dimension();
    let retained_b = patch_b.retained_dimension();
    let mut joined = Array2::<f64>::zeros((ambient, retained_a + retained_b));
    joined
        .slice_mut(s![.., 0..retained_a])
        .assign(&patch_a.projection_frame);
    joined
        .slice_mut(s![.., retained_a..])
        .assign(&patch_b.projection_frame);
    let (union_left, union_singular, _) = joined
        .svd(true, false)
        .map_err(|error| format!("edge ({}, {}) projection SVD failed: {error}", spec.a, spec.b))?;
    let union_left = union_left.ok_or_else(|| {
        format!(
            "edge ({}, {}) projection SVD did not return requested left vectors",
            spec.a, spec.b
        )
    })?;
    let largest_union_singular = union_singular.first().copied().unwrap_or(0.0);
    let union_rank_threshold = f64::EPSILON
        * ambient.max(retained_a + retained_b) as f64
        * largest_union_singular;
    let projected_dimension = union_singular
        .iter()
        .take_while(|&&value| value > union_rank_threshold)
        .count();
    if projected_dimension < INTRINSIC_DIMENSION {
        return Err(format!(
            "edge ({}, {}) projected union rank {projected_dimension} is below intrinsic dimension {INTRINSIC_DIMENSION}",
            spec.a, spec.b
        ));
    }
    let union = union_left.slice(s![.., 0..projected_dimension]);
    let tangent_a = patch_a.tangent_frame();
    let tangent_b = patch_b.tangent_frame();
    let tangent_a_projected = union.t().dot(&tangent_a);
    let tangent_b_projected = union.t().dot(&tangent_b);
    // Coordinates are mapped from chart a to chart b, hence M = U_b^T U_a.
    let cross = tangent_b_projected.t().dot(&tangent_a_projected);
    let (left, singular, right_t) = cross
        .svd(true, true)
        .map_err(|error| format!("edge ({}, {}) cross-Gram SVD failed: {error}", spec.a, spec.b))?;
    let left = left.ok_or_else(|| {
        format!(
            "edge ({}, {}) cross-Gram SVD omitted requested left vectors",
            spec.a, spec.b
        )
    })?;
    let right_t = right_t.ok_or_else(|| {
        format!(
            "edge ({}, {}) cross-Gram SVD omitted requested right vectors",
            spec.a, spec.b
        )
    })?;
    if singular.len() != INTRINSIC_DIMENSION {
        return Err(format!(
            "edge ({}, {}) cross-Gram has {} singular values, expected {INTRINSIC_DIMENSION}",
            spec.a,
            spec.b,
            singular.len()
        ));
    }
    let largest_singular_value = singular[0];
    let smallest_singular_value = singular[INTRINSIC_DIMENSION - 1];
    let numerical_rank_threshold = f64::EPSILON
        * INTRINSIC_DIMENSION as f64
        * largest_singular_value.max(1.0);
    let orientation_margin = determinant_2(cross.view()).abs();
    let principal_angle_cosines = [
        singular[0].clamp(0.0, 1.0),
        singular[1].clamp(0.0, 1.0),
    ];
    if smallest_singular_value <= numerical_rank_threshold {
        return Ok(EdgeWork {
            public: ProjectedAtlasEdgeGeometry {
                a: spec.a,
                b: spec.b,
                overlap: spec.overlap,
                projected_dimension,
                principal_angle_cosines,
                orientation_margin,
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
    let raw_a = tangent_b.dot(&angle_gradient);
    let raw_b = tangent_a.dot(&angle_gradient.t());
    let patch_gradient_a = project_normal(tangent_a.view(), &raw_a);
    let patch_gradient_b = project_normal(tangent_b.view(), &raw_b);
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
                    tree_edges.insert(edge_index);
                    queue.push_back(next);
                }
            }
        }
    }
    let tree_edge_lookup: BTreeMap<(usize, usize), usize> = tree_edges
        .iter()
        .map(|&index| ((edges[index].a, edges[index].b), index))
        .collect();
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
                    "non-tree edge ({}, {}) joins different spanning-forest components",
                    chord.a, chord.b
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
            let key = canonical_edge_key(endpoints[0], endpoints[1]);
            let edge_index = *tree_edge_lookup.get(&key).ok_or_else(|| {
                format!("fundamental-cycle tree edge {key:?} is absent")
            })?;
            let forward = endpoints[0] == edges[edge_index].a;
            steps.push((edge_index, forward));
        }
        let chord_from = walk[walk.len() - 2];
        steps.push((chord_index, chord_from == chord.a));
        cycles.push(FundamentalCycle {
            charts: walk,
            steps,
        });
    }
    Ok(cycles)
}

fn aligned_frame_error(projector_error: f64) -> f64 {
    let q = projector_error.clamp(0.0, 1.0);
    (2.0 - 2.0 * (1.0 - q * q).sqrt()).sqrt()
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
) -> PatchTail {
    let degrees_of_freedom = patch.covariance_degrees_of_freedom() as f64;
    let u = ((projected_dimension as f64).sqrt() + (2.0 * tail_parameter).sqrt())
        / degrees_of_freedom.sqrt();
    let covariance_error =
        patch.population_bounds.spectral_radius_upper() * (2.0 * u + u * u);
    let projector_error =
        2.0 * covariance_error / patch.population_bounds.eigengap_lower;
    PatchTail {
        covariance_error,
        projector_error,
        aligned_frame_error: aligned_frame_error(projector_error),
    }
}

fn orientation_tail_and_prescription(
    patches: &[GaussianPcaPatch],
    edges: &[EdgeWork],
    allocated_alpha: f64,
) -> (
    AtlasStatisticalDecision<AtlasOrientability>,
    f64,
    Vec<AtlasPatchSamplePrescription>,
) {
    let mut reasons = Vec::new();
    for edge in edges {
        if edge.public.estimated_sign.is_none() {
            reasons.push(AtlasStatisticalRefusal::SingularProjectedCrossGram {
                a: edge.public.a,
                b: edge.public.b,
                smallest_singular_value: edge.smallest_singular_value,
                numerical_rank_threshold: edge.numerical_rank_threshold,
            });
        }
    }
    if edges.is_empty() {
        return (
            AtlasStatisticalDecision::Certified {
                value: AtlasOrientability::Orientable,
                error_probability_bound: 0.0,
            },
            0.0,
            Vec::new(),
        );
    }
    let mut frame_budgets = vec![f64::INFINITY; patches.len()];
    let mut projected_dimensions = vec![0usize; patches.len()];
    for edge in edges {
        // If both endpoints stay within m/4 in aligned-frame norm, their
        // cross-Gram differs by at most m/2 < m, so its determinant cannot
        // cross zero.  The unused half-margin makes the strict inequality
        // explicit without a floating threshold.
        let endpoint_budget = edge.public.orientation_margin / 4.0;
        for chart in [edge.public.a, edge.public.b] {
            frame_budgets[chart] = frame_budgets[chart].min(endpoint_budget);
            projected_dimensions[chart] =
                projected_dimensions[chart].max(edge.public.projected_dimension);
        }
    }
    let active_patches = projected_dimensions.iter().filter(|&&rank| rank > 0).count();
    let requested_tail_parameter =
        (2.0 * active_patches as f64 / allocated_alpha).ln();
    let mut prescriptions = Vec::with_capacity(active_patches);
    let mut flip_probability_bound = 0.0_f64;
    for (chart, patch) in patches.iter().enumerate() {
        let projected_dimension = projected_dimensions[chart];
        if projected_dimension == 0 {
            continue;
        }
        let frame_budget = frame_budgets[chart];
        let projector_budget = projector_error_for_aligned_frame_error(frame_budget);
        let covariance_budget =
            patch.population_bounds.eigengap_lower * projector_budget / 2.0;
        let normalized_budget =
            covariance_budget / patch.population_bounds.spectral_radius_upper();
        let u_budget = (1.0 + normalized_budget).sqrt() - 1.0;
        let numerator = (projected_dimension as f64).sqrt()
            + (2.0 * requested_tail_parameter).sqrt();
        let required_dof_f64 = (numerator / u_budget).powi(2).ceil();
        let required_dof = if required_dof_f64.is_finite()
            && required_dof_f64 <= usize::MAX as f64
        {
            required_dof_f64 as usize
        } else {
            usize::MAX
        };
        let required_rows = patch
            .centering
            .rows_for_degrees_of_freedom(required_dof)
            .unwrap_or(usize::MAX);
        prescriptions.push(AtlasPatchSamplePrescription {
            chart,
            current_rows: patch.inference_rows,
            required_rows,
            current_covariance_degrees_of_freedom: patch.covariance_degrees_of_freedom(),
            required_covariance_degrees_of_freedom: required_dof,
            projected_dimension,
            aligned_frame_error_budget: frame_budget,
        });
        let available = (patch.covariance_degrees_of_freedom() as f64).sqrt() * u_budget
            - (projected_dimension as f64).sqrt();
        let supported_tail_parameter = if available > 0.0 {
            available * available / 2.0
        } else {
            0.0
        };
        flip_probability_bound += (2.0 * (-supported_tail_parameter).exp()).min(1.0);
    }
    flip_probability_bound = flip_probability_bound.min(1.0);
    if flip_probability_bound > allocated_alpha {
        reasons.push(
            AtlasStatisticalRefusal::OrientationFlipBoundExceedsLevel {
                flip_probability_bound,
                allocated_alpha,
            },
        );
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
    (decision, flip_probability_bound, prescriptions)
}

fn edge_step_matrix(edge: &EdgeWork, forward: bool) -> Option<Array2<f64>> {
    let transition = edge.transition.as_ref()?;
    Some(if forward {
        transition.clone()
    } else {
        transition.t().to_owned()
    })
}

fn analyze_cycle(
    cycle_index: usize,
    cycle: &FundamentalCycle,
    patches: &[GaussianPcaPatch],
    edges: &[EdgeWork],
    allocated_alpha: f64,
) -> Result<AtlasCycleHolonomy, String> {
    let gaussian_error_budget = allocated_alpha / 2.0;
    let subspace_tail_probability_bound = allocated_alpha - gaussian_error_budget;
    let geometric_remainder_bound: f64 = cycle
        .steps
        .iter()
        .map(|&(edge, _)| edges[edge].public.geometric_remainder_bound)
        .sum();
    let mut reasons = Vec::new();
    for &(edge_index, _) in &cycle.steps {
        let edge = &edges[edge_index];
        if edge.transition.is_none() {
            reasons.push(AtlasStatisticalRefusal::SingularProjectedCrossGram {
                a: edge.public.a,
                b: edge.public.b,
                smallest_singular_value: edge.smallest_singular_value,
                numerical_rank_threshold: edge.numerical_rank_threshold,
            });
        }
    }
    if !reasons.is_empty() {
        return Ok(AtlasCycleHolonomy {
            cycle_index,
            charts: cycle.charts.clone(),
            absolute_angle: None,
            first_order_variance: None,
            naive_edgewise_first_order_variance: None,
            shared_patch_covariance_adjustment: None,
            second_order_variance: None,
            standard_error: None,
            polar_linearization_remainder_bound: None,
            geometric_remainder_bound,
            gaussian_error_budget,
            subspace_tail_probability_bound,
            decision: AtlasStatisticalDecision::Refused { reasons },
        });
    }
    let step_matrices: Vec<Array2<f64>> = cycle
        .steps
        .iter()
        .map(|&(edge, forward)| {
            edge_step_matrix(&edges[edge], forward).ok_or_else(|| {
                format!(
                    "cycle {cycle_index} edge ({}, {}) lost its validated polar transition",
                    edges[edge].public.a, edges[edge].public.b
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
        return Ok(AtlasCycleHolonomy {
            cycle_index,
            charts: cycle.charts.clone(),
            absolute_angle: None,
            first_order_variance: None,
            naive_edgewise_first_order_variance: None,
            shared_patch_covariance_adjustment: None,
            second_order_variance: None,
            standard_error: None,
            polar_linearization_remainder_bound: None,
            geometric_remainder_bound,
            gaussian_error_budget,
            subspace_tail_probability_bound,
            decision: AtlasStatisticalDecision::Refused {
                reasons: vec![AtlasStatisticalRefusal::ImproperCycleHolonomy {
                    cycle_index,
                }],
            },
        });
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
        let canonical = edges[edge_index].transition.as_ref().ok_or_else(|| {
            format!(
                "cycle {cycle_index} edge ({}, {}) lost its validated transition during derivative assembly",
                edges[edge_index].public.a, edges[edge_index].public.b
            )
        })?;
        let canonical_tangent = canonical.dot(&generator);
        let step_tangent = if forward {
            canonical_tangent
        } else {
            -canonical_tangent.t().to_owned()
        };
        coefficients.push(frobenius_inner(
            transition_gradient.view(),
            step_tangent.view(),
        ));
    }
    let ambient = patches[0].ambient_dimension();
    let mut patch_gradients = vec![
        Array2::<f64>::zeros((ambient, INTRINSIC_DIMENSION));
        patches.len()
    ];
    let mut naive_first_order_variance = 0.0_f64;
    let mut second_order_variance = 0.0_f64;
    for (position, &(edge_index, _)) in cycle.steps.iter().enumerate() {
        let coefficient = coefficients[position];
        let edge = &edges[edge_index];
        let gradient_a = edge.patch_gradient_a.as_ref().ok_or_else(|| {
            format!(
                "cycle {cycle_index} edge ({}, {}) lost patch-a projector gradient",
                edge.public.a, edge.public.b
            )
        })?;
        let gradient_b = edge.patch_gradient_b.as_ref().ok_or_else(|| {
            format!(
                "cycle {cycle_index} edge ({}, {}) lost patch-b projector gradient",
                edge.public.a, edge.public.b
            )
        })?;
        patch_gradients[edge.public.a].scaled_add(coefficient, gradient_a);
        patch_gradients[edge.public.b].scaled_add(coefficient, gradient_b);
        naive_first_order_variance += coefficient * coefficient
            * (patches[edge.public.a].projector_variance_scale()
                * frobenius_squared(gradient_a.view())
                + patches[edge.public.b].projector_variance_scale()
                    * frobenius_squared(gradient_b.view()));
        second_order_variance += coefficient * coefficient
            * (edge.public.projected_dimension - INTRINSIC_DIMENSION) as f64
            * patches[edge.public.a].projector_variance_scale()
            * patches[edge.public.b].projector_variance_scale()
            / 2.0;
    }
    let first_order_variance: f64 = patch_gradients
        .iter()
        .zip(patches)
        .map(|(gradient, patch)| {
            patch.projector_variance_scale() * frobenius_squared(gradient.view())
        })
        .sum();
    let shared_patch_covariance_adjustment =
        first_order_variance - naive_first_order_variance;
    let standard_error = (first_order_variance + second_order_variance).sqrt();
    let absolute_angle = (holonomy[[1, 0]] - holonomy[[0, 1]])
        .atan2(holonomy[[0, 0]] + holonomy[[1, 1]])
        .abs();

    let cycle_charts: BTreeSet<usize> = cycle.charts.iter().copied().collect();
    let tail_parameter =
        (2.0 * cycle_charts.len() as f64 / subspace_tail_probability_bound).ln();
    let mut patch_projected_dimension = vec![0usize; patches.len()];
    for &(edge_index, _) in &cycle.steps {
        let edge = &edges[edge_index].public;
        patch_projected_dimension[edge.a] =
            patch_projected_dimension[edge.a].max(edge.projected_dimension);
        patch_projected_dimension[edge.b] =
            patch_projected_dimension[edge.b].max(edge.projected_dimension);
    }
    let mut patch_tails = vec![None; patches.len()];
    for chart in cycle_charts {
        let tail = patch_tail(
            &patches[chart],
            patch_projected_dimension[chart],
            tail_parameter,
        );
        if tail.covariance_error >= patches[chart].population_bounds.eigengap_lower / 2.0
            || tail.projector_error >= 1.0
        {
            reasons.push(AtlasStatisticalRefusal::PatchTailCrossesEigengap {
                chart,
                covariance_error_bound: tail.covariance_error,
                eigengap_lower: patches[chart].population_bounds.eigengap_lower,
            });
        }
        patch_tails[chart] = Some(tail);
    }
    let mut polar_linearization_remainder_bound = 0.0_f64;
    for (position, &(edge_index, _)) in cycle.steps.iter().enumerate() {
        let edge = &edges[edge_index];
        let tail_a = patch_tails[edge.public.a];
        let tail_b = patch_tails[edge.public.b];
        let (Some(tail_a), Some(tail_b)) = (tail_a, tail_b) else {
            continue;
        };
        let cross_error = tail_a.aligned_frame_error + tail_b.aligned_frame_error;
        if cross_error >= edge.smallest_singular_value {
            reasons.push(AtlasStatisticalRefusal::PolarLinearizationUnresolved {
                cycle_index,
                a: edge.public.a,
                b: edge.public.b,
                cross_gram_error_bound: cross_error,
                smallest_singular_value: edge.smallest_singular_value,
            });
            continue;
        }
        let polar_difference =
            2.0 * cross_error / (2.0 * edge.smallest_singular_value - cross_error);
        if polar_difference >= 2.0 {
            reasons.push(AtlasStatisticalRefusal::PolarLinearizationUnresolved {
                cycle_index,
                a: edge.public.a,
                b: edge.public.b,
                cross_gram_error_bound: cross_error,
                smallest_singular_value: edge.smallest_singular_value,
            });
            continue;
        }
        let total_angle_change = 2.0 * (polar_difference / 2.0).asin();
        let gradient_norm = edge
            .angle_gradient
            .as_ref()
            .map(|gradient| frobenius_squared(gradient.view()).sqrt())
            .unwrap_or(0.0);
        let linear_change_bound =
            gradient_norm * (INTRINSIC_DIMENSION as f64).sqrt() * cross_error;
        polar_linearization_remainder_bound += coefficients[position].abs()
            * (total_angle_change + linear_change_bound);
    }
    if !reasons.is_empty() {
        return Ok(AtlasCycleHolonomy {
            cycle_index,
            charts: cycle.charts.clone(),
            absolute_angle: Some(absolute_angle),
            first_order_variance: Some(first_order_variance),
            naive_edgewise_first_order_variance: Some(naive_first_order_variance),
            shared_patch_covariance_adjustment: Some(shared_patch_covariance_adjustment),
            second_order_variance: Some(second_order_variance),
            standard_error: Some(standard_error),
            polar_linearization_remainder_bound: Some(
                polar_linearization_remainder_bound,
            ),
            geometric_remainder_bound,
            gaussian_error_budget,
            subspace_tail_probability_bound,
            decision: AtlasStatisticalDecision::Refused { reasons },
        });
    }
    let normal = Normal::new(0.0, 1.0)
        .map_err(|error| format!("standard-normal construction failed: {error}"))?;
    let critical = normal.inverse_cdf(1.0 - gaussian_error_budget / 2.0);
    let rejection_boundary = critical * standard_error
        + polar_linearization_remainder_bound
        + geometric_remainder_bound;
    if std::f64::consts::PI - absolute_angle <= rejection_boundary {
        return Ok(AtlasCycleHolonomy {
            cycle_index,
            charts: cycle.charts.clone(),
            absolute_angle: Some(absolute_angle),
            first_order_variance: Some(first_order_variance),
            naive_edgewise_first_order_variance: Some(naive_first_order_variance),
            shared_patch_covariance_adjustment: Some(shared_patch_covariance_adjustment),
            second_order_variance: Some(second_order_variance),
            standard_error: Some(standard_error),
            polar_linearization_remainder_bound: Some(
                polar_linearization_remainder_bound,
            ),
            geometric_remainder_bound,
            gaussian_error_budget,
            subspace_tail_probability_bound,
            decision: AtlasStatisticalDecision::Refused {
                reasons: vec![AtlasStatisticalRefusal::CycleAngleBranchCutCrossed {
                    cycle_index,
                    absolute_angle,
                    uncertainty_radius: rejection_boundary,
                }],
            },
        });
    }
    let conclusion = if absolute_angle > rejection_boundary {
        AtlasCycleConclusion::NonTrivialHolonomy
    } else {
        AtlasCycleConclusion::NotRejected
    };
    Ok(AtlasCycleHolonomy {
        cycle_index,
        charts: cycle.charts.clone(),
        absolute_angle: Some(absolute_angle),
        first_order_variance: Some(first_order_variance),
        naive_edgewise_first_order_variance: Some(naive_first_order_variance),
        shared_patch_covariance_adjustment: Some(shared_patch_covariance_adjustment),
        second_order_variance: Some(second_order_variance),
        standard_error: Some(standard_error),
        polar_linearization_remainder_bound: Some(polar_linearization_remainder_bound),
        geometric_remainder_bound,
        gaussian_error_budget,
        subspace_tail_probability_bound,
        decision: AtlasStatisticalDecision::Certified {
            value: conclusion,
            error_probability_bound: gaussian_error_budget
                + subspace_tail_probability_bound,
        },
    })
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
    let variance_scale = naive_contribution_variance.abs().max(1.0);
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
    let standard_error = first_order_variance.sqrt();
    let integer_f64 = (total_curvature_estimate / std::f64::consts::TAU).round();
    if !(integer_f64.is_finite()
        && integer_f64 >= i64::MIN as f64
        && integer_f64 <= i64::MAX as f64)
    {
        return Err("Gauss-Bonnet integer candidate is outside i64 range".to_string());
    }
    let nearest_integer_candidate = integer_f64 as i64;
    let residual_to_integer_curvature = (total_curvature_estimate
        - std::f64::consts::TAU * nearest_integer_candidate as f64)
        .abs();
    let total_remainder =
        polar_linearization_remainder_bound + geometric_remainder_bound;
    let rounding_margin = std::f64::consts::PI
        - residual_to_integer_curvature
        - total_remainder;
    let (misround_probability_bound, decision) = if rounding_margin <= 0.0 {
        (
            1.0,
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
        let probability = if standard_error == 0.0 {
            0.0
        } else {
            let normal = Normal::new(0.0, 1.0)
                .map_err(|error| format!("standard-normal construction failed: {error}"))?;
            (2.0 * (1.0 - normal.cdf(rounding_margin / standard_error))).clamp(0.0, 1.0)
        };
        if probability <= allocated_alpha {
            (
                probability,
                AtlasStatisticalDecision::Certified {
                    value: nearest_integer_candidate,
                    error_probability_bound: probability,
                },
            )
        } else {
            (
                probability,
                AtlasStatisticalDecision::Refused {
                    reasons: vec![
                        AtlasStatisticalRefusal::GaussBonnetErrorBoundExceedsLevel {
                            misround_probability_bound: probability,
                            allocated_alpha,
                        },
                    ],
                },
            )
        }
    };
    Ok(GaussBonnetConfidence {
        total_curvature_estimate,
        nearest_integer_candidate,
        residual_to_integer_curvature,
        first_order_variance,
        naive_contribution_variance,
        shared_source_covariance_adjustment: first_order_variance
            - naive_contribution_variance,
        standard_error,
        polar_linearization_remainder_bound,
        geometric_remainder_bound,
        rounding_margin,
        misround_probability_bound,
        decision,
    })
}

impl GaussianPcaHolonomyAnalysis {
    /// Build the complete projected PCA holonomy analysis.
    #[must_use = "Gaussian PCA holonomy construction errors must be handled"]
    pub fn certify(
        mut patches: Vec<GaussianPcaPatch>,
        mut edge_specs: Vec<ProjectedAtlasEdgeSpec>,
        familywise_level: AtlasFamilywiseLevel,
        gauss_bonnet_input: Option<GaussBonnetInput>,
    ) -> Result<Self, String> {
        patches.sort_by_key(|patch| patch.chart);
        for (expected, patch) in patches.iter().enumerate() {
            if patch.chart != expected {
                return Err(format!(
                    "Gaussian PCA patch indices must be contiguous: position {expected} contains chart {}",
                    patch.chart
                ));
            }
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
        edge_specs.sort_by_key(|edge| (edge.a, edge.b, edge.overlap));
        for (position, edge) in edge_specs.iter().enumerate() {
            if edge.b >= chart_count {
                return Err(format!(
                    "projected atlas edge ({}, {}) is outside the {chart_count}-chart atlas",
                    edge.a, edge.b
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
            orientation_tail_and_prescription(&patches, &edge_work, allocated_alpha);
        let cycles = fundamental
            .iter()
            .enumerate()
            .map(|(index, cycle)| {
                analyze_cycle(index, cycle, &patches, &edge_work, allocated_alpha)
            })
            .collect::<Result<_, _>>()?;
        let gauss_bonnet = gauss_bonnet_input
            .as_ref()
            .map(|input| gauss_bonnet_confidence(input, allocated_alpha))
            .transpose()?;
        Ok(Self {
            familywise_level,
            chart_count,
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
        familywise_level: AtlasFamilywiseLevel,
        gauss_bonnet_input: Option<GaussBonnetInput>,
    ) -> Result<Self, String> {
        Ok(Self::GaussianPcaPlugin(
            GaussianPcaHolonomyAnalysis::certify(
                patches,
                edge_specs,
                familywise_level,
                gauss_bonnet_input,
            )?,
        ))
    }
}
