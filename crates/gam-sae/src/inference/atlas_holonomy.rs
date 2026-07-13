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
    pub sign: i8,
}

impl AtlasSignedEdge {
    #[must_use = "signed-edge validation errors must be handled"]
    pub fn new(a: usize, b: usize, sign: i8) -> Result<Self, String> {
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
        edges.sort_by_key(|edge| (edge.a, edge.b));
        for (position, edge) in edges.iter().enumerate() {
            if edge.b >= chart_count {
                return Err(format!(
                    "exact atlas edge ({}, {}) is outside the {chart_count}-chart atlas",
                    edge.a, edge.b
                ));
            }
            if position > 0
                && (edges[position - 1].a, edges[position - 1].b) == (edge.a, edge.b)
            {
                return Err(format!(
                    "duplicate exact atlas edge ({}, {})",
                    edge.a, edge.b
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

/// One locally projected PCA patch in the isotropic-spike model.
///
/// The first two columns of `retained_frame` are the estimated tangent frame;
/// remaining columns are caller-retained normal PCs.  The caller, rather than a
/// hidden `q`, chooses how many PCs are retained.  Rows are assumed independently
/// Gaussian and centered by a known or independently estimated mean, so
/// `sample_count` is the covariance sample size appearing in the Wishart tail.
#[derive(Clone, Debug, PartialEq)]
pub struct GaussianPcaPatch {
    pub chart: usize,
    pub sample_count: usize,
    pub retained_frame: Array2<f64>,
    pub noise_variance_estimate: f64,
    pub signal_variance_estimate: f64,
    pub population_bounds: GaussianPcaPopulationBounds,
}

impl GaussianPcaPatch {
    #[must_use = "Gaussian PCA patch validation errors must be handled"]
    pub fn new(
        chart: usize,
        sample_count: usize,
        retained_frame: Array2<f64>,
        noise_variance_estimate: f64,
        signal_variance_estimate: f64,
        population_bounds: GaussianPcaPopulationBounds,
    ) -> Result<Self, String> {
        let (ambient, retained) = retained_frame.dim();
        if sample_count == 0 {
            return Err(format!(
                "Gaussian PCA patch {chart} must contain at least one independent row"
            ));
        }
        if ambient == 0 || retained < INTRINSIC_DIMENSION || retained > ambient {
            return Err(format!(
                "Gaussian PCA patch {chart} frame shape ({ambient}, {retained}) must satisfy ambient >= retained >= {INTRINSIC_DIMENSION}"
            ));
        }
        if retained_frame.iter().any(|value| !value.is_finite()) {
            return Err(format!(
                "Gaussian PCA patch {chart} retained frame must be finite"
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
        let gram = retained_frame.t().dot(&retained_frame);
        let frame_scale: f64 = retained_frame.iter().map(|value| value * value).sum();
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
        Ok(Self {
            chart,
            sample_count,
            retained_frame,
            noise_variance_estimate,
            signal_variance_estimate,
            population_bounds,
        })
    }

    #[must_use]
    pub fn ambient_dimension(&self) -> usize {
        self.retained_frame.nrows()
    }

    #[must_use]
    pub fn retained_dimension(&self) -> usize {
        self.retained_frame.ncols()
    }

    #[must_use]
    pub fn projector_variance_scale(&self) -> f64 {
        let noise = self.noise_variance_estimate;
        let signal = self.signal_variance_estimate;
        noise * (signal + noise) / (signal * signal * self.sample_count as f64)
    }

    fn tangent_frame(&self) -> ArrayView2<'_, f64> {
        self.retained_frame.slice(s![.., 0..INTRINSIC_DIMENSION])
    }
}

/// One edge requested from the Gaussian-PCA atlas.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectedAtlasEdgeSpec {
    pub a: usize,
    pub b: usize,
    /// Deterministic continuum/discretization error for this edge, in radians.
    /// This is supplied by the geometry builder; it is never inferred from a
    /// grid or hidden angular threshold.
    pub geometric_remainder_bound: f64,
}

impl ProjectedAtlasEdgeSpec {
    #[must_use = "projected-edge validation errors must be handled"]
    pub fn new(a: usize, b: usize, geometric_remainder_bound: f64) -> Result<Self, String> {
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
            geometric_remainder_bound,
        })
    }
}

/// Public relative geometry of one projected two-patch edge.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectedAtlasEdgeGeometry {
    pub a: usize,
    pub b: usize,
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
        let mut queue = VecDeque::from([root]);
        while let Some(chart) = queue.pop_front() {
            let here = orientations[chart].expect("queued chart is oriented");
            for &(next, sign) in &adjacency[chart] {
                let required = here * sign;
                match orientations[next] {
                    Some(existing) if existing != required => {
                        return AtlasOrientability::NonOrientable;
                    }
                    Some(_) => {}
                    None => {
                        orientations[next] = Some(required);
                        queue.push_back(next);
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

