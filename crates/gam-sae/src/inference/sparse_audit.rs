//! Sparse-route SAE audit: the measurement pipeline behind the `audit_sae` /
//! `atlas_nerve_diagram` Python surface.
//!
//! An external SAE arrives frozen — a decoder, a fixed-width sparse route, the
//! activations it was read from, and an architecture-matched donor route. This
//! module measures it: the routability floor, the global-optimality dual
//! certificate, per-block circle coordinates, per-atom topology records, the
//! atlas nerve (with cross-fitted Gaussian-PCA holonomy when ambient rows and a
//! familywise level are supplied), feature-absorption pairs, the
//! circle-transport class, and the standing architecture-matched null plus
//! residual spike-in calibration that prices the topological claims.
//!
//! The route is never densified: no step materializes the logical `N x K` code
//! matrix, so cost tracks the live entries rather than the dictionary width.
//!
//! The route type and the surrogate resampler live in [`crate::null_sampler`]
//! (#2470, `b94f8714a`); this module is the audit that consumes them. The
//! standing calibration stays HERE rather than in [`crate::null_battery`]
//! because its observed statistic is the atlas-nerve richness — a null module
//! must not depend upward on the atlas machinery that already consumes it.
//! Parameterizing `null_battery` over a caller-supplied statistic is the
//! follow-up that would let the calibration descend.
//!
//! This lives here rather than in the binding because every number above is a
//! measurement, not marshalling (SPEC rule 8). The PyO3 layer decodes the
//! options dict, calls [`run_sparse_sae_audit`], and turns
//! [`SparseSaeAuditReport`] into Python objects.


pub struct AbsorptionPairReport {
    pub a: usize,
    pub b: usize,
    pub n_obs: usize,
    pub n_a: usize,
    pub n_b: usize,
    pub n_joint: usize,
    pub p_a_given_b: f64,
    pub p_b_given_a: f64,
    pub lift: f64,
    pub weight_correlation: f64,
    pub dependence: f64,
    pub fusion_evidence: f64,
    pub absorption_asymmetry: f64,
}

pub struct AbsorptionAuditReport {
    pub n_units: usize,
    pub activation_threshold: f32,
    pub pairs: Vec<AbsorptionPairReport>,
}

#[derive(Clone)]
pub struct AuditTopologyRecord {
    pub atom: usize,
    pub support_size: usize,
    pub landmark_count: usize,
    pub covering_side: String,
    pub measured_betti: crate::manifold::BettiSignature,
    pub expected_betti: crate::manifold::BettiSignature,
    pub contested: bool,
    pub dominant_h1_persistence: f64,
    pub dominant_h2_persistence: f64,
    pub note: String,
}

pub struct AuditAtlasReport {
    pub chart_blocks: Vec<usize>,
    pub diagram: crate::inference::atlas_nerve::AtlasNerveDiagram,
    pub holonomy_unavailable_reason: Option<String>,
}

pub fn atlas_refusal_code(
    refusal: &crate::inference::atlas_holonomy::AtlasStatisticalRefusal,
) -> &'static str {
    use crate::inference::atlas_holonomy::AtlasStatisticalRefusal;
    match refusal {
        AtlasStatisticalRefusal::PilotProjectionUncertified { .. } => {
            "pilot_projection_uncertified"
        }
        AtlasStatisticalRefusal::PopulationSpectrumUncertified { .. } => {
            "population_spectrum_uncertified"
        }
        AtlasStatisticalRefusal::GaussianLinearizationIsPlugin { .. } => {
            "gaussian_linearization_is_plugin"
        }
        AtlasStatisticalRefusal::DegenerateFirstOrderLimitUnresolved { .. } => {
            "degenerate_first_order_limit_unresolved"
        }
        AtlasStatisticalRefusal::PopulationCrossGramMarginUncertified { .. } => {
            "population_cross_gram_margin_uncertified"
        }
        AtlasStatisticalRefusal::SingularProjectedCrossGram { .. } => {
            "singular_projected_cross_gram"
        }
        AtlasStatisticalRefusal::PatchTailCrossesEigengap { .. } => "patch_tail_crosses_eigengap",
        AtlasStatisticalRefusal::OrientationFlipBoundExceedsLevel { .. } => {
            "orientation_flip_bound_exceeds_level"
        }
        AtlasStatisticalRefusal::ImproperCycleHolonomy { .. } => "improper_cycle_holonomy",
        AtlasStatisticalRefusal::PolarLinearizationUnresolved { .. } => {
            "polar_linearization_unresolved"
        }
        AtlasStatisticalRefusal::CycleAngleBranchCutCrossed { .. } => {
            "cycle_angle_branch_cut_crossed"
        }
        AtlasStatisticalRefusal::GaussBonnetRoundingMarginExhausted { .. } => {
            "gauss_bonnet_rounding_margin_exhausted"
        }
        AtlasStatisticalRefusal::GaussBonnetErrorBoundExceedsLevel { .. } => {
            "gauss_bonnet_error_bound_exceeds_level"
        }
        AtlasStatisticalRefusal::GaussBonnetGaussianLinearizationIsPlugin => {
            "gauss_bonnet_gaussian_linearization_is_plugin"
        }
        AtlasStatisticalRefusal::GaussBonnetFirstOrderLimitDegenerate { .. } => {
            "gauss_bonnet_first_order_limit_degenerate"
        }
    }
}

/// Monte-Carlo operating points for the standing null battery / spike-in
/// calibration `audit_sae` attaches to its topology and atlas-nerve claims.
/// Surfaced on the FFI so callers can widen the null replicate count or move the
/// spike-in operating point; the FFI defaults are the reporting operating
/// points.
#[derive(Clone, Copy, Debug)]
pub struct StandingCalibrationConfig {
    pub null_replicates: usize,
    pub null_seed: u64,
    pub spikein_trials: usize,
    pub spikein_snr: f64,
    pub spikein_false_positive_rate: f64,
}

/// Knobs the sparse-route audit runs under. Every field is a reporting operating
/// point rather than a correctness switch: the caller may widen the null
/// replicate count or move the spike-in point without changing what the audit
/// certifies.
#[derive(Clone, Debug)]
pub struct SparseSaeAuditConfig {
    pub block_size: usize,
    pub delta: f64,
    /// Optimality-ratio quantiles reported by the routability audit.
    pub quantile_levels: Vec<f64>,
    pub max_candidates: usize,
    /// Dictionary blocks promoted to atlas charts; `None` selects every block.
    pub coordinate_blocks: Option<Vec<usize>>,
    pub activation_threshold: f32,
    pub max_absorption_pairs: usize,
    pub transport_theta_in: Option<Vec<f64>>,
    pub transport_theta_out: Option<Vec<f64>>,
    pub transport_layer_from: usize,
    pub transport_layer_to: usize,
    pub calibration: StandingCalibrationConfig,
}

/// The frozen artifacts one audit reads: the dictionary, the observed route, the
/// activations it was read from, and the architecture-matched donor route the
/// standing null resamples. Bundled so the entry keeps a one-argument signature
/// and each array is named at the call site rather than positional among six
/// same-shaped neighbours.
pub struct SparseSaeAuditRequest {
    pub decoder: ndarray::Array2<f32>,
    pub route_indices: ndarray::Array2<u32>,
    pub route_values: ndarray::Array3<f32>,
    pub data: ndarray::Array2<f32>,
    pub donor_indices: ndarray::Array2<u32>,
    pub donor_values: ndarray::Array3<f32>,
    pub config: SparseSaeAuditConfig,
}

/// Everything one sparse-route audit measures. The binding marshals these
/// fields; it recomputes none of them.
pub struct SparseSaeAuditReport {
    pub block_size: usize,
    pub n_units: usize,
    pub route_rows: usize,
    pub route_width: usize,
    pub decoder_shape: (usize, usize),
    pub routability: crate::routability::RoutabilityAudit,
    pub dual: crate::dual_certificate::DualCertificateReport,
    pub coordinate_reports: Vec<crate::sparse_dict::BlockCoordinateReport>,
    pub topology_records: Vec<AuditTopologyRecord>,
    pub atlas_nerve: Option<AuditAtlasReport>,
    pub absorption: AbsorptionAuditReport,
    pub transport: Option<crate::inference::transport_class::CircleTransportReport>,
    pub calibration: Option<crate::null_battery::ClaimNullCalibration>,
}

