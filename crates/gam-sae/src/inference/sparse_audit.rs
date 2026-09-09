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

use crate::null_sampler::AuditSparseRoute;
use crate::null_sampler::resample_sparse_architecture_null;

#[derive(Clone, Copy, Default)]
struct SparsePairAccum {
    n_joint: usize,
    sum_a: f64,
    sum_b: f64,
    sum_a2: f64,
    sum_b2: f64,
    sum_ab: f64,
}
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

pub fn atlas_nerve_from_sparse_route(
    route: &AuditSparseRoute,
    activation_threshold: f32,
    requested_blocks: Option<&[usize]>,
    ambient_data: Option<ndarray::ArrayView2<'_, f64>>,
    familywise_alpha: Option<f64>,
) -> Result<Option<AuditAtlasReport>, String> {
    if route.block_size == 1 {
        return Ok(None);
    }
    let n_blocks = route.n_units;
    let chart_blocks: Vec<usize> = match requested_blocks {
        Some(blocks) => blocks.to_vec(),
        None => (0..n_blocks).collect(),
    };
    if chart_blocks.len() < 2 {
        return Ok(None);
    }
    let mut chart_positions = std::collections::HashMap::with_capacity(chart_blocks.len());
    for (chart_idx, &block) in chart_blocks.iter().enumerate() {
        if block >= n_blocks {
            return Err(format!(
                "audit_sae atlas block {block} out of range 0..{n_blocks}"
            ));
        }
        if chart_positions.insert(block, chart_idx).is_some() {
            return Err(format!(
                "audit_sae atlas block {block} is selected more than once"
            ));
        }
    }
    let mut chart_rows = vec![Vec::<usize>::new(); chart_blocks.len()];
    let mut chart_weights = vec![Vec::<f64>::new(); chart_blocks.len()];
    let mut coactive_pairs = std::collections::BTreeSet::<(usize, usize)>::new();
    for row in 0..route.nrows() {
        let mut live_charts = Vec::with_capacity(route.width());
        for slot in 0..route.width() {
            let unit = route.indices[[row, slot]] as usize;
            if let Some(&chart_idx) = chart_positions.get(&unit) {
                let gate = route.gate(row, slot);
                if gate > activation_threshold as f64 {
                    chart_rows[chart_idx].push(row);
                    chart_weights[chart_idx].push(gate);
                    live_charts.push(chart_idx);
                }
            }
        }
        live_charts.sort_unstable();
        for left in 0..live_charts.len() {
            for right in (left + 1)..live_charts.len() {
                coactive_pairs.insert((live_charts[left], live_charts[right]));
            }
        }
    }
    let mut charts = Vec::with_capacity(chart_blocks.len());
    for (chart_idx, (rows, weights)) in chart_rows.into_iter().zip(chart_weights).enumerate() {
        charts.push(
            crate::inference::atlas_nerve::AtlasChart::from_sparse_weights(
                chart_idx,
                route.nrows(),
                rows,
                weights,
            )?,
        );
    }
    let mut gates = Vec::with_capacity(coactive_pairs.len());
    for (a, b) in coactive_pairs {
        if let Some(gate) = chart_transfer_gate_sparse(
            route,
            &charts[a],
            &charts[b],
            chart_blocks[a],
            chart_blocks[b],
            a,
            b,
        )? {
            gates.push(gate);
        }
    }
    let preliminary =
        crate::inference::atlas_nerve::build_atlas_nerve(&charts, &gates, None, None)?;
    let admitted_edges: Vec<_> = preliminary
        .edges
        .iter()
        .filter(|edge| edge.admitted)
        .map(|edge| {
            crate::inference::atlas_holonomy::AtlasHolonomyEdgeId::new(
                edge.a,
                edge.b,
                edge.overlap,
            )
        })
        .collect::<Result<_, _>>()?;
    let (holonomy_certificate, holonomy_unavailable_reason) = match (
        ambient_data,
        familywise_alpha,
    ) {
        (Some(data), Some(alpha)) => {
            match cross_fitted_holonomy_from_ambient(&charts, &admitted_edges, data, alpha) {
                Ok(certificate) => (Some(certificate), None),
                Err(reason) => (None, Some(reason)),
            }
        }
        (None, None) => (
            None,
            Some(
                "ambient observations and a familywise level were not supplied for cross-fitted holonomy"
                    .to_string(),
            ),
        ),
        _ => {
            return Err(
                "cross-fitted atlas holonomy requires ambient observations and familywise alpha together"
                    .to_string(),
            );
        }
    };
    let diagram = match holonomy_certificate {
        Some(certificate) => crate::inference::atlas_nerve::build_atlas_nerve(
            &charts,
            &gates,
            None,
            Some(certificate),
        )?,
        None => preliminary,
    };
    Ok(Some(AuditAtlasReport {
        chart_blocks,
        diagram,
        holonomy_unavailable_reason,
    }))
}

/// Run the whole sparse-route SAE audit: routability floor, dual certificate,
/// per-block circle coordinates, topology records, atlas nerve (with cross-fitted
/// Gaussian-PCA holonomy when ambient rows and a familywise level are supplied),
/// absorption pairs, optional circle-transport class, and the standing
/// architecture-matched null + residual spike-in calibration for the topological
/// claims.
///
/// Validation order is load-bearing: when several inputs are wrong at once the
/// first check that fires picks the message, and the messages are the audit's
/// documented contract.
pub fn run_sparse_sae_audit(
    request: SparseSaeAuditRequest,
) -> Result<SparseSaeAuditReport, String> {
    let SparseSaeAuditRequest {
        decoder: decoder_values,
        route_indices,
        route_values,
        data: data_values,
        donor_indices,
        donor_values,
        config,
    } = request;
    let SparseSaeAuditConfig {
        block_size,
        delta,
        quantile_levels: quantiles,
        max_candidates,
        coordinate_blocks,
        activation_threshold,
        max_absorption_pairs,
        transport_theta_in: theta_in_values,
        transport_theta_out: theta_out_values,
        transport_layer_from,
        transport_layer_to,
        calibration: calibration_cfg,
    } = config;

    if decoder_values.nrows() == 0 || decoder_values.ncols() == 0 {
        return Err("audit_sae requires a non-empty decoder matrix".to_string());
    }
    if block_size == 0 {
        return Err("audit_sae block_size must be >= 1".to_string());
    }
    if decoder_values.nrows() % block_size != 0 {
        return Err(format!(
            "audit_sae decoder has K={} rows, not a multiple of block_size {block_size}",
            decoder_values.nrows()
        ));
    }
    if decoder_values.iter().any(|value| !value.is_finite())
        || data_values.iter().any(|value| !value.is_finite())
    {
        return Err("audit_sae decoder and activations must be finite".to_string());
    }
    let n_units = decoder_values.nrows() / block_size;
    let route = AuditSparseRoute::new(
        route_indices,
        route_values,
        n_units,
        block_size,
        "observed route",
    )?;
    let donor = AuditSparseRoute::new(
        donor_indices,
        donor_values,
        n_units,
        block_size,
        "random-weight donor route",
    )?;
    if data_values.nrows() != route.nrows() || data_values.ncols() != decoder_values.ncols() {
        return Err(format!(
            "audit_sae data shape {:?} is incompatible with {} route rows and decoder {:?}",
            data_values.dim(),
            route.nrows(),
            decoder_values.dim()
        ));
    }
    if !delta.is_finite() || delta <= 0.0 {
        return Err("audit_sae requires a finite delta > 0".to_string());
    }
    if activation_threshold < 0.0 || !activation_threshold.is_finite() {
        return Err("audit_sae activation_threshold must be finite and non-negative".to_string());
    }
    if calibration_cfg.null_replicates == 0 {
        return Err("audit_sae null_replicates must be >= 1".to_string());
    }
    if calibration_cfg.spikein_trials == 0 {
        return Err("audit_sae spikein_trials must be >= 1".to_string());
    }
    if !calibration_cfg.spikein_snr.is_finite() || calibration_cfg.spikein_snr < 0.0 {
        return Err("audit_sae spikein_snr must be finite and non-negative".to_string());
    }
    if !calibration_cfg.spikein_false_positive_rate.is_finite()
        || calibration_cfg.spikein_false_positive_rate <= 0.0
        || calibration_cfg.spikein_false_positive_rate >= 1.0
    {
        return Err("audit_sae spikein_false_positive_rate must be in (0, 1)".to_string());
    }

    let decoder_shape = decoder_values.dim();
    let route_rows = route.nrows();
    let route_width = route.width();

    let residuals = residuals_from_sparse_sae(data_values.view(), decoder_values.view(), &route)?;
    let routability = crate::routability::routability_audit(
        decoder_values.view(),
        residuals.view(),
        block_size,
        delta,
        &quantiles,
    )?;

    let (dual, coordinate_reports) = if block_size == 1 {
        let report = crate::dual_certificate::sparse_route_dual_certificate(
            data_values.view(),
            decoder_values.view(),
            route.indices.view(),
            route.values.index_axis(ndarray::Axis(2), 0),
            max_candidates,
        )?;
        (report, Vec::new())
    } else {
        let report = crate::dual_certificate::block_route_dual_certificate(
            data_values.view(),
            decoder_values.view(),
            route.indices.view(),
            route.values.view(),
            block_size,
            max_candidates,
        )?;
        let mut coordinates = Vec::new();
        if block_size >= 2 && block_size % 2 == 0 {
            let total_blocks = decoder_values.nrows() / block_size;
            let blocks = coordinate_blocks
                .clone()
                .unwrap_or_else(|| (0..total_blocks).collect());
            for block in blocks {
                if block >= total_blocks {
                    return Err(format!(
                        "audit_sae coordinate block {block} out of range 0..{total_blocks}"
                    ));
                }
                coordinates.push(crate::sparse_dict::harmonic_route_firing_coordinates(
                    route.indices.view(),
                    route.values.view(),
                    n_units,
                    block,
                )?);
            }
        }
        (report, coordinates)
    };

    let topology_records =
        topology_records_from_codes(&coordinate_reports, block_size, activation_threshold);
    let atlas_data = data_values.mapv(f64::from);
    let atlas_nerve = atlas_nerve_from_sparse_route(
        &route,
        activation_threshold,
        coordinate_blocks.as_deref(),
        Some(atlas_data.view()),
        Some(delta),
    )?;
    let absorption = absorption_audit(&route, activation_threshold, max_absorption_pairs);

    let transport = match (theta_in_values, theta_out_values) {
        (Some(theta_in), Some(theta_out)) => Some(
            crate::inference::transport_class::classify_circle_transport(
                &theta_in,
                &theta_out,
                transport_layer_from,
                transport_layer_to,
            )?,
        ),
        (None, None) => None,
        _ => {
            return Err(
                "audit_sae transport requires both transport_theta_in and transport_theta_out"
                    .to_string(),
            );
        }
    };

    // Standing null battery + spike-in calibration for the audit's topological
    // claims: re-invoke the atlas-richness audit on the architecture-matched
    // random-weight donor and plant a circle into the real residuals. Gated on a
    // selected atlas chart set (block dictionaries with >= 2 charts);
    // scalar/degenerate shapes carry no such claim.
    let calibration = match atlas_nerve.as_ref() {
        Some(atlas) => {
            let residuals_f64 = residuals.mapv(|value| value as f64);
            standing_sparse_null_calibration(
                &route,
                &donor,
                residuals_f64.view(),
                &atlas.chart_blocks,
                activation_threshold as f64,
                &calibration_cfg,
            )?
        }
        None => None,
    };

    Ok(SparseSaeAuditReport {
        block_size,
        n_units,
        route_rows,
        route_width,
        decoder_shape,
        routability,
        dual,
        coordinate_reports,
        topology_records,
        atlas_nerve,
        absorption,
        transport,
        calibration,
    })
}

fn absorption_audit(
    route: &AuditSparseRoute,
    activation_threshold: f32,
    max_pairs: usize,
) -> AbsorptionAuditReport {
    let mut marginals = vec![0usize; route.n_units];
    let mut accumulators = std::collections::BTreeMap::<(usize, usize), SparsePairAccum>::new();
    for row in 0..route.nrows() {
        let mut live = Vec::with_capacity(route.width());
        for slot in 0..route.width() {
            let weight = route.gate(row, slot);
            if weight > activation_threshold as f64 {
                let unit = route.indices[[row, slot]] as usize;
                marginals[unit] += 1;
                live.push((unit, weight));
            }
        }
        live.sort_unstable_by_key(|(unit, _)| *unit);
        for left in 0..live.len() {
            for right in (left + 1)..live.len() {
                let (a, wa) = live[left];
                let (b, wb) = live[right];
                let acc = accumulators.entry((a, b)).or_default();
                acc.n_joint += 1;
                acc.sum_a += wa;
                acc.sum_b += wb;
                acc.sum_a2 += wa * wa;
                acc.sum_b2 += wb * wb;
                acc.sum_ab += wa * wb;
            }
        }
    }
    let n_obs = route.nrows();
    let mut pairs = accumulators
        .into_iter()
        .map(|((a, b), acc)| {
            let n_a = marginals[a];
            let n_b = marginals[b];
            let conditional = |joint: usize, marginal: usize| {
                if marginal == 0 {
                    0.0
                } else {
                    joint as f64 / marginal as f64
                }
            };
            let p_a_given_b = conditional(acc.n_joint, n_b);
            let p_b_given_a = conditional(acc.n_joint, n_a);
            let lift = if n_a == 0 || n_b == 0 || n_obs == 0 {
                0.0
            } else {
                acc.n_joint as f64 * n_obs as f64 / (n_a as f64 * n_b as f64)
            };
            let weight_correlation = if acc.n_joint < 2 {
                0.0
            } else {
                let n = acc.n_joint as f64;
                let covariance = acc.sum_ab - acc.sum_a * acc.sum_b / n;
                let variance_a = acc.sum_a2 - acc.sum_a * acc.sum_a / n;
                let variance_b = acc.sum_b2 - acc.sum_b * acc.sum_b / n;
                if variance_a > 0.0 && variance_b > 0.0 {
                    (covariance / (variance_a.sqrt() * variance_b.sqrt())).clamp(-1.0, 1.0)
                } else {
                    0.0
                }
            };
            let dependence = p_a_given_b.min(p_b_given_a);
            AbsorptionPairReport {
                a,
                b,
                n_obs,
                n_a,
                n_b,
                n_joint: acc.n_joint,
                p_a_given_b,
                p_b_given_a,
                lift,
                weight_correlation,
                dependence,
                fusion_evidence: dependence * weight_correlation.abs(),
                absorption_asymmetry: (p_a_given_b - p_b_given_a).abs(),
            }
        })
        .collect::<Vec<_>>();
    pairs.sort_by(|left, right| {
        right
            .absorption_asymmetry
            .total_cmp(&left.absorption_asymmetry)
    });
    pairs.truncate(max_pairs);
    AbsorptionAuditReport {
        n_units: route.n_units,
        activation_threshold,
        pairs,
    }
}

/// Genuine chart-transfer certificate for one atlas-nerve gate between two
/// charts, read from the frozen code matrix.
///
/// A gate stamped `valid = true` with zero transport/equivariance defect is a
/// FABRICATED certificate: it admits every co-active chart pair as a nerve edge
/// without running any transport test, so the reported topology is manufactured,
/// not measured. The real certificate needs a square (≤2-D) chart-to-chart
/// operator; only the harmonic circle lane (`block_size == 2`) exposes a 2-D
/// per-row coordinate from which the empirical transfer operator `A` (least
/// squares `X_a A ≈ X_b` over the rows that fire in BOTH charts) can be formed.
/// `A` is certified against isometry (`‖AᵀA − I‖_F`) and SO(2) equivariance
/// (`‖A·G − G·A‖_F`) by [`certify_square_transfer`], and validity is the
/// library's own gate ([`AtlasTransferGate::from_square_transfer`]). Any other
/// block width, fewer than two co-firing rows, a singular coordinate Gram, or a
/// non-finite operator exposes no transfer gate at this boundary; the nerve
/// records the observed overlap as rejected because no certificate exists.
/// `block_a`/`block_b` index the dictionary blocks the two charts read;
/// `chart_a`/`chart_b` are the nerve-vertex labels.
fn chart_transfer_gate_sparse(
    route: &AuditSparseRoute,
    support_a: &crate::inference::atlas_nerve::AtlasChart,
    support_b: &crate::inference::atlas_nerve::AtlasChart,
    block_a: usize,
    block_b: usize,
    chart_a: usize,
    chart_b: usize,
) -> Result<Option<crate::inference::atlas_nerve::AtlasTransferGate>, String> {
    use crate::inference::atlas_holonomy::AtlasHolonomyEdgeId;
    use crate::inference::atlas_nerve::AtlasTransferGate;
    let edge = AtlasHolonomyEdgeId::new(chart_a, chart_b, 0)?;
    if route.block_size != 2 {
        return Ok(None);
    }
    let mut xa: Vec<f64> = Vec::new();
    let mut xb: Vec<f64> = Vec::new();
    let mut position_a = 0usize;
    let mut position_b = 0usize;
    while position_a < support_a.support_rows().len() && position_b < support_b.support_rows().len()
    {
        let row_a = support_a.support_rows()[position_a];
        let row_b = support_b.support_rows()[position_b];
        if row_a < row_b {
            position_a += 1;
            continue;
        }
        if row_b < row_a {
            position_b += 1;
            continue;
        }
        let row = row_a;
        let mut a = None;
        let mut b = None;
        for slot in 0..route.width() {
            let unit = route.indices[[row, slot]] as usize;
            let value = [
                route.values[[row, slot, 0]] as f64,
                route.values[[row, slot, 1]] as f64,
            ];
            if value[0] * value[0] + value[1] * value[1] == 0.0 {
                continue;
            }
            if unit == block_a {
                a = Some(value);
            } else if unit == block_b {
                b = Some(value);
            }
        }
        if let (Some([a0, a1]), Some([b0, b1])) = (a, b) {
            xa.extend([a0, a1]);
            xb.extend([b0, b1]);
        }
        position_a += 1;
        position_b += 1;
    }
    let n_co = xa.len() / 2;
    if n_co < 2 {
        return Ok(None);
    }
    let x_a = ndarray::Array2::from_shape_vec((n_co, 2), xa)
        .map_err(|error| format!("chart {chart_a} overlap coordinates are malformed: {error}"))?;
    let x_b = ndarray::Array2::from_shape_vec((n_co, 2), xb)
        .map_err(|error| format!("chart {chart_b} overlap coordinates are malformed: {error}"))?;
    // Empirical chart-to-chart transfer operator `A = (X_aᵀX_a)⁻¹ X_aᵀX_b`
    // solving `X_a A ≈ X_b` over the co-firing rows.
    let Ok(operator) =
        crate::chart_transfer::pulled_back_operator(x_a.view(), x_b.view())
    else {
        return Ok(None);
    };
    // Both charts are circles, so the shared infinitesimal-rotation generator is
    // the SO(2) generator `[[0,−1],[1,0]]`.
    let generator = ndarray::array![[0.0_f64, -1.0], [1.0, 0.0]];
    match crate::chart_transfer::certify_square_transfer(
        operator.view(),
        generator.view(),
        generator.view(),
    ) {
        Ok(cert) => Ok(Some(AtlasTransferGate::from_square_transfer(edge, cert, 2))),
        Err(_) => Ok(None),
    }
}

/// Build the fitted Gaussian-PCA holonomy analysis on a deterministic global
/// cross-fit. Even rows fit every live chart's pilot frame; odd rows are
/// assigned to exactly one live chart for inference. Consequently no pilot row
/// is reused for inference and patch inference sets are pairwise disjoint.
/// Spectrum values remain typed plug-in estimates, so the core reports an
/// analyzed refusal rather than manufacturing a finite-sample certificate.
fn cross_fitted_holonomy_from_ambient(
    charts: &[crate::inference::atlas_nerve::AtlasChart],
    admitted_edges: &[crate::inference::atlas_holonomy::AtlasHolonomyEdgeId],
    data: ndarray::ArrayView2<'_, f64>,
    familywise_alpha: f64,
) -> Result<crate::inference::atlas_holonomy::AtlasHolonomyCertificate, String> {
    use crate::inference::atlas_holonomy::{
        AtlasFamilywiseLevel, AtlasHolonomyCertificate, GaussianPatchRowSplit,
        GaussianPcaErrorModel, GaussianPcaPatch, PopulationCrossGramProvenance,
        ProjectedAtlasEdgeSpec,
    };
    if data.ncols() < 3 {
        return Err(format!(
            "cross-fitted atlas holonomy needs at least three ambient coordinates, got {}",
            data.ncols()
        ));
    }
    let mut live_by_row = vec![Vec::<usize>::new(); data.nrows()];
    for (chart, patch) in charts.iter().enumerate() {
        for &row in patch.support_rows() {
            if row >= data.nrows() {
                return Err(format!(
                    "atlas chart {chart} support row {row} exceeds ambient data height {}",
                    data.nrows()
                ));
            }
            live_by_row[row].push(chart);
        }
    }
    let mut pilot_rows = vec![Vec::<usize>::new(); charts.len()];
    let mut inference_rows = vec![Vec::<usize>::new(); charts.len()];
    for (row, live) in live_by_row.iter().enumerate() {
        if row % 2 == 0 {
            for &chart in live {
                pilot_rows[chart].push(row);
            }
        } else if !live.is_empty() {
            let owner = live[(row / 2) % live.len()];
            inference_rows[owner].push(row);
        }
    }
    let mut patches = Vec::with_capacity(charts.len());
    for chart in 0..charts.len() {
        let split = GaussianPatchRowSplit::new(
            std::mem::take(&mut pilot_rows[chart]),
            std::mem::take(&mut inference_rows[chart]),
        )?;
        patches.push(GaussianPcaPatch::fit_cross_fitted_plugin(
            chart,
            split,
            data.view(),
            data.ncols(),
        )?);
    }
    let error_model = GaussianPcaErrorModel::independent(&patches)?;
    let edge_specs = admitted_edges
        .iter()
        .copied()
        .map(|edge| {
            ProjectedAtlasEdgeSpec::new(
                edge.a(),
                edge.b(),
                edge.overlap(),
                PopulationCrossGramProvenance::EstimatedOnly,
                0.0,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    AtlasHolonomyCertificate::gaussian_pca(
        patches,
        edge_specs,
        error_model,
        AtlasFamilywiseLevel::new(familywise_alpha)?,
        None,
    )
}

fn residuals_from_sparse_sae(
    data: ndarray::ArrayView2<'_, f32>,
    decoder: ndarray::ArrayView2<'_, f32>,
    route: &AuditSparseRoute,
) -> Result<ndarray::Array2<f32>, String> {
    if data.ncols() != decoder.ncols() || data.nrows() != route.nrows() {
        return Err(format!(
            "audit_sae data shape {:?} is incompatible with decoder {:?} and {} route rows",
            data.dim(),
            decoder.dim(),
            route.nrows()
        ));
    }
    let fitted = route.reconstruct(decoder)?;
    let mut residuals = data.to_owned();
    residuals -= &fitted;
    Ok(residuals)
}

/// Build the standing sparse donor null + residual spike-in calibration. Both
/// observed and donor routing remain `N×s×b`; implicit zeros are never expanded.
fn standing_sparse_null_calibration(
    route: &AuditSparseRoute,
    donor: &AuditSparseRoute,
    residuals_f64: ndarray::ArrayView2<'_, f64>,
    chart_blocks: &[usize],
    activation_threshold: f64,
    cfg: &StandingCalibrationConfig,
) -> Result<Option<crate::null_battery::ClaimNullCalibration>, String> {
    use crate::null_battery as nb;
    if chart_blocks.len() < 2
        || donor.n_units != route.n_units
        || donor.block_size != route.block_size
        || residuals_f64.nrows() < 4
        || residuals_f64.ncols() < 2
        || cfg.null_replicates == 0
        || cfg.spikein_trials == 0
    {
        return Ok(None);
    }
    use rand::SeedableRng;
    let observed =
        sparse_atlas_nerve_richness_statistic(route, chart_blocks, activation_threshold)?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.null_seed);
    let mut samples = Vec::with_capacity(cfg.null_replicates);
    for _ in 0..cfg.null_replicates {
        let surrogate = resample_sparse_architecture_null(route, donor, &mut rng)?;
        samples.push(sparse_atlas_nerve_richness_statistic(
            &surrogate,
            chart_blocks,
            activation_threshold,
        )?);
    }
    let null_summary = nb::summarize_null_distribution(
        nb::NullKind::ArchitectureMatchedRandomWeight,
        observed,
        samples,
        nb::Tail::Larger,
    )?;
    let nulls = nb::NullBatteryReport {
        observed,
        summaries: vec![null_summary],
    };
    // Spike-in power: plant a synthetic circle into the real audit residuals and
    // measure the default block-chart/topology detector's recovery rate at the
    // requested false-positive operating point. Bootstrapping the empirical
    // residual rows keeps the real post-fit covariance and tails in the loop.
    let mut roc_config = nb::SpikeInRocConfig::circle(
        vec![0.0, cfg.spikein_snr],
        cfg.spikein_trials,
        cfg.null_seed,
    );
    roc_config.noise_mode = nb::SpikeInNoiseMode::EmpiricalResidualBootstrap;
    roc_config.fpr_levels = vec![cfg.spikein_false_positive_rate];
    let roc = nb::default_spike_in_roc_curve(residuals_f64, &roc_config)?;
    let report = nb::calibrated_roc_claim_report(
        "audit_sae.topology_atlas_nerve",
        cfg.spikein_snr,
        cfg.spikein_false_positive_rate,
        nulls,
        roc,
    )?;
    Ok(Some(nb::ClaimNullCalibration::from_calibrated_roc(report)?))
}

fn topology_records_from_codes(
    coordinate_reports: &[crate::sparse_dict::BlockCoordinateReport],
    block_size: usize,
    activation_threshold: f32,
) -> Vec<AuditTopologyRecord> {
    let threshold = activation_threshold as f64;
    if block_size == 1 {
        // A scalar external SAE feature is a point/line chart, not a manifold:
        // there is no circle/torus topology to audit. This mirrors the atlas
        // nerve, which likewise declines the scalar `block_size == 1` shape, so a
        // scalar dictionary reports zero topology atoms rather than one trivial
        // point record per column.
        return Vec::new();
    }

    let expected = crate::manifold::BettiSignature {
        b0: 1,
        b1: 1,
        b2: None,
    };
    let mut records = Vec::with_capacity(coordinate_reports.len());
    for report in coordinate_reports {
        let live: Vec<_> = report
            .firings
            .iter()
            .filter(|firing| firing.amplitude > threshold)
            .collect();
        if live.len() < 4 {
            let measured = crate::manifold::BettiSignature {
                b0: if live.is_empty() { 0 } else { 1 },
                b1: 0,
                b2: None,
            };
            records.push(AuditTopologyRecord {
                atom: report.firings.first().map_or(0, |firing| firing.block),
                support_size: live.len(),
                landmark_count: live.len(),
                covering_side: "below_covering_number".to_string(),
                measured_betti: measured,
                expected_betti: expected,
                contested: measured != expected,
                dominant_h1_persistence: 0.0,
                dominant_h2_persistence: 0.0,
                note: "under-resolved harmonic-circle block: fewer than four firing coordinates"
                    .to_string(),
            });
            continue;
        }

        let mut points = ndarray::Array2::<f64>::zeros((live.len(), 2));
        for (row_idx, firing) in live.iter().enumerate() {
            let phase = std::f64::consts::TAU * firing.t;
            let (sin_phase, cos_phase) = phase.sin_cos();
            points[[row_idx, 0]] = cos_phase;
            points[[row_idx, 1]] = sin_phase;
        }
        if let Some(verdict) = crate::manifold::topology_persistence_verdict(
            points.view(),
            &crate::manifold::SaeAtomBasisKind::Periodic,
        ) {
            records.push(AuditTopologyRecord {
                atom: live[0].block,
                support_size: verdict.support_size,
                landmark_count: verdict.landmark_count,
                covering_side: verdict.covering_side.as_str().to_string(),
                measured_betti: verdict.measured_betti,
                expected_betti: verdict.expected_betti,
                contested: verdict.contested,
                dominant_h1_persistence: verdict.dominant_h1_persistence,
                dominant_h2_persistence: verdict.dominant_h2_persistence,
                note: verdict.note,
            });
        }
    }
    records
}

/// Atlas-nerve topological-richness statistic over a fixed-width sparse route.
/// The statistic never materializes the logical `N×K` code matrix.
fn sparse_atlas_nerve_richness_statistic(
    route: &AuditSparseRoute,
    chart_blocks: &[usize],
    activation_threshold: f64,
) -> Result<f64, String> {
    let report = atlas_nerve_from_sparse_route(
        route,
        activation_threshold as f32,
        Some(chart_blocks),
        None,
        None,
    )?
    .ok_or_else(|| "atlas null statistic requires at least two block charts".to_string())?;
    let richness = report
        .diagram
        .simplex_counts
        .iter()
        .skip(1)
        .map(|&count| count as f64)
        .sum::<f64>();
    if !richness.is_finite() {
        return Err("atlas nerve richness overflowed finite reporting range".to_string());
    }
    Ok(richness)
}