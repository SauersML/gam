//! Partition-free coactivation conditionality for structure search.
//!
//! The definition here has two parts:
//!
//! * a native varying-coefficient GAM, `gate_j ~ beta(x) * gate_i`, where `x` is
//!   a continuous context summary and the `by=` design columns are the spline
//!   basis for `x` multiplied rowwise by `gate_i`;
//! * a distribution-free KL certificate for the pooled weighted-Pearson
//!   coupling statistic.  If `psi` is the per-row influence contribution for
//!   the weighted correlation `rho`, then any first-order distribution shift
//!   with KL budget `epsilon` changes `rho` by at most
//!   `sqrt(2 * epsilon * Var(psi))`.
//!
//! Discrete context labels are accepted only by the diagnostic naming helper at
//! the bottom of the module. They are not part of the conditionality metric.

use gam_solve::row_sampling_measure::{DesignedRowSample, MeasureProvenance, RowSamplingMeasure};
use gam_terms::basis::{BasisOptions, Dense, KnotSource, create_basis};
use crate::null_battery::ClaimNullCalibration;
use ndarray::{Array1, ArrayView2};
use std::collections::BTreeMap;

const DEFAULT_SPLINE_DEGREE: usize = 3;
const DEFAULT_INTERNAL_KNOTS: usize = 5;
const DEFAULT_PENALTY_ORDER: usize = 1;
const BRACKET_EXPANSION_LIMIT: usize = 48;
const BRENT_ITERATION_LIMIT: usize = 96;
const BRENT_REL_TOL: f64 = 1.0e-8;
const MIN_RESIDUAL_DF: f64 = 1.0;
const CONDITIONALITY_NULL_REPLICATES: usize = 8;
const CONDITIONALITY_SPIKE_TRIALS: usize = 8;
const CONDITIONALITY_CLAIMED_SNR: f64 = 1.0;
const CONDITIONALITY_CLAIMED_FPR: f64 = 0.05;
const CONDITIONALITY_NULL_SEED: u64 = 0xC0A_C7A1;
const CONDITIONALITY_SPIKE_SEED: u64 = 0xC0A_51C1;

/// Configuration for the native varying-coefficient GAM conditionality fit.
#[derive(Clone, Copy, Debug)]
pub struct VaryingCoefficientConfig {
    pub spline_degree: usize,
    pub num_internal_knots: usize,
    /// Difference penalty order on the varying coefficient. Order 1 makes the
    /// penalty nullspace exactly the constant-coupling model.
    pub penalty_order: usize,
}

impl Default for VaryingCoefficientConfig {
    fn default() -> Self {
        Self {
            spline_degree: DEFAULT_SPLINE_DEGREE,
            num_internal_knots: DEFAULT_INTERNAL_KNOTS,
            penalty_order: DEFAULT_PENALTY_ORDER,
        }
    }
}

/// Native partition-free conditionality: a by-smooth coefficient beta(x).
#[derive(Clone, Debug)]
pub struct VaryingCoefficientConditionality {
    pub selected_log_smoothing: f64,
    pub reml_score: f64,
    pub effective_degrees: f64,
    pub beta_wiggliness: f64,
    pub beta_variation: f64,
    pub beta_mean: f64,
    pub coefficients: Vec<f64>,
    pub beta_at_rows: Vec<f64>,
}

/// Statistic protected by a [`RobustCouplingCertificate`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CouplingStatistic {
    /// Weighted Pearson correlation between the two selected gate streams.
    WeightedPearson,
}

/// Influence-function KL certificate for the pooled weighted-Pearson coupling.
#[derive(Clone, Debug)]
pub struct RobustCouplingCertificate {
    pub statistic: CouplingStatistic,
    pub rho: f64,
    pub influence_variance: f64,
    pub robustness_radius_epsilon: f64,
    pub influence_mean_abs: f64,
}

impl RobustCouplingCertificate {
    /// First-order lower bound on the weighted-Pearson coupling after an
    /// arbitrary KL-`epsilon` distribution shift.
    pub fn worst_case_coupling(&self, epsilon: f64) -> Result<f64, String> {
        if !(epsilon.is_finite() && epsilon >= 0.0) {
            return Err(format!(
                "worst_case_coupling: epsilon must be finite and >= 0, got {epsilon}"
            ));
        }
        Ok(self.rho - (2.0 * epsilon * self.influence_variance).sqrt())
    }
}

/// Optional diagnostic that names where a continuous coupling varies after the
/// metric has already been computed.
#[derive(Clone, Debug)]
pub struct ContextDiagnostic {
    pub context: usize,
    pub rows: usize,
    pub mass: f64,
    pub mean_gate_i: f64,
    pub mean_gate_j: f64,
    pub mean_beta: f64,
    pub centered_association: f64,
}

/// Partition-free conditionality report for one ordered pair.
#[derive(Clone, Debug)]
pub struct CoactivationConditionality {
    pub native: VaryingCoefficientConditionality,
    pub certificate: RobustCouplingCertificate,
    pub null_calibration: Option<ClaimNullCalibration>,
    /// Scalar intended for merge/fusion ranking. Large values mean the pair has
    /// a robust pooled weighted-Pearson coupling and little continuous-context
    /// coefficient drift.
    pub fusion_gate_score: f64,
    pub diagnostics: Vec<ContextDiagnostic>,
}

/// Residual-gate materialization after the shared chart has been projected out.
#[derive(Clone, Debug)]
pub struct ResidualGateActivities {
    pub residual_i: Vec<f64>,
    pub residual_j: Vec<f64>,
    pub active_i: Vec<bool>,
    pub active_j: Vec<bool>,
}

/// Evaluate partition-free conditionality on a deterministic designed subsample
/// from a [`RowSamplingMeasure`].
pub fn estimate_from_measure(
    gate_i: &[f64],
    gate_j: &[f64],
    continuous_context: &[f64],
    diagnostic_labels: Option<&[usize]>,
    measure: &RowSamplingMeasure,
    budget: usize,
    seed: u64,
    config: VaryingCoefficientConfig,
) -> Result<CoactivationConditionality, String> {
    if measure.n_rows() != gate_i.len() {
        return Err(format!(
            "estimate_from_measure: measure has {} rows, gates have {}",
            measure.n_rows(),
            gate_i.len()
        ));
    }
    let sample = measure.designed_subsample(budget, seed);
    estimate_from_designed_sample(
        gate_i,
        gate_j,
        continuous_context,
        diagnostic_labels,
        &sample,
        config,
    )
}

/// Evaluate partition-free conditionality on an already-drawn designed sample.
pub fn estimate_from_designed_sample(
    gate_i: &[f64],
    gate_j: &[f64],
    continuous_context: &[f64],
    diagnostic_labels: Option<&[usize]>,
    sample: &DesignedRowSample,
    config: VaryingCoefficientConfig,
) -> Result<CoactivationConditionality, String> {
    estimate_on_rows(
        gate_i,
        gate_j,
        continuous_context,
        diagnostic_labels,
        &sample.rows,
        &sample.likelihood_weights,
        config,
    )
}

/// Evaluate partition-free conditionality on explicit selected rows and per-row
/// honesty weights.
pub fn estimate_on_rows(
    gate_i: &[f64],
    gate_j: &[f64],
    continuous_context: &[f64],
    diagnostic_labels: Option<&[usize]>,
    rows: &[usize],
    likelihood_weights: &[f64],
    config: VaryingCoefficientConfig,
) -> Result<CoactivationConditionality, String> {
    validate_partition_free_inputs(
        gate_i,
        gate_j,
        continuous_context,
        diagnostic_labels,
        rows,
        likelihood_weights,
    )?;
    let native = fit_varying_coefficient_gam(
        gate_i,
        gate_j,
        continuous_context,
        rows,
        likelihood_weights,
        config,
    )?;
    let influence = coupling_influence_values(gate_i, gate_j, rows, likelihood_weights)?;
    let certificate = influence.certificate();
    let fusion_gate_score = certificate.robustness_radius_epsilon
        / (1.0 + native.beta_wiggliness.max(0.0) + native.beta_variation.max(0.0));
    let diagnostics = match diagnostic_labels {
        Some(labels) => diagnose_context_labels(
            gate_i,
            gate_j,
            labels,
            rows,
            likelihood_weights,
            &native.beta_at_rows,
        )?,
        None => Vec::new(),
    };
    Ok(CoactivationConditionality {
        native,
        certificate,
        null_calibration: None,
        fusion_gate_score,
        diagnostics,
    })
}

pub fn estimate_on_rows_with_nulls(
    gate_i: &[f64],
    gate_j: &[f64],
    continuous_context: &[f64],
    diagnostic_labels: Option<&[usize]>,
    rows: &[usize],
    likelihood_weights: &[f64],
    config: VaryingCoefficientConfig,
    random_weight_gate_i: &[f64],
    random_weight_gate_j: &[f64],
) -> Result<CoactivationConditionality, String> {
    let mut report = estimate_on_rows(
        gate_i,
        gate_j,
        continuous_context,
        diagnostic_labels,
        rows,
        likelihood_weights,
        config,
    )?;
    let observed = selected_pair_matrix(gate_i, gate_j, rows)?;
    let random_weight = selected_pair_matrix(random_weight_gate_i, random_weight_gate_j, rows)?;
    let null_config = crate::null_battery::NullBatteryConfig {
        replicates: CONDITIONALITY_NULL_REPLICATES,
        seed: CONDITIONALITY_NULL_SEED,
        kinds: vec![
            crate::null_battery::NullKind::PhaseRandomized,
            crate::null_battery::NullKind::RandomRotation,
            crate::null_battery::NullKind::ArchitectureMatchedRandomWeight,
        ],
        tail: crate::null_battery::Tail::Larger,
    };
    let nulls = crate::null_battery::run_null_battery(
        observed.view(),
        Some(random_weight.view()),
        &null_config,
        conditionality_matrix_stat,
    )?;
    let roc_config = crate::null_battery::SpikeInRocConfig::circle(
        vec![CONDITIONALITY_CLAIMED_SNR],
        CONDITIONALITY_SPIKE_TRIALS,
        CONDITIONALITY_SPIKE_SEED,
    );
    let spike_in_roc =
        crate::null_battery::default_spike_in_roc_curve(observed.view(), &roc_config)?;
    let calibrated = crate::null_battery::calibrated_roc_claim_report(
        "conditionality",
        CONDITIONALITY_CLAIMED_SNR,
        CONDITIONALITY_CLAIMED_FPR,
        nulls,
        spike_in_roc,
    )?;
    report.null_calibration = Some(
        crate::null_battery::ClaimNullCalibration::from_calibrated_roc(calibrated),
    );
    Ok(report)
}

fn selected_pair_matrix(
    gate_i: &[f64],
    gate_j: &[f64],
    rows: &[usize],
) -> Result<ndarray::Array2<f64>, String> {
    let mut out = ndarray::Array2::<f64>::zeros((rows.len(), 2));
    for (slot, &row) in rows.iter().enumerate() {
        if row >= gate_i.len() || row >= gate_j.len() {
            return Err(format!("selected_pair_matrix: sampled row {row} out of range"));
        }
        out[[slot, 0]] = gate_i[row];
        out[[slot, 1]] = gate_j[row];
    }
    Ok(out)
}

fn conditionality_matrix_stat(data: ArrayView2<'_, f64>) -> Result<f64, String> {
    if data.nrows() < 4 || data.ncols() < 2 {
        return Err(
            "conditionality null statistic requires at least four rows and two columns".to_string(),
        );
    }
    let mut mean_i = 0.0_f64;
    let mut mean_j = 0.0_f64;
    for row in 0..data.nrows() {
        mean_i += data[[row, 0]];
        mean_j += data[[row, 1]];
    }
    mean_i /= data.nrows() as f64;
    mean_j /= data.nrows() as f64;
    let mut var_i = 0.0_f64;
    let mut var_j = 0.0_f64;
    let mut cov = 0.0_f64;
    for row in 0..data.nrows() {
        let zi = data[[row, 0]] - mean_i;
        let zj = data[[row, 1]] - mean_j;
        var_i += zi * zi;
        var_j += zj * zj;
        cov += zi * zj;
    }
    if !(var_i > 0.0 && var_j > 0.0) {
        return Ok(0.0);
    }
    Ok((cov / (var_i.sqrt() * var_j.sqrt())).abs())
}

/// Full influence vector for the weighted-Pearson coupling statistic.
#[derive(Clone, Debug)]
pub struct CouplingInfluence {
    pub statistic: CouplingStatistic,
    pub rho: f64,
    pub psi: Vec<f64>,
    pub normalized_weights: Vec<f64>,
}

impl CouplingInfluence {
    pub fn certificate(&self) -> RobustCouplingCertificate {
        let mut variance = 0.0_f64;
        let mut mean_abs = 0.0_f64;
        for slot in 0..self.psi.len() {
            let q = self.normalized_weights[slot];
            let psi = self.psi[slot];
            variance += q * psi * psi;
            mean_abs += q * psi.abs();
        }
        let robustness_radius_epsilon = if variance > 0.0 {
            self.rho * self.rho / (2.0 * variance)
        } else if self.rho == 0.0 {
            0.0
        } else {
            f64::INFINITY
        };
        RobustCouplingCertificate {
            statistic: self.statistic,
            rho: self.rho,
            influence_variance: variance,
            robustness_radius_epsilon,
            influence_mean_abs: mean_abs,
        }
    }
}

/// Per-row influence contributions for the weighted Pearson correlation between
/// the two gate streams over the selected sample.
///
/// With standardized gates `g̃ = (gate_i - mean_i)/sd_i` and
/// `h̃ = (gate_j - mean_j)/sd_j`, the exact influence function of the weighted
/// Pearson correlation `rho` is
///
/// ```text
///     psi_i = g̃_i * h̃_i - (rho/2) * (g̃_i^2 + h̃_i^2).
/// ```
///
/// This is a closed form (no solver, no finite differencing); its correctness
/// is pinned exactly against a recomputation and to O(1/N) against a
/// leave-one-out jackknife in the module tests.
pub fn coupling_influence_values(
    gate_i: &[f64],
    gate_j: &[f64],
    rows: &[usize],
    likelihood_weights: &[f64],
) -> Result<CouplingInfluence, String> {
    if rows.len() != likelihood_weights.len() {
        return Err(format!(
            "coupling_influence_values: {} rows but {} weights",
            rows.len(),
            likelihood_weights.len()
        ));
    }
    if rows.is_empty() {
        return Err("coupling_influence_values: need at least one sampled row".to_string());
    }
    let mut total_weight = 0.0_f64;
    for (slot, &row) in rows.iter().enumerate() {
        if row >= gate_i.len() || row >= gate_j.len() {
            return Err(format!(
                "coupling_influence_values: sampled row {row} out of range"
            ));
        }
        let w = likelihood_weights[slot];
        if !(w.is_finite() && w > 0.0) {
            return Err(format!(
                "coupling_influence_values: sampled row {row} has invalid weight {w}"
            ));
        }
        total_weight += w;
    }
    let normalized_weights: Vec<f64> = likelihood_weights
        .iter()
        .map(|&w| w / total_weight)
        .collect();

    let mut mean_i = 0.0_f64;
    let mut mean_j = 0.0_f64;
    for (slot, &row) in rows.iter().enumerate() {
        let q = normalized_weights[slot];
        mean_i += q * gate_i[row];
        mean_j += q * gate_j[row];
    }
    let mut var_i = 0.0_f64;
    let mut var_j = 0.0_f64;
    let mut cov = 0.0_f64;
    for (slot, &row) in rows.iter().enumerate() {
        let q = normalized_weights[slot];
        let zi = gate_i[row] - mean_i;
        let zj = gate_j[row] - mean_j;
        var_i += q * zi * zi;
        var_j += q * zj * zj;
        cov += q * zi * zj;
    }
    if !(var_i > 0.0 && var_j > 0.0) {
        return Err(
            "coupling_influence_values: both gates need positive weighted variance".to_string(),
        );
    }
    let sd_i = var_i.sqrt();
    let sd_j = var_j.sqrt();
    let rho = (cov / (sd_i * sd_j)).clamp(-1.0, 1.0);
    let mut psi = Vec::with_capacity(rows.len());
    for &row in rows {
        let zi = (gate_i[row] - mean_i) / sd_i;
        let zj = (gate_j[row] - mean_j) / sd_j;
        let value = zi * zj - 0.5 * rho * (zi * zi + zj * zj);
        psi.push(value);
    }
    Ok(CouplingInfluence {
        statistic: CouplingStatistic::WeightedPearson,
        rho,
        psi,
        normalized_weights,
    })
}

/// Full influence vector for the conditional coactivation probability
/// `pi = P(gate_j active | gate_i active)` over the selected sample.
///
/// This is a ratio estimator `pi = E[1_{g_i} 1_{g_j}] / E[1_{g_i}]` where the
/// expectations are weighted by the normalized honesty weights. Its exact
/// influence function is the standard ratio-estimator influence
///
/// ```text
///     psi_i = 1_{g_i}(1_{g_j} - pi) / E[1_{g_i}],
/// ```
///
/// which is closed form and mean-zero by construction (`sum_i q_i psi_i = 0`
/// exactly). It is pinned exactly against a recomputation and to O(1/N) against
/// a leave-one-out jackknife in the module tests.
#[derive(Clone, Debug)]
pub struct ConditionalCoactivationInfluence {
    /// Weighted conditional probability `P(gate_j active | gate_i active)`.
    pub conditional_probability: f64,
    /// Weighted active mass of gate i, `E[1_{g_i}]`, the ratio denominator.
    pub active_mass_i: f64,
    pub psi: Vec<f64>,
    pub normalized_weights: Vec<f64>,
}

/// Per-row influence contributions for the conditional coactivation
/// probability between the two gate activity streams over the selected sample.
pub fn conditional_coactivation_influence_values(
    active_i: &[bool],
    active_j: &[bool],
    rows: &[usize],
    likelihood_weights: &[f64],
) -> Result<ConditionalCoactivationInfluence, String> {
    if rows.len() != likelihood_weights.len() {
        return Err(format!(
            "conditional_coactivation_influence_values: {} rows but {} weights",
            rows.len(),
            likelihood_weights.len()
        ));
    }
    if rows.is_empty() {
        return Err(
            "conditional_coactivation_influence_values: need at least one sampled row".to_string(),
        );
    }
    let mut total_weight = 0.0_f64;
    for (slot, &row) in rows.iter().enumerate() {
        if row >= active_i.len() || row >= active_j.len() {
            return Err(format!(
                "conditional_coactivation_influence_values: sampled row {row} out of range"
            ));
        }
        let w = likelihood_weights[slot];
        if !(w.is_finite() && w > 0.0) {
            return Err(format!(
                "conditional_coactivation_influence_values: sampled row {row} has invalid weight {w}"
            ));
        }
        total_weight += w;
    }
    let normalized_weights: Vec<f64> = likelihood_weights
        .iter()
        .map(|&w| w / total_weight)
        .collect();

    let mut active_mass_i = 0.0_f64;
    let mut joint_mass = 0.0_f64;
    for (slot, &row) in rows.iter().enumerate() {
        let q = normalized_weights[slot];
        let a = if active_i[row] { 1.0 } else { 0.0 };
        let b = if active_j[row] { 1.0 } else { 0.0 };
        active_mass_i += q * a;
        joint_mass += q * a * b;
    }
    if !(active_mass_i > 0.0) {
        return Err(
            "conditional_coactivation_influence_values: gate i has zero active mass".to_string(),
        );
    }
    let conditional_probability = joint_mass / active_mass_i;
    let mut psi = Vec::with_capacity(rows.len());
    for &row in rows {
        let a = if active_i[row] { 1.0 } else { 0.0 };
        let b = if active_j[row] { 1.0 } else { 0.0 };
        psi.push(a * (b - conditional_probability) / active_mass_i);
    }
    Ok(ConditionalCoactivationInfluence {
        conditional_probability,
        active_mass_i,
        psi,
        normalized_weights,
    })
}

/// Build residual gate indicators by regressing each gate on the shared chart
/// basis plus an intercept, then thresholding the positive residual. Passing an
/// empty chart leaves the gates unchanged.
pub fn residual_gate_activities(
    gate_i: &[f64],
    gate_j: &[f64],
    shared_chart: Option<ArrayView2<'_, f64>>,
    likelihood_weights: &[f64],
    active_threshold: f64,
) -> Result<ResidualGateActivities, String> {
    if gate_i.len() != gate_j.len() {
        return Err(format!(
            "residual_gate_activities: gate lengths differ ({} vs {})",
            gate_i.len(),
            gate_j.len()
        ));
    }
    if likelihood_weights.len() != gate_i.len() {
        return Err(format!(
            "residual_gate_activities: {} weights for {} gates",
            likelihood_weights.len(),
            gate_i.len()
        ));
    }
    if !active_threshold.is_finite() {
        return Err("residual_gate_activities: active threshold must be finite".to_string());
    }
    for (row, &w) in likelihood_weights.iter().enumerate() {
        if !(w.is_finite() && w > 0.0) {
            return Err(format!(
                "residual_gate_activities: row {row} has invalid weight {w}"
            ));
        }
    }
    let residual_i = residualize_gate(gate_i, shared_chart.clone(), likelihood_weights)?;
    let residual_j = residualize_gate(gate_j, shared_chart, likelihood_weights)?;
    let active_i: Vec<bool> = residual_i.iter().map(|&g| g > active_threshold).collect();
    let active_j: Vec<bool> = residual_j.iter().map(|&g| g > active_threshold).collect();
    Ok(ResidualGateActivities {
        residual_i,
        residual_j,
        active_i,
        active_j,
    })
}

/// Deterministic residual-cluster labels from explicit centroids. These labels
/// are naming diagnostics only; they never define conditionality.
pub fn derive_residual_cluster_labels(
    residuals: ArrayView2<'_, f64>,
    centroids: ArrayView2<'_, f64>,
) -> Result<Vec<usize>, String> {
    let (n, p) = residuals.dim();
    let (k, cp) = centroids.dim();
    if p != cp {
        return Err(format!(
            "derive_residual_cluster_labels: residual width {p} != centroid width {cp}"
        ));
    }
    if k == 0 {
        return Err("derive_residual_cluster_labels: need at least one centroid".to_string());
    }
    let mut labels = Vec::with_capacity(n);
    for row in 0..n {
        let mut best = 0usize;
        let mut best_dist = f64::INFINITY;
        for c in 0..k {
            let mut dist = 0.0;
            for col in 0..p {
                let d = residuals[[row, col]] - centroids[[c, col]];
                dist += d * d;
            }
            if dist < best_dist {
                best_dist = dist;
                best = c;
            }
        }
        labels.push(best);
    }
    Ok(labels)
}

/// Convenience full-pass sample for tests and exact in-memory callers.
pub fn full_pass_rows(n: usize) -> DesignedRowSample {
    DesignedRowSample {
        provenance: MeasureProvenance::Uniform,
        rows: (0..n).collect(),
        likelihood_weights: vec![1.0; n],
        expected_size: n as f64,
    }
}

fn validate_partition_free_inputs(
    gate_i: &[f64],
    gate_j: &[f64],
    continuous_context: &[f64],
    diagnostic_labels: Option<&[usize]>,
    rows: &[usize],
    likelihood_weights: &[f64],
) -> Result<(), String> {
    let n = gate_i.len();
    if gate_j.len() != n || continuous_context.len() != n {
        return Err(format!(
            "coactivation conditionality: lengths differ gate_i={} gate_j={} context={}",
            gate_i.len(),
            gate_j.len(),
            continuous_context.len()
        ));
    }
    if let Some(labels) = diagnostic_labels
        && labels.len() != n
    {
        return Err(format!(
            "coactivation conditionality: diagnostic labels length {} != gates {n}",
            labels.len()
        ));
    }
    if rows.len() != likelihood_weights.len() {
        return Err(format!(
            "coactivation conditionality: {} rows but {} weights",
            rows.len(),
            likelihood_weights.len()
        ));
    }
    if rows.is_empty() {
        return Err("coactivation conditionality: need at least one sampled row".to_string());
    }
    for (slot, &row) in rows.iter().enumerate() {
        if row >= n {
            return Err(format!(
                "coactivation conditionality: sampled row {row} out of range {n}"
            ));
        }
        let w = likelihood_weights[slot];
        if !(w.is_finite() && w > 0.0) {
            return Err(format!(
                "coactivation conditionality: sampled row {row} has invalid weight {w}"
            ));
        }
        let gi = gate_i[row];
        let gj = gate_j[row];
        let x = continuous_context[row];
        if !(gi.is_finite() && gj.is_finite() && x.is_finite()) {
            return Err(format!(
                "coactivation conditionality: sampled row {row} has non-finite gate/context"
            ));
        }
    }
    Ok(())
}

fn fit_varying_coefficient_gam(
    gate_i: &[f64],
    gate_j: &[f64],
    continuous_context: &[f64],
    rows: &[usize],
    likelihood_weights: &[f64],
    config: VaryingCoefficientConfig,
) -> Result<VaryingCoefficientConditionality, String> {
    if config.spline_degree < 1 {
        return Err(format!(
            "fit_varying_coefficient_gam: spline degree must be >= 1, got {}",
            config.spline_degree
        ));
    }
    if config.penalty_order == 0 {
        return Err("fit_varying_coefficient_gam: penalty order must be >= 1".to_string());
    }

    let x_sample: Vec<f64> = rows.iter().map(|&row| continuous_context[row]).collect();
    let x_min = x_sample.iter().copied().fold(f64::INFINITY, f64::min);
    let x_max = x_sample.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !(x_max > x_min) {
        return Err("fit_varying_coefficient_gam: context needs positive range".to_string());
    }
    let x_array = Array1::from_vec(x_sample);
    let (basis_arc, knot_vector) = create_basis::<Dense>(
        x_array.view(),
        KnotSource::Generate {
            data_range: (x_min, x_max),
            num_internal_knots: config.num_internal_knots,
        },
        config.spline_degree,
        BasisOptions::value(),
    )
    .map_err(|e| format!("fit_varying_coefficient_gam: basis build failed: {e}"))?;
    if knot_vector.len() <= config.spline_degree {
        return Err(format!(
            "fit_varying_coefficient_gam: knot vector has {} entries for degree {}",
            knot_vector.len(),
            config.spline_degree
        ));
    }
    let basis = basis_arc.as_ref();
    let n = rows.len();
    let beta_cols = basis.ncols();
    if beta_cols <= config.penalty_order {
        return Err(format!(
            "fit_varying_coefficient_gam: basis has {beta_cols} columns but penalty order is {}",
            config.penalty_order
        ));
    }
    let design_cols = beta_cols + 1;
    let mut design = vec![vec![0.0_f64; design_cols]; n];
    for slot in 0..n {
        let row = rows[slot];
        design[slot][0] = 1.0;
        for col in 0..beta_cols {
            design[slot][col + 1] = gate_i[row] * basis[[slot, col]];
        }
    }
    let penalty_beta = difference_penalty(beta_cols, config.penalty_order)?;
    let mut penalty = vec![vec![0.0_f64; design_cols]; design_cols];
    for r in 0..beta_cols {
        for c in 0..beta_cols {
            penalty[r + 1][c + 1] = penalty_beta[r][c];
        }
    }
    let y: Vec<f64> = rows.iter().map(|&row| gate_j[row]).collect();
    let fit_for_log_lambda = |log_lambda: f64| -> Result<PenalizedFit, String> {
        penalized_gaussian_fit(&design, &y, likelihood_weights, &penalty, log_lambda)
    };
    let selected_log_smoothing = minimize_reml_log_smoothing(fit_for_log_lambda)?;
    let final_fit =
        penalized_gaussian_fit(&design, &y, likelihood_weights, &penalty, selected_log_smoothing)?;
    let mut coefficients = vec![0.0_f64; beta_cols];
    coefficients.copy_from_slice(&final_fit.coef[1..]);
    let mut beta_at_rows = Vec::with_capacity(n);
    for slot in 0..n {
        let mut beta = 0.0_f64;
        for col in 0..beta_cols {
            beta += basis[[slot, col]] * coefficients[col];
        }
        beta_at_rows.push(beta);
    }
    let total_weight: f64 = likelihood_weights.iter().sum();
    let beta_mean = beta_at_rows
        .iter()
        .zip(likelihood_weights.iter())
        .map(|(&b, &w)| w * b)
        .sum::<f64>()
        / total_weight;
    let beta_variation = beta_at_rows
        .iter()
        .zip(likelihood_weights.iter())
        .map(|(&b, &w)| {
            let d = b - beta_mean;
            w * d * d
        })
        .sum::<f64>()
        / total_weight;
    let penalty_energy = quadratic_form(&coefficients, &penalty_beta);
    let beta_norm = coefficients.iter().map(|v| v * v).sum::<f64>();
    let beta_wiggliness = if beta_norm > 0.0 {
        penalty_energy / beta_norm
    } else {
        0.0
    };
    Ok(VaryingCoefficientConditionality {
        selected_log_smoothing,
        reml_score: final_fit.reml_score,
        effective_degrees: final_fit.effective_degrees,
        beta_wiggliness,
        beta_variation,
        beta_mean,
        coefficients,
        beta_at_rows,
    })
}

#[derive(Clone, Debug)]
struct PenalizedFit {
    coef: Vec<f64>,
    reml_score: f64,
    effective_degrees: f64,
}

fn penalized_gaussian_fit(
    design: &[Vec<f64>],
    y: &[f64],
    weights: &[f64],
    penalty: &[Vec<f64>],
    log_lambda: f64,
) -> Result<PenalizedFit, String> {
    let n = design.len();
    let p = design
        .first()
        .map(|row| row.len())
        .ok_or_else(|| "penalized_gaussian_fit: empty design".to_string())?;
    let lambda = log_lambda.exp();
    let mut xtwx = vec![vec![0.0_f64; p]; p];
    let mut xtwy = vec![0.0_f64; p];
    let mut ywy = 0.0_f64;
    for row in 0..n {
        let w = weights[row];
        let yy = y[row];
        ywy += w * yy * yy;
        for a in 0..p {
            let xa = design[row][a];
            xtwy[a] += w * xa * yy;
            for b in 0..p {
                xtwx[a][b] += w * xa * design[row][b];
            }
        }
    }
    let mut a = xtwx.clone();
    for r in 0..p {
        for c in 0..p {
            a[r][c] += lambda * penalty[r][c];
        }
    }
    let factor = cholesky_decompose(&a)?;
    let coef = cholesky_solve(&factor, &xtwy);
    let xty_beta = dot(&xtwy, &coef);
    let penalty_energy = quadratic_form(&coef, penalty);
    let sse = (ywy - 2.0 * xty_beta + quadratic_form_matrix(&coef, &xtwx)).max(0.0);
    let penalized_sse = (sse + lambda * penalty_energy).max(f64::MIN_POSITIVE);
    let mut effective_degrees = 0.0_f64;
    for col in 0..p {
        let mut rhs = vec![0.0_f64; p];
        for row in 0..p {
            rhs[row] = xtwx[row][col];
        }
        let solved = cholesky_solve(&factor, &rhs);
        effective_degrees += solved[col];
    }
    let penalty_rank = penalty_rank_from_diagonal(penalty);
    let df = (n as f64 - effective_degrees).max(MIN_RESIDUAL_DF);
    let logdet = cholesky_logdet(&factor);
    let reml_score = logdet - penalty_rank as f64 * log_lambda + df * (penalized_sse / df).ln();
    Ok(PenalizedFit {
        coef,
        reml_score,
        effective_degrees,
    })
}

fn minimize_reml_log_smoothing<F>(mut evaluate: F) -> Result<f64, String>
where
    F: FnMut(f64) -> Result<PenalizedFit, String>,
{
    let mut center = 0.0_f64;
    let mut step = 1.0_f64;
    let mut f_center = evaluate(center)?.reml_score;
    let mut left = center - step;
    let mut right = center + step;
    let mut f_left = evaluate(left)?.reml_score;
    let mut f_right = evaluate(right)?.reml_score;
    for _iteration in 0..BRACKET_EXPANSION_LIMIT {
        if f_left >= f_center && f_right >= f_center {
            return golden_section_minimize(&mut evaluate, left, right);
        }
        if f_left < f_center {
            right = center;
            center = left;
            f_center = f_left;
            step *= 2.0;
            left = center - step;
            f_left = evaluate(left)?.reml_score;
            f_right = evaluate(right)?.reml_score;
        } else {
            left = center;
            center = right;
            f_center = f_right;
            step *= 2.0;
            right = center + step;
            f_right = evaluate(right)?.reml_score;
            f_left = evaluate(left)?.reml_score;
        }
    }
    golden_section_minimize(&mut evaluate, left, right)
}

fn golden_section_minimize<F>(evaluate: &mut F, mut left: f64, mut right: f64) -> Result<f64, String>
where
    F: FnMut(f64) -> Result<PenalizedFit, String>,
{
    let inv_phi = (5.0_f64.sqrt() - 1.0) * 0.5;
    let mut c = right - inv_phi * (right - left);
    let mut d = left + inv_phi * (right - left);
    let mut f_c = evaluate(c)?.reml_score;
    let mut f_d = evaluate(d)?.reml_score;
    for _iteration in 0..BRENT_ITERATION_LIMIT {
        let scale = 1.0 + left.abs().max(right.abs());
        if (right - left).abs() <= BRENT_REL_TOL * scale {
            break;
        }
        if f_c <= f_d {
            right = d;
            d = c;
            f_d = f_c;
            c = right - inv_phi * (right - left);
            f_c = evaluate(c)?.reml_score;
        } else {
            left = c;
            c = d;
            f_c = f_d;
            d = left + inv_phi * (right - left);
            f_d = evaluate(d)?.reml_score;
        }
    }
    Ok(0.5 * (left + right))
}

fn difference_penalty(width: usize, order: usize) -> Result<Vec<Vec<f64>>, String> {
    if order == 0 || width <= order {
        return Err(format!(
            "difference_penalty: width {width} cannot support order {order}"
        ));
    }
    let rows = width - order;
    let mut diff = vec![vec![0.0_f64; width]; rows];
    let coeff = difference_coefficients(order);
    for r in 0..rows {
        for c in 0..=order {
            diff[r][r + c] = coeff[c];
        }
    }
    let mut penalty = vec![vec![0.0_f64; width]; width];
    for r in 0..rows {
        for a in 0..width {
            for b in 0..width {
                penalty[a][b] += diff[r][a] * diff[r][b];
            }
        }
    }
    Ok(penalty)
}

fn difference_coefficients(order: usize) -> Vec<f64> {
    let mut coeff = vec![1.0_f64];
    for _ in 0..order {
        let mut next = vec![0.0_f64; coeff.len() + 1];
        for (idx, &value) in coeff.iter().enumerate() {
            next[idx] -= value;
            next[idx + 1] += value;
        }
        coeff = next;
    }
    coeff
}

fn penalty_rank_from_diagonal(penalty: &[Vec<f64>]) -> usize {
    let diag_max = penalty
        .iter()
        .enumerate()
        .map(|(idx, row)| row[idx].abs())
        .fold(0.0_f64, f64::max);
    let floor = f64::EPSILON * penalty.len().max(1) as f64 * diag_max.max(1.0);
    penalty
        .iter()
        .enumerate()
        .filter(|(idx, row)| row[*idx].abs() > floor)
        .count()
}

fn diagnose_context_labels(
    gate_i: &[f64],
    gate_j: &[f64],
    labels: &[usize],
    rows: &[usize],
    likelihood_weights: &[f64],
    beta_at_rows: &[f64],
) -> Result<Vec<ContextDiagnostic>, String> {
    let mut accum: BTreeMap<usize, DiagnosticAccum> = BTreeMap::new();
    for (slot, &row) in rows.iter().enumerate() {
        let w = likelihood_weights[slot];
        let entry = accum.entry(labels[row]).or_default();
        entry.rows += 1;
        entry.mass += w;
        entry.sum_i += w * gate_i[row];
        entry.sum_j += w * gate_j[row];
        entry.sum_ij += w * gate_i[row] * gate_j[row];
        entry.sum_beta += w * beta_at_rows[slot];
    }
    let mut diagnostics = Vec::with_capacity(accum.len());
    for (&context, a) in accum.iter() {
        if !(a.mass > 0.0) {
            return Err(format!(
                "diagnose_context_labels: context {context} has non-positive mass"
            ));
        }
        let mean_i = a.sum_i / a.mass;
        let mean_j = a.sum_j / a.mass;
        diagnostics.push(ContextDiagnostic {
            context,
            rows: a.rows,
            mass: a.mass,
            mean_gate_i: mean_i,
            mean_gate_j: mean_j,
            mean_beta: a.sum_beta / a.mass,
            centered_association: a.sum_ij / a.mass - mean_i * mean_j,
        });
    }
    Ok(diagnostics)
}

#[derive(Default)]
struct DiagnosticAccum {
    rows: usize,
    mass: f64,
    sum_i: f64,
    sum_j: f64,
    sum_ij: f64,
    sum_beta: f64,
}

fn residualize_gate(
    gate: &[f64],
    shared_chart: Option<ArrayView2<'_, f64>>,
    weights: &[f64],
) -> Result<Vec<f64>, String> {
    let Some(chart) = shared_chart else {
        return Ok(gate.to_vec());
    };
    let (n, q) = chart.dim();
    if n != gate.len() {
        return Err(format!(
            "residualize_gate: chart has {n} rows but gate has {}",
            gate.len()
        ));
    }
    if q == 0 {
        return Ok(gate.to_vec());
    }
    let cols = q + 1;
    let mut xtx = vec![vec![0.0_f64; cols]; cols];
    let mut xty = vec![0.0_f64; cols];
    for row in 0..n {
        let y = gate[row];
        if !y.is_finite() {
            return Err(format!("residualize_gate: row {row} has non-finite gate {y}"));
        }
        let w = weights[row];
        for a in 0..cols {
            let xa = if a == 0 { 1.0 } else { chart[[row, a - 1]] };
            if !xa.is_finite() {
                return Err(format!(
                    "residualize_gate: row {row} chart column {} is non-finite",
                    a - 1
                ));
            }
            xty[a] += w * xa * y;
            for b in 0..cols {
                let xb = if b == 0 { 1.0 } else { chart[[row, b - 1]] };
                xtx[a][b] += w * xa * xb;
            }
        }
    }
    let beta = solve_symmetric_system(xtx, xty)?;
    let mut residual = vec![0.0_f64; n];
    let gate_scale = gate.iter().fold(1.0_f64, |acc, &v| acc.max(v.abs()));
    let residual_floor = f64::EPSILON * cols.max(1) as f64 * gate_scale;
    for row in 0..n {
        let mut fitted = beta[0];
        for col in 0..q {
            fitted += beta[col + 1] * chart[[row, col]];
        }
        let r = gate[row] - fitted;
        residual[row] = if r.abs() <= residual_floor { 0.0 } else { r };
    }
    Ok(residual)
}

fn solve_symmetric_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>, String> {
    let n = b.len();
    let mut scale = 1.0_f64;
    for row in 0..n {
        for col in 0..n {
            scale = scale.max(a[row][col].abs());
        }
    }
    let pivot_floor = f64::EPSILON * n.max(1) as f64 * scale;
    for col in 0..n {
        let mut pivot = col;
        let mut pivot_abs = a[col][col].abs();
        for row in (col + 1)..n {
            let candidate_abs = a[row][col].abs();
            if candidate_abs > pivot_abs {
                pivot = row;
                pivot_abs = candidate_abs;
            }
        }
        if !(pivot_abs > pivot_floor) {
            return Err(format!(
                "solve_symmetric_system: singular shared-chart normal equation at column {col}"
            ));
        }
        if pivot != col {
            a.swap(pivot, col);
            b.swap(pivot, col);
        }
        let diag = a[col][col];
        for row in (col + 1)..n {
            let factor = a[row][col] / diag;
            a[row][col] = 0.0;
            for k in (col + 1)..n {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = vec![0.0_f64; n];
    for row in (0..n).rev() {
        let mut rhs = b[row];
        for col in (row + 1)..n {
            rhs -= a[row][col] * x[col];
        }
        x[row] = rhs / a[row][row];
    }
    Ok(x)
}

fn cholesky_decompose(a: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
    let n = a.len();
    let mut l = vec![vec![0.0_f64; n]; n];
    let scale = a
        .iter()
        .enumerate()
        .map(|(idx, row)| row[idx].abs())
        .fold(1.0_f64, f64::max);
    let floor = f64::EPSILON * n.max(1) as f64 * scale;
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                if !(sum > floor) {
                    return Err(format!(
                        "cholesky_decompose: non-SPD matrix at diagonal {i} with value {sum}"
                    ));
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    Ok(l)
}

fn cholesky_solve(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i][k] * y[k];
        }
        y[i] = sum / l[i][i];
    }
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for k in (i + 1)..n {
            sum -= l[k][i] * x[k];
        }
        x[i] = sum / l[i][i];
    }
    x
}

fn cholesky_logdet(l: &[Vec<f64>]) -> f64 {
    2.0 * l
        .iter()
        .enumerate()
        .map(|(idx, row)| row[idx].ln())
        .sum::<f64>()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

fn quadratic_form(x: &[f64], a: &[Vec<f64>]) -> f64 {
    quadratic_form_matrix(x, a)
}

fn quadratic_form_matrix(x: &[f64], a: &[Vec<f64>]) -> f64 {
    let mut value = 0.0_f64;
    for r in 0..x.len() {
        for c in 0..x.len() {
            value += x[r] * a[r][c] * x[c];
        }
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn invariant_coupling_selects_constant_beta_and_large_radius() {
        let n = 96usize;
        let mut x = Vec::with_capacity(n);
        let mut gate_i = Vec::with_capacity(n);
        let mut gate_j = Vec::with_capacity(n);
        for row in 0..n {
            let t = row as f64 / (n - 1) as f64;
            x.push(t);
            let gi = 0.8 + 0.3 * (2.0 * std::f64::consts::PI * t).sin();
            gate_i.push(gi);
            gate_j.push(0.2 + 1.7 * gi);
        }
        let sample = full_pass_rows(n);
        let report = estimate_from_designed_sample(
            &gate_i,
            &gate_j,
            &x,
            None,
            &sample,
            VaryingCoefficientConfig::default(),
        )
        .expect("partition-free conditionality");
        println!(
            "case=invariant beta_wiggliness={:.6e} beta_variation={:.6e} epsilon_star={:.6e} rho={:.6e}",
            report.native.beta_wiggliness,
            report.native.beta_variation,
            report.certificate.robustness_radius_epsilon,
            report.certificate.rho
        );
        assert!(report.native.beta_wiggliness < 1.0e-6);
        assert!(report.native.beta_variation < 1.0e-8);
        assert!(report.certificate.robustness_radius_epsilon > 1.0e10);
    }

    #[test]
    fn context_varying_coupling_selects_wiggly_beta_and_small_radius() {
        let n = 120usize;
        let mut x = Vec::with_capacity(n);
        let mut gate_i = Vec::with_capacity(n);
        let mut gate_j = Vec::with_capacity(n);
        for row in 0..n {
            let t = row as f64 / (n - 1) as f64;
            x.push(t);
            let gi = 0.8 + 0.25 * (6.0 * std::f64::consts::PI * t).cos();
            let beta = if t < 0.5 { 1.8 } else { -1.8 };
            gate_i.push(gi);
            gate_j.push(0.1 + beta * gi);
        }
        let labels: Vec<usize> = x.iter().map(|&t| if t >= 0.5 { 1 } else { 0 }).collect();
        let sample = full_pass_rows(n);
        let report = estimate_from_designed_sample(
            &gate_i,
            &gate_j,
            &x,
            Some(&labels),
            &sample,
            VaryingCoefficientConfig::default(),
        )
        .expect("partition-free conditionality");
        println!(
            "case=context_varying beta_wiggliness={:.6e} beta_variation={:.6e} epsilon_star={:.6e} rho={:.6e} diagnostics={}",
            report.native.beta_wiggliness,
            report.native.beta_variation,
            report.certificate.robustness_radius_epsilon,
            report.certificate.rho,
            report.diagnostics.len()
        );
        assert!(report.native.beta_wiggliness > 1.0e-2);
        assert!(report.native.beta_variation > 0.5);
        assert!(report.certificate.robustness_radius_epsilon < 0.05);
        assert_eq!(report.diagnostics.len(), 2);
    }

    #[test]
    fn robustness_radius_matches_direct_adversarial_reweighting_search() {
        let n = 400usize;
        let mut gate_i = Vec::with_capacity(n);
        let mut gate_j = Vec::with_capacity(n);
        for row in 0..n {
            let t = row as f64 / n as f64;
            let a = (2.0 * std::f64::consts::PI * t).sin();
            let b = (4.0 * std::f64::consts::PI * t).cos();
            gate_i.push(a + 0.35 * b);
            gate_j.push(0.06 * a + b);
        }
        let rows: Vec<usize> = (0..n).collect();
        let weights = vec![1.0_f64; n];
        let influence = coupling_influence_values(&gate_i, &gate_j, &rows, &weights)
            .expect("influence values");
        let certificate = influence.certificate();
        let direct = direct_exponential_tilt_radius_to_kill(
            certificate.rho,
            &influence.psi,
            &influence.normalized_weights,
        )
        .expect("direct tilt radius");
        println!(
            "case=adversarial formula_epsilon_star={:.6e} direct_epsilon={:.6e} rho={:.6e} var_psi={:.6e}",
            certificate.robustness_radius_epsilon,
            direct,
            certificate.rho,
            certificate.influence_variance
        );
        let rel = (direct - certificate.robustness_radius_epsilon).abs()
            / certificate.robustness_radius_epsilon.max(1.0e-12);
        assert!(rel < 0.08, "relative error {rel}");
    }

    #[test]
    fn weighted_correlation_influence_matches_leave_one_out_jackknife() {
        let n = 701usize;
        let mut gate_i = Vec::with_capacity(n);
        let mut gate_j = Vec::with_capacity(n);
        let mut weights = Vec::with_capacity(n);
        for row in 0..n {
            let t = row as f64 / (n - 1) as f64;
            let low = (2.0 * std::f64::consts::PI * t).sin();
            let mid = (10.0 * std::f64::consts::PI * t).cos();
            let high = (row as f64 * 0.173).sin();
            gate_i.push(0.3 + 0.7 * low + 0.19 * mid);
            gate_j.push(-0.2 + 0.45 * low - 0.31 * mid + 0.23 * high);
            weights.push(0.75 + 0.35 * (row % 11) as f64 / 10.0);
        }
        let rows: Vec<usize> = (0..n).collect();
        let influence = coupling_influence_values(&gate_i, &gate_j, &rows, &weights)
            .expect("weighted correlation influence");
        assert_eq!(influence.statistic, CouplingStatistic::WeightedPearson);
        assert_eq!(
            influence.certificate().statistic,
            CouplingStatistic::WeightedPearson
        );
        let rho = weighted_correlation_stat_excluding(&gate_i, &gate_j, &rows, &weights, None)
            .expect("weighted correlation statistic");
        assert!((rho - influence.rho).abs() < 1.0e-14);

        let mut max_closed_form_diff = 0.0_f64;
        let mut max_jackknife_diff = 0.0_f64;
        let mut jackknife_sse = 0.0_f64;
        let mut weighted_mean = 0.0_f64;
        let (mean_i, mean_j, sd_i, sd_j) =
            weighted_standardization(&gate_i, &gate_j, &rows, &weights, None)
                .expect("weighted standardization");
        for slot in 0..n {
            let row = rows[slot];
            let zi = (gate_i[row] - mean_i) / sd_i;
            let zj = (gate_j[row] - mean_j) / sd_j;
            let closed_form = zi * zj - 0.5 * influence.rho * (zi * zi + zj * zj);
            let closed_form_diff = (closed_form - influence.psi[slot]).abs();
            max_closed_form_diff = max_closed_form_diff.max(closed_form_diff);
            weighted_mean += influence.normalized_weights[slot] * influence.psi[slot];

            let leave_one_out =
                weighted_correlation_stat_excluding(&gate_i, &gate_j, &rows, &weights, Some(slot))
                    .expect("leave-one-out weighted correlation");
            let q = influence.normalized_weights[slot];
            let jackknife = ((1.0 - q) / q) * (influence.rho - leave_one_out);
            let diff = jackknife - influence.psi[slot];
            max_jackknife_diff = max_jackknife_diff.max(diff.abs());
            jackknife_sse += diff * diff;
        }
        let jackknife_rms = (jackknife_sse / n as f64).sqrt();
        println!(
            "case=weighted_corr_if rho={:.6e} max_closed_form_diff={:.6e} mean_psi={:.6e} max_jackknife_diff={:.6e} jackknife_rms={:.6e}",
            influence.rho,
            max_closed_form_diff,
            weighted_mean,
            max_jackknife_diff,
            jackknife_rms
        );
        // Exact checks: the closed-form influence recomputed independently must
        // match the returned psi to machine precision, and every influence
        // function is mean-zero under the sampling measure. These are the
        // honest "both exact" pins.
        assert!(max_closed_form_diff < 1.0e-14);
        assert!(weighted_mean.abs() < 1.0e-14);
        // Finite-sample jackknife pin. The leave-one-out jackknife recovers the
        // influence function only to first order, so it agrees with the exact
        // closed form to O(1/N) (here 1/701 ~ 1.4e-3), NOT to machine precision.
        // The bound below is that O(1/N) tolerance; asserting a 1e-6 bound here
        // would falsely conflate a finite-sample estimator with the exact IF and
        // is banned by SPEC (never paper over). The exact 1e-6 comparison lives
        // in the two machine-precision assertions above.
        assert!(max_jackknife_diff < 1.2e-2);
        assert!(jackknife_rms < 3.0e-3);
    }

    #[test]
    fn conditional_influence_matches_leave_one_out_jackknife() {
        let n = 701usize;
        let mut active_i = Vec::with_capacity(n);
        let mut active_j = Vec::with_capacity(n);
        let mut weights = Vec::with_capacity(n);
        for row in 0..n {
            let t = row as f64 / (n - 1) as f64;
            // Gate i active roughly half the time (denominator mass ~ 0.5) so the
            // ratio estimator is well conditioned; gate j depends on i plus its
            // own structure so the conditional probability is a nontrivial ratio.
            let ai = (2.0 * std::f64::consts::PI * t).sin() + 0.2 * (row as f64 * 0.37).cos() > 0.0;
            let aj = if ai {
                (7.0 * std::f64::consts::PI * t).cos() + 0.15 * (row as f64 * 0.11).sin() > -0.3
            } else {
                (5.0 * std::f64::consts::PI * t).sin() > 0.0
            };
            active_i.push(ai);
            active_j.push(aj);
            weights.push(0.75 + 0.35 * (row % 11) as f64 / 10.0);
        }
        let rows: Vec<usize> = (0..n).collect();
        let influence =
            conditional_coactivation_influence_values(&active_i, &active_j, &rows, &weights)
                .expect("conditional coactivation influence");

        // The conditional probability the influence function protects must equal
        // the plain ratio statistic computed directly.
        let pi = conditional_probability_excluding(&active_i, &active_j, &rows, &weights, None)
            .expect("conditional probability statistic");
        assert!((pi - influence.conditional_probability).abs() < 1.0e-14);

        let mut max_closed_form_diff = 0.0_f64;
        let mut max_jackknife_diff = 0.0_f64;
        let mut jackknife_sse = 0.0_f64;
        let mut weighted_mean = 0.0_f64;
        for slot in 0..n {
            let row = rows[slot];
            let a = if active_i[row] { 1.0 } else { 0.0 };
            let b = if active_j[row] { 1.0 } else { 0.0 };
            // Exact closed form: psi_i = 1_{g_i}(1_{g_j} - pi) / E[1_{g_i}].
            let closed_form =
                a * (b - influence.conditional_probability) / influence.active_mass_i;
            max_closed_form_diff =
                max_closed_form_diff.max((closed_form - influence.psi[slot]).abs());
            weighted_mean += influence.normalized_weights[slot] * influence.psi[slot];

            let leave_one_out = conditional_probability_excluding(
                &active_i,
                &active_j,
                &rows,
                &weights,
                Some(slot),
            )
            .expect("leave-one-out conditional probability");
            let q = influence.normalized_weights[slot];
            let jackknife = ((1.0 - q) / q) * (influence.conditional_probability - leave_one_out);
            let diff = jackknife - influence.psi[slot];
            max_jackknife_diff = max_jackknife_diff.max(diff.abs());
            jackknife_sse += diff * diff;
        }
        let jackknife_rms = (jackknife_sse / n as f64).sqrt();
        println!(
            "case=conditional_if pi={:.6e} mass_i={:.6e} max_closed_form_diff={:.6e} mean_psi={:.6e} max_jackknife_diff={:.6e} jackknife_rms={:.6e}",
            influence.conditional_probability,
            influence.active_mass_i,
            max_closed_form_diff,
            weighted_mean,
            max_jackknife_diff,
            jackknife_rms
        );

        // Exact pins: the recomputed closed form matches the returned psi to
        // machine precision, and the ratio influence function is exactly
        // mean-zero (sum_i q_i psi_i = (joint - pi * mass_i)/mass_i = 0).
        assert!(max_closed_form_diff < 1.0e-14);
        assert!(weighted_mean.abs() < 1.0e-14);
        // Finite-sample jackknife pin. The leave-one-out jackknife recovers the
        // ratio influence function only to first order, so it agrees to O(1/N),
        // NOT to machine precision. The tolerance is written as an explicit
        // multiple of 1/N to make the O(1/N) scaling honest rather than a magic
        // constant; a 1e-6 bound here would falsely equate a finite-sample
        // estimator with the exact IF (SPEC: never paper over).
        let jackknife_tol = 10.0 / n as f64;
        assert!(
            max_jackknife_diff < jackknife_tol,
            "max_jackknife_diff {max_jackknife_diff} exceeds O(1/N) bound {jackknife_tol}"
        );
        assert!(jackknife_rms < jackknife_tol);
    }

    fn conditional_probability_excluding(
        active_i: &[bool],
        active_j: &[bool],
        rows: &[usize],
        weights: &[f64],
        excluded_slot: Option<usize>,
    ) -> Result<f64, String> {
        let mut denom = 0.0_f64;
        let mut numer = 0.0_f64;
        for slot in 0..rows.len() {
            if excluded_slot == Some(slot) {
                continue;
            }
            let row = rows[slot];
            let weight = weights[slot];
            let a = if active_i[row] { 1.0 } else { 0.0 };
            let b = if active_j[row] { 1.0 } else { 0.0 };
            denom += weight * a;
            numer += weight * a * b;
        }
        if !(denom > 0.0) {
            return Err(
                "conditional_probability_excluding: zero active mass in retained sample".to_string(),
            );
        }
        Ok(numer / denom)
    }

    #[test]
    fn residual_gate_denominator_removes_same_chart_anchor_binding() {
        let n = 12usize;
        let chart_gate: Vec<f64> = (0..n)
            .map(|row| if row % 3 == 0 { 1.0 } else { 0.0 })
            .collect();
        let sample = full_pass_rows(n);
        let chart = Array2::from_shape_vec((n, 1), chart_gate.clone()).unwrap();
        let residual = residual_gate_activities(
            &chart_gate,
            &chart_gate,
            Some(chart.view()),
            &sample.likelihood_weights,
            0.0,
        )
        .unwrap();
        assert!(residual.active_i.iter().all(|&active| !active));
        assert!(residual.active_j.iter().all(|&active| !active));
    }

    fn weighted_correlation_stat_excluding(
        gate_i: &[f64],
        gate_j: &[f64],
        rows: &[usize],
        weights: &[f64],
        excluded_slot: Option<usize>,
    ) -> Result<f64, String> {
        let (mean_i, mean_j, sd_i, sd_j) =
            weighted_standardization(gate_i, gate_j, rows, weights, excluded_slot)?;
        let mut total_weight = 0.0_f64;
        let mut covariance = 0.0_f64;
        for slot in 0..rows.len() {
            if excluded_slot == Some(slot) {
                continue;
            }
            let row = rows[slot];
            let weight = weights[slot];
            total_weight += weight;
            covariance += weight * (gate_i[row] - mean_i) * (gate_j[row] - mean_j);
        }
        Ok(covariance / total_weight / (sd_i * sd_j))
    }

    fn weighted_standardization(
        gate_i: &[f64],
        gate_j: &[f64],
        rows: &[usize],
        weights: &[f64],
        excluded_slot: Option<usize>,
    ) -> Result<(f64, f64, f64, f64), String> {
        let mut total_weight = 0.0_f64;
        let mut mean_i = 0.0_f64;
        let mut mean_j = 0.0_f64;
        for slot in 0..rows.len() {
            if excluded_slot == Some(slot) {
                continue;
            }
            let row = rows[slot];
            let weight = weights[slot];
            total_weight += weight;
            mean_i += weight * gate_i[row];
            mean_j += weight * gate_j[row];
        }
        if !(total_weight > 0.0) {
            return Err("weighted_standardization: empty retained sample".to_string());
        }
        mean_i /= total_weight;
        mean_j /= total_weight;
        let mut var_i = 0.0_f64;
        let mut var_j = 0.0_f64;
        for slot in 0..rows.len() {
            if excluded_slot == Some(slot) {
                continue;
            }
            let row = rows[slot];
            let weight = weights[slot];
            let zi = gate_i[row] - mean_i;
            let zj = gate_j[row] - mean_j;
            var_i += weight * zi * zi;
            var_j += weight * zj * zj;
        }
        var_i /= total_weight;
        var_j /= total_weight;
        if !(var_i > 0.0 && var_j > 0.0) {
            return Err("weighted_standardization: zero variance".to_string());
        }
        Ok((mean_i, mean_j, var_i.sqrt(), var_j.sqrt()))
    }

    fn direct_exponential_tilt_radius_to_kill(
        rho: f64,
        psi: &[f64],
        weights: &[f64],
    ) -> Result<f64, String> {
        if rho == 0.0 {
            return Ok(0.0);
        }
        let direction = if rho > 0.0 { -1.0 } else { 1.0 };
        let target = -rho;
        let shifted_mean = |eta: f64| -> (f64, f64) {
            let mut log_terms = Vec::with_capacity(psi.len());
            for &value in psi {
                log_terms.push(direction * eta * value);
            }
            let max_log = log_terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut z = 0.0_f64;
            let mut mean = 0.0_f64;
            for slot in 0..psi.len() {
                let un = weights[slot] * (log_terms[slot] - max_log).exp();
                z += un;
                mean += un * psi[slot];
            }
            mean /= z;
            let log_z = max_log + z.ln();
            let kl = direction * eta * mean - log_z;
            (mean, kl)
        };
        let mut lo = 0.0_f64;
        let mut hi = 1.0_f64;
        let mut hi_mean = shifted_mean(hi).0;
        for _iteration in 0..64 {
            let crossed = if rho > 0.0 {
                hi_mean <= target
            } else {
                hi_mean >= target
            };
            if crossed {
                break;
            }
            hi *= 2.0;
            hi_mean = shifted_mean(hi).0;
        }
        for _iteration in 0..96 {
            let mid = 0.5 * (lo + hi);
            let mid_mean = shifted_mean(mid).0;
            let crossed = if rho > 0.0 {
                mid_mean <= target
            } else {
                mid_mean >= target
            };
            if crossed {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        Ok(shifted_mean(hi).1)
    }
}
