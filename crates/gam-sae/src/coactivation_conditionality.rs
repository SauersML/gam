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

use crate::null_battery::ClaimNullCalibration;
use gam_terms::basis::{BasisOptions, Dense, KnotSource, create_basis};
use ndarray::{Array1, Array2, ArrayView2};
use std::collections::BTreeMap;

const DEFAULT_SPLINE_DEGREE: usize = 3;
const DEFAULT_INTERNAL_KNOTS: usize = 5;
const DEFAULT_PENALTY_ORDER: usize = 1;

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
    // D_m has `beta_cols - m` independent rows, hence rank(D_m^T D_m) is
    // exactly `beta_cols - m`. Reading non-zero diagonal entries instead counts
    // every coefficient touched by the difference operator and overstates rank.
    let penalty_rank = beta_cols - config.penalty_order;
    let mut penalty = vec![vec![0.0_f64; design_cols]; design_cols];
    for r in 0..beta_cols {
        for c in 0..beta_cols {
            penalty[r + 1][c + 1] = penalty_beta[r][c];
        }
    }
    let y: Vec<f64> = rows.iter().map(|&row| gate_j[row]).collect();
    let null_basis = penalty_null_basis(beta_cols, config.penalty_order);
    let reml = GaussianRemlProblem::new(
        &design,
        &y,
        likelihood_weights,
        &penalty,
        penalty_rank,
        &null_basis,
    )?;
    let (selected_log_smoothing, final_fit) = reml.select_and_fit()?;
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

/// The profiled Gaussian REML criterion of the varying-coefficient GAM in
/// `ρ = log λ`:
///
/// `V(ρ) = log|A| − r·ρ + ν·log P`,  `A = XᵀWX + λS`,  `P = yᵀWy − bᵀA⁻¹b`,
///
/// with `b = XᵀWy`, `P` the penalized residual sum of squares at `β̂ = A⁻¹b`,
/// `r = rank S`, and `ν = Σw − (p − r)` the residual degrees of freedom (the
/// likelihood mass less the unpenalized null space). The scale is profiled out.
/// The weighted sufficient statistics are accumulated once, so an evaluation
/// costs `O(p³)` in the coefficient dimension alone.
struct GaussianRemlProblem<'a> {
    xtwx: Vec<Vec<f64>>,
    xtwy: Vec<f64>,
    ywy: f64,
    penalty: &'a [Vec<f64>],
    penalty_rank: usize,
    residual_df: f64,
    null_space: NullSpaceFit,
}

/// The fit of the penalty null space `N` (`p × k`), the smooth's unpenalized
/// model. A null-space coefficient pays no penalty, so it is feasible at every
/// `λ` and every penalized fit keeps `P(ρ) ≤ P_null`.
struct NullSpaceFit {
    /// `N·ĉ`, in design coordinates.
    coef: Vec<f64>,
    /// `P_null = yᵀWy − gᵀĉ`, `g = NᵀXᵀWy`, `ĉ = (NᵀXᵀWXN)⁻¹g`.
    penalized_rss: f64,
    /// The arithmetic resolution of the cancelling difference `penalized_rss`.
    resolution: f64,
}

/// The penalized solve at one `ρ`, shared by the criterion value, its
/// derivatives and the reported fit.
struct GaussianRemlSolve {
    lambda: f64,
    factor: Vec<Vec<f64>>,
    coef: Vec<f64>,
    penalized_rss: f64,
    value: f64,
}

impl<'a> GaussianRemlProblem<'a> {
    fn new(
        design: &[Vec<f64>],
        y: &[f64],
        weights: &[f64],
        penalty: &'a [Vec<f64>],
        penalty_rank: usize,
        null_basis: &[Vec<f64>],
    ) -> Result<Self, String> {
        let p = design
            .first()
            .map(|row| row.len())
            .ok_or_else(|| "coactivation REML: empty design".to_string())?;
        let mut xtwx = vec![vec![0.0_f64; p]; p];
        let mut xtwx_magnitude = vec![vec![0.0_f64; p]; p];
        let mut xtwy = vec![0.0_f64; p];
        let mut xtwy_magnitude = vec![0.0_f64; p];
        let mut ywy = 0.0_f64;
        for (row, values) in design.iter().enumerate() {
            let w = weights[row];
            let yy = y[row];
            ywy += w * yy * yy;
            for a in 0..p {
                let xa = values[a];
                let cross = w * xa * yy;
                xtwy[a] += cross;
                xtwy_magnitude[a] += cross.abs();
                for b in 0..p {
                    let gram = w * xa * values[b];
                    xtwx[a][b] += gram;
                    xtwx_magnitude[a][b] += gram.abs();
                }
            }
        }
        // Frequency/HT weights define the likelihood mass represented by this
        // sample. The residual degrees of freedom must live on that same measure;
        // using the selected row count mixed a full-corpus weighted SSE with
        // selected-sample df and changed the criterion under a common weight scale.
        let likelihood_mass: f64 = weights.iter().sum();
        let null_dimension = p - penalty_rank;
        let residual_df = likelihood_mass - null_dimension as f64;
        if !(residual_df.is_finite() && residual_df > 0.0) {
            return Err(format!(
                "coactivation REML: likelihood mass {likelihood_mass} does not exceed the \
                 {null_dimension} unpenalized coefficients, so the profiled criterion has no \
                 residual degrees of freedom"
            ));
        }
        let null_space = null_space_fit(
            &xtwx,
            &xtwx_magnitude,
            &xtwy,
            &xtwy_magnitude,
            ywy,
            null_basis,
            design.len(),
        )?;
        Ok(Self {
            xtwx,
            xtwy,
            ywy,
            penalty,
            penalty_rank,
            residual_df,
            null_space,
        })
    }

    fn solve(&self, log_lambda: f64) -> Result<GaussianRemlSolve, String> {
        let p = self.xtwy.len();
        let lambda = gam_problem::checked_exp_log_strength(log_lambda)
            .map_err(|error| format!("coactivation REML: {error}"))?;
        let mut a = self.xtwx.clone();
        for r in 0..p {
            for c in 0..p {
                a[r][c] += lambda * self.penalty[r][c];
            }
        }
        let factor = cholesky_decompose(&a)?;
        let coef = cholesky_solve(&factor, &self.xtwy);
        let penalized_rss = self.ywy - dot(&self.xtwy, &coef);
        if !(penalized_rss.is_finite() && penalized_rss > 0.0) {
            return Err(format!(
                "coactivation REML: penalized residual sum of squares {penalized_rss} at \
                 log λ = {log_lambda} is not positive: the sample is fit exactly and the \
                 profiled criterion is unbounded"
            ));
        }
        let value = cholesky_logdet(&factor) - self.penalty_rank as f64 * log_lambda
            + self.residual_df * penalized_rss.ln();
        Ok(GaussianRemlSolve {
            lambda,
            factor,
            coef,
            penalized_rss,
            value,
        })
    }

    /// `(V, V′, V″)` at `ρ`, in closed form. With `M = A⁻¹S`, `q = β̂ᵀSβ̂`,
    /// `s = β̂ᵀSA⁻¹Sβ̂`, `dβ̂/dρ = −λA⁻¹Sβ̂` and the envelope identity
    /// `dP/dρ = λq`:
    ///
    /// `V′ = λ·tr M − r + ν·λq/P`,
    /// `V″ = λ·tr M − λ²·tr(M²) + ν·(λq/P − 2λ²s/P − (λq/P)²)`.
    fn jet(&self, log_lambda: f64) -> Result<(f64, f64, f64), String> {
        let solve = self.solve(log_lambda)?;
        let p = self.xtwy.len();
        let mut inverse_penalty = vec![vec![0.0_f64; p]; p];
        let mut column = vec![0.0_f64; p];
        for c in 0..p {
            for r in 0..p {
                column[r] = self.penalty[r][c];
            }
            let solved = cholesky_solve(&solve.factor, &column);
            for r in 0..p {
                inverse_penalty[r][c] = solved[r];
            }
        }
        let trace = (0..p).map(|i| inverse_penalty[i][i]).sum::<f64>();
        let mut trace_square = 0.0_f64;
        for i in 0..p {
            for j in 0..p {
                trace_square += inverse_penalty[i][j] * inverse_penalty[j][i];
            }
        }
        let penalty_beta: Vec<f64> = self
            .penalty
            .iter()
            .map(|row| dot(row, &solve.coef))
            .collect();
        let penalty_energy = dot(&solve.coef, &penalty_beta);
        let penalty_curvature = dot(
            &penalty_beta,
            &cholesky_solve(&solve.factor, &penalty_beta),
        );
        let lambda = solve.lambda;
        let energy_share = lambda * penalty_energy / solve.penalized_rss;
        let gradient =
            lambda * trace - self.penalty_rank as f64 + self.residual_df * energy_share;
        let hessian = lambda * trace - lambda * lambda * trace_square
            + self.residual_df
                * (energy_share
                    - 2.0 * lambda * lambda * penalty_curvature / solve.penalized_rss
                    - energy_share * energy_share);
        Ok((solve.value, gradient, hessian))
    }

    fn fit(&self, log_lambda: f64) -> Result<PenalizedFit, String> {
        let solve = self.solve(log_lambda)?;
        let p = self.xtwy.len();
        let mut effective_degrees = 0.0_f64;
        let mut column = vec![0.0_f64; p];
        for c in 0..p {
            for r in 0..p {
                column[r] = self.xtwx[r][c];
            }
            effective_degrees += cholesky_solve(&solve.factor, &column)[c];
        }
        Ok(PenalizedFit {
            coef: solve.coef,
            reml_score: solve.value,
            effective_degrees,
        })
    }

    /// Select `ρ̂` and return it with the fit there.
    ///
    /// When the penalty null space interpolates the sample, with `P_null` inside
    /// its own arithmetic resolution, every `P(ρ) ≤ P_null` is inside it too. The
    /// profiled criterion then has no finite value at any `ρ`, and there is no
    /// smoothing parameter to select. That is a structural result, not a failed
    /// search: the coefficient is the null space's (exactly constant under the
    /// default order-1 penalty), which is the `λ → ∞` limit of the penalized fit.
    /// It is reported at `log λ = +∞`, with the criterion's infimum `−∞` as its
    /// score and the null dimension as its effective degrees of freedom.
    /// Otherwise `ρ̂` is selected by the outer engine.
    fn select_and_fit(&self) -> Result<(f64, PenalizedFit), String> {
        let null_space = &self.null_space;
        if null_space.penalized_rss.is_finite()
            && !(null_space.penalized_rss > null_space.resolution)
        {
            return Ok((
                f64::INFINITY,
                PenalizedFit {
                    coef: null_space.coef.clone(),
                    reml_score: f64::NEG_INFINITY,
                    effective_degrees: (self.xtwy.len() - self.penalty_rank) as f64,
                },
            ));
        }
        let selected = self.select_log_smoothing()?;
        Ok((selected, self.fit(selected)?))
    }

    /// Select `ρ̂` with the workspace's outer engine on the analytic `V′`, `V″`.
    ///
    /// The domain is the term's own resolvability interval (#2812): the
    /// generalized eigenvalues `γ_j` of `XᵀWX` against `S` on the penalty range
    /// give `[ln(√ε·γ_min), ln(γ_max/√ε)]`, past whose faces the criterion's
    /// gradient is under its own round-off, so a railed `ρ̂` is a structural
    /// result (the coefficient is constant, or unpenalized, to working
    /// precision). The single start is the interval's midpoint. A search the
    /// engine cannot certify is an error, never a returned `λ`.
    fn select_log_smoothing(&self) -> Result<f64, String> {
        use gam_solve::estimate::{EstimationError, rho_domain};
        use gam_solve::rho_optimizer::{
            DeclaredHessianForm, Derivative, HessianValue, OuterEval, OuterProblem,
        };
        let p = self.xtwy.len();
        let gram = Array2::from_shape_fn((p, p), |(r, c)| self.xtwx[r][c]);
        let penalty = Array2::from_shape_fn((p, p), |(r, c)| self.penalty[r][c]);
        let interval = rho_domain::penalty_range_gammas_from_gram(&gram, &penalty)
            .as_deref()
            .and_then(rho_domain::resolvability_interval)
            .ok_or_else(|| {
                "coactivation REML: no penalized direction carries data curvature, so the \
                 varying coefficient's smoothing parameter is not identified"
                    .to_string()
            })?;
        let (lower, upper) = rho_domain::coordinate_domain(Some(interval), None);
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Dense)
            .with_bounds(Array1::from_vec(vec![lower]), Array1::from_vec(vec![upper]))
            .with_initial_rho(Array1::from_vec(vec![0.5 * (lower + upper)]))
            .with_seed_config(gam_solve::seeding::SeedConfig {
                max_seeds: 1,
                seed_budget: 1,
                ..Default::default()
            });
        // A trial ρ whose penalized system cannot be factored or whose criterion
        // is unbounded is a property of that trial, so the search retreats from it.
        let refuse = |reason: String| EstimationError::TrialPointRefused { reason };
        let mut objective = problem.build_objective(
            (),
            |_: &mut (), rho: &Array1<f64>| {
                self.solve(rho[0]).map(|solve| solve.value).map_err(refuse)
            },
            |_: &mut (), rho: &Array1<f64>| {
                let (cost, gradient, hessian) = self.jet(rho[0]).map_err(refuse)?;
                Ok(OuterEval {
                    cost,
                    gradient: Array1::from_vec(vec![gradient]),
                    hessian: HessianValue::Dense(Array2::from_elem((1, 1), hessian)),
                    inner_beta_hint: None,
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<gam_problem::EfsEval, EstimationError>>,
        );
        let result = problem
            .run(&mut objective, "coactivation varying-coefficient REML")
            .map_err(|error| format!("coactivation REML: smoothing selection failed: {error}"))?;
        if !result.converged() {
            return Err(format!(
                "coactivation REML: the outer search stopped at log λ = {} without a \
                 convergence certificate",
                result.rho[0]
            ));
        }
        Ok(result.rho[0])
    }
}

/// Fit the penalty null space `N` (`p × k`) from the sufficient statistics.
///
/// `P_null = yᵀWy − gᵀĉ`, with `G = NᵀXᵀWXN`, `g = NᵀXᵀWy` and `ĉ = G⁻¹g`, is a
/// cancelling difference. To first order, rounding `δg`, `δG` in the accumulated
/// statistics moves it by `−2ĉᵀδg + ĉᵀδGĉ`. Under the `γ_m = m·ε/(1 − m·ε)` model
/// each accumulation is off by at most `γ_m` times the absolute sum of its
/// elementary terms. The resolution is therefore
/// `γ_m·(yᵀWy + 2|ĉ|ᵀ|g| + |ĉ|ᵀ|G||ĉ|)`, where `|g|` and `|G|` are those absolute
/// term sums. `m` adds up the accumulation depths and the solve's backward
/// errors: `n` rows, `p²` products per entry of `G` and `p` per entry of `g`,
/// `γ_{k+1}` for the `k×k` Cholesky factor, `γ_k` for each of its two triangular
/// solves (Higham, *Accuracy and Stability of Numerical Algorithms*, chs. 8 and
/// 10), `k` for the fitted energy, and one subtraction. Both factors are
/// derived, and the bar scales with the response energy, so it is invariant to
/// a rescaling of `y`.
fn null_space_fit(
    xtwx: &[Vec<f64>],
    xtwx_magnitude: &[Vec<f64>],
    xtwy: &[f64],
    xtwy_magnitude: &[f64],
    ywy: f64,
    null_basis: &[Vec<f64>],
    rows: usize,
) -> Result<NullSpaceFit, String> {
    let p = xtwy.len();
    if null_basis.len() != p {
        return Err(format!(
            "coactivation REML: the null-space basis has {} rows for {p} coefficients",
            null_basis.len()
        ));
    }
    let k = null_basis.first().map_or(0, |row| row.len());
    let mut gram = vec![vec![0.0_f64; k]; k];
    let mut gram_magnitude = vec![vec![0.0_f64; k]; k];
    let mut rhs = vec![0.0_f64; k];
    let mut rhs_magnitude = vec![0.0_f64; k];
    for i in 0..k {
        for a in 0..p {
            let n_ai = null_basis[a][i];
            rhs[i] += n_ai * xtwy[a];
            rhs_magnitude[i] += n_ai.abs() * xtwy_magnitude[a];
            for j in 0..k {
                for b in 0..p {
                    let n_bj = null_basis[b][j];
                    gram[i][j] += n_ai * xtwx[a][b] * n_bj;
                    gram_magnitude[i][j] += (n_ai * n_bj).abs() * xtwx_magnitude[a][b];
                }
            }
        }
    }
    let factor = cholesky_decompose(&gram)?;
    let c = cholesky_solve(&factor, &rhs);
    let penalized_rss = ywy - dot(&rhs, &c);
    let mut magnitude = ywy;
    for i in 0..k {
        magnitude += 2.0 * c[i].abs() * rhs_magnitude[i];
        for j in 0..k {
            magnitude += c[i].abs() * gram_magnitude[i][j] * c[j].abs();
        }
    }
    let operations = rows + p * p + p + (k + 1) + 2 * k + k + 1;
    let n_eps = operations as f64 * f64::EPSILON;
    let resolution = if n_eps < 1.0 {
        n_eps / (1.0 - n_eps) * magnitude
    } else {
        f64::INFINITY
    };
    let coef = null_basis.iter().map(|row| dot(row, &c)).collect();
    Ok(NullSpaceFit {
        coef,
        penalized_rss,
        resolution,
    })
}

/// A basis of the null space of the penalty `diag(0, D_mᵀD_m)` on the design
/// `[1, gate·B(x)]`, as `(width + 1) × (1 + m)` columns. The columns are the
/// intercept and the coefficient sequences `k^j` (`j < m`) that every order-`m`
/// difference annihilates.
fn penalty_null_basis(width: usize, order: usize) -> Vec<Vec<f64>> {
    let mut basis = vec![vec![0.0_f64; 1 + order]; width + 1];
    basis[0][0] = 1.0;
    for k in 0..width {
        let mut power = 1.0_f64;
        for j in 0..order {
            basis[k + 1][j + 1] = power;
            power *= k as f64;
        }
    }
    basis
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
            return Err(format!(
                "residualize_gate: row {row} has non-finite gate {y}"
            ));
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

    /// The closed-form REML jet against central differences of its own value
    /// and gradient, and the engine's selection a local minimum of the value.
    #[test]
    fn varying_coefficient_reml_jet_matches_central_differences_2902() {
        let n = 300usize;
        let mut state = 0x2902_2902_u64;
        let mut unit = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut design = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for _ in 0..n {
            let x = 2.0 * unit() - 1.0;
            let gate = 0.5 + unit();
            let noise = unit() - 0.5;
            design.push(vec![1.0, gate, gate * x, gate * x * x]);
            y.push((1.0 + 0.5 * x) * gate + 0.3 * noise);
        }
        let weights = vec![1.0_f64; n];
        let beta_penalty = difference_penalty(3, 1).expect("first-difference penalty");
        let mut penalty = vec![vec![0.0_f64; 4]; 4];
        for r in 0..3 {
            for c in 0..3 {
                penalty[r + 1][c + 1] = beta_penalty[r][c];
            }
        }
        let problem = GaussianRemlProblem::new(
            &design,
            &y,
            &weights,
            &penalty,
            2,
            &penalty_null_basis(3, 1),
        )
        .expect("REML problem");
        let step = 1.0e-5;
        for rho in [-3.0_f64, 0.0, 4.0] {
            let (value, gradient, hessian) = problem.jet(rho).expect("jet");
            assert_eq!(
                value.to_bits(),
                problem.solve(rho).expect("solve").value.to_bits()
            );
            let (value_up, gradient_up, _) = problem.jet(rho + step).expect("jet up");
            let (value_down, gradient_down, _) = problem.jet(rho - step).expect("jet down");
            let gradient_fd = (value_up - value_down) / (2.0 * step);
            let hessian_fd = (gradient_up - gradient_down) / (2.0 * step);
            assert!(
                (gradient - gradient_fd).abs() <= 1.0e-5 * (1.0 + gradient.abs()),
                "rho={rho}: V' {gradient} vs central difference {gradient_fd}"
            );
            assert!(
                (hessian - hessian_fd).abs() <= 1.0e-5 * (1.0 + hessian.abs()),
                "rho={rho}: V'' {hessian} vs central difference {hessian_fd}"
            );
        }
        let selected = problem
            .select_log_smoothing()
            .expect("certified selection");
        let selected_value = problem.solve(selected).expect("solve at selection").value;
        for offset in [-1.0e-2_f64, 1.0e-2] {
            let neighbour = problem.solve(selected + offset).expect("solve at neighbour");
            assert!(
                selected_value <= neighbour.value,
                "selected log λ {selected} (V={selected_value}) is beaten at offset {offset} \
                 (V={})",
                neighbour.value
            );
        }
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
        let influence =
            coupling_influence_values(&gate_i, &gate_j, &rows, &weights).expect("influence values");
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
            influence.rho, max_closed_form_diff, weighted_mean, max_jackknife_diff, jackknife_rms
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
            let closed_form = a * (b - influence.conditional_probability) / influence.active_mass_i;
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
                "conditional_probability_excluding: zero active mass in retained sample"
                    .to_string(),
            );
        }
        Ok(numer / denom)
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
