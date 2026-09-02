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
