//! Context-conditional coactivation diagnostics for structure search.
//!
//! Pooled coactivation is a marginal statistic: it cannot distinguish an
//! invariant binding from a pair of atoms that only co-occur in one partition of
//! the corpus. This module estimates the directional conditional
//! `P(j active | i active, context)` on the same designed row reservoirs used by
//! the certifiers, carrying Horvitz-Thompson row weights through every count.
//!
//! The intended gate is the dispersion across contexts, not the pooled
//! correlation. Low directional variance reads as structural binding; a sign
//! flip in the centered association across contexts is reported as direct
//! evidence that the pooled statistic was a partition artifact.

use gam_solve::row_sampling_measure::{DesignedRowSample, MeasureProvenance, RowSamplingMeasure};
use ndarray::ArrayView2;
use std::collections::BTreeMap;

/// One context's weighted directional conditional estimates.
#[derive(Clone, Debug)]
pub struct ContextConditional {
    pub context: usize,
    pub rows: usize,
    pub mass: f64,
    pub denom_i: f64,
    pub denom_j: f64,
    pub pi_j_given_i: Option<f64>,
    pub pi_i_given_j: Option<f64>,
    pub centered_association: f64,
}

/// Across-context conditionality report for one ordered pair and its reverse.
#[derive(Clone, Debug)]
pub struct CoactivationConditionality {
    pub contexts: Vec<ContextConditional>,
    pub var_j_given_i: f64,
    pub var_i_given_j: f64,
    pub sampling_var_j_given_i: f64,
    pub sampling_var_i_given_j: f64,
    pub sign_instability: bool,
    pub stable_for_structure_search: bool,
    pub valid_j_given_i_contexts: usize,
    pub valid_i_given_j_contexts: usize,
}

/// Residual-gate materialization after the shared chart has been projected out.
#[derive(Clone, Debug)]
pub struct ResidualGateActivities {
    pub residual_i: Vec<f64>,
    pub residual_j: Vec<f64>,
    pub active_i: Vec<bool>,
    pub active_j: Vec<bool>,
}

/// Evaluate conditionality on a deterministic designed subsample from a
/// [`RowSamplingMeasure`]. The returned estimate uses exactly the sample rows
/// and `1/pi` honesty weights produced by the measure.
pub fn estimate_from_measure(
    active_i: &[bool],
    active_j: &[bool],
    context_labels: &[usize],
    measure: &RowSamplingMeasure,
    budget: usize,
    seed: u64,
) -> Result<CoactivationConditionality, String> {
    if measure.n_rows() != active_i.len() {
        return Err(format!(
            "estimate_from_measure: measure has {} rows, gates have {}",
            measure.n_rows(),
            active_i.len()
        ));
    }
    let sample = measure.designed_subsample(budget, seed);
    estimate_from_designed_sample(active_i, active_j, context_labels, &sample)
}

/// Evaluate conditionality on an already-drawn designed sample.
pub fn estimate_from_designed_sample(
    active_i: &[bool],
    active_j: &[bool],
    context_labels: &[usize],
    sample: &DesignedRowSample,
) -> Result<CoactivationConditionality, String> {
    estimate_on_rows(
        active_i,
        active_j,
        context_labels,
        &sample.rows,
        &sample.likelihood_weights,
    )
}

/// Evaluate conditionality on explicit selected rows and per-row honesty
/// weights. This is the common implementation used by both synthetic tests and
/// production designed reservoirs.
pub fn estimate_on_rows(
    active_i: &[bool],
    active_j: &[bool],
    context_labels: &[usize],
    rows: &[usize],
    likelihood_weights: &[f64],
) -> Result<CoactivationConditionality, String> {
    validate_inputs(active_i, active_j, context_labels, rows, likelihood_weights)?;

    let mut accum: BTreeMap<usize, ContextAccum> = BTreeMap::new();
    for (slot, &row) in rows.iter().enumerate() {
        let w = likelihood_weights[slot];
        let entry = accum.entry(context_labels[row]).or_default();
        entry.rows += 1;
        entry.mass += w;
        let ai = active_i[row];
        let aj = active_j[row];
        if ai {
            entry.i += w;
            entry.i_sq += w * w;
        }
        if aj {
            entry.j += w;
            entry.j_sq += w * w;
        }
        if ai && aj {
            entry.ij += w;
        }
    }

    let total_mass: f64 = accum.values().map(|a| a.mass).sum();
    let mut contexts = Vec::with_capacity(accum.len());
    let mut positives = 0usize;
    let mut negatives = 0usize;

    for (&context, a) in accum.iter() {
        let pi = if a.mass > 0.0 { a.i / a.mass } else { 0.0 };
        let pj = if a.mass > 0.0 { a.j / a.mass } else { 0.0 };
        let pij = if a.mass > 0.0 { a.ij / a.mass } else { 0.0 };
        let association = pij - pi * pj;
        if association > association_sign_floor(a.mass, total_mass) {
            positives += 1;
        }
        if association < -association_sign_floor(a.mass, total_mass) {
            negatives += 1;
        }
        contexts.push(ContextConditional {
            context,
            rows: a.rows,
            mass: a.mass,
            denom_i: a.i,
            denom_j: a.j,
            pi_j_given_i: (a.i > 0.0).then_some(a.ij / a.i),
            pi_i_given_j: (a.j > 0.0).then_some(a.ij / a.j),
            centered_association: association,
        });
    }

    let stats_j_given_i = directional_stats(&contexts, accum.values(), Direction::JGivenI);
    let stats_i_given_j = directional_stats(&contexts, accum.values(), Direction::IGivenJ);
    let sign_instability = positives > 0 && negatives > 0;
    let stable_for_structure_search = stats_j_given_i.valid_contexts > 0
        && stats_i_given_j.valid_contexts > 0
        && !sign_instability
        && stats_j_given_i.observed_var <= stats_j_given_i.sampling_var
        && stats_i_given_j.observed_var <= stats_i_given_j.sampling_var;

    Ok(CoactivationConditionality {
        contexts,
        var_j_given_i: stats_j_given_i.observed_var,
        var_i_given_j: stats_i_given_j.observed_var,
        sampling_var_j_given_i: stats_j_given_i.sampling_var,
        sampling_var_i_given_j: stats_i_given_j.sampling_var,
        sign_instability,
        stable_for_structure_search,
        valid_j_given_i_contexts: stats_j_given_i.valid_contexts,
        valid_i_given_j_contexts: stats_i_given_j.valid_contexts,
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

/// Deterministic residual-cluster labels from explicit centroids. This accepts
/// centroids instead of learning them so structure search can use the same
/// chart/residual summaries that produced its strata.
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

#[derive(Default)]
struct ContextAccum {
    rows: usize,
    mass: f64,
    i: f64,
    i_sq: f64,
    j: f64,
    j_sq: f64,
    ij: f64,
}

#[derive(Clone, Copy)]
enum Direction {
    JGivenI,
    IGivenJ,
}

struct DirectionalStats {
    observed_var: f64,
    sampling_var: f64,
    valid_contexts: usize,
}

fn validate_inputs(
    active_i: &[bool],
    active_j: &[bool],
    context_labels: &[usize],
    rows: &[usize],
    likelihood_weights: &[f64],
) -> Result<(), String> {
    let n = active_i.len();
    if active_j.len() != n || context_labels.len() != n {
        return Err(format!(
            "coactivation conditionality: lengths differ active_i={} active_j={} labels={}",
            active_i.len(),
            active_j.len(),
            context_labels.len()
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
    }
    Ok(())
}

fn directional_stats<'a>(
    contexts: &[ContextConditional],
    accum: impl Iterator<Item = &'a ContextAccum>,
    direction: Direction,
) -> DirectionalStats {
    let accums: Vec<&ContextAccum> = accum.collect();
    let total_context_mass: f64 = contexts.iter().map(|c| c.mass).sum();
    let mut valid_mass = 0.0;
    let mut mean = 0.0;
    let mut valid_contexts = 0usize;
    for context in contexts {
        let value = match direction {
            Direction::JGivenI => context.pi_j_given_i,
            Direction::IGivenJ => context.pi_i_given_j,
        };
        if let Some(v) = value {
            valid_contexts += 1;
            valid_mass += context.mass;
            mean += context.mass * v;
        }
    }
    if !(valid_mass > 0.0) {
        return DirectionalStats {
            observed_var: f64::INFINITY,
            sampling_var: 0.0,
            valid_contexts,
        };
    }
    mean /= valid_mass;

    let mut observed_var = 0.0;
    let mut sampling_var = 0.0;
    for (idx, context) in contexts.iter().enumerate() {
        let value = match direction {
            Direction::JGivenI => context.pi_j_given_i,
            Direction::IGivenJ => context.pi_i_given_j,
        };
        if let Some(v) = value {
            let context_prob = if total_context_mass > 0.0 {
                context.mass / total_context_mass
            } else {
                0.0
            };
            let d = v - mean;
            observed_var += context_prob * d * d;
            let a = accums[idx];
            let (denom, denom_sq) = match direction {
                Direction::JGivenI => (a.i, a.i_sq),
                Direction::IGivenJ => (a.j, a.j_sq),
            };
            let n_eff = if denom_sq > 0.0 {
                denom * denom / denom_sq
            } else {
                0.0
            };
            if n_eff > 0.0 {
                sampling_var += context_prob * v * (1.0 - v) / n_eff;
            }
        }
    }
    DirectionalStats {
        observed_var,
        sampling_var,
        valid_contexts,
    }
}

fn association_sign_floor(context_mass: f64, total_mass: f64) -> f64 {
    if total_mass > 0.0 {
        f64::EPSILON * (1.0 + total_mass / context_mass.max(f64::MIN_POSITIVE))
    } else {
        f64::EPSILON
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn invariant_cofire_has_low_conditional_variance() {
        let n = 12usize;
        let labels: Vec<usize> = (0..n).map(|row| row % 3).collect();
        let active_i = vec![true; n];
        let active_j = vec![true; n];
        let sample = full_pass_rows(n);
        let report =
            estimate_from_designed_sample(&active_i, &active_j, &labels, &sample).unwrap();
        assert_eq!(report.var_j_given_i, 0.0);
        assert_eq!(report.var_i_given_j, 0.0);
        assert!(!report.sign_instability);
        assert!(report.stable_for_structure_search);
    }

    #[test]
    fn context_specific_cofire_and_anticorrelation_is_unstable() {
        let n = 16usize;
        let mut labels = vec![0usize; n];
        for label in labels.iter_mut().skip(n / 2) {
            *label = 1;
        }
        let mut active_i = vec![false; n];
        let mut active_j = vec![false; n];
        for row in 0..(n / 2) {
            let on = row % 2 == 0;
            active_i[row] = on;
            active_j[row] = on;
        }
        for row in (n / 2)..n {
            active_i[row] = row % 2 == 0;
            active_j[row] = !active_i[row];
        }
        let sample = full_pass_rows(n);
        let report =
            estimate_from_designed_sample(&active_i, &active_j, &labels, &sample).unwrap();
        assert!(report.var_j_given_i > 0.20);
        assert!(report.var_i_given_j > 0.20);
        assert!(report.sign_instability);
        assert!(!report.stable_for_structure_search);
    }

    #[test]
    fn residual_gate_denominator_removes_same_chart_anchor_binding() {
        let n = 12usize;
        let labels: Vec<usize> = (0..n).map(|row| row % 2).collect();
        let chart_gate: Vec<f64> = (0..n)
            .map(|row| if row % 3 == 0 { 1.0 } else { 0.0 })
            .collect();
        let raw_active: Vec<bool> = chart_gate.iter().map(|&g| g > 0.5).collect();
        let sample = full_pass_rows(n);
        let raw =
            estimate_from_designed_sample(&raw_active, &raw_active, &labels, &sample).unwrap();

        let chart = Array2::from_shape_vec((n, 1), chart_gate.clone()).unwrap();
        let residual = residual_gate_activities(
            &chart_gate,
            &chart_gate,
            Some(chart.view()),
            &sample.likelihood_weights,
            0.0,
        )
        .unwrap();
        let adjusted = estimate_from_designed_sample(
            &residual.active_i,
            &residual.active_j,
            &labels,
            &sample,
        )
        .unwrap();
        assert!(raw.stable_for_structure_search);
        assert_eq!(adjusted.valid_j_given_i_contexts, 0);
        assert!(!adjusted.stable_for_structure_search);
        assert!(residual.active_i.iter().all(|&active| !active));
        assert!(residual.active_j.iter().all(|&active| !active));
    }
}
