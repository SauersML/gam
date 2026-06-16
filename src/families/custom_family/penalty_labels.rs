//! Penalty-label layout and labeled log-λ (de)aggregation helpers for the
//! custom-family outer objective.
//!
//! These functions translate between the *physical* per-block penalty
//! coordinates and the *outer* (label-tied) ρ coordinates: penalties sharing a
//! `precision_label` collapse to one outer coordinate, fixed penalties drop out
//! of the outer vector entirely, and gradients aggregate back the other way.
//! Extracted from `outer_objective` so that module stays under the tracked-file
//! line limit; the items are re-exported through the parent module, so existing
//! `use super::*` callers are unaffected.

use super::*;
use ndarray::Array1;
use std::collections::BTreeMap;

pub(crate) fn flatten_log_lambdas(specs: &[ParameterBlockSpec]) -> Array1<f64> {
    let total = specs
        .iter()
        .map(|s| s.initial_log_lambdas.len())
        .sum::<usize>();
    let mut out = Array1::<f64>::zeros(total);
    let mut at = 0usize;
    for spec in specs {
        let len = spec.initial_log_lambdas.len();
        if len > 0 {
            out.slice_mut(ndarray::s![at..at + len])
                .assign(&spec.initial_log_lambdas);
        }
        at += len;
    }
    out
}

#[derive(Clone, Debug)]
pub(crate) struct PenaltyLabelLayout {
    pub(crate) penalty_counts: Vec<usize>,
    pub(crate) physical_to_outer: Vec<Option<usize>>,
    pub(crate) fixed_log_lambdas: Vec<Option<f64>>,
    pub(crate) initial_rho: Array1<f64>,
}

impl PenaltyLabelLayout {
    pub(crate) fn physical_count(&self) -> usize {
        self.physical_to_outer.len()
    }

    pub(crate) fn has_tied_coordinates(&self) -> bool {
        self.initial_rho.len() != self.physical_to_outer.len()
    }
}

pub(crate) fn penalty_label_layout(
    specs: &[ParameterBlockSpec],
    penalty_counts: Vec<usize>,
) -> Result<PenaltyLabelLayout, String> {
    let mut label_to_outer = BTreeMap::<String, usize>::new();
    let mut physical_to_outer = Vec::<Option<usize>>::new();
    let mut fixed_log_lambdas = Vec::<Option<f64>>::new();
    let mut initial = Vec::<f64>::new();

    for (block_idx, spec) in specs.iter().enumerate() {
        for penalty_idx in 0..spec.penalties.len() {
            if let Some(fixed) = spec.penalties[penalty_idx].fixed_log_lambda() {
                if !fixed.is_finite() {
                    return Err(CustomFamilyError::ConstraintViolation {
                        reason: format!(
                            "block {block_idx} penalty {penalty_idx} fixed log-precision is non-finite: {fixed}"
                        ),
                    }
                    .into());
                }
                physical_to_outer.push(None);
                fixed_log_lambdas.push(Some(fixed));
                continue;
            }
            let label = spec.penalties[penalty_idx]
                .precision_label()
                .map(str::to_owned)
                .unwrap_or_else(|| format!("__block_{block_idx}_penalty_{penalty_idx}"));
            let rho0 = spec.initial_log_lambdas[penalty_idx];
            let outer = if let Some(&outer) = label_to_outer.get(&label) {
                let first = initial[outer];
                if first.is_finite() && rho0.is_finite() && (first - rho0).abs() > 1e-10 {
                    return Err(CustomFamilyError::ConstraintViolation { reason: format!(
                        "precision label '{label}' has inconsistent initial log-precisions: {first} and {rho0}"
                    ) }.into());
                }
                outer
            } else {
                let outer = initial.len();
                label_to_outer.insert(label, outer);
                initial.push(rho0);
                outer
            };
            physical_to_outer.push(Some(outer));
            fixed_log_lambdas.push(None);
        }
    }

    Ok(PenaltyLabelLayout {
        penalty_counts,
        physical_to_outer,
        fixed_log_lambdas,
        initial_rho: Array1::from_vec(initial),
    })
}

pub(crate) fn expand_labeled_log_lambdas(
    rho: &Array1<f64>,
    layout: &PenaltyLabelLayout,
) -> Result<Array1<f64>, String> {
    if rho.len() != layout.initial_rho.len() {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: format!(
                "log-lambda label coordinate mismatch: got {}, expected {}",
                rho.len(),
                layout.initial_rho.len()
            ),
        }
        .into());
    }
    let mut expanded = Array1::<f64>::zeros(layout.physical_count());
    for (physical, outer) in layout.physical_to_outer.iter().enumerate() {
        expanded[physical] = match *outer {
            Some(outer) => rho[outer],
            None => layout.fixed_log_lambdas[physical].ok_or_else(|| {
                CustomFamilyError::ConstraintViolation {
                    reason: format!(
                        "fixed penalty layout missing value at physical slot {physical}"
                    ),
                }
                .to_string()
            })?,
        };
    }
    Ok(expanded)
}

pub(crate) fn split_labeled_log_lambdas(
    rho: &Array1<f64>,
    layout: &PenaltyLabelLayout,
) -> Result<Vec<Array1<f64>>, String> {
    let expanded = expand_labeled_log_lambdas(rho, layout)?;
    split_log_lambdas(&expanded, &layout.penalty_counts)
}

pub(crate) fn aggregate_labeled_gradient(
    gradient: &Array1<f64>,
    layout: &PenaltyLabelLayout,
) -> Result<Array1<f64>, String> {
    if gradient.len() != layout.physical_count() {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: format!(
                "physical gradient length mismatch: got {}, expected {}",
                gradient.len(),
                layout.physical_count()
            ),
        }
        .into());
    }
    let mut out = Array1::<f64>::zeros(layout.initial_rho.len());
    for (physical, outer) in layout.physical_to_outer.iter().enumerate() {
        if let Some(outer) = *outer {
            out[outer] += gradient[physical];
        }
    }
    Ok(out)
}
