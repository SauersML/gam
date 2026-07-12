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
    /// Full-width cross-block joint penalties (gam#1587: the reference-symmetric
    /// `M⊗S_t` multinomial penalty). Empty for every family whose
    /// `joint_penalty_specs()` returns no specs (all paths below are then
    /// byte-identical to the per-block-only layout). When non-empty, each joint
    /// spec is tied to an outer ρ coordinate by its `label` (sharing the
    /// coordinate with any per-block penalty carrying the same label), recorded
    /// in `joint_to_outer` parallel to this vector.
    pub(crate) joint_specs: Vec<gam_problem::JointPenaltySpec>,
    /// Outer ρ coordinate each joint spec maps to, parallel to `joint_specs`.
    pub(crate) joint_to_outer: Vec<usize>,
}

impl PenaltyLabelLayout {
    /// Number of per-block physical penalty slots (the per-block contribution to
    /// the evaluator's positional penalty list / gradient). The joint coords are
    /// appended AFTER these in the assembly, so the full evaluator-gradient
    /// length is `physical_count() + joint_specs.len()`.
    pub(crate) fn physical_count(&self) -> usize {
        self.physical_to_outer.len()
    }

    pub(crate) fn has_tied_coordinates(&self) -> bool {
        self.initial_rho.len() != self.physical_to_outer.len()
    }

    /// Joint-penalty `log λ` values for the current outer ρ, parallel to
    /// `joint_specs` (each pulled from its tied outer coordinate). Empty when no
    /// joint penalties are present.
    pub(crate) fn joint_log_lambdas(&self, rho: &Array1<f64>) -> Vec<f64> {
        self.joint_to_outer.iter().map(|&o| rho[o]).collect()
    }
}

pub(crate) fn penalty_label_layout_with_joint(
    specs: &[ParameterBlockSpec],
    penalty_counts: Vec<usize>,
    joint_specs: Vec<gam_problem::JointPenaltySpec>,
) -> Result<PenaltyLabelLayout, String> {
    let mut label_to_outer = BTreeMap::<String, usize>::new();
    let mut physical_to_outer = Vec::<Option<usize>>::new();
    let mut fixed_log_lambdas = Vec::<Option<f64>>::new();
    let mut initial = Vec::<f64>::new();

    for (block_idx, spec) in specs.iter().enumerate() {
        for penalty_idx in 0..spec.penalties.len() {
            if let Some(fixed) = spec.penalties[penalty_idx].fixed_log_lambda() {
                if let Err(error) = gam_problem::validate_log_strength(fixed) {
                    return Err(CustomFamilyError::ConstraintViolation {
                        reason: format!(
                            "block {block_idx} penalty {penalty_idx} fixed log-precision: {error}"
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
            gam_problem::validate_log_strength(rho0).map_err(|error| {
                CustomFamilyError::ConstraintViolation {
                    reason: format!(
                        "block {block_idx} penalty {penalty_idx} initial log-precision: {error}"
                    ),
                }
            })?;
            let outer = if let Some(&outer) = label_to_outer.get(&label) {
                let first = initial[outer];
                if (first - rho0).abs() > 1e-10 {
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

    // gam#1587: tie each full-width joint penalty to an outer ρ coordinate by
    // its label. A joint spec carrying a label already seen on a per-block
    // penalty shares that coordinate (the multinomial centered penalty shares
    // `multinomial_term_{t}` with — now empty — per-block slots, so one shared
    // λ_t smooths every class). A new label creates a fresh coordinate.
    let mut joint_to_outer = Vec::<usize>::with_capacity(joint_specs.len());
    for (joint_idx, spec) in joint_specs.iter().enumerate() {
        spec.validate()
            .map_err(|error| CustomFamilyError::ConstraintViolation {
                reason: format!("joint penalty {joint_idx}: {error}"),
            })?;
        let label = spec
            .label
            .clone()
            .unwrap_or_else(|| format!("__joint_penalty_{joint_idx}"));
        let rho0 = spec.initial_log_lambda;
        let outer = if let Some(&outer) = label_to_outer.get(&label) {
            let first = initial[outer];
            if first.is_finite() && rho0.is_finite() && (first - rho0).abs() > 1e-10 {
                return Err(CustomFamilyError::ConstraintViolation {
                    reason: format!(
                        "joint penalty label '{label}' has inconsistent initial log-precisions: {first} and {rho0}"
                    ),
                }
                .into());
            }
            outer
        } else {
            let outer = initial.len();
            label_to_outer.insert(label, outer);
            initial.push(rho0);
            outer
        };
        joint_to_outer.push(outer);
    }

    Ok(PenaltyLabelLayout {
        penalty_counts,
        physical_to_outer,
        fixed_log_lambdas,
        initial_rho: Array1::from_vec(initial),
        joint_specs,
        joint_to_outer,
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
    // The evaluator gradient is the per-block physical coords followed by the
    // appended joint coords (gam#1587). When no joint penalties are present this
    // is exactly the legacy per-block length.
    let expected = layout.physical_count() + layout.joint_specs.len();
    if gradient.len() != expected {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: format!(
                "physical gradient length mismatch: got {}, expected {} (per-block {} + joint {})",
                gradient.len(),
                expected,
                layout.physical_count(),
                layout.joint_specs.len(),
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
    let joint_base = layout.physical_count();
    for (joint_idx, &outer) in layout.joint_to_outer.iter().enumerate() {
        out[outer] += gradient[joint_base + joint_idx];
    }
    Ok(out)
}
