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

/// Stable physical-penalty label used by every custom-family layout producer.
/// Explicit labels are preserved; unlabeled slots receive a collision-visible
/// name derived from their authoritative flattened block position.
pub(crate) fn resolved_physical_penalty_label(
    penalty: &PenaltyMatrix,
    block_idx: usize,
    penalty_idx: usize,
) -> String {
    penalty
        .precision_label()
        .map(str::to_owned)
        .unwrap_or_else(|| format!("__block_{block_idx}_penalty_{penalty_idx}"))
}

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

    /// Whether an optimizer-coordinate rho must be expanded before it can be
    /// stored in a warm start consumed by the physical per-block solver.
    ///
    /// Cardinality alone is not sufficient: a fixed physical slot and a new
    /// joint coordinate can cancel in the counts while the physical map still
    /// contains a hole. The map is direct only when every physical slot maps
    /// to the optimizer coordinate at the identical index and the vectors have
    /// equal lengths.
    pub(crate) fn physical_rho_requires_remap(&self) -> bool {
        self.initial_rho.len() != self.physical_to_outer.len()
            || self
                .physical_to_outer
                .iter()
                .enumerate()
                .any(|(physical_idx, outer_idx)| *outer_idx != Some(physical_idx))
    }

    /// Whether rho can be passed directly to the legacy physical-coordinate
    /// EFS evaluator. That evaluator sees only per-block penalties, so even an
    /// identity rho map is ineligible when a joint penalty shares an existing
    /// label and therefore adds no optimizer coordinate of its own.
    pub(crate) fn supports_direct_physical_efs(&self) -> bool {
        self.joint_specs.is_empty() && !self.physical_rho_requires_remap()
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
            let label = resolved_physical_penalty_label(
                &spec.penalties[penalty_idx],
                block_idx,
                penalty_idx,
            );
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn one_block(
        penalties: Vec<PenaltyMatrix>,
        initial_log_lambdas: Vec<f64>,
    ) -> ParameterBlockSpec {
        let penalty_count = penalties.len();
        ParameterBlockSpec {
            name: "layout".to_string(),
            design: crate::DesignMatrix::from(Array2::<f64>::eye(2)),
            offset: Array1::zeros(2),
            penalties,
            nullspace_dims: vec![0; penalty_count],
            initial_log_lambdas: Array1::from_vec(initial_log_lambdas),
            initial_beta: None,
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        }
    }

    fn joint_penalty(label: &str, initial_log_lambda: f64) -> gam_problem::JointPenaltySpec {
        gam_problem::JointPenaltySpec {
            label: Some(label.to_string()),
            matrix: Array2::<f64>::eye(2),
            initial_log_lambda,
            nullspace_dim: 0,
        }
    }

    #[test]
    fn fixed_and_joint_count_cancellation_requires_physical_rho_remap_2315() {
        let specs = vec![one_block(
            vec![
                PenaltyMatrix::Dense(Array2::<f64>::eye(2)).with_fixed_log_lambda(2.0),
                PenaltyMatrix::Dense(Array2::<f64>::eye(2)),
            ],
            vec![0.5, -1.0],
        )];
        let layout = penalty_label_layout_with_joint(
            &specs,
            vec![2],
            vec![joint_penalty("new_joint", -2.0)],
        )
        .expect("fixed plus joint layout must be valid");

        // One fixed physical slot removes an outer coordinate while the new
        // joint penalty adds one, so the old cardinality predicate reported a
        // false identity despite the hole in the physical map.
        assert_eq!(layout.initial_rho.len(), layout.physical_count());
        assert_eq!(layout.physical_to_outer, vec![None, Some(0)]);
        assert_eq!(layout.joint_to_outer, vec![1]);
        assert!(layout.physical_rho_requires_remap());
        assert!(!layout.supports_direct_physical_efs());
    }

    #[test]
    fn joint_reusing_physical_label_is_not_direct_efs_compatible_2315() {
        let specs = vec![one_block(
            vec![PenaltyMatrix::Dense(Array2::<f64>::eye(2)).with_precision_label("shared")],
            vec![-1.0],
        )];
        let direct = penalty_label_layout_with_joint(&specs, vec![1], Vec::new())
            .expect("ordinary one-to-one physical layout must be valid");
        assert!(!direct.physical_rho_requires_remap());
        assert!(direct.supports_direct_physical_efs());

        let with_joint =
            penalty_label_layout_with_joint(&specs, vec![1], vec![joint_penalty("shared", -1.0)])
                .expect("joint penalty may share an existing physical precision label");

        // The rho representation remains an exact identity, but the raw EFS
        // evaluator would omit the joint penalty bundle entirely.
        assert_eq!(with_joint.initial_rho.len(), with_joint.physical_count());
        assert_eq!(with_joint.physical_to_outer, vec![Some(0)]);
        assert_eq!(with_joint.joint_to_outer, vec![0]);
        assert!(!with_joint.physical_rho_requires_remap());
        assert!(!with_joint.supports_direct_physical_efs());
    }
}
