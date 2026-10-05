//! Native local targets evaluated on the inputs reached by a candidate.
//! This is training supervision, never an execution-time input to the explanation.
use crate::{artifact::Artifact, operator_program::OperatorProgram};
use crate::operator_program::FamilyInputs;
use ndarray::{Array2, Axis};
use serde::Serialize;
use std::collections::BTreeMap;

pub struct Panel {
    /// Ordered exactly as `arguments`, including argument permutations.
    pub inputs: Vec<Array2<f64>>,
    /// Native outputs concatenated in the caller's declared order.
    pub targets: Array2<f64>,
    pub report: Report,
}

#[derive(Clone, Debug, Serialize)]
pub struct Report {
    pub rows: usize,
    pub arguments: Vec<usize>,
    pub native_outputs: Vec<usize>,
    pub output_widths: Vec<usize>,
    pub planned_trace_and_panel_bytes: usize,
    pub scope: &'static str,
}

/// Capture a complete sequence family without changing its attention layout.
/// The native teacher is clamped to the candidate's incoming values at every
/// declared boundary, then evaluated downstream. Targets are NOT the clean
/// trajectory outputs. Callers must supply a complete validated region boundary.
///
/// The budget covers retained prefix traces and copied panels, not existing
/// weights, input storage, allocator overhead, or evaluator scratch buffers.
pub fn capture(
    native: &OperatorProgram,
    candidate: &Artifact,
    arguments: &[usize],
    outputs: &[usize],
    family: &FamilyInputs,
    numeric_bytes: usize,
) -> Result<Panel, String> {
    if arguments.is_empty() || outputs.is_empty() || family.rows == 0 {
        return Err("local supervision needs nonempty boundaries and rows".into());
    }
    if native.declarations.parameters != 0 || candidate.program.declarations.parameters != 0 {
        return Err("local supervision requires controls lowered to explicit inputs".into());
    }
    let nt = native.interfaces().map_err(|e| e.to_string())?;
    let ct = candidate.program.interfaces().map_err(|e| e.to_string())?;
    let mut reads = BTreeMap::new();
    for &n in arguments {
        let c = candidate.place(n).ok_or("candidate erased a required native input")?;
        let nw = nt.get(n).ok_or("native input absent")?.width();
        let cw = ct.get(c).ok_or("candidate input absent")?.width();
        if nw != cw || reads.insert(n, c).is_some() {
            return Err("local supervision inputs must be unique and width-preserving".into());
        }
    }
    let output_widths = outputs.iter().map(|&n| nt.get(n)
        .map(|t| t.width()).ok_or("native output absent".to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    let last_native = *outputs.iter().max().ok_or("native outputs absent")?;
    if arguments.iter().any(|n| *n >= last_native) {
        return Err("local supervision inputs must precede the last output".into());
    }
    let last_candidate = *reads.values().max().ok_or("candidate inputs absent")?;
    let trace_widths = nt[..=last_native].iter().map(|t| t.width())
        .chain(ct[..=last_candidate].iter().map(|t| t.width()));
    // Two boundary copies (patch and returned input); two target copies during concatenate.
    let mut widths = trace_widths.chain(arguments.iter().flat_map(|n| [nt[*n].width(); 2]))
        .chain(output_widths.iter().flat_map(|w| [*w; 2]));
    let elements_per_row = widths.try_fold(0usize, |a, b| a.checked_add(b)
        .ok_or("local supervision size overflow"))?;
    let planned = elements_per_row.checked_mul(family.rows).and_then(|v| v.checked_mul(8))
        .ok_or("local supervision size overflow")?;
    if planned > numeric_bytes {
        return Err(format!("local supervision trace/panel plan {planned} exceeds {numeric_bytes}"));
    }
    let prefix = |p: &OperatorProgram, last: usize| {
        let mut p = p.clone();
        p.nodes.truncate(last + 1);
        p.output = last;
        p
    };
    let candidate_trace = prefix(&candidate.program, last_candidate)
        .execute(family, false).map_err(|e| e.to_string())?;
    let patches = reads.iter().map(|(&n, &c)| (n, candidate_trace.values[c].clone()))
        .collect::<BTreeMap<_, _>>();
    if patches.values().any(|v| v.iter().any(|v| !v.is_finite())) {
        return Err("nonfinite candidate local inputs".into());
    }
    let native_trace = prefix(native, last_native).execute_edited(family, |n, value, _| {
        if let Some(patch) = patches.get(&n) { value.assign(patch); }
        Ok(())
    }).map_err(|e| e.to_string())?;
    let targets = ndarray::concatenate(Axis(1), &outputs.iter()
        .map(|n| native_trace.values[*n].view()).collect::<Vec<_>>())
        .map_err(|e| e.to_string())?;
    if targets.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite same-input native local targets".into());
    }
    Ok(Panel {
        inputs: arguments.iter().map(|n| patches[n].clone()).collect(),
        targets,
        report: Report {
            rows: family.rows, arguments: arguments.to_vec(), native_outputs: outputs.to_vec(),
            output_widths, planned_trace_and_panel_bytes: planned,
            scope: "Native local outputs on candidate-reached incoming states; training-only. Complete sequence layout retained. Budget excludes weights, evaluator scratch, inputs, and allocator overhead.",
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator_program::{Declarations, Node, Slot, SlotValues};
    use ndarray::array;

    fn fixture() -> (OperatorProgram, FamilyInputs) {
        (OperatorProgram {
            declarations: Declarations { domains: vec![], slots: vec![Slot::Raw { width: 1 }, Slot::Raw { width: 1 }], parameters: 0 },
            bases: vec![], rules: vec![], operators: vec![],
            nodes: vec![Node::Raw { slot: 0 }, Node::Raw { slot: 1 },
                Node::Hadamard { left: 0, right: 1 }], output: 2,
        }, FamilyInputs { rows: 3, slots: vec![SlotValues::Raw(array![[1.], [2.], [3.]]),
            SlotValues::Raw(array![[4.], [5.], [6.]])], layout: None })
    }

    #[test]
    fn native_targets_follow_candidate_inputs_not_clean_trajectory() {
        let (native, family) = fixture();
        let mut candidate = Artifact::native(&native).unwrap();
        candidate.program.nodes[0] = Node::Raw { slot: 1 };
        let panel = capture(&native, &candidate, &[0, 1], &[2], &family, 1 << 20).unwrap();
        assert_eq!(panel.targets, array![[16.], [25.], [36.]]);
        assert_ne!(panel.targets, native.execute(&family, false).unwrap().values[2]);
        assert_eq!(panel.inputs[0], array![[4.], [5.], [6.]]);
    }

    #[test]
    fn permutation_changes_local_argument_order_not_teacher_wiring() {
        let (native, family) = fixture();
        let candidate = Artifact::native(&native).unwrap();
        let panel = capture(&native, &candidate, &[1, 0], &[0, 2], &family, 1 << 20).unwrap();
        assert_eq!(panel.inputs[0], array![[4.], [5.], [6.]]);
        assert_eq!(panel.targets, array![[1., 4.], [2., 10.], [3., 18.]]);
        assert_eq!(panel.report.output_widths, vec![1, 1]);
    }

    #[test]
    fn rejects_missing_duplicate_and_over_budget_boundaries() {
        let (native, family) = fixture();
        let mut candidate = Artifact::native(&native).unwrap();
        assert!(capture(&native, &candidate, &[0, 0], &[2], &family, 1 << 20).is_err());
        assert!(capture(&native, &candidate, &[0, 1], &[2], &family, 1).is_err());
        candidate.places.retain(|(n, _)| *n != 0);
        assert!(capture(&native, &candidate, &[0, 1], &[2], &family, 1 << 20).is_err());
    }
}
