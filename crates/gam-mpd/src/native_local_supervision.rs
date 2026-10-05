//! Native local targets evaluated on the inputs reached by a candidate.
//! This is training supervision, never an execution-time input to the explanation.
use crate::{artifact::Artifact, operator_program::OperatorProgram};
use crate::operator_program::{FamilyInputs, Interface, Node};

#[path = "native_local_supervision_device.rs"]
mod device;
pub use device::PreparedDeviceCapture;
use ndarray::{Array2, Axis};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

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
    /// Native top-level operations actually evaluated; boundary producers are skipped.
    pub native_evaluated_nodes: Vec<usize>,
    pub native_prefix_nodes_avoided: usize,
    pub planned_trace_and_panel_bytes: usize,
    pub scope: &'static str,
}

struct BoundaryPlan {
    reads: BTreeMap<usize, usize>,
    nt: Vec<Interface>,
    ct: Vec<Interface>,
    output_widths: Vec<usize>,
    last_native: usize,
    native_evaluated_nodes: Vec<usize>,
}

/// Shared closure/interface validation for ordinary and device execution.
fn boundary_plan(native: &OperatorProgram, candidate: &Artifact, arguments: &[usize], outputs: &[usize])
    -> Result<BoundaryPlan, String>
{
    if arguments.is_empty() || outputs.is_empty() {
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
    // A region validated against an already-rewritten candidate can omit a read
    // that the native equation still needs. Refuse labels with ambient native
    // state hidden from the local predictor. Rule bodies can read slots directly,
    // so checking only a Call's declared arguments would leave that bypass open.
    let mut pending: Vec<(Option<usize>, usize)> = outputs.iter().map(|&n| (None, n)).collect();
    let mut seen = BTreeSet::new();
    while let Some((rule, n)) = pending.pop() {
        if (rule.is_none() && reads.contains_key(&n)) || !seen.insert((rule, n)) {
            continue;
        }
        let node = match rule {
            Some(r) => &native.rules[r].nodes[n],
            None => &native.nodes[n],
        };
        match node {
            Node::Raw { .. } | Node::Feature { .. } => {
                return Err(format!("incomplete native local boundary: unbound ambient input at rule {rule:?}, node {n}"));
            }
            Node::Param { .. } if rule.is_none() => {
                return Err(format!("incomplete native local boundary: unbound parameter at node {n}"));
            }
            Node::Call { rule: callee, .. } => {
                pending.push((Some(*callee), native.rules[*callee].output));
            }
            _ => {}
        }
        pending.extend(node.arguments().into_iter().map(|p| (rule, p)));
    }
    let native_evaluated_nodes = seen.iter().filter_map(|(rule,n)| rule.is_none().then_some(*n))
        .collect::<Vec<_>>();
    Ok(BoundaryPlan { reads, nt, ct, output_widths, last_native, native_evaluated_nodes })
}

/// Capture a complete sequence family without changing its attention layout.
/// The native teacher is clamped to the candidate's incoming values at every
/// declared boundary, then evaluated downstream. Targets are NOT the clean
/// trajectory outputs. Every native output dependency must terminate at a
/// declared boundary or fixed constant, including dependencies inside calls.
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
    let BoundaryPlan { reads, nt, ct, output_widths, last_native, native_evaluated_nodes } =
        boundary_plan(native, candidate, arguments, outputs)?;
    let last_candidate = *reads.values().max().ok_or("candidate inputs absent")?;
    let trace_widths = native_evaluated_nodes.iter().map(|&n| nt[n].width())
        .chain(ct[..=last_candidate].iter().map(|t| t.width()));
    // Boundary patches, evaluator boundary buffers, and returned inputs; two
    // target copies during concatenate. Native upstream trace is never allocated.
    let mut widths = trace_widths.chain(arguments.iter().flat_map(|n| [nt[*n].width(); 3]))
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
    if patches.iter().any(|(&n,v)| v.dim() != (family.rows,nt[n].width())) {
        return Err("local supervision needs materialized boundary matrices".into());
    }
    if patches.values().any(|v| v.iter().any(|v| !v.is_finite())) {
        return Err("nonfinite candidate local inputs".into());
    }
    drop(candidate_trace);
    let native_values = native.execute_clamped_outputs(family, &patches, outputs)
        .map_err(|e| e.to_string())?;
    let targets = ndarray::concatenate(Axis(1), &native_values.iter()
        .map(|v| v.view()).collect::<Vec<_>>())
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
            native_prefix_nodes_avoided: last_native + 1 - native_evaluated_nodes.len(),
            native_evaluated_nodes,
            scope: "Native dependency slice below complete candidate-reached boundaries; training-only. Boundary producers and unrelated native nodes are not evaluated. Complete sequence layout retained. Budget excludes weights, evaluator scratch, inputs, and allocator overhead.",
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator_program::{Coefficient, Declarations, Interface, Law, Rule, Scale, SequenceLayout, Slot, SlotValues};
    use ndarray::array;

    fn device_matches(native: &OperatorProgram, candidate: &Artifact, arguments: &[usize],
        outputs: &[usize], family: &FamilyInputs) -> Panel
    {
        let expected = capture(native, candidate, arguments, outputs, family, 1 << 24).unwrap();
        let prepared = PreparedDeviceCapture::new(&gam_gpu::tensor::Device::host(), native,
            candidate, arguments, outputs, 1 << 24).unwrap();
        let actual = prepared.capture(family, 1 << 24).unwrap();
        assert_eq!(actual.inputs.len(), expected.inputs.len());
        for (a, b) in actual.inputs.iter().zip(&expected.inputs) {
            assert_eq!(a.dim(), b.dim());
            for (a, b) in a.iter().zip(b) {
                assert!((a-b).abs() <= 1e-12 * (1. + b.abs()), "boundary device {a}, reference {b}");
            }
        }
        assert_eq!(actual.targets.dim(), expected.targets.dim());
        for (a, b) in actual.targets.iter().zip(&expected.targets) {
            assert!((a-b).abs() <= 1e-12 * (1. + b.abs()), "device {a}, reference {b}");
        }
        assert_eq!(actual.report.native_evaluated_nodes, expected.report.native_evaluated_nodes);
        assert_eq!(actual.report.native_prefix_nodes_avoided, expected.report.native_prefix_nodes_avoided);
        assert_eq!(actual.report.arguments, arguments);
        assert_eq!(actual.report.native_outputs, outputs);
        let again = prepared.capture(family, 1 << 24).unwrap();
        assert_eq!(again.inputs, actual.inputs);
        assert_eq!(again.targets, actual.targets, "prepared programs are reusable");
        actual
    }

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
        device_matches(&native, &candidate, &[0, 1], &[2], &family);
    }

    #[test]
    fn permutation_changes_local_argument_order_not_teacher_wiring() {
        let (native, family) = fixture();
        let candidate = Artifact::native(&native).unwrap();
        let panel = capture(&native, &candidate, &[1, 0], &[0, 2], &family, 1 << 20).unwrap();
        assert_eq!(panel.inputs[0], array![[4.], [5.], [6.]]);
        assert_eq!(panel.targets, array![[1., 4.], [2., 10.], [3., 18.]]);
        assert_eq!(panel.report.output_widths, vec![1, 1]);
        device_matches(&native, &candidate, &[1, 0], &[0, 2], &family);
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

    #[test]
    fn rejects_native_dependency_dropped_by_an_earlier_candidate_rewrite() {
        let (native, family) = fixture();
        let mut candidate = Artifact::native(&native).unwrap();
        candidate.program.nodes[2] = Node::Pointwise { input: 0, laws: vec![Law::Relu] };
        candidate = candidate.bind("earlier replacement", &[0, 1], 2).unwrap();
        candidate.validate_coverage(&native).unwrap();
        // [0] is a complete executable boundary in this candidate, but native
        // multiplication still reads 1. The teacher must not silently use its
        // clean family values as labels for a predictor given only input 0.
        let error = capture(&native, &candidate, &[0], &[2], &family, 1 << 20)
            .err().expect("incomplete native boundary must fail");
        assert!(error.contains("incomplete native local boundary"), "{error}");
    }

    #[test]
    fn computed_boundary_stops_native_ancestor_traversal() {
        let (mut native, family) = fixture();
        native.nodes.push(Node::Pointwise { input: 2, laws: vec![Law::Relu] });
        native.output = 3;
        let mut candidate = Artifact::native(&native).unwrap();
        candidate.program.nodes[2] = Node::Raw { slot: 1 };
        let panel = capture(&native, &candidate, &[2], &[3], &family, 1 << 20).unwrap();
        assert_eq!(panel.inputs[0], array![[4.], [5.], [6.]]);
        assert_eq!(panel.targets, panel.inputs[0]);
        assert_eq!(panel.report.native_evaluated_nodes, vec![3]);
        assert_eq!(panel.report.native_prefix_nodes_avoided, 3);
    }

    #[test]
    fn sliced_native_teacher_matches_clamped_prefix_with_grouped_and_duplicate_outputs() {
        let (mut native, family) = fixture();
        for _ in 0..64 {
            native.nodes.push(Node::Gain { input: native.nodes.len()-1,
                coefficient: Coefficient::Number(1.01) });
        }
        let boundary = native.nodes.len()-1;
        let grouped = native.nodes.len();
        native.nodes.push(Node::Concat { parts: vec![boundary, boundary] });
        native.nodes.push(Node::Pointwise { input: grouped, laws: vec![Law::Relu, Law::GeluTanh] });
        native.output = native.nodes.len()-1;
        let mut candidate = Artifact::native(&native).unwrap();
        candidate.program.nodes[boundary] = Node::Gain { input: 1,
            coefficient: Coefficient::Number(-0.5) };
        let outputs = [native.output, grouped, native.output];
        let panel = capture(&native, &candidate, &[boundary], &outputs, &family, 1 << 20).unwrap();
        let reference = native.execute_edited(&family, |n,value,_| {
            if n == boundary { value.assign(&panel.inputs[0]); }
            Ok(())
        }).unwrap();
        let expected = ndarray::concatenate(Axis(1), &outputs.iter()
            .map(|n|reference.values[*n].view()).collect::<Vec<_>>()).unwrap();
        assert_eq!(panel.targets, expected);
        assert_eq!(panel.report.native_evaluated_nodes, vec![grouped, native.output]);
        assert_eq!(panel.report.native_prefix_nodes_avoided, boundary+1);
        device_matches(&native, &candidate, &[boundary], &outputs, &family);
    }

    #[test]
    fn sliced_attention_teacher_preserves_sequence_boundaries_and_causality() {
        let native = OperatorProgram {
            declarations: Declarations { domains:vec![],slots:vec![Slot::Raw {width:1};3],parameters:0 },
            bases:vec![],rules:vec![],operators:vec![],
            nodes:vec![Node::Raw {slot:0},Node::Raw {slot:1},Node::Raw {slot:2},
                Node::Attend {query:0,key:1,value:2,scale:Scale::One,rotary:None,causal:true}],output:3,
        };
        let family = FamilyInputs { rows:4,
            slots:vec![SlotValues::Raw(array![[1.],[2.],[-1.],[-2.]]),
                SlotValues::Raw(array![[0.5],[-0.5],[0.25],[-0.25]]),
                SlotValues::Raw(array![[10.],[20.],[100.],[200.]])],
            layout:Some(SequenceLayout {sequence:vec![0,0,1,1],position:vec![0,1,0,1]}),
        };
        let mut candidate = Artifact::native(&native).unwrap();
        candidate.program.nodes[1] = Node::Raw {slot:0};
        let panel = capture(&native,&candidate,&[2,0,1],&[3],&family,1 << 20).unwrap();
        let reference = native.execute_edited(&family, |n,value,_| {
            if let Some(i) = [2,0,1].iter().position(|p|*p==n) { value.assign(&panel.inputs[i]); }
            Ok(())
        }).unwrap();
        assert_eq!(panel.targets,reference.values[3]);
        assert_eq!(panel.targets[(0,0)],10.);
        assert_eq!(panel.targets[(2,0)],100.);
        assert_eq!(panel.report.native_evaluated_nodes,vec![3]);
        assert_ne!(panel.targets,native.execute(&family,false).unwrap().values[3]);
        let device_panel = device_matches(&native, &candidate, &[2,0,1], &[3], &family);
        // If only the second token is scored, its unscored first token still contributes
        // attention context. Capturing the complete family preserves that dependence.
        let mut changed_prefix = family.clone();
        if let SlotValues::Raw(value) = &mut changed_prefix.slots[2] { value[(0,0)] += 7.; }
        let changed = device_matches(&native, &candidate, &[2,0,1], &[3], &changed_prefix);
        assert_ne!(device_panel.targets[(1,0)], changed.targets[(1,0)]);
        assert_eq!(device_panel.targets[(3,0)], changed.targets[(3,0)],
            "attention must not leak across complete sequence blocks");
    }

    #[test]
    fn called_rule_cannot_bypass_boundary_via_ambient_slot() {
        let (mut native, family) = fixture();
        native.rules.push(Rule {
            name: "ambient bypass".into(), inputs: vec![Interface::native(1).unwrap()],
            nodes: vec![Node::Param { index: 0 }, Node::Raw { slot: 1 },
                Node::Hadamard { left: 0, right: 1 }], output: 2,
        });
        native.nodes[2] = Node::Call { rule: 0, arguments: vec![0] };
        let candidate = Artifact::native(&native).unwrap();
        let error = capture(&native, &candidate, &[0], &[2], &family, 1 << 20)
            .err().expect("ambient rule input must fail");
        assert!(error.contains("incomplete native local boundary"), "{error}");
        // Supplying all reads through actual call arguments restores closure.
        let error = PreparedDeviceCapture::new(&gam_gpu::tensor::Device::host(), &native,
            &candidate, &[0], &[2], 1 << 20).err().expect("device rejects ambient rule input");
        assert!(error.contains("incomplete native local boundary"), "{error}");
        native.rules[0].inputs.push(Interface::native(1).unwrap());
        native.rules[0].nodes[1] = Node::Param { index: 1 };
        native.nodes[2] = Node::Call { rule: 0, arguments: vec![0, 1] };
        let candidate = Artifact::native(&native).unwrap();
        let panel = capture(&native, &candidate, &[0, 1], &[2], &family, 1 << 20).unwrap();
        assert_eq!(panel.targets, array![[4.], [10.], [18.]]);
        device_matches(&native, &candidate, &[0, 1], &[2], &family);
    }

    #[test]
    fn device_nested_nonlinear_teacher_preserves_grouped_boundaries_and_output_order() {
        let (mut native, family) = fixture();
        let one = Interface::native(1).unwrap();
        native.rules = vec![
            Rule { name: "inner".into(), inputs: vec![one.clone(), one.clone()],
                nodes: vec![Node::Param { index: 0 }, Node::Param { index: 1 },
                    Node::Hadamard { left: 0, right: 1 },
                    Node::Pointwise { input: 2, laws: vec![Law::GeluTanh] }], output: 3 },
            Rule { name: "outer".into(), inputs: vec![one.clone(), one],
                nodes: vec![Node::Param { index: 0 }, Node::Param { index: 1 },
                    Node::Call { rule: 0, arguments: vec![1,0] }], output: 2 },
        ];
        native.nodes[2] = Node::Call { rule: 1, arguments: vec![0,1] };
        native.nodes.push(Node::Concat { parts: vec![2,0] });
        native.nodes.push(Node::Pointwise { input: 3, laws: vec![Law::GeluTanh, Law::Relu] });
        native.output = 4;
        let mut candidate = Artifact::native(&native).unwrap();
        candidate.program.nodes[0] = Node::Raw { slot: 1 };
        device_matches(&native, &candidate, &[1,0], &[2,0,2], &family);
        device_matches(&native, &candidate, &[3], &[4,3,4], &family);
        // A nested ambient source must still be refused before any compilation/execution.
        native.rules[0].nodes[1] = Node::Raw { slot: 1 };
        let candidate = Artifact::native(&native).unwrap();
        let error = PreparedDeviceCapture::new(&gam_gpu::tensor::Device::host(), &native,
            &candidate, &[0,1], &[2], 1 << 20).err().expect("nested ambient bypass");
        assert!(error.contains("incomplete native local boundary"), "{error}");
    }

    #[test]
    fn device_capture_refuses_budgets_and_unsupported_teacher_without_fallback() {
        let (native, family) = fixture();
        let candidate = Artifact::native(&native).unwrap();
        assert!(PreparedDeviceCapture::new(&gam_gpu::tensor::Device::host(), &native,
            &candidate, &[0,1], &[2], 1).is_err());
        let prepared = PreparedDeviceCapture::new(&gam_gpu::tensor::Device::host(), &native,
            &candidate, &[0,1], &[2], 1 << 20).unwrap();
        let planned = prepared.planned_trace_and_panel_bytes(&family).unwrap();
        assert!(prepared.capture(&family, planned-1).is_err());
        prepared.capture(&family, planned).unwrap();
        let mut native = native;
        native.nodes[2] = Node::Softmax { scores: vec![0,1] };
        let candidate = Artifact::native(&native).unwrap();
        let error = PreparedDeviceCapture::new(&gam_gpu::tensor::Device::host(), &native,
            &candidate, &[0,1], &[2], 1 << 20).err().expect("unsupported teacher primitive");
        assert!(error.contains("has no device rule"), "{error}");
    }

    #[test]
    fn device_teacher_prunes_unrelated_unsupported_nodes_instead_of_running_native_prefix() {
        let (mut native, family) = fixture();
        native.nodes[2] = Node::Softmax { scores: vec![0,1] };
        native.nodes.push(Node::Hadamard { left: 0, right: 1 });
        native.output = 3;
        let mut candidate = Artifact::native(&native).unwrap();
        candidate.program.nodes[0] = Node::Raw { slot: 1 };
        let result = device_matches(&native, &candidate, &[0,1], &[3], &family);
        assert_eq!(result.targets, array![[16.], [25.], [36.]]);
        assert_eq!(result.report.native_evaluated_nodes, vec![3]);
    }
}
