//! Executable response functions for a declared native down-weight edit family.
//! The program reads x and scalar edit controls, never native hidden activations.
//! It computes f(x) + sum_j a_j U_j s_j(x). The supplied f and s_j are actual
//! learned programs; this compiler does not fit or assume their native fidelity.
use crate::operator_program::{
    Declarations, Interface, Node, Operator, OperatorBody, OperatorProgram, Slot, remap_node,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Root-node identity before any graft, inlining, or pruning. The clean
/// affine writer's terms are fused into the composed output, so its standalone
/// value is not an output dependency and ordinary root pruning can erase it.
/// Function grafting retains its body index; observe it through that Call path.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CleanOutputMetadata {
    pub node: usize,
    pub retained_by_composed_output: bool,
}
/// These IDs name values actually computed by `program`, not native hidden
/// activations or a reconstruction from the controlled final output. Grafting
/// must explicitly remap/retain requested IDs; they are not post-graft IDs.
#[derive(Clone, Debug)]
pub struct Composition {
    pub program: OperatorProgram,
    pub response_nodes: Vec<usize>,
    pub clean_output: CleanOutputMetadata,
    pub composed_output: usize,
}

fn signature(function: &OperatorProgram) -> Result<(usize, Interface), String> {
    let width =
        match function.declarations.slots.as_slice() {
            [Slot::Raw { width }]
                if function.declarations.parameters == 0
                    && function.declarations.domains.is_empty()
                    && function.bases.is_empty() =>
            {
                *width
            }
            _ => return Err(
                "response functions require one Raw input and no ambient parameters/domains/bases"
                    .into(),
            ),
        };
    let interfaces = function.interfaces().map_err(|e| e.to_string())?;
    if function
        .rules
        .iter()
        .flat_map(|r| &r.nodes)
        .any(|n| matches!(n, Node::Raw { .. } | Node::Feature { .. }))
    {
        return Err("response rules must use explicit arguments".into());
    }
    Ok((width, interfaces[function.output].clone()))
}

fn import(out: &mut OperatorProgram, function: &OperatorProgram) -> Result<usize, String> {
    signature(function)?;
    let mut ops = vec![];
    for op in &function.operators {
        let id = match out.operators.iter().position(|held| Arc::ptr_eq(held, op)) {
            Some(id) => id,
            None => {
                let id = out.operators.len();
                out.operators.push(op.clone());
                id
            }
        };
        ops.push(id);
    }
    let mut rules = vec![];
    for rule in &function.rules {
        let mut translated = rule.clone();
        let nodes: Vec<_> = (0..translated.nodes.len()).collect();
        for node in &mut translated.nodes {
            remap_node(node, &nodes, &ops, &[], &rules);
        }
        let id = out
            .rules
            .iter()
            .position(|r| {
                r.inputs == translated.inputs
                    && r.nodes == translated.nodes
                    && r.output == translated.output
            })
            .unwrap_or_else(|| {
                let id = out.rules.len();
                out.rules.push(translated);
                id
            });
        rules.push(id);
    }
    // Expose boundary maps to native-interface specialization at graft time while
    // preserving all nested shared calls. Root inputs all refer to the same x.
    let mut nodes = vec![usize::MAX; function.nodes.len()];
    for (index, node) in function.nodes.iter().enumerate() {
        if matches!(node, Node::Raw { slot: 0 }) {
            nodes[index] = 0;
        } else {
            let mut translated = node.clone();
            remap_node(&mut translated, &nodes, &ops, &[], &rules);
            nodes[index] = out.nodes.len();
            out.nodes.push(translated);
        }
    }
    Ok(nodes[function.output])
}

/// Same executable composition as [`compose`], with pre-graft provenance for
/// each imported scalar `s_j(x)` before multiplying by `a_j` or `U_j`.
/// No new output arithmetic or native activation input is introduced.
pub fn compose_with_outputs(
    clean: &OperatorProgram,
    responses: &[OperatorProgram],
    output_directions: &[Arc<Operator>],
) -> Result<Composition, String> {
    let (width, output) = signature(clean)?;
    let scalar = Interface::native(1).map_err(|e| e.to_string())?;
    if responses.is_empty() || responses.len() != output_directions.len() {
        return Err("one response per nonempty edit family required".into());
    }
    for (response, direction) in responses.iter().zip(output_directions) {
        if signature(response)? != (width, scalar.clone())
            || direction.rows.width() != output.width()
            || direction.cols.width() != 1
        {
            return Err("scalar response input/output or direction interface mismatch".into());
        }
        if !matches!(&direction.body, OperatorBody::Dense { present, .. } if present.iter().all(|x| *x))
        {
            return Err("response direction requires a complete dense column".into());
        }
    }
    let mut slots = vec![Slot::Raw { width }];
    slots.extend((0..responses.len()).map(|_| Slot::Raw { width: 1 }));
    let mut out = OperatorProgram {
        declarations: Declarations {
            domains: vec![],
            slots,
            parameters: 0,
        },
        bases: vec![],
        operators: vec![],
        rules: vec![],
        nodes: vec![Node::Raw { slot: 0 }],
        output: 0,
    };
    let clean_output = import(&mut out, clean)?;
    // Reuse the clean writer instead of storing an artificial dense identity. This
    // fuses the final sum in real arithmetic; the resulting float program, including
    // its accumulation order, must itself be fitted and independently evaluated.
    let Node::Affine { mut terms, bias } = out.nodes[clean_output].clone() else {
        return Err("clean response function must expose an affine output writer".into());
    };
    let mut response_nodes = Vec::with_capacity(responses.len());
    for (j, (response, direction)) in responses.iter().zip(output_directions).enumerate() {
        let value = import(&mut out, response)?;
        response_nodes.push(value);
        let control = out.nodes.len();
        out.nodes.push(Node::Raw { slot: j + 1 });
        let scaled = out.nodes.len();
        out.nodes.push(Node::Hadamard {
            left: value,
            right: control,
        });
        let direction = if direction.rows == output && direction.cols == scalar {
            direction.clone()
        } else {
            let mut adjusted = (**direction).clone();
            adjusted.rows = output.clone();
            adjusted.cols = scalar.clone();
            if let OperatorBody::Dense { present, .. } = &mut adjusted.body {
                *present =
                    ndarray::Array2::from_elem((output.group_count(), scalar.group_count()), true);
            }
            Arc::new(adjusted)
        };
        let op = out
            .operators
            .iter()
            .position(|held| Arc::ptr_eq(held, &direction))
            .unwrap_or_else(|| {
                let id = out.operators.len();
                out.operators.push(direction);
                id
            });
        terms.push((scaled, op));
    }
    out.output = out.nodes.len();
    out.nodes.push(Node::Affine { terms, bias });
    out.interfaces().map_err(|e| e.to_string())?;
    let composed_output = out.output;
    Ok(Composition {
        program: out,
        response_nodes,
        clean_output: CleanOutputMetadata {
            node: clean_output,
            retained_by_composed_output: false,
        },
        composed_output,
    })
}

/// Bind the clean output writer to the native output grouping before fusing it
/// with the response writers. The standalone clean observation and final output
/// consequently refer to the same stored coefficients after function grafting.
/// This changes coordinate labels only, never coordinate order or values.
/// A writer used elsewhere is refused: specializing only this occurrence would
/// split a shared parameter into independently trainable copies.
pub fn compose_with_outputs_on_interface(
    clean: &OperatorProgram,
    responses: &[OperatorProgram],
    output_directions: &[Arc<Operator>],
    target: &Interface,
) -> Result<Composition, String> {
    let (_, output) = signature(clean)?;
    if output == *target {
        return compose_with_outputs(clean, responses, output_directions);
    }
    if output.width() != target.width() {
        return Err("response output regrouping width mismatch".into());
    }
    for response in responses {
        signature(response)?;
    }
    let Node::Affine { terms, bias } = &clean.nodes[clean.output] else {
        return Err("clean response function must expose an affine output writer".into());
    };
    let writer_ids = terms
        .iter()
        .map(|(_, op)| *op)
        .chain(*bias)
        .collect::<Vec<_>>();
    for &writer in &writer_ids {
        let source = &clean.operators[writer];
        let uses_writer = |program: &OperatorProgram, node: &Node| {
            node.operators()
                .iter()
                .any(|&op| Arc::ptr_eq(&program.operators[op], source))
        };
        if clean
            .nodes
            .iter()
            .enumerate()
            .any(|(id, node)| id != clean.output && uses_writer(clean, node))
            || clean
                .rules
                .iter()
                .flat_map(|rule| &rule.nodes)
                .any(|node| uses_writer(clean, node))
            || responses.iter().any(|response| {
                response
                    .nodes
                    .iter()
                    .chain(response.rules.iter().flat_map(|rule| &rule.nodes))
                    .any(|node| uses_writer(response, node))
            })
            || output_directions
                .iter()
                .any(|direction| Arc::ptr_eq(direction, source))
        {
            return Err("response output regrouping would split a shared writer parameter".into());
        }
    }
    let mut adapted = clean.clone();
    let mut specialized: Vec<(Arc<Operator>, Arc<Operator>)> = Vec::new();
    for writer in writer_ids {
        let source = &clean.operators[writer];
        if let Some((_, held)) = specialized.iter().find(|(old, _)| Arc::ptr_eq(old, source)) {
            adapted.operators[writer] = held.clone();
            continue;
        }
        let mut bound = (**source).clone();
        bound.rows = target.clone();
        match &mut bound.body {
            OperatorBody::Dense { present, .. } if present.iter().all(|keep| *keep) => {
                *present = ndarray::Array2::from_elem(
                    (target.group_count(), bound.cols.group_count()),
                    true,
                );
            }
            _ => {
                return Err(
                    "response output regrouping requires complete dense boundary operators".into(),
                );
            }
        }
        let bound = Arc::new(bound);
        adapted.operators[writer] = bound.clone();
        specialized.push((source.clone(), bound));
    }
    compose_with_outputs(&adapted, responses, output_directions)
}

/// Attach an exact finite edit family to one jointly learned multi-output graph.
/// `clean_node` and `response_nodes` identify its actual computed values. Importing
/// the graph once preserves shared nonlinear intermediates, not just tied weights.
/// Native correspondence is explicit: clean is the typed down output and each
/// response is one scalar coefficient in the declared direction order.
/// Raw slot zero is x; appended slots are globally constant edit amplitudes.
pub fn compose_joint_with_outputs(
    function: &OperatorProgram,
    clean_node: usize,
    response_nodes: &[usize],
    output_directions: &[Arc<Operator>],
    target: &Interface,
) -> Result<Composition, String> {
    signature(function)?;
    let types = function.interfaces().map_err(|e| e.to_string())?;
    let scalar = Interface::native(1).map_err(|e| e.to_string())?;
    if response_nodes.is_empty() || response_nodes.len() != output_directions.len() {
        return Err("one response per nonempty edit family required".into());
    }
    let clean_type = types.get(clean_node).ok_or("joint clean node absent")?;
    if clean_type.width() != target.width() {
        return Err("joint clean/native output width mismatch".into());
    }
    for (&response, direction) in response_nodes.iter().zip(output_directions) {
        if types.get(response) != Some(&scalar)
            || direction.rows.width() != target.width()
            || direction.cols.width() != 1
            || !matches!(&direction.body, OperatorBody::Dense { present, .. } if present.iter().all(|p| *p))
        {
            return Err("joint scalar response or direction interface mismatch".into());
        }
    }
    let mut out = function.clone();
    // Fuse a writer only when its typed output already matches. Otherwise a paid
    // identity boundary map preserves every existing shared parameter owner.
    let (mut final_terms, bias) = match &function.nodes[clean_node] {
        Node::Affine { terms, bias } if clean_type == target => (terms.clone(), *bias),
        _ => {
            let mut identity = Operator::identity("joint clean boundary", clean_type.clone());
            if clean_type != target {
                identity = Operator::dense(
                    "joint clean boundary", target.clone(), clean_type.clone(),
                    ndarray::Array2::eye(target.width()),
                    crate::operator_program::exact_precision([0., 1.]).map_err(|e| e.to_string())?,
                    Default::default(),
                ).map_err(|e| e.to_string())?;
            }
            let op = out.operators.len();
            out.operators.push(Arc::new(identity));
            (vec![(clean_node, op)], None)
        }
    };
    for (&response, direction) in response_nodes.iter().zip(output_directions) {
        let slot = out.declarations.slots.len();
        out.declarations.slots.push(Slot::Raw { width: 1 });
        let control = out.nodes.len();
        out.nodes.push(Node::Raw { slot });
        let gated = out.nodes.len();
        out.nodes.push(Node::Hadamard { left: response, right: control });
        let held = if direction.rows == *target && direction.cols == scalar {
            direction.clone()
        } else {
            let mut adjusted = (**direction).clone();
            adjusted.rows = target.clone();
            adjusted.cols = scalar.clone();
            if let OperatorBody::Dense { present, .. } = &mut adjusted.body {
                *present = ndarray::Array2::from_elem((target.group_count(), scalar.group_count()), true);
            }
            Arc::new(adjusted)
        };
        // Direction ownership is explicit; never deduplicate equal learned matrices.
        let op = out.operators.len();
        out.operators.push(held);
        final_terms.push((gated, op));
    }
    out.output = out.nodes.len();
    out.nodes.push(Node::Affine { terms: final_terms, bias });
    out.interfaces().map_err(|e| e.to_string())?;
    let retained = final_terms_retain_clean(&out, clean_node);
    Ok(Composition {
        composed_output: out.output,
        program: out,
        response_nodes: response_nodes.to_vec(),
        clean_output: CleanOutputMetadata { node: clean_node, retained_by_composed_output: retained },
    })
}

fn final_terms_retain_clean(program: &OperatorProgram, clean: usize) -> bool {
    let mut pending = vec![program.output];
    let mut seen = std::collections::BTreeSet::new();
    while let Some(node) = pending.pop() {
        if node == clean { return true; }
        if seen.insert(node) { pending.extend(program.nodes[node].arguments()); }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        artifact::Artifact,
        down_edit_family::{self, Direction},
        operator_program::{FamilyInputs, Law, SlotValues, exact_precision},
    };
    use ndarray::{Array2, array};

    fn dense(values: Array2<f64>) -> Arc<Operator> {
        Arc::new(
            Operator::dense(
                "fixture",
                Interface::native(values.nrows()).expect("rows"),
                Interface::native(values.ncols()).expect("cols"),
                values.clone(),
                exact_precision(values.iter().copied()).expect("precision"),
                Default::default(),
            )
            .expect("operator"),
        )
    }
    fn function() -> OperatorProgram {
        OperatorProgram {
            declarations: Declarations {
                domains: vec![],
                slots: vec![Slot::Raw { width: 2 }],
                parameters: 0,
            },
            bases: vec![],
            rules: vec![],
            operators: vec![
                dense(array![[1., -0.5], [0.25, 2.]]),
                dense(array![[1., 0.5], [-0.5, 1.]]),
            ],
            nodes: vec![
                Node::Raw { slot: 0 },
                Node::Affine {
                    terms: vec![(0, 0)],
                    bias: None,
                },
                Node::Pointwise {
                    input: 1,
                    laws: vec![Law::Relu],
                },
                Node::Affine {
                    terms: vec![(2, 1)],
                    bias: None,
                },
            ],
            output: 3,
        }
    }
    #[test]
    fn joint_graph_keeps_shared_nonlinearity_and_signed_controls_after_codec() {
        let mut joint = function();
        joint.operators.push(dense(array![[1., -0.25]]));
        joint.operators.push(dense(array![[-0.5, 1.]]));
        joint.nodes.push(Node::Affine { terms: vec![(2, 2)], bias: None });
        joint.nodes.push(Node::Affine { terms: vec![(2, 3)], bias: None });
        joint.nodes.push(Node::Concat { parts: vec![3, 4, 5] });
        joint.output = 6;
        let directions = vec![dense(array![[0.5], [-1.]]), dense(array![[-0.25], [0.75]])];
        let target = Interface::native(2).unwrap();
        let composition = compose_joint_with_outputs(&joint, 3, &[4, 5], &directions, &target).unwrap();
        assert_eq!(composition.program.nodes.iter().filter(|n| matches!(n, Node::Pointwise { .. })).count(), 1);
        assert_eq!(composition.response_nodes, vec![4, 5]);
        assert!(!composition.clean_output.retained_by_composed_output);
        assert!(Arc::ptr_eq(&composition.program.operators[joint.operators.len()], &directions[0]));
        let saved = Artifact::native(&composition.program).unwrap();
        let saved = Artifact::from_bytes(&saved.to_bytes().unwrap(), &composition.program.declarations).unwrap();
        for amplitudes in [[-0.75, 1.25], [0., 0.], [1.5, -0.5]] {
            let input = FamilyInputs { rows: 2, slots: vec![
                SlotValues::Raw(array![[1., -2.], [-0.5, 1.25]]),
                SlotValues::Raw(Array2::from_elem((2, 1), amplitudes[0])),
                SlotValues::Raw(Array2::from_elem((2, 1), amplitudes[1])),
            ], layout: None };
            let original = composition.program.execute(&input, false).unwrap();
            let replay = saved.program.execute(&input, false).unwrap();
            assert_eq!(original.values[composition.composed_output], replay.values[saved.program.output]);
            let (observed_program, _, observed) = crate::artifact_device::mapped_inlined_observed(
                &saved.program, &[vec![3], vec![4], vec![5]],
            ).unwrap();
            let observed_trace = observed_program.execute(&input, false).unwrap();
            for (&source, &retained) in [3, 4, 5].iter().zip(&observed) {
                assert_eq!(original.values[source], observed_trace.values[retained]);
            }
            for row in 0..2 { for col in 0..2 {
                let expected = original.values[3][[row, col]]
                    + amplitudes[0] * directions[0].matrix()[[col, 0]] * original.values[4][[row, 0]]
                    + amplitudes[1] * directions[1].matrix()[[col, 0]] * original.values[5][[row, 0]];
                assert!((original.values[composition.composed_output][[row, col]] - expected).abs() < 1e-12);
            }}
        }
        // General clean expressions remain valid and retain their actual value.
        let direct = compose_joint_with_outputs(&joint, 2, &[4, 5], &directions, &target).unwrap();
        assert!(direct.clean_output.retained_by_composed_output);
    }

    #[test]
    fn grouped_output_binding_keeps_one_writer_and_actual_observations_after_graft() {
        use crate::operator_program::LabelKind;
        let clean = function();
        let mut response = clean.clone();
        response.operators[1] = dense(array![[0.5, -0.25]]);
        let grouped = Interface::uniform(2, 1, LabelKind::Unit, 0).unwrap();
        let mut native = clean.clone();
        native.operators = clean
            .operators
            .iter()
            .map(|op| Arc::new((**op).clone()))
            .collect();
        let mut writer = (*native.operators[1]).clone();
        writer.rows = grouped.clone();
        if let OperatorBody::Dense { present, .. } = &mut writer.body {
            *present = Array2::from_elem((2, 1), true);
        }
        native.operators[1] = Arc::new(writer);
        let family = down_edit_family::build(
            &native,
            0,
            3,
            &[Direction {
                output: array![1., -0.5],
                hidden: array![0.5, -0.25],
            }],
        )
        .unwrap();
        let directions = vec![family.program.operators[family.direction_operators[0].0].clone()];
        let composed =
            compose_with_outputs_on_interface(&clean, &[response.clone()], &directions, &grouped)
                .unwrap();
        let types = composed.program.interfaces().unwrap();
        assert_eq!(types[composed.clean_output.node], grouped);
        assert_eq!(types[composed.composed_output], grouped);
        let Node::Affine {
            terms: clean_terms, ..
        } = &composed.program.nodes[composed.clean_output.node]
        else {
            panic!("clean writer")
        };
        let Node::Affine {
            terms: final_terms, ..
        } = &composed.program.nodes[composed.composed_output]
        else {
            panic!("composed writer")
        };
        assert_eq!(clean_terms[0].1, final_terms[0].1);
        let candidate = Artifact::native(&family.program)
            .unwrap()
            .replace_function_inputs(
                "grouped observed responses",
                &composed.program,
                &[family.native_read, family.control_nodes[0]],
                family.native_write,
            )
            .unwrap();
        let new_dense = candidate
            .program
            .operators
            .iter()
            .filter(|op| {
                matches!(op.body, OperatorBody::Dense { .. })
                    && !family
                        .program
                        .operators
                        .iter()
                        .any(|held| Arc::ptr_eq(held, op))
            })
            .count();
        assert_eq!(
            new_dense, 3,
            "one reader, one clean writer, one scalar writer; no dead-writer copy"
        );
        let saved =
            Artifact::from_bytes(&candidate.to_bytes().unwrap(), &family.program.declarations)
                .unwrap();
        let call = saved.place(family.native_write).unwrap();
        let paths = vec![
            vec![call, composed.clean_output.node],
            vec![call, composed.response_nodes[0]],
        ];
        let (flat, _, observed) =
            crate::artifact_device::mapped_inlined_observed(&saved.program, &paths).unwrap();
        let x = array![[1., -2.], [-0.5, 3.]];
        let base = FamilyInputs {
            rows: 2,
            slots: vec![SlotValues::Raw(x)],
            layout: None,
        };
        let clean_value = clean.execute(&base, false).unwrap().values[clean.output].clone();
        let response_value =
            response.execute(&base, false).unwrap().values[response.output].clone();
        for amplitude in [-1., 0., 0.75] {
            let inputs = family.inputs(&base, &[amplitude]).unwrap();
            let trace = flat.execute(&inputs, false).unwrap();
            assert_eq!(trace.values[observed[0]], clean_value);
            assert_eq!(trace.values[observed[1]], response_value);
            let expected = composed.program.execute(&inputs, false).unwrap();
            assert_eq!(
                trace.values[flat.output],
                expected.values[composed.program.output]
            );
        }
        let mut internally_shared = clean.clone();
        internally_shared.nodes.insert(
            3,
            Node::Affine {
                terms: vec![(2, 1)],
                bias: None,
            },
        );
        internally_shared.output = 4;
        assert!(
            compose_with_outputs_on_interface(
                &internally_shared,
                &[response],
                &directions,
                &grouped
            )
            .unwrap_err()
            .contains("split a shared writer")
        );
        assert!(
            compose_with_outputs_on_interface(&clean, &[clean.clone()], &directions, &grouped)
                .unwrap_err()
                .contains("split a shared writer")
        );
        let mut malformed = clean.clone();
        malformed.nodes[1] = Node::Affine {
            terms: vec![(0, 99)],
            bias: None,
        };
        assert!(
            compose_with_outputs_on_interface(&clean, &[malformed], &directions, &grouped).is_err()
        );
        let mut scalar_input = clean.clone();
        scalar_input.declarations.slots = vec![Slot::Raw { width: 1 }];
        scalar_input.operators = vec![dense(array![[1.], [2.]])];
        scalar_input.nodes = vec![
            Node::Raw { slot: 0 },
            Node::Affine {
                terms: vec![(0, 0)],
                bias: None,
            },
        ];
        scalar_input.output = 1;
        let mut scalar_response = scalar_input.clone();
        scalar_response.operators = vec![dense(array![[0.5]])];
        assert!(
            compose_with_outputs_on_interface(
                &scalar_input,
                &[scalar_response],
                &[scalar_input.operators[0].clone()],
                &grouped
            )
            .unwrap_err()
            .contains("split a shared writer")
        );
    }
}
