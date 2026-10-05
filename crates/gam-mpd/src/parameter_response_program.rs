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

/// Raw slot zero is x; slot j+1 is scalar a_j. Callers enforce globally constant
/// a_j for shared parameter edits. Per-row settings execute but are NOT that family.
/// Direction columns U_j are stored operators and paid by ordinary encoding/C32.
/// The hidden read directions defining the native edits belong to the family declaration
/// and must also be retained by the caller (down_edit_family::retain_directions).
pub fn compose(
    clean: &OperatorProgram,
    responses: &[OperatorProgram],
    output_directions: &[Arc<Operator>],
) -> Result<OperatorProgram, String> {
    Ok(compose_with_outputs(clean, responses, output_directions)?.program)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        artifact::Artifact,
        down_edit_family::{self, Direction},
        operator_program::{FamilyInputs, Law, Rule, SlotValues, exact_precision},
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
    fn observed_composition_and_decoded_graft_expose_actual_clean_and_scalar_responses() {
        let mut clean = function();
        let ty = Interface::native(2).unwrap();
        clean.rules = vec![Rule {
            name: "shared learned reader".into(),
            inputs: vec![ty],
            nodes: vec![
                Node::Param { index: 0 },
                Node::Affine {
                    terms: vec![(0, 0)],
                    bias: None,
                },
                Node::Pointwise {
                    input: 1,
                    laws: vec![Law::Relu],
                },
            ],
            output: 2,
        }];
        clean.nodes = vec![
            Node::Raw { slot: 0 },
            Node::Call {
                rule: 0,
                arguments: vec![0],
            },
            Node::Affine {
                terms: vec![(1, 1)],
                bias: None,
            },
        ];
        clean.output = 2;
        let family = down_edit_family::build(
            &clean,
            0,
            2,
            &[
                Direction {
                    output: array![0.5, -1.],
                    hidden: array![1., -0.25],
                },
                Direction {
                    output: array![-1., 0.25],
                    hidden: array![0.25, 0.5],
                },
            ],
        )
        .unwrap();
        let mut responses = vec![];
        let mut directions = vec![];
        for &(u, v) in &family.direction_operators {
            let mut response = clean.clone();
            response.operators[1] = family.program.operators[v].clone();
            responses.push(response);
            directions.push(family.program.operators[u].clone());
        }
        let composed = compose_with_outputs(&clean, &responses, &directions).unwrap();
        assert_eq!(
            composed.program,
            compose(&clean, &responses, &directions).unwrap()
        );
        assert_eq!(composed.composed_output, composed.program.output);
        assert!(!composed.clean_output.retained_by_composed_output);
        assert!(
            composed
                .program
                .nodes
                .iter()
                .all(|n| !n.arguments().contains(&composed.clean_output.node))
        );
        assert_eq!(composed.program.rules.len(), 1);
        assert_eq!(
            composed
                .program
                .nodes
                .iter()
                .filter(|n| matches!(n, Node::Call { .. }))
                .count(),
            3
        );
        assert_eq!(
            composed
                .program
                .operators
                .iter()
                .filter(|op| Arc::ptr_eq(op, &clean.operators[0]))
                .count(),
            1
        );
        let mut reads = vec![family.native_read];
        reads.extend(&family.control_nodes);
        let graft = Artifact::native(&family.program)
            .unwrap()
            .replace_function_inputs(
                "observed responses",
                &composed.program,
                &reads,
                family.native_write,
            )
            .unwrap();
        let saved =
            Artifact::from_bytes(&graft.to_bytes().unwrap(), &family.program.declarations).unwrap();
        let call = saved.place(family.native_write).unwrap();
        assert!(matches!(saved.program.nodes[call], Node::Call { .. }));
        let mut paths = composed
            .response_nodes
            .iter()
            .map(|n| vec![call, *n])
            .collect::<Vec<_>>();
        paths.push(vec![call, composed.clean_output.node]);
        let (flat, _, observed) =
            crate::artifact_device::mapped_inlined_observed(&saved.program, &paths).unwrap();
        let x = array![[1., -2.], [0.5, 3.]];
        let base = FamilyInputs {
            rows: 2,
            slots: vec![SlotValues::Raw(x.clone())],
            layout: None,
        };
        let clean_trace = clean.execute(&base, false).unwrap();
        for amplitudes in [[0., 0.], [-1.5, 0.75], [0.25, -2.]] {
            let inputs = family.inputs(&base, &amplitudes).unwrap();
            let direct = composed.program.execute(&inputs, false).unwrap();
            let replay = flat.execute(&inputs, false).unwrap();
            for (j, response) in responses.iter().enumerate() {
                let target =
                    response.execute(&base, false).unwrap().values[response.output].clone();
                assert_eq!(direct.values[composed.response_nodes[j]], target);
                assert_eq!(replay.values[observed[j]], target);
            }
            assert_eq!(
                direct.values[composed.clean_output.node],
                clean_trace.values[clean.output]
            );
            assert_eq!(
                replay.values[*observed.last().unwrap()],
                clean_trace.values[clean.output]
            );
            assert_eq!(
                direct.values[composed.program.output],
                replay.values[flat.output]
            );
        }
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
    #[test]
    fn saved_multiargument_replacement_predicts_literal_native_weight_edits() {
        let clean = function();
        let mut native = clean.clone();
        // Keep a downstream consumer and skip path: compare autonomous composition,
        // not merely the altered block's direct output.
        native.nodes.push(Node::Affine {
            terms: vec![(3, 0), (0, 0)],
            bias: None,
        });
        native.output = 4;
        let directions = vec![
            Direction {
                output: array![0.5, -1.],
                hidden: array![1., -0.25],
            },
            Direction {
                output: array![-1., 0.25],
                hidden: array![0.25, 0.5],
            },
        ];
        let family =
            down_edit_family::build(&native, 0, 3, &directions).expect("native edit family");
        let mut responses = vec![];
        let mut outputs = vec![];
        for &(u, v) in &family.direction_operators {
            let mut response = clean.clone();
            response.operators[1] = family.program.operators[v].clone();
            responses.push(response);
            outputs.push(family.program.operators[u].clone());
        }
        let program = compose(&clean, &responses, &outputs).expect("response program");
        // Shared actual reader coefficients remain one stored object across three functions.
        assert_eq!(
            program
                .operators
                .iter()
                .filter(|op| Arc::ptr_eq(op, &clean.operators[0]))
                .count(),
            1
        );
        let mut reads = vec![family.native_read];
        reads.extend(&family.control_nodes);
        let mut candidate = Artifact::native(&family.program)
            .expect("native artifact")
            .replace_function_inputs("learned responses", &program, &reads, family.native_write)
            .expect("graft all explicit inputs");
        family
            .retain_directions(&mut candidate)
            .expect("paid directions");
        candidate
            .validate_coverage(&family.program)
            .expect("native coverage");
        assert!(candidate.place(family.node_mapping[1]).is_none());
        assert!(candidate.place(family.node_mapping[2]).is_none());
        let bytes = candidate.to_bytes().expect("encode");
        let saved = Artifact::from_bytes(&bytes, &family.program.declarations).expect("decode");
        for x in [array![[1., -2.], [0.5, 3.]], array![[-3., 0.7], [2.5, 1.]]] {
            let base = FamilyInputs {
                rows: 2,
                slots: vec![SlotValues::Raw(x)],
                layout: None,
            };
            for amplitudes in [[0., 0.], [1., -0.5], [-1.3, 0.7]] {
                let target = family
                    .literal_native(&amplitudes)
                    .expect("literal edit")
                    .execute(&base, false)
                    .expect("native run")
                    .values[native.output]
                    .clone();
                let inputs = family.inputs(&base, &amplitudes).expect("control inputs");
                let predicted = saved.execute(&inputs).expect("saved autonomous run").values
                    [saved.program.output]
                    .clone();
                assert!(
                    target
                        .iter()
                        .zip(&predicted)
                        .all(|(a, b)| (a - b).abs() < 1e-12)
                );
            }
        }
        assert!(
            Artifact::native(&family.program)
                .expect("artifact")
                .replace_function_inputs(
                    "missing control",
                    &program,
                    &[family.native_read],
                    family.native_write
                )
                .is_err()
        );
    }
    #[test]
    fn refuses_nonscalar_response_or_ambient_rule_inputs() {
        let clean = function();
        let u = dense(array![[1.], [-1.]]);
        assert!(compose(&clean, &[clean.clone()], &[u.clone()]).is_err());
        assert!(compose(&clean, &[], &[]).is_err());
        let mut response = clean.clone();
        response.operators[1] = dense(array![[1., 0.]]);
        response.rules.push(Rule {
            name: "ambient".into(),
            inputs: vec![],
            nodes: vec![Node::Raw { slot: 0 }],
            output: 0,
        });
        assert!(compose(&clean, &[response], &[u]).is_err());
    }
}
