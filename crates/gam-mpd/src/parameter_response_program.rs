//! Executable response functions for a declared native down-weight edit family.
//! The program reads x and scalar edit controls, never native hidden activations.
//! It computes f(x) + sum_j a_j U_j s_j(x). The supplied f and s_j are actual
//! learned programs; this compiler does not fit or assume their native fidelity.
use crate::operator_program::{
    Declarations, Interface, Node, Operator, OperatorBody, OperatorProgram, Slot, remap_node,
};
use std::sync::Arc;

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
    for (j, (response, direction)) in responses.iter().zip(output_directions).enumerate() {
        let value = import(&mut out, response)?;
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
    Ok(out)
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
