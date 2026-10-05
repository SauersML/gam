//! Exact learned numerical-body sharing; no additional fitting, snapping or activation discovery.
use super::super::operator_program::OperatorBody;
use super::*;
use std::sync::Arc;

fn same_operator(a: &Operator, b: &Operator) -> bool {
    if a.rows != b.rows || a.cols != b.cols || a.body != b.body || a.provenance != b.provenance {
        return false;
    }
    match (&a.body, &b.body) {
        (OperatorBody::Dense { values: x, .. }, OperatorBody::Dense { values: y, .. }) => {
            x.iter().zip(y).all(|(x, y)| x.to_bits() == y.to_bits())
        }
        _ => false,
    }
}

fn same_body(a: &Rule, b: &Rule, operators: &[Arc<Operator>]) -> bool {
    if a.inputs != b.inputs || a.output != b.output || a.nodes.len() != b.nodes.len() {
        return false;
    }
    a.nodes.iter().zip(&b.nodes).all(|(x, y)| match (x, y) {
        (Node::Constant { operator: x }, Node::Constant { operator: y }) => {
            same_operator(&operators[*x], &operators[*y])
        }
        (Node::Affine { terms: x, bias: xb }, Node::Affine { terms: y, bias: yb }) => {
            x.len() == y.len()
                && x.iter().zip(y).all(|((xn, xo), (yn, yo))| {
                    xn == yn && same_operator(&operators[*xo], &operators[*yo])
                })
                && match (xb, yb) {
                    (Some(x), Some(y)) => same_operator(&operators[*x], &operators[*y]),
                    (None, None) => true,
                    _ => false,
                }
        }
        _ => x == y,
    })
}

/// Compile repeated, bit-identical learned scalar functions into shared Rule bodies and Calls.
/// Actual input bindings are excluded from body identity and serialized at each Call. Readers,
/// writes, offsets, all function coefficients and graph wiring are paid by ordinary C32.
/// Hidden native-unit intervention places remain omitted, as in the flat account compiler.
pub fn with_account_shared(
    artifact: &Artifact,
    name: &str,
    layer: &LayerNodes,
    account: &Account,
) -> Result<Artifact, String> {
    let read = artifact
        .place(layer.normed)
        .ok_or_else(|| format!("{name}: absent MLP input"))?;
    let write = artifact
        .place(layer.mlp)
        .ok_or_else(|| format!("{name}: absent MLP output"))?;
    let interfaces = artifact.program.interfaces().map_err(|e| e.to_string())?;
    let input = &interfaces[read];
    let output = &interfaces[write];
    let m = account.reads.nrows();
    let k = account.rules.len();
    if account.reads.ncols() != input.width()
        || account.writes.dim() != (k, output.width())
        || account.offset.len() != output.width()
    {
        return Err(format!("{name}: account shape mismatch"));
    }
    if !account
        .reads
        .iter()
        .chain(account.writes.iter())
        .chain(account.offset.iter())
        .all(|v| v.is_finite())
    {
        return Err(format!("{name}: nonfinite account map"));
    }
    for r in &account.rules {
        if r.inputs.iter().any(|i| *i >= m)
            || r.linear.len() != r.inputs.len()
            || r.units.iter().any(|u| u.w.len() != r.inputs.len())
        {
            return Err(format!("{name}: malformed scalar function"));
        }
        if !std::iter::once(&r.beta)
            .chain(r.linear.iter())
            .chain(r.units.iter().flat_map(|u| u.w.iter().chain([&u.d, &u.c])))
            .all(|v| v.is_finite())
        {
            return Err(format!("{name}: nonfinite scalar function"));
        }
    }
    let scalar = Interface::native(1).map_err(|e| e.to_string())?;
    let constant = Interface::constant();
    let mut result = artifact.clone();
    let mut nodes = vec![Node::Param { index: 0 }];
    let mut reads = Vec::new();
    for i in 0..m {
        let op = result.program.operators.len();
        result.program.operators.push(Arc::new(dense(
            &format!("{name} read {i}"),
            &scalar,
            input,
            account.reads.slice(ndarray::s![i..i + 1, ..]).to_owned(),
        )?));
        reads.push(nodes.len());
        nodes.push(Node::Affine {
            terms: vec![(0, op)],
            bias: None,
        });
    }
    let mut calls = Vec::new();
    for (j, r) in account.rules.iter().enumerate() {
        let d = r.inputs.len();
        let first = result.program.operators.len();
        let (body, ops) = if d == 0 {
            let beta = dense(
                &format!("{name} beta {j}"),
                &scalar,
                &constant,
                Array2::from_elem((1, 1), r.beta),
            )?;
            let mut ops = vec![beta];
            let mut body_nodes = vec![Node::Constant { operator: first }];
            if !r.units.is_empty() {
                let hidden = units(r.units.len(), LabelKind::Unit)?;
                ops.push(dense(
                    &format!("{name} offsets {j}"),
                    &hidden,
                    &constant,
                    Array2::from_shape_vec(
                        (r.units.len(), 1),
                        r.units.iter().map(|u| u.d).collect(),
                    )
                    .map_err(|e| e.to_string())?,
                )?);
                ops.push(dense(
                    &format!("{name} scales {j}"),
                    &scalar,
                    &hidden,
                    Array2::from_shape_vec(
                        (1, r.units.len()),
                        r.units.iter().map(|u| u.c).collect(),
                    )
                    .map_err(|e| e.to_string())?,
                )?);
                body_nodes.push(Node::Constant {
                    operator: first + 1,
                });
                body_nodes.push(Node::Pointwise {
                    input: 1,
                    laws: vec![Law::GeluTanh; r.units.len()],
                });
                body_nodes.push(Node::Affine {
                    terms: vec![(2, first + 2)],
                    bias: Some(first),
                });
            }
            let last = body_nodes.len() - 1;
            (
                Rule {
                    name: format!("{name} scalar {j}"),
                    inputs: vec![],
                    nodes: body_nodes,
                    output: last,
                },
                ops,
            )
        } else {
            let packed = Interface::new((0..d).map(|_| scalar.groups()[0].clone()).collect())
                .map_err(|e| e.to_string())?;
            let mut local = r.clone();
            local.inputs = (0..d).collect();
            let sub = Account {
                reads: Array2::eye(d),
                rules: vec![local],
                writes: Array2::ones((1, 1)),
                offset: Array1::zeros(1),
            };
            account_rule(&format!("{name} scalar {j}"), &sub, &packed, &scalar, first)?
        };
        result
            .program
            .operators
            .extend(ops.into_iter().map(Arc::new));
        let found = result
            .program
            .rules
            .iter()
            .position(|existing| same_body(existing, &body, &result.program.operators));
        let rule = if let Some(i) = found {
            result.program.operators.truncate(first);
            i
        } else {
            let i = result.program.rules.len();
            result.program.rules.push(body);
            i
        };
        let arguments = if d == 0 {
            vec![]
        } else if d == 1 {
            vec![reads[r.inputs[0]]]
        } else {
            let packed = nodes.len();
            nodes.push(Node::Concat {
                parts: r.inputs.iter().map(|i| reads[*i]).collect(),
            });
            vec![packed]
        };
        calls.push(nodes.len());
        nodes.push(Node::Call { rule, arguments });
    }
    let bias = result.program.operators.len();
    result.program.operators.push(Arc::new(dense(
        &format!("{name} offset"),
        output,
        &constant,
        account.offset.clone().insert_axis(ndarray::Axis(1)),
    )?));
    if k == 0 {
        nodes.push(Node::Constant { operator: bias });
    } else {
        let outgoing = Interface::new((0..k).map(|_| scalar.groups()[0].clone()).collect())
            .map_err(|e| e.to_string())?;
        let write_op = result.program.operators.len();
        result.program.operators.push(Arc::new(dense(
            &format!("{name} writes"),
            output,
            &outgoing,
            account.writes.t().as_standard_layout().to_owned(),
        )?));
        let joined = nodes.len();
        nodes.push(Node::Concat { parts: calls });
        nodes.push(Node::Affine {
            terms: vec![(joined, write_op)],
            bias: Some(bias),
        });
    }
    let output_node = nodes.len() - 1;
    result.replace_block(
        name,
        Callee::New(Rule {
            name: format!("{name} shared"),
            inputs: vec![input.clone()],
            nodes,
            output: output_node,
        }),
        vec![Argument::Native(layer.normed)],
        layer.mlp,
        vec![],
    )
}

#[cfg(test)]
mod tests {
    use super::super::super::acceptance::structural_cost;
    use super::super::super::operator_program::{
        Declarations, FamilyInputs, OperatorProgram, Slot, SlotValues,
    };
    use super::*;
    fn fixture() -> (Artifact, LayerNodes, Account, FamilyInputs) {
        let interface = Interface::native(2).expect("native interface");
        let op = dense("native write", &interface, &interface, Array2::eye(2)).expect("operator");
        let p = OperatorProgram {
            declarations: Declarations {
                domains: vec![],
                slots: vec![Slot::Raw { width: 2 }],
                parameters: 0,
            },
            bases: vec![],
            operators: vec![Arc::new(op)],
            rules: vec![],
            nodes: vec![
                Node::Raw { slot: 0 },
                Node::Affine {
                    terms: vec![(0, 0)],
                    bias: None,
                },
            ],
            output: 1,
        };
        let scalar = AccountRule {
            inputs: vec![0, 1],
            beta: 0.25,
            linear: vec![0.125, -0.5],
            units: vec![
                super::super::super::mlp_account::Unit {
                    w: vec![0.75, 0.25],
                    d: 0.1,
                    c: 1.,
                },
                super::super::super::mlp_account::Unit {
                    w: vec![-0.4, 0.8],
                    d: -0.2,
                    c: 0.5,
                },
            ],
        };
        let mut rules = vec![scalar; 64];
        for (i, r) in rules.iter_mut().enumerate() {
            if i % 2 == 1 {
                r.inputs.reverse();
            }
        }
        let account = Account {
            reads: ndarray::array![[1., 0.25], [-0.5, 0.75]],
            rules,
            writes: Array2::from_shape_fn((64, 2), |(i, j)| ((i + j) % 5 + 1) as f64 / 128.),
            offset: ndarray::array![0.1, -0.2],
        };
        let family = FamilyInputs {
            rows: 3,
            slots: vec![SlotValues::Raw(ndarray::array![
                [1., -2.],
                [0.3, 0.7],
                [-1.2, 0.4]
            ])],
            layout: None,
        };
        (
            Artifact::native(&p).expect("artifact"),
            LayerNodes {
                normed: 0,
                mlp: 1,
                ..LayerNodes::default()
            },
            account,
            family,
        )
    }
    fn decoded(a: &Artifact) -> Artifact {
        Artifact::from_bytes(&a.to_bytes().expect("encode"), &a.program.declarations)
            .expect("independent decode")
    }
    #[test]
    fn learned_nonlinear_body_shares_different_bindings_and_pays_less() {
        let (native, layer, account, family) = fixture();
        let flat = decoded(&with_account(&native, "flat", &layer, &account).expect("flat"));
        let shared =
            decoded(&with_account_shared(&native, "shared", &layer, &account).expect("shared"));
        let proposer = AccountProposer {
            layers: vec![layer.clone()],
            accounts: vec![(0, "learned".into(), account.clone())],
        };
        assert!(crate::candidate_frontier::account_bank(&native, &proposer, 2).is_err());
        let bank = crate::candidate_frontier::account_bank(&native, &proposer, 3)
            .expect("native, flat and shared");
        assert_eq!(bank.len(), 3);
        assert_eq!(shared.program.rules.len(), 2);
        let outer = shared.program.rules.last().expect("outer");
        let calls: Vec<_> = outer
            .nodes
            .iter()
            .filter_map(|n| {
                if let Node::Call { rule, arguments } = n {
                    Some((*rule, arguments.clone()))
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(calls.len(), 64);
        assert!(calls.iter().all(|c| c.0 == calls[0].0));
        assert_ne!(calls[0].1, calls[1].1);
        assert!(
            structural_cost(&shared, &mut Default::default())
                .expect("shared cost")
                .total()
                < structural_cost(&flat, &mut Default::default())
                    .expect("flat cost")
                    .total()
        );
        let f = flat.execute(&family).expect("flat execution");
        let s = shared.execute(&family).expect("shared execution");
        assert!(
            f.values[flat.program.output]
                .iter()
                .zip(&s.values[shared.program.output])
                .all(|(a, b)| (a - b).abs() < 1e-12)
        );
        // Held-out represented-input intervention: it changes an actual input, independently
        // of how the scalar functions were fitted. Omitted native hidden units stay omitted.
        let edited = |a: &Artifact| {
            a.execute_edited(&family, |node, value, _| {
                if node == a.place(0).expect("held input") {
                    value[[0, 0]] *= 0.3;
                }
                Ok(())
            })
            .expect("intervention")
        };
        let ef = edited(&flat);
        let es = edited(&shared);
        assert!(
            ef.values[flat.program.output]
                .iter()
                .zip(&es.values[shared.program.output])
                .all(|(a, b)| (a - b).abs() < 1e-12)
        );
        assert!(
            (es.values[shared.program.output][[0, 0]] - s.values[shared.program.output][[0, 0]])
                .abs()
                > 1e-3
        );
    }
    #[test]
    fn differing_coefficients_and_signed_zero_are_not_merged() {
        let (native, layer, mut account, _) = fixture();
        account.rules[1].units[0].w[0] += 0.125;
        assert_eq!(
            with_account_shared(&native, "different", &layer, &account)
                .expect("compile")
                .program
                .rules
                .len(),
            3
        );
        for r in &mut account.rules {
            r.linear[0] = 0.0;
        }
        account.rules[1].linear[0] = -0.0;
        assert_eq!(
            with_account_shared(&native, "signed zero", &layer, &account)
                .expect("compile")
                .program
                .rules
                .len(),
            3
        );
        account.rules[0].units[0].w.pop();
        assert!(with_account_shared(&native, "bad", &layer, &account).is_err());
    }
    #[test]
    fn shared_body_survives_cross_layer_composition_and_device_execution() {
        let (mut native, first, account, family) = fixture();
        native.program.nodes.push(Node::Affine {
            terms: vec![(1, 0)],
            bias: None,
        });
        native.program.output = 2;
        native = Artifact::native(&native.program).expect("two native layers");
        let second = LayerNodes {
            normed: 1,
            mlp: 2,
            ..LayerNodes::default()
        };
        let one = with_account_shared(&native, "first", &first, &account).expect("first account");
        let two = decoded(
            &with_account_shared(&one, "second", &second, &account).expect("second account"),
        );
        assert_eq!(
            two.program.rules.len(),
            3,
            "one scalar body and two outer bindings"
        );
        let expected = two.execute(&family).expect("CPU").values[two.program.output].clone();
        let (expanded, _) = crate::artifact_device::expanded_artifact(&two).expect("expand calls");
        let mut devices = vec![gam_gpu::tensor::Device::host()];
        if let Some(device) = gam_gpu::tensor::Device::accelerator(gam_gpu::GpuPolicy::Auto)
            .expect("device discovery")
        {
            if device.float64() {
                devices.push(device);
            }
        }
        for device in devices {
            let program =
                crate::device_program::DeviceProgram::compile_values(&device, &expanded.program)
                    .expect("values compile");
            let trace = program.forward(&family).expect("resident execution");
            let actual = device
                .download(trace.value(program.hidden()).expect("output"))
                .expect("diagnostic download");
            assert!(
                actual
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| (a - b).abs() < 1e-10)
            );
            let mut edited = family.clone();
            let SlotValues::Raw(values) = &mut edited.slots[0] else {
                panic!("raw fixture")
            };
            values[[0, 0]] *= 0.3;
            let expected_edit = two
                .execute_edited(&family, |node, value, _| {
                    if node == two.place(0).expect("held input") {
                        value[[0, 0]] *= 0.3;
                    }
                    Ok(())
                })
                .expect("CPU input intervention")
                .values[two.program.output]
                .clone();
            let trace = program
                .forward(&edited)
                .expect("resident input intervention");
            let actual_edit = device
                .download(trace.value(program.hidden()).expect("output"))
                .expect("diagnostic download");
            assert!(
                actual_edit
                    .iter()
                    .zip(&expected_edit)
                    .all(|(a, b)| (a - b).abs() < 1e-10)
            );
        }
    }
}
