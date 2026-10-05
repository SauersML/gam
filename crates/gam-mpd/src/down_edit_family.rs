//! A declared finite-coordinate native down-weight edit family in ordinary IR.
//! Controls are globally uniform scalar inputs, not activation masks. Floating-point
//! execution is measured against literal edits; real-algebra equivalence is not bit parity.
use crate::{
    artifact::Artifact,
    operator_program::{
        FamilyInputs, Interface, Node, Operator, OperatorBody, OperatorProgram, Slot, SlotValues,
        exact_precision, remap_node,
    },
};
use ndarray::{Array1, Array2};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct Direction {
    pub output: Array1<f64>,
    pub hidden: Array1<f64>,
}
pub struct Family {
    pub program: OperatorProgram,
    pub node_mapping: Vec<usize>,
    pub control_nodes: Vec<usize>,
    pub control_slots: Vec<usize>,
    pub native_read: usize,
    pub native_write: usize,
    pub direction_operators: Vec<(usize, usize)>,
    original: OperatorProgram,
    target_operator: usize,
}
impl Family {
    /// Same scalar on EVERY row: a genuine shared weight edit, not rowwise activation editing.
    pub fn inputs(&self, base: &FamilyInputs, amplitudes: &[f64]) -> Result<FamilyInputs, String> {
        if amplitudes.len() != self.control_slots.len()
            || amplitudes.iter().any(|a| !a.is_finite())
            || base.rows == 0
            || base.slots.len() != self.original.declarations.slots.len()
        {
            return Err("invalid family controls/base slots".into());
        }
        for (slot, values) in self.original.declarations.slots.iter().zip(&base.slots) {
            let valid = match (slot, values) {
                (Slot::Raw { width }, SlotValues::Raw(x)) => {
                    x.dim() == (base.rows, *width) && x.iter().all(|v| v.is_finite())
                }
                (Slot::Token { domain }, SlotValues::Tokens(x)) => {
                    x.len() == base.rows
                        && x.iter().all(|t| {
                            (*t as usize) < self.original.declarations.domains[*domain].size
                        })
                }
                _ => false,
            };
            if !valid {
                return Err("invalid base family slot".into());
            }
        }
        if base
            .layout
            .as_ref()
            .is_some_and(|l| l.sequence.len() != base.rows || l.position.len() != base.rows)
        {
            return Err("invalid family layout".into());
        }
        let mut out = base.clone();
        for (&slot, &a) in self.control_slots.iter().zip(amplitudes) {
            if slot != out.slots.len() {
                return Err("control slot mismatch".into());
            }
            out.slots
                .push(SlotValues::Raw(Array2::from_elem((base.rows, 1), a)));
        }
        Ok(out)
    }
    /// Independent original graph with its actual stored down matrix edited.
    pub fn literal_native(&self, amplitudes: &[f64]) -> Result<OperatorProgram, String> {
        if amplitudes.len() != self.direction_operators.len()
            || amplitudes.iter().any(|a| !a.is_finite())
        {
            return Err("invalid literal controls".into());
        }
        let mut p = self.original.clone();
        let mut op = (*p.operators[self.target_operator]).clone();
        let OperatorBody::Dense {
            values, present, ..
        } = &op.body
        else {
            return Err("dense target required".into());
        };
        let mut edited = values.clone();
        for (&a, &(u, v)) in amplitudes.iter().zip(&self.direction_operators) {
            let OperatorBody::Dense { values: u, .. } = &self.program.operators[u].body else {
                return Err("dense direction required".into());
            };
            let OperatorBody::Dense { values: v, .. } = &self.program.operators[v].body else {
                return Err("dense direction required".into());
            };
            for row in 0..edited.nrows() {
                for col in 0..edited.ncols() {
                    edited[[row, col]] += a * u[[row, 0]] * v[[0, col]];
                }
            }
        }
        if edited.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite literal edit".into());
        }
        let precision = exact_precision(edited.iter().copied()).map_err(|e| e.to_string())?;
        op.body = OperatorBody::Dense {
            values: edited,
            present: present.clone(),
            precision,
        };
        p.operators[self.target_operator] = Arc::new(op);
        Ok(p)
    }
    /// Explicit stored family-definition literals after replacement/pruning.
    /// Ordinary program encoding and C32 include UNUSED operators too. Call after the
    /// last pruning operation. This stores numerical directions, not a hidden predictor.
    pub fn retain_directions(
        &self,
        artifact: &mut Artifact,
    ) -> Result<Vec<(usize, usize)>, String> {
        self.retain_directions_excluding(artifact, &std::collections::BTreeSet::new())
    }
    /// Never alias fixed family literals to independently fitted parameters, even
    /// when their current arrays happen to be equal.
    pub fn retain_directions_excluding(
        &self,
        artifact: &mut Artifact,
        trainable: &std::collections::BTreeSet<usize>,
    ) -> Result<Vec<(usize, usize)>, String> {
        if artifact.program.declarations != self.program.declarations {
            return Err("wrong augmented declarations".into());
        }
        let mut ids = vec![];
        for &(u, v) in &self.direction_operators {
            let mut retain = |source: usize| {
                let direction = &self.program.operators[source];
                let existing =
                    artifact
                        .program
                        .operators
                        .iter()
                        .enumerate()
                        .find_map(|(id, op)| {
                            if trainable.contains(&id) {
                                return None;
                            }
                            let matches = op.rows == direction.rows
                                && op.cols == direction.cols
                                && op.body == direction.body
                                && match (&op.body, &direction.body) {
                                    (
                                        OperatorBody::Dense { values: a, .. },
                                        OperatorBody::Dense { values: b, .. },
                                    ) => a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()),
                                    _ => false,
                                };
                            matches.then_some(id)
                        });
                if let Some(id) = existing {
                    id
                } else {
                    let id = artifact.program.operators.len();
                    artifact.program.operators.push(direction.clone());
                    id
                }
            };
            ids.push((retain(u), retain(v)));
        }
        Ok(ids)
    }
}
/// One direct, uniquely-used Dense down map only. The native block reads `read`;
/// its complete hidden computation must depend on that read and fixed constants only.
/// Directions are rank-one EDIT coordinates; this imposes no rank cap on payload/output.
pub fn build(
    native: &OperatorProgram,
    read: usize,
    write: usize,
    directions: &[Direction],
) -> Result<Family, String> {
    native.interfaces().map_err(|e| e.to_string())?;
    if read >= write || write >= native.nodes.len() || directions.is_empty() {
        return Err("invalid block or empty family".into());
    }
    let (active, target, bias) = match &native.nodes[write] {
        Node::Affine { terms, bias } if terms.len() == 1 => (terms[0].0, terms[0].1, *bias),
        _ => return Err("one direct down Affine required".into()),
    };
    if active <= read {
        return Err("hidden activation must follow declared block read".into());
    }
    if bias == Some(target) {
        return Err("down target also used as bias; unsupported".into());
    }
    let op = &native.operators[target];
    if !matches!(&op.body,OperatorBody::Dense{present,..} if present.iter().all(|p|*p)) {
        return Err("all-present Dense down target required".into());
    }
    let uses = native
        .nodes
        .iter()
        .chain(native.rules.iter().flat_map(|r| r.nodes.iter()))
        .filter(|n| n.operators().contains(&target))
        .count();
    if uses != 1 {
        return Err("shared native target requires all-use edit lowering; unsupported here".into());
    }
    let mut pending = vec![active];
    let mut seen = std::collections::BTreeSet::new();
    while let Some(node) = pending.pop() {
        if node == read || !seen.insert(node) {
            continue;
        }
        let n = &native.nodes[node];
        if n.arguments().is_empty() && !matches!(n, Node::Constant { .. }) {
            return Err("hidden state has an undeclared input".into());
        }
        pending.extend(n.arguments());
    }
    if native
        .rules
        .iter()
        .flat_map(|r| &r.nodes)
        .any(|n| matches!(n, Node::Raw { .. } | Node::Feature { .. }))
    {
        return Err("ambient rule inputs unsupported".into());
    }
    let mut p = native.clone();
    let mut control_nodes = vec![];
    let mut control_slots = vec![];
    p.nodes.clear();
    for _ in directions {
        let slot = p.declarations.slots.len();
        p.declarations.slots.push(Slot::Raw { width: 1 });
        control_slots.push(slot);
        control_nodes.push(p.nodes.len());
        p.nodes.push(Node::Raw { slot });
    }
    let mut pairs = vec![];
    for d in directions {
        if d.output.len() != op.rows.width()
            || d.hidden.len() != op.cols.width()
            || d.output
                .iter()
                .chain(d.hidden.iter())
                .any(|v| !v.is_finite())
            || !d.output.iter().any(|v| *v != 0.)
            || !d.hidden.iter().any(|v| *v != 0.)
        {
            return Err("invalid direction shape/values".into());
        }
        let u = Array2::from_shape_vec((d.output.len(), 1), d.output.to_vec())
            .map_err(|e| e.to_string())?;
        let v = Array2::from_shape_vec((1, d.hidden.len()), d.hidden.to_vec())
            .map_err(|e| e.to_string())?;
        let make = |rows, cols, values: Array2<f64>| -> Result<Arc<Operator>, String> {
            let precision = exact_precision(values.iter().copied()).map_err(|e| e.to_string())?;
            let result = Operator::dense(
                "declared edit direction",
                rows,
                cols,
                values.clone(),
                precision,
                Default::default(),
            )
            .map_err(|e| e.to_string())?;
            let OperatorBody::Dense { values: stored, .. } = &result.body else {
                return Err("dense direction construction".into());
            };
            if !stored
                .iter()
                .zip(&values)
                .all(|(a, b)| a.to_bits() == b.to_bits())
            {
                return Err("direction not exactly representable on declared precision".into());
            }
            Ok(Arc::new(result))
        };
        let u_id = p.operators.len();
        p.operators.push(make(
            op.rows.clone(),
            Interface::native(1).map_err(|e| e.to_string())?,
            u,
        )?);
        let v_id = p.operators.len();
        p.operators.push(make(
            Interface::native(1).map_err(|e| e.to_string())?,
            op.cols.clone(),
            v,
        )?);
        pairs.push((u_id, v_id));
    }
    let ops: Vec<_> = (0..p.operators.len()).collect();
    let bases: Vec<_> = (0..p.bases.len()).collect();
    let rules: Vec<_> = (0..p.rules.len()).collect();
    let mut map = vec![usize::MAX; native.nodes.len()];
    for (index, node) in native.nodes.iter().enumerate() {
        if index == write {
            let mut terms = vec![(map[active], target)];
            for (j, &(u, v)) in pairs.iter().enumerate() {
                let projection = p.nodes.len();
                p.nodes.push(Node::Affine {
                    terms: vec![(map[active], v)],
                    bias: None,
                });
                let gated = p.nodes.len();
                p.nodes.push(Node::Hadamard {
                    left: projection,
                    right: control_nodes[j],
                });
                terms.push((gated, u));
            }
            map[index] = p.nodes.len();
            p.nodes.push(Node::Affine { terms, bias });
        } else {
            let mut n = node.clone();
            remap_node(&mut n, &map, &ops, &bases, &rules);
            map[index] = p.nodes.len();
            p.nodes.push(n);
        }
    }
    p.output = map[native.output];
    p.interfaces().map_err(|e| e.to_string())?;
    Ok(Family {
        native_read: map[read],
        native_write: map[write],
        program: p,
        node_mapping: map,
        control_nodes,
        control_slots,
        direction_operators: pairs,
        original: native.clone(),
        target_operator: target,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        artifact::{Argument, Callee},
        operator_program::{Declarations, Law, Rule},
    };
    use ndarray::array;
    fn dense(rows: Interface, cols: Interface, values: Array2<f64>) -> Arc<Operator> {
        let precision = exact_precision(values.iter().copied()).expect("precision");
        Arc::new(
            Operator::dense("native", rows, cols, values, precision, Default::default())
                .expect("dense"),
        )
    }
    fn native() -> OperatorProgram {
        let i = Interface::native(2).expect("width");
        let h = Interface::native(3).expect("hidden");
        OperatorProgram {
            declarations: Declarations {
                domains: vec![],
                slots: vec![Slot::Raw { width: 2 }],
                parameters: 0,
            },
            bases: vec![],
            rules: vec![],
            operators: vec![
                dense(
                    h.clone(),
                    i.clone(),
                    array![[1., -0.5], [0.25, 2.], [-1., 1.]],
                ),
                dense(i.clone(), h, array![[1., 0.5, -1.], [2., -0.25, 0.5]]),
                Arc::new(Operator::identity("tail", i)),
            ],
            nodes: vec![
                Node::Raw { slot: 0 },
                Node::Affine {
                    terms: vec![(0, 0)],
                    bias: None,
                },
                Node::Pointwise {
                    input: 1,
                    laws: vec![Law::Gelu],
                },
                Node::Affine {
                    terms: vec![(2, 1)],
                    bias: None,
                },
                Node::Affine {
                    terms: vec![(3, 2), (0, 2)],
                    bias: None,
                },
            ],
            output: 4,
        }
    }
    fn base() -> FamilyInputs {
        FamilyInputs {
            rows: 3,
            slots: vec![SlotValues::Raw(array![[1., -2.], [0.5, 3.], [-1., 0.25]])],
            layout: None,
        }
    }
    fn family() -> Family {
        build(
            &native(),
            0,
            3,
            &[
                Direction {
                    output: array![0.5, -1.],
                    hidden: array![1., -0.25, 0.5],
                },
                Direction {
                    output: array![-1., 0.25],
                    hidden: array![0.25, 0.5, -1.],
                },
            ],
        )
        .expect("family")
    }
    #[test]
    fn augmented_full_program_matches_literal_weight_edits_and_zero_control() {
        let f = family();
        assert_eq!(f.program.operators[..3], native().operators);
        for amplitudes in [[0., 0.], [0.25, -0.5], [-1., 1.]] {
            let inputs = f.inputs(&base(), &amplitudes).expect("inputs");
            let augmented = f.program.execute(&inputs, false).expect("augmented");
            let literal = f.literal_native(&amplitudes).expect("literal");
            let target = literal.execute(&base(), false).expect("target");
            for (index, &mapped) in f.node_mapping.iter().enumerate() {
                assert!(
                    augmented.values[mapped]
                        .iter()
                        .zip(&target.values[index])
                        .all(|(a, b)| (a - b).abs() < 1e-12)
                );
            }
            let a = Artifact::native(&f.program).expect("artifact");
            let bytes = a.to_bytes().expect("encode");
            let decoded = Artifact::from_bytes(&bytes, &f.program.declarations).expect("decode");
            decoded.validate_coverage(&f.program).expect("coverage");
            assert_eq!(
                decoded.execute(&inputs).expect("saved").values[decoded.program.output],
                augmented.values[f.program.output]
            );
        }
        assert!(f.inputs(&base(), &[f64::NAN, 0.]).is_err());
        assert!(f.inputs(&base(), &[0.]).is_err());
    }
    #[test]
    fn independent_contexts_and_new_amplitudes_match_literal_native_edits() {
        let f = family();
        let heldout = FamilyInputs {
            rows: 2,
            slots: vec![SlotValues::Raw(array![[-3.5, 0.75], [2.25, 1.5]])],
            layout: None,
        };
        let amplitudes = [0.75, -0.125];
        let augmented_inputs = f.inputs(&heldout, &amplitudes).expect("new inputs");
        let actual = f
            .program
            .execute(&augmented_inputs, false)
            .expect("augmented");
        let literal = f.literal_native(&amplitudes).expect("literal");
        let reference = literal
            .execute(&heldout, false)
            .expect("native weight edit");
        assert!(
            actual.values[f.program.output]
                .iter()
                .zip(&reference.values[literal.output])
                .all(|(a, b)| (a - b).abs() < 1e-12)
        );
    }
    #[test]
    fn replacement_uses_own_input_controls_and_preserves_priced_direction_literals() {
        let f = family();
        let a = Artifact::native(&f.program).expect("artifact");
        let i = Interface::native(2).expect("width");
        let scalar = Interface::native(1).expect("scalar");
        let id = a.program.operators.len();
        let response = Rule {
            name: "learned response fixture".into(),
            inputs: vec![i.clone(), scalar.clone(), scalar],
            nodes: vec![
                Node::Param { index: 0 },
                Node::Param { index: 1 },
                Node::Param { index: 2 },
                Node::Affine {
                    terms: vec![(1, id)],
                    bias: None,
                },
                Node::Affine {
                    terms: vec![(0, id + 1), (3, id + 1)],
                    bias: None,
                },
            ],
            output: 4,
        };
        let mut candidate = a
            .replace_block(
                "entire augmented MLP",
                Callee::New(response),
                vec![
                    Argument::Native(f.native_read),
                    Argument::Native(f.control_nodes[0]),
                    Argument::Native(f.control_nodes[1]),
                ],
                f.native_write,
                vec![
                    (*dense(
                        i.clone(),
                        Interface::native(1).expect("scalar"),
                        array![[0.5], [-1.]],
                    ))
                    .clone(),
                    Operator::identity("reader", i),
                ],
            )
            .expect("replacement");
        candidate.validate_coverage(&f.program).expect("coverage");
        assert!(candidate.place(f.node_mapping[1]).is_none());
        assert!(candidate.place(f.node_mapping[2]).is_none());
        let before_c32 =
            crate::acceptance::structural_cost(&candidate, &mut Default::default()).expect("C32");
        let before = candidate.program.code_bits().expect("cost");
        let ids = f
            .retain_directions(&mut candidate)
            .expect("retained directions");
        let after = candidate.program.code_bits().expect("cost");
        assert!(after > before);
        let after_c32 =
            crate::acceptance::structural_cost(&candidate, &mut Default::default()).expect("C32");
        assert_eq!(after_c32.literals - before_c32.literals, 8); // Ten direction literals, minus the already-paid two-literal writer.
        assert!(after_c32.total() > before_c32.total());
        assert_eq!(
            f.retain_directions(&mut candidate)
                .expect("idempotent retention"),
            ids
        );
        assert_eq!(candidate.program.code_bits().expect("stable cost"), after);
        let bytes = candidate.to_bytes().expect("encode");
        let saved = Artifact::from_bytes(&bytes, &f.program.declarations).expect("decode");
        saved.validate_coverage(&f.program).expect("saved coverage");
        assert_eq!(
            crate::acceptance::structural_cost(&saved, &mut Default::default()).expect("saved C32"),
            after_c32
        );
        for (&(u, v), &(su, sv)) in f.direction_operators.iter().zip(&ids) {
            assert_eq!(
                saved.program.operators[su].body,
                f.program.operators[u].body
            );
            assert_eq!(
                saved.program.operators[sv].body,
                f.program.operators[v].body
            );
        }
        let x = f.inputs(&base(), &[0.25, -0.5]).expect("inputs");
        assert_eq!(
            candidate.execute(&x).expect("candidate").values[candidate.program.output],
            saved.execute(&x).expect("saved").values[saved.program.output]
        );
    }
    #[test]
    fn refuses_shared_target_undeclared_inputs_and_bad_directions() {
        let mut p = native();
        p.nodes.push(Node::Affine {
            terms: vec![(2, 1)],
            bias: None,
        });
        p.output = 5;
        let d = Direction {
            output: array![1., 0.],
            hidden: array![1., 0., 0.],
        };
        assert!(build(&p, 0, 3, std::slice::from_ref(&d)).is_err());
        assert!(build(&native(), 1, 3, std::slice::from_ref(&d)).is_ok());
        assert!(
            build(
                &native(),
                0,
                3,
                &[Direction {
                    output: array![1.],
                    hidden: array![1., 0., 0.]
                }]
            )
            .is_err()
        );
        let mut external = native();
        external.nodes[1] = Node::Raw { slot: 0 };
        assert!(build(&external, 0, 3, std::slice::from_ref(&d)).is_err());
        assert!(build(&native(), 0, 3, &[]).is_err());
    }
    #[test]
    fn fixed_direction_never_aliases_equal_independently_fitted_writer() {
        let f = family();
        let source_v = f.direction_operators[0].1;
        let mut p = f.program.clone();
        p.nodes = vec![Node::Raw { slot: 0 }];
        p.output = 0;
        p.rules.clear();
        p.operators = vec![f.program.operators[source_v].clone()];
        let mut candidate = Artifact::native(&p).expect("stored learned writer");
        let fixed = f
            .retain_directions_excluding(&mut candidate, &std::collections::BTreeSet::from([0]))
            .expect("fixed family literals");
        assert_ne!(fixed[0].1, 0);
        let before = candidate.program.operators[fixed[0].1].clone();
        let mut learned = (*candidate.program.operators[0]).clone();
        let OperatorBody::Dense { values, .. } = &mut learned.body else {
            panic!("dense test writer")
        };
        values.fill(0.);
        candidate.program.operators[0] = Arc::new(learned);
        assert_eq!(candidate.program.operators[fixed[0].1], before);
        assert_eq!(
            f.retain_directions_excluding(&mut candidate, &std::collections::BTreeSet::from([0]))
                .expect("stable retention"),
            fixed
        );
    }
}
