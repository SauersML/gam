//! Frozen numerical-body transfer to new, fully paid native-use bindings.
//! This preserves supplied GELU architecture and tests numerical reuse, not discovery.
use crate::{
    operator_program::{OperatorBody, OperatorProgram},
    run_check::LayerNodes,
    shared_geometry_pilot::{self, Arm, Proposal},
};

/// Instantiate fresh input/output bindings for disjoint uses while retaining every
/// discovery body coefficient (including its offset) and its rule verbatim.
pub fn build(
    discovery: &Proposal,
    native: &OperatorProgram,
    layers: &[LayerNodes],
    uses: &[usize],
    seed: u64,
) -> Result<Proposal, String> {
    if uses.iter().any(|id| discovery.uses.contains(id)) {
        return Err("transfer uses must be disjoint from discovery uses".into());
    }
    discovery.program.interfaces().map_err(|e| e.to_string())?;
    if discovery.body_operators != [0, 1] || discovery.program.rules.len() != 1 {
        return Err("transfer requires the declared shared geometry body pool".into());
    }
    let mut transfer = shared_geometry_pilot::build(native, layers, uses, Arm::FrozenNative, seed)?;
    let source_rule = &discovery.program.rules[0];
    let fresh_rule = &transfer.program.rules[0];
    if source_rule.inputs != fresh_rule.inputs
        || source_rule.nodes != fresh_rule.nodes
        || source_rule.output != fresh_rule.output
    {
        return Err(
            "discovery body architecture differs from transfer native interfaces/law".into(),
        );
    }
    for &id in &discovery.body_operators {
        let source = discovery
            .program
            .operators
            .get(id)
            .ok_or("discovery body operator absent")?;
        let fresh = transfer
            .program
            .operators
            .get(id)
            .ok_or("transfer body operator absent")?;
        if source.rows != fresh.rows
            || source.cols != fresh.cols
            || !matches!(source.body, OperatorBody::Dense { .. })
        {
            return Err("discovery body operator interface/kind mismatch".into());
        }
        transfer.program.operators[id] = source.clone();
    }
    transfer.program.rules[0] = source_rule.clone();
    verify_frozen(discovery, &transfer)?;
    Ok(transfer)
}

/// Check a fitted or ordinarily decoded transfer program against the discovery
/// snapshot. Numeric bits, masks, precision and rule syntax must remain identical;
/// decoded display names/provenance need not match. No body ID may be trainable.
pub fn verify_frozen(discovery: &Proposal, transfer: &Proposal) -> Result<(), String> {
    if discovery.body_operators != [0, 1]
        || transfer.body_operators != discovery.body_operators
        || discovery.program.rules.len() != 1
        || transfer.program.rules.len() != 1
    {
        return Err("frozen body pool differs".into());
    }
    if transfer
        .trainable
        .iter()
        .any(|id| transfer.body_operators.contains(id))
    {
        return Err("frozen body operator is trainable".into());
    }
    let a = &discovery.program.rules[0];
    let b = &transfer.program.rules[0];
    if a.inputs != b.inputs || a.nodes != b.nodes || a.output != b.output {
        return Err("frozen body rule changed".into());
    }
    for &id in &discovery.body_operators {
        let a = discovery
            .program
            .operators
            .get(id)
            .ok_or("discovery body absent")?;
        let b = transfer
            .program
            .operators
            .get(id)
            .ok_or("transfer body absent")?;
        if a.rows != b.rows || a.cols != b.cols {
            return Err("frozen body interface changed".into());
        }
        match (&a.body, &b.body) {
            (
                OperatorBody::Dense {
                    values: av,
                    present: ap,
                    precision: aq,
                },
                OperatorBody::Dense {
                    values: bv,
                    present: bp,
                    precision: bq,
                },
            ) if av.dim() == bv.dim()
                && ap == bp
                && aq == bq
                && av.iter().zip(bv).all(|(x, y)| x.to_bits() == y.to_bits()) => {}
            _ => return Err("frozen body numeric bits/mask/precision changed".into()),
        }
    }
    transfer.program.interfaces().map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator_program::{
        Declarations, FamilyInputs, Interface, Law, Node, Operator, Slot, SlotValues,
        exact_precision,
    };
    use ndarray::Array2;
    use std::sync::Arc;
    fn native() -> (OperatorProgram, Vec<LayerNodes>) {
        let mut p = OperatorProgram {
            declarations: Declarations {
                domains: vec![],
                slots: vec![Slot::Raw { width: 2 }],
                parameters: 0,
            },
            bases: vec![],
            operators: vec![],
            rules: vec![],
            nodes: vec![Node::Raw { slot: 0 }],
            output: 0,
        };
        let mut layers = vec![];
        for layer in 0..4 {
            let op = p.operators.len();
            for (r, c) in [(3, 2), (2, 3)] {
                let values =
                    Array2::from_shape_fn((r, c), |(i, j)| (layer + i + j + 1) as f64 / 8.0);
                p.operators.push(Arc::new(
                    Operator::dense(
                        "native",
                        Interface::native(r).expect("rows"),
                        Interface::native(c).expect("cols"),
                        values.clone(),
                        exact_precision(values.iter().copied()).expect("precision"),
                        Default::default(),
                    )
                    .expect("operator"),
                ));
            }
            let pre = p.nodes.len();
            p.nodes.push(Node::Affine {
                terms: vec![(0, op)],
                bias: None,
            });
            let active = p.nodes.len();
            p.nodes.push(Node::Pointwise {
                input: pre,
                laws: vec![Law::GeluTanh],
            });
            let mlp = p.nodes.len();
            p.nodes.push(Node::Affine {
                terms: vec![(active, op + 1)],
                bias: None,
            });
            layers.push(LayerNodes {
                stream: 0,
                normed_stream: 0,
                queries: vec![],
                keys: vec![],
                values: vec![],
                reads: vec![],
                attention: 0,
                attended: 0,
                normed: 0,
                pre,
                active,
                mlp,
                residual: mlp,
            });
        }
        p.output = p.nodes.len() - 1;
        (p, layers)
    }
    #[test]
    fn new_uses_retain_body_arcs_and_ordinary_replay_cost() {
        let (native, layers) = native();
        let discovery = shared_geometry_pilot::build(&native, &layers, &[0, 1], Arm::Learned, 7)
            .expect("discovery");
        let mut transfer = build(&discovery, &native, &layers, &[2, 3], 7).expect("transfer");
        assert_eq!(transfer.trainable, (2..10).collect::<Vec<_>>());
        for id in [0, 1] {
            assert!(Arc::ptr_eq(
                &discovery.program.operators[id],
                &transfer.program.operators[id]
            ));
        }
        let inputs = FamilyInputs {
            rows: 2,
            slots: vec![
                SlotValues::Raw(
                    Array2::from_shape_vec((2, 2), vec![0.2, -0.3, 0.7, 0.1]).expect("input")
                );
                2
            ],
            layout: None,
        };
        let expected = transfer
            .program
            .execute(&inputs, false)
            .expect("execute")
            .values[transfer.program.output]
            .clone();
        let message = transfer.program.encode().expect("encode");
        transfer.program = OperatorProgram::decode(&message, &transfer.program.declarations)
            .expect("ordinary decode");
        verify_frozen(&discovery, &transfer).expect("all body bits frozen after replay");
        assert_eq!(transfer.program.encode().expect("reencode"), message);
        assert_eq!(
            transfer
                .program
                .execute(&inputs, false)
                .expect("replay execution")
                .values[transfer.program.output],
            expected
        );
        let standalone =
            crate::artifact::Artifact::native(&transfer.program).expect("standalone artifact");
        let ordinary = crate::artifact::Artifact::from_bytes(
            &standalone.to_bytes().expect("standalone bytes"),
            &transfer.program.declarations,
        )
        .expect("standalone decode");
        assert_eq!(
            crate::acceptance::structural_cost(&standalone, &mut Default::default()).expect("cost"),
            crate::acceptance::structural_cost(&ordinary, &mut Default::default())
                .expect("decoded cost")
        );
        let fresh = shared_geometry_pilot::build(&native, &layers, &[2, 3], Arm::FrozenNative, 7)
            .expect("fresh");
        assert_ne!(
            fresh.program.operators[0].matrix(),
            discovery.program.operators[0].matrix()
        );
        assert_eq!(transfer.program.operators.len(), 10);
        assert!(message.len_bits() > 0);
        transfer.trainable.push(1);
        assert!(verify_frozen(&discovery, &transfer).is_err());
        assert!(build(&discovery, &native, &layers, &[1, 2], 7).is_err());
    }
    #[test]
    fn offset_bits_cannot_change_silently() {
        let (native, layers) = native();
        let discovery = shared_geometry_pilot::build(&native, &layers, &[0, 1], Arm::Learned, 7)
            .expect("discovery");
        let mut transfer = build(&discovery, &native, &layers, &[2, 3], 7).expect("transfer");
        let body = Arc::make_mut(&mut transfer.program.operators[1]);
        if let OperatorBody::Dense { values, .. } = &mut body.body {
            values[(0, 0)] = -0.0;
        }
        assert!(verify_frozen(&discovery, &transfer).is_err());
    }
}
