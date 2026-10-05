use super::*;
use crate::operator_program::{
    Declarations, FamilyInputs, LabelKind, Law, OperatorProgram, Slot, SlotValues, exact_precision,
};
use ndarray::{Array2, array};

fn settings() -> Settings {
    Settings {
        max_edit_checks: 512,
        max_proposals: 128,
        max_body_nodes: 32,
        max_parameter_elements: 128,
    }
}

fn dense(name: &str, rows: Interface, cols: Interface, values: Array2<f64>) -> Arc<Operator> {
    Arc::new(
        Operator::dense(
            name,
            rows,
            cols,
            values.clone(),
            exact_precision(values.iter().copied()).unwrap(),
            Default::default(),
        )
        .unwrap(),
    )
}

/// Deliberately uses independent bias owners for a shared gate/value matrix;
/// the downstream map is bias-free. This cannot be encoded as one global bias
/// choice or as a sharing partition of bundled matrix+bias affine owners.
fn fixture(gated: bool, outside_shared_use: bool) -> (Artifact, Region) {
    let input = Interface::native(2).unwrap();
    let hidden = Interface::uniform(2, 1, LabelKind::Unit, 0).unwrap();
    let operators = vec![
        dense(
            "gate/value",
            hidden.clone(),
            input.clone(),
            array![[1., 0.5], [-0.5, 2.]],
        ),
        dense(
            "gate bias",
            hidden.clone(),
            Interface::constant(),
            array![[0.25], [-0.25]],
        ),
        dense(
            "value bias",
            hidden.clone(),
            Interface::constant(),
            array![[-0.5], [0.75]],
        ),
        dense(
            "down",
            input.clone(),
            hidden,
            array![[2., -0.5], [0.25, 1.]],
        ),
    ];
    let mut nodes = vec![
        Node::Raw { slot: 0 },
        Node::Affine {
            terms: vec![(0, 0)],
            bias: Some(1),
        },
        Node::Pointwise {
            input: 1,
            laws: vec![if gated { Law::Silu } else { Law::GeluTanh }; 2],
        },
    ];
    let active = if gated {
        nodes.push(Node::Affine {
            terms: vec![(0, 0)],
            bias: Some(2),
        });
        nodes.push(Node::Hadamard { left: 2, right: 3 });
        4
    } else {
        2
    };
    let down = nodes.len();
    nodes.push(Node::Affine {
        terms: vec![(active, 3)],
        bias: None,
    });
    let mut output = down;
    if outside_shared_use {
        let external = nodes.len();
        nodes.push(Node::Affine {
            terms: vec![(0, 0)],
            bias: None,
        });
        output = nodes.len();
        nodes.push(Node::Concat {
            parts: vec![down, external],
        });
    }
    let program = OperatorProgram {
        declarations: Declarations {
            domains: vec![],
            slots: vec![Slot::Raw { width: 2 }],
            parameters: 0,
        },
        operators,
        nodes,
        output,
        bases: vec![],
        rules: vec![],
    };
    let source = Artifact::native(&program).unwrap();
    let region = Region {
        anchor_mode: joint::AnchorMode::Observable,
        native_reads: vec![0],
        current_reads: vec![0],
        native_writes: vec![active, down],
        current_writes: vec![active, down],
        current_internal_nodes: (1..=down).collect(),
        producer_native: active,
        erased_native_places: (1..=down).filter(|n| *n != active && *n != down).collect(),
        source_program_nodes: source.program.nodes.len(),
    };
    joint::exact_body(&source, &region).unwrap();
    (source, region)
}

fn family() -> FamilyInputs {
    FamilyInputs {
        rows: 3,
        slots: vec![SlotValues::Raw(array![[-1., 0.5], [0., 0.], [2., -0.75]])],
        layout: None,
    }
}

#[test]
fn exact_gelu_and_gated_controls_preserve_coefficients_and_all_exits() {
    for gated in [false, true] {
        let (source, region) = fixture(gated, false);
        let inventory = enumerate(&source, &region, &settings()).unwrap();
        assert!(inventory.control.edit.is_none());
        assert!(inventory.proposals.iter().all(|p| p.edit.is_some()));
        let result = apply(&source, &region, &inventory.control, &settings()).unwrap();
        assert_eq!(result.initialization.random_elements, 0);
        assert_eq!(
            result.initialization.inherited_elements,
            inventory.control.parameter_elements
        );
        for (local, owner) in inventory.control.owners.iter().enumerate() {
            assert_eq!(
                result.local_fit.program.operators[local],
                source.program.operators[owner.parent_operator]
            );
        }
        let native = source.program.execute(&family(), false).unwrap();
        let local = result.local_fit.program.execute(&family(), false).unwrap();
        let grafted = result.artifact.program.execute(&family(), false).unwrap();
        for ((&native_node, &local_node), &native_place) in region
            .current_writes
            .iter()
            .zip(&result.local_fit.output_nodes)
            .zip(&region.native_writes)
        {
            assert_eq!(native.values[native_node], local.values[local_node]);
            assert_eq!(
                native.values[native_node],
                grafted.values[result.artifact.place(native_place).unwrap()]
            );
        }
    }
}

#[test]
fn independent_matrix_and_bias_owners_survive_mixed_bias_control() {
    let (source, region) = fixture(true, false);
    let inventory = enumerate(&source, &region, &settings()).unwrap();
    assert_eq!(inventory.control.owners.len(), 4);
    assert_eq!(inventory.control.parameter_elements, 12);
    let result = apply(&source, &region, &inventory.control, &settings()).unwrap();
    let affine = result
        .local_fit
        .program
        .nodes
        .iter()
        .filter_map(|node| match node {
            Node::Affine { terms, bias } => Some((terms, bias)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(affine.len(), 3);
    assert_eq!(
        affine[0].0[0].1, affine[1].0[0].1,
        "one gate/value matrix owner"
    );
    assert_ne!(affine[0].1, affine[1].1, "independent nonzero bias owners");
    assert!(
        affine[2].1.is_none(),
        "native bias-free down map stays bias-free"
    );
    assert_eq!(result.local_fit.trainable_operator_ids.len(), 4);
}

#[test]
fn inner_edge_edit_inherits_untouched_down_matrix_and_is_not_control() {
    let (source, region) = fixture(true, false);
    let inventory = enumerate(&source, &region, &settings()).unwrap();
    let proposal = inventory
        .proposals
        .iter()
        .find(|p| {
            p.edit
                == Some(Reuse {
                    consumer: 4,
                    operand: 1,
                    previous: 3,
                    replacement: 2,
                })
        })
        .expect("reuse gate activation as product's other factor");
    let result = apply(&source, &region, proposal, &settings()).unwrap();
    let local_down = proposal
        .owners
        .iter()
        .position(|o| o.parent_operator == 3)
        .unwrap();
    assert_eq!(
        result.local_fit.program.operators[local_down],
        source.program.operators[3]
    );
    assert_eq!(result.initialization.random_elements, 0);
    assert!(
        !proposal.owners.iter().any(|o| o.parent_operator == 2),
        "unread value bias is removed"
    );
    let product = result
        .local_fit
        .program
        .nodes
        .iter()
        .find_map(|node| match node {
            Node::Hadamard { left, right } => Some((*left, *right)),
            _ => None,
        })
        .unwrap();
    assert_eq!(product.0, product.1, "one shared internal computation");
    let native = source.program.execute(&family(), false).unwrap();
    let edited = result.local_fit.program.execute(&family(), false).unwrap();
    assert_ne!(
        native.values[5], edited.values[result.local_fit.output_nodes[1]],
        "a changed hypothesis must not be labeled an exact preservation control"
    );
}

#[test]
fn external_shared_owner_stays_frozen_and_same_grafted_identity() {
    let (source, region) = fixture(true, true);
    let inventory = enumerate(&source, &region, &settings()).unwrap();
    let shared = inventory
        .control
        .owners
        .iter()
        .position(|o| o.parent_operator == 0)
        .unwrap();
    assert!(!inventory.control.owners[shared].trainable);
    let result = apply(&source, &region, &inventory.control, &settings()).unwrap();
    assert!(!result.local_fit.trainable_operator_ids.contains(&shared));
    assert_eq!(
        result
            .artifact
            .program
            .operators
            .iter()
            .filter(|o| o.name == "gate/value")
            .count(),
        1
    );
    let native = source.program.execute(&family(), false).unwrap();
    let grafted = result.artifact.program.execute(&family(), false).unwrap();
    assert_eq!(
        native.values[source.program.output],
        grafted.values[result.artifact.program.output]
    );
    let mut bad_fit = result.local_fit.program.clone();
    let mut changed = bad_fit.operators[shared].as_ref().clone();
    if let OperatorBody::Dense { values, .. } = &mut changed.body {
        values[[0, 0]] += 1.;
    }
    bad_fit.operators[shared] = Arc::new(changed);
    let mut candidate = result.artifact.clone();
    assert!(result.local_fit.transfer(&bad_fit, &mut candidate).is_err());
}

#[test]
fn independent_same_shaped_native_owners_do_not_merge() {
    let (mut source, region) = fixture(true, false);
    let mut distinct = source.program.operators[0].as_ref().clone();
    distinct.name = "different owner with identical coefficients".into();
    let id = source.program.operators.len();
    source.program.operators.push(Arc::new(distinct));
    let Node::Affine { terms, .. } = &mut source.program.nodes[3] else {
        unreachable!()
    };
    terms[0].1 = id;
    let inventory = enumerate(&source, &region, &settings()).unwrap();
    assert!(
        inventory
            .control
            .owners
            .iter()
            .any(|o| o.parent_operator == 0)
    );
    assert!(
        inventory
            .control
            .owners
            .iter()
            .any(|o| o.parent_operator == id)
    );
    assert_eq!(inventory.control.owners.len(), 5);
    let result = apply(&source, &region, &inventory.control, &settings()).unwrap();
    assert_eq!(result.local_fit.trainable_operator_ids.len(), 5);
}

#[test]
fn edits_are_typed_acyclic_bounded_and_deterministic() {
    let (source, region) = fixture(true, false);
    let mut small = settings();
    small.max_edit_checks = 5;
    small.max_proposals = 2;
    let first = enumerate(&source, &region, &small).unwrap();
    let second = enumerate(&source, &region, &small).unwrap();
    assert_eq!(
        serde_json::to_string(&first).unwrap(),
        serde_json::to_string(&second).unwrap()
    );
    assert!(first.truncated);
    assert!(first.checked_edits <= small.max_edit_checks);
    assert!(first.proposals.len() <= small.max_proposals);
    assert!(first.rejection_counts.values().sum::<usize>() > 0);
    let parent = extract(&source, &region).unwrap();
    assert!(
        materialize(
            &source,
            &parent,
            Some(&Reuse {
                consumer: 4,
                operand: 1,
                previous: 3,
                replacement: 0,
            })
        )
        .unwrap_err()
        .contains("interfaces differ")
    );
    assert!(
        materialize(
            &source,
            &parent,
            Some(&Reuse {
                consumer: 4,
                operand: 1,
                previous: 3,
                replacement: 5,
            })
        )
        .unwrap_err()
        .contains("forward reference")
    );
    let mut forged = enumerate(&source, &region, &settings()).unwrap().control;
    forged.owners[0].parent_operator = 3;
    assert!(apply(&source, &region, &forged, &settings()).is_err());
}

#[test]
fn changed_source_bits_or_laws_reject_old_proposals_and_control_cannot_be_hypothesis() {
    let (mut source, region) = fixture(true, false);
    let inventory = enumerate(&source, &region, &settings()).unwrap();
    assert!(apply_hypothesis(&source, &region, &inventory.control, &settings()).is_err());
    let proposal = inventory.proposals.first().unwrap();
    let old = Arc::clone(&source.program.operators[3]);
    let mut changed = old.as_ref().clone();
    if let OperatorBody::Dense { values, .. } = &mut changed.body {
        values[[1, 1]] += 0.25;
    }
    source.program.operators[3] = Arc::new(changed);
    assert!(apply_hypothesis(&source, &region, proposal, &settings()).is_err());
    source.program.operators[3] = old;
    let Node::Pointwise { laws, .. } = &mut source.program.nodes[2] else {
        unreachable!()
    };
    laws[0] = Law::Relu;
    assert!(apply_hypothesis(&source, &region, proposal, &settings()).is_err());
}

#[test]
fn removed_argument_dependency_preserves_complete_input_abi() {
    let (mut source, _) = fixture(true, false);
    source
        .program
        .declarations
        .slots
        .push(Slot::Raw { width: 2 });
    source.program.nodes = vec![
        Node::Raw { slot: 0 },
        Node::Raw { slot: 1 },
        Node::Affine {
            terms: vec![(0, 0)],
            bias: Some(1),
        },
        Node::Affine {
            terms: vec![(1, 0)],
            bias: Some(2),
        },
        Node::Hadamard { left: 2, right: 3 },
        Node::Affine {
            terms: vec![(4, 3)],
            bias: None,
        },
    ];
    source.program.output = 5;
    source = Artifact::native(&source.program).unwrap();
    let region = Region {
        anchor_mode: joint::AnchorMode::Observable,
        native_reads: vec![0, 1],
        current_reads: vec![0, 1],
        native_writes: vec![4, 5],
        current_writes: vec![4, 5],
        current_internal_nodes: vec![2, 3, 4, 5],
        producer_native: 4,
        erased_native_places: vec![2, 3],
        source_program_nodes: 6,
    };
    let inventory = enumerate(&source, &region, &settings()).unwrap();
    let proposal = inventory
        .proposals
        .iter()
        .find(|p| {
            p.edit
                == Some(Reuse {
                    consumer: 4,
                    operand: 1,
                    previous: 3,
                    replacement: 2,
                })
        })
        .unwrap();
    assert_eq!(proposal.used_arguments, vec![0]);
    let applied = apply_hypothesis(&source, &region, proposal, &settings()).unwrap();
    assert_eq!(applied.local_fit.input_native_places, vec![0, 1]);
    assert_eq!(applied.local_fit.program.declarations.slots.len(), 2);
    for slot in 0..2 {
        assert_eq!(
            applied
                .local_fit
                .program
                .nodes
                .iter()
                .filter(|n| matches!(n, Node::Raw { slot: held } if *held == slot))
                .count(),
            1
        );
    }
}

#[test]
fn multi_term_native_affine_keeps_frozen_identity_and_term_order() {
    let (mut source, region) = fixture(true, false);
    let identity = source.program.operators.len();
    source.program.operators.push(Arc::new(Operator::identity(
        "residual",
        Interface::native(2).unwrap(),
    )));
    let Node::Affine { terms, .. } = &mut source.program.nodes[5] else {
        unreachable!()
    };
    terms.insert(0, (0, identity));
    let inventory = enumerate(&source, &region, &settings()).unwrap();
    assert!(
        !inventory
            .control
            .owners
            .iter()
            .find(|o| o.parent_operator == identity)
            .unwrap()
            .trainable
    );
    let applied = apply(&source, &region, &inventory.control, &settings()).unwrap();
    let native = source.program.execute(&family(), false).unwrap();
    let local = applied.local_fit.program.execute(&family(), false).unwrap();
    assert_eq!(
        native.values[5],
        local.values[applied.local_fit.output_nodes[1]]
    );
}

#[test]
fn parent_over_target_budget_does_not_block_simplifying_edits() {
    let (source, region) = fixture(true, false);
    let mut small = settings();
    small.max_parameter_elements = 10;
    small.max_body_nodes = 5;
    let inventory = enumerate(&source, &region, &small).unwrap();
    assert!(inventory.control.parameter_elements > small.max_parameter_elements);
    assert!(inventory.control.compiled_node_count > small.max_body_nodes);
    assert!(!inventory.proposals.is_empty());
    assert!(
        inventory
            .proposals
            .iter()
            .all(|p| p.parameter_elements <= small.max_parameter_elements
                && p.compiled_node_count <= small.max_body_nodes)
    );
    apply(&source, &region, &inventory.control, &small).unwrap();
}

#[test]
fn masked_native_owners_remain_exact_and_frozen_for_existing_local_fit() {
    let (mut source, region) = fixture(false, false);
    let op = &source.program.operators[0];
    let OperatorBody::Dense {
        values,
        present,
        precision,
    } = &op.body
    else {
        unreachable!()
    };
    let mut mask = present.clone();
    mask[[0, 0]] = false;
    source.program.operators[0] = Arc::new(
        Operator::blocks(
            op.name.clone(),
            op.rows.clone(),
            op.cols.clone(),
            values.clone(),
            mask,
            *precision,
            op.provenance.clone(),
        )
        .unwrap(),
    );
    let inventory = enumerate(&source, &region, &settings()).unwrap();
    let local_id = inventory
        .control
        .owners
        .iter()
        .position(|o| o.parent_operator == 0)
        .unwrap();
    assert!(!inventory.control.owners[local_id].trainable);
    let applied = apply(&source, &region, &inventory.control, &settings()).unwrap();
    assert!(!applied.local_fit.trainable_operator_ids.contains(&local_id));
    assert_eq!(
        applied.local_fit.program.operators[local_id],
        source.program.operators[0]
    );
    let local = applied.local_fit.program.execute(&family(), false).unwrap();
    let native = source.program.execute(&family(), false).unwrap();
    for (&n, &l) in region
        .current_writes
        .iter()
        .zip(&applied.local_fit.output_nodes)
    {
        assert_eq!(native.values[n], local.values[l]);
    }
    for &id in &applied.local_fit.trainable_operator_ids {
        let OperatorBody::Dense { present, .. } = &applied.local_fit.program.operators[id].body
        else {
            unreachable!()
        };
        assert!(
            present.iter().all(|keep| *keep),
            "match existing local fitter eligibility"
        );
    }
}

fn trainable_sources(proposal: &Proposal) -> Vec<usize> {
    proposal
        .owners
        .iter()
        .filter_map(|owner| owner.trainable.then_some(owner.parent_operator))
        .collect()
}

#[test]
fn inner_edit_fits_affected_down_only_and_freezes_unchanged_donor() {
    let (source, region) = fixture(true, false);
    let inventory = enumerate(&source, &region, &settings()).unwrap();
    let proposal = inventory
        .proposals
        .iter()
        .find(|p| {
            p.edit
                == Some(Reuse {
                    consumer: 4,
                    operand: 1,
                    previous: 3,
                    replacement: 2,
                })
        })
        .unwrap();
    assert_eq!(trainable_sources(proposal), vec![3]);
    let applied = apply_hypothesis(&source, &region, proposal, &settings()).unwrap();
    let down = proposal
        .owners
        .iter()
        .position(|o| o.parent_operator == 3)
        .unwrap();
    let gate = proposal
        .owners
        .iter()
        .position(|o| o.parent_operator == 0)
        .unwrap();
    assert_eq!(applied.local_fit.trainable_operator_ids, vec![down]);
    assert_eq!(applied.local_fit.owner_mapping.len(), 1);
    assert_eq!(
        applied.trainable_operator_ids,
        vec![applied.local_fit.owner_mapping[0].1]
    );
    let mut fitted = applied.local_fit.program.clone();
    let mut changed = fitted.operators[gate].as_ref().clone();
    if let OperatorBody::Dense { values, .. } = &mut changed.body {
        values[[0, 0]] += 0.25;
    }
    fitted.operators[gate] = Arc::new(changed);
    let mut candidate = applied.artifact.clone();
    assert!(
        applied.local_fit.transfer(&fitted, &mut candidate).is_err(),
        "untouched donor is frozen in the actual local transfer contract"
    );
}

#[test]
fn direct_affine_edge_edit_trains_that_term_even_with_clean_donor() {
    let (source, region) = fixture(true, false);
    let inventory = enumerate(&source, &region, &settings()).unwrap();
    let proposal = inventory
        .proposals
        .iter()
        .find(|p| {
            p.edit
                == Some(Reuse {
                    consumer: 5,
                    operand: 0,
                    previous: 4,
                    replacement: 2,
                })
        })
        .unwrap();
    assert_eq!(
        trainable_sources(proposal),
        vec![3],
        "the edited affine input edge is dirty although its donor node stays clean"
    );
    assert!(
        proposal
            .owners
            .iter()
            .filter(|o| o.parent_operator != 3)
            .all(|o| !o.trainable)
    );
    apply_hypothesis(&source, &region, proposal, &settings()).unwrap();
}

#[test]
fn clean_affine_sibling_term_frozen_while_affected_bias_can_compensate() {
    let (mut source, region) = fixture(true, false);
    let residual = source.program.operators.len();
    source.program.operators.push(dense(
        "clean residual",
        Interface::native(2).unwrap(),
        Interface::native(2).unwrap(),
        array![[0.5, 0.25], [0., 0.5]],
    ));
    let bias = source.program.operators.len();
    source.program.operators.push(dense(
        "down bias",
        Interface::native(2).unwrap(),
        Interface::constant(),
        array![[0.25], [-0.5]],
    ));
    let Node::Affine {
        terms,
        bias: down_bias,
    } = &mut source.program.nodes[5]
    else {
        unreachable!()
    };
    terms.push((0, residual));
    *down_bias = Some(bias);
    let inventory = enumerate(&source, &region, &settings()).unwrap();
    let proposal = inventory
        .proposals
        .iter()
        .find(|p| {
            p.edit
                == Some(Reuse {
                    consumer: 4,
                    operand: 1,
                    previous: 3,
                    replacement: 2,
                })
        })
        .unwrap();
    assert_eq!(trainable_sources(proposal), vec![3, bias]);
    assert!(
        !proposal
            .owners
            .iter()
            .find(|o| o.parent_operator == residual)
            .unwrap()
            .trainable
    );
    apply_hypothesis(&source, &region, proposal, &settings()).unwrap();
}

#[test]
fn an_owner_with_both_dirty_and_clean_uses_remains_one_frozen_owner() {
    let (mut source, region) = fixture(true, false);
    let Node::Affine { terms, .. } = &mut source.program.nodes[5] else {
        unreachable!()
    };
    terms.push((2, 3));
    let inventory = enumerate(&source, &region, &settings()).unwrap();
    let proposal = inventory
        .proposals
        .iter()
        .find(|p| {
            p.edit
                == Some(Reuse {
                    consumer: 4,
                    operand: 1,
                    previous: 3,
                    replacement: 2,
                })
        })
        .unwrap();
    assert!(
        trainable_sources(proposal).is_empty(),
        "dirty product and clean gate activation share down owner, so the entire owner freezes"
    );
    let applied = apply_hypothesis(&source, &region, proposal, &settings()).unwrap();
    assert!(applied.local_fit.trainable_operator_ids.is_empty());
    assert!(applied.trainable_operator_ids.is_empty());
    assert_eq!(
        applied
            .artifact
            .program
            .operators
            .iter()
            .filter(|o| o.name == "down")
            .count(),
        1
    );
}

#[test]
fn unrelated_downstream_branch_stays_frozen() {
    let (mut source, mut region) = fixture(true, false);
    let other_down = source.program.operators.len();
    let mut other = source.program.operators[3].as_ref().clone();
    other.name = "other down".into();
    source.program.operators.push(Arc::new(other));
    source.program.nodes.push(Node::Affine {
        terms: vec![(4, other_down)],
        bias: None,
    });
    source
        .program
        .nodes
        .push(Node::Concat { parts: vec![5, 6] });
    source.program.output = 7;
    source = Artifact::native(&source.program).unwrap();
    region.current_internal_nodes.push(6);
    region.native_writes.push(6);
    region.current_writes.push(6);
    region.source_program_nodes = 8;
    let inventory = enumerate(&source, &region, &settings()).unwrap();
    let proposal = inventory
        .proposals
        .iter()
        .find(|p| {
            p.edit
                == Some(Reuse {
                    consumer: 5,
                    operand: 0,
                    previous: 4,
                    replacement: 2,
                })
        })
        .unwrap();
    assert_eq!(trainable_sources(proposal), vec![3]);
    assert!(
        !proposal
            .owners
            .iter()
            .find(|o| o.parent_operator == other_down)
            .unwrap()
            .trainable
    );
    apply_hypothesis(&source, &region, proposal, &settings()).unwrap();
}
