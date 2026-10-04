use super::*;
fn budget(n: usize) -> Budget {
    Budget {
        max_nodes: n,
        max_c32: u64::MAX,
        max_prefixes: 5_000_000,
        max_bodies: 100_000,
        scale_coefficients: vec![],
    }
}
#[test]
fn matrix_enumeration_complete_small_grammar_and_caps() {
    let t = Type::Matrix { rows: 2, cols: 3 };
    let one = enumerate(&[t.clone()], &t, &budget(1)).unwrap();
    assert!(one.complete);
    assert_eq!(one.bodies.len(), 1);
    let two = enumerate(&[t.clone()], &t, &budget(2)).unwrap();
    assert!(two.complete);
    assert_eq!(two.bodies.len(), 3); // Param, reciprocal, self divide.
    let mut b = budget(2);
    b.max_bodies = 1;
    let stopped = enumerate(&[t.clone()], &t, &b).unwrap();
    assert!(!stopped.complete);
    assert!(stopped.stop_reason.is_some());
    b.max_bodies = 100;
    b.max_prefixes = 0;
    assert!(!enumerate(&[t.clone()], &t, &b).unwrap().complete);
}
#[test]
fn matrix_enumeration_generic_seven_node_composite_reachable() {
    let inputs = vec![
        Type::Vector { len: 2 },
        Type::Vector { len: 2 },
        Type::Matrix { rows: 1, cols: 2 },
    ];
    let wanted = MatrixRule {
        inputs: inputs.clone(),
        nodes: vec![
            Node::Param { index: 0 },
            Node::Param { index: 1 },
            Node::Param { index: 2 },
            Node::Divide {
                numerator: 0,
                denominator: 1,
            },
            Node::Diag { input: 3 },
            Node::Pinv {
                input: 2,
                convention: PinvConvention::SvdResolutionBand,
            },
            Node::MatMul { left: 4, right: 5 },
        ],
        output: 6,
    };
    let key = canonicalize(&wanted).unwrap().rule.encode().unwrap();
    let all = enumerate(&inputs, &Type::Matrix { rows: 2, cols: 1 }, &budget(7)).unwrap();
    println!(
        "seven-node metadata: {} canonical bodies, {} visited prefixes",
        all.bodies.len(),
        all.visited_prefixes
    );
    assert!(
        all.complete,
        "{:?} prefixes {}",
        all.stop_reason, all.visited_prefixes
    );
    assert!(all.bodies.iter().any(|b| b.rule.encode().unwrap() == key));
}
#[test]
fn matrix_enumeration_canonical_slots_preserve_binding_order() {
    let t = Type::Matrix { rows: 2, cols: 2 };
    let body = canonicalize(&MatrixRule {
        inputs: vec![t.clone(), t],
        nodes: vec![
            Node::Param { index: 0 },
            Node::Param { index: 1 },
            Node::Divide {
                numerator: 1,
                denominator: 0,
            },
        ],
        output: 2,
    })
    .unwrap();
    assert_eq!(body.input_permutation, vec![1, 0]);
    let binding = [19, 23];
    assert_eq!(
        body.input_permutation
            .iter()
            .map(|&i| binding[i])
            .collect::<Vec<_>>(),
        vec![23, 19]
    );
}
#[test]
fn matrix_enumeration_transitive_exclusion_repeated_bindings_and_guard() {
    let t = Type::Vector { len: 2 };
    let sources = vec![
        Source {
            id: 0,
            ty: t.clone(),
            dependencies: vec![],
        },
        Source {
            id: 1,
            ty: t.clone(),
            dependencies: vec![0],
        },
        Source {
            id: 2,
            ty: t.clone(),
            dependencies: vec![1],
        },
        Source {
            id: 3,
            ty: t.clone(),
            dependencies: vec![],
        },
    ];
    let eligible = eligible_sources(&sources, 0).unwrap();
    assert_eq!(eligible.iter().map(|s| s.id).collect::<Vec<_>>(), vec![3]);
    let rule = MatrixRule {
        inputs: vec![t.clone(), t],
        nodes: vec![
            Node::Param { index: 0 },
            Node::Param { index: 1 },
            Node::Divide {
                numerator: 0,
                denominator: 1,
            },
        ],
        output: 2,
    };
    assert_eq!(
        Bindings::new(&rule, &eligible, 1)
            .unwrap()
            .collect::<Vec<_>>(),
        vec![vec![3, 3]]
    );
    assert!(Bindings::new(&rule, &eligible, 0).is_err());
    let broken = vec![Source {
        id: 9,
        ty: Type::Vector { len: 1 },
        dependencies: vec![9],
    }];
    assert!(eligible_sources(&broken, 0).is_err());
}
#[test]
fn matrix_enumeration_inventory_is_complete_and_budget_failure_unresolved() {
    let t = Type::Matrix { rows: 2, cols: 3 };
    let sources = vec![
        Source {
            id: 0,
            ty: t.clone(),
            dependencies: vec![],
        },
        Source {
            id: 1,
            ty: t.clone(),
            dependencies: vec![],
        },
        Source {
            id: 2,
            ty: t.clone(),
            dependencies: vec![],
        },
    ];
    let targets = vec![Target { id: 0, ty: t }];
    let all = inventory(&targets, &sources, 1, &budget(2), 6).unwrap();
    assert!(all.complete);
    assert_eq!(all.binding_count, 6);
    assert_eq!(all.families.len(), 3);
    let partial = inventory(&targets, &sources, 1, &budget(2), 5).unwrap();
    assert!(!partial.complete);
    assert_eq!(partial.binding_count, 4);
    assert!(partial.stop_reason.is_some());
}
