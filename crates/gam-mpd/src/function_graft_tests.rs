use super::*;
use crate::operator_program::{Law, Slot};
use ndarray::array;

fn native() -> OperatorProgram {
    let i = Interface::native(2).expect("two-coordinate native interface");
    let op = |name: &str, gain: f64| Arc::new(Operator::dense(name, i.clone(), i.clone(),
        Array2::eye(2) * gain, exact_precision([gain, 0.]).expect("finite gain precision"), Default::default()).expect("square native map"));
    OperatorProgram {
        declarations: Declarations { domains: vec![], slots: vec![Slot::Raw { width: 2 }], parameters: 0 },
        bases: vec![], rules: vec![], operators: vec![op("parent", 3.), op("consumer", 2.)],
        nodes: vec![Node::Raw { slot: 0 }, Node::Affine { terms: vec![(0, 0)], bias: None },
            Node::Pointwise { input: 1, laws: vec![Law::Relu] }, Node::Affine { terms: vec![(2, 1)], bias: None }],
        output: 3,
    }
}
fn function() -> OperatorProgram {
    let i = Interface::native(2).expect("two-coordinate function interface");
    OperatorProgram {
        declarations: Declarations { domains: vec![], slots: vec![Slot::Raw { width: 2 }], parameters: 0 },
        bases: vec![], operators: vec![Arc::new(Operator::identity("sum", i.clone()))],
        rules: vec![
            Rule { name: "square".into(), inputs: vec![i.clone()], nodes: vec![Node::Param { index: 0 },
                Node::Hadamard { left: 0, right: 0 }], output: 1 },
            Rule { name: "reused square".into(), inputs: vec![i], nodes: vec![Node::Param { index: 0 },
                Node::Call { rule: 0, arguments: vec![0] }, Node::Call { rule: 0, arguments: vec![0] },
                Node::Affine { terms: vec![(1, 0), (2, 0)], bias: None }], output: 3 },
        ],
        nodes: vec![Node::Raw { slot: 0 }, Node::Call { rule: 1, arguments: vec![0] }], output: 1,
    }
}

#[test]
fn saved_function_graft_preserves_nested_sharing_native_parents_and_downstream_use() {
    let model = native();
    let f = function();
    let graft = Artifact::native(&model).unwrap().replace_function("learned", &f, 1, 2).unwrap();
    graft.validate_coverage(&model).unwrap();
    assert_eq!(graft.program.rules.len(), 3); // Two library bodies and one binding wrapper.
    assert_eq!(graft.program.operators.len(), 3); // Two retained native maps, one shared identity.
    assert_eq!(graft.program.rules.iter().filter(|r| r.nodes.iter().any(|n| matches!(n, Node::Hadamard { .. }))).count(), 1);
    assert_eq!(graft.blocks[0].native_reads, vec![1]);
    assert_eq!(graft.blocks[0].native_write, 2);
    let bytes = graft.to_bytes().unwrap();
    let decoded = Artifact::from_bytes(&bytes, &model.declarations).unwrap();
    decoded.validate_coverage(&model).unwrap();
    assert_eq!(decoded.to_bytes().unwrap(), bytes);
    let inputs = array![[1., -2.], [0.5, 3.]];
    let family = FamilyInputs { rows: 2, slots: vec![SlotValues::Raw(inputs.clone())], layout: None };
    let trace = decoded.program.execute(&family, false).unwrap();
    assert_eq!(trace.values[decoded.program.output], inputs.mapv(|x| 36. * x * x));
    let on_parents = FamilyInputs { rows: 2, slots: vec![SlotValues::Raw(inputs * 3.)], layout: None };
    let function_trace = f.execute(&on_parents, false).unwrap();
    assert_eq!(trace.values[decoded.place(2).unwrap()], function_trace.values[f.output]);
    let existing_rule = match decoded.program.nodes[decoded.place(2).unwrap()] { Node::Call { rule, .. } => rule, _ => panic!("call") };
    let repeated = decoded.replace_block("same body again", Callee::Existing(existing_rule), vec![Argument::Native(1)], 2, vec![]).unwrap();
    assert_eq!(repeated.program.rules.len(), 3);
    assert_eq!(repeated.program.execute(&family, false).unwrap().values[repeated.program.output], trace.values[decoded.program.output]);
}

#[test]
fn function_graft_refuses_external_context_and_bad_boundaries() {
    let base = Artifact::native(&native()).unwrap();
    let f = function();
    assert!(base.replace_function("backward", &f, 3, 2).is_err());
    assert!(base.replace_function("absent", &f, 999, 2).is_err());
    let mut malformed = f.clone(); malformed.declarations.parameters = 1;
    assert!(base.replace_function("external", &malformed, 1, 2).is_err());
    let mut malformed = f.clone(); malformed.declarations.slots.push(Slot::Raw { width: 2 });
    assert!(base.replace_function("extra input", &malformed, 1, 2).is_err());
    let mut malformed = f.clone(); malformed.rules[0].nodes[0] = Node::Raw { slot: 0 };
    assert!(base.replace_function("ambient", &malformed, 1, 2).is_err());
    let mut malformed = f; malformed.output = 999;
    assert!(base.replace_function("bad output", &malformed, 1, 2).is_err());
}
