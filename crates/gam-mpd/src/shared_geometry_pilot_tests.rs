use super::*;
use crate::operator_program::{FamilyInputs, SlotValues};
fn native() -> (OperatorProgram, Vec<LayerNodes>) {
    let mut operators = vec![];
    let mut nodes = vec![Node::Raw {slot:0}];
    let mut layers = vec![];
    for layer in 0..2 {
        let op = operators.len();
        operators.push(dense(format!("up{layer}"),Array2::from_shape_fn((3,2),|(r,c)|0.2*(r+c+layer+1) as f64)).expect("reader"));
        operators.push(dense(format!("down{layer}"),Array2::from_shape_fn((2,3),|(r,c)|0.1*(r+c+layer+1) as f64)).expect("writer"));
        let pre=nodes.len();nodes.push(Node::Affine {terms:vec![(0,op)],bias:None});
        let active=nodes.len();nodes.push(Node::Pointwise {input:pre,laws:vec![Law::GeluTanh]});
        let mlp=nodes.len();nodes.push(Node::Affine {terms:vec![(active,op+1)],bias:None});
        layers.push(LayerNodes {stream:0,normed_stream:0,queries:vec![],keys:vec![],values:vec![],reads:vec![],attention:0,attended:0,normed:0,pre,active,mlp,residual:mlp});
    }
    let output=nodes.len()-1;
    (OperatorProgram {declarations:Declarations {domains:vec![],slots:vec![Slot::Raw {width:2}],parameters:0},bases:vec![],operators,rules:vec![],nodes,output},layers)
}
#[test]
fn shared_geometry_controls_and_function_pool() {
    let (native,layers)=native();
    let learned=build(&native,&layers,&[0,1],Arm::Learned,17).expect("learned");
    let frozen=build(&native,&layers,&[0,1],Arm::FrozenNative,17).expect("frozen");
    let untied=build(&native,&layers,&[0,1],Arm::Untied,17).expect("untied");
    assert_eq!(learned.program,frozen.program);
    assert_eq!(learned.trainable.len(),10);assert_eq!(frozen.trainable.len(),8);
    assert_eq!(learned.body_operators,vec![0,1]);
    let a=Array2::from_shape_vec((2,2),vec![1.0,2.0,-1.0,0.5]).expect("rows");
    let b=Array2::from_shape_vec((2,2),vec![0.2,0.3,1.5,-0.7]).expect("rows");
    let inputs=FamilyInputs {rows:2,slots:vec![SlotValues::Raw(a.clone()),SlotValues::Raw(b.clone())],layout:None};
    let grouped=learned.program.execute(&inputs,false).expect("joint execute");
    let direct=untied.program.execute(&inputs,false).expect("untied execute");
    assert_eq!(grouped.values[learned.program.output],direct.values[untied.program.output]);
    for (slot,x) in [a,b].into_iter().enumerate() {
        let function=function(&learned,slot).expect("function export");
        assert!(Arc::ptr_eq(&function.operators[0],&learned.program.operators[0]));
        let f=function.execute(&FamilyInputs {rows:2,slots:vec![SlotValues::Raw(x)],layout:None},false).expect("function execute");
        assert_eq!(f.values[function.output],grouped.values[learned.outputs[slot]]);
    }
}
#[test]
fn shared_geometry_random_seed_and_scope() {
    let (native,layers)=native();
    let a=build(&native,&layers,&[0,1],Arm::FrozenRandom,17).expect("random");
    let b=build(&native,&layers,&[0,1],Arm::FrozenRandom,17).expect("same random");
    let c=build(&native,&layers,&[0,1],Arm::FrozenRandom,18).expect("other random");
    assert_eq!(a.program,b.program);assert_ne!(a.program,c.program);
    assert!(build(&native,&layers,&[0,0],Arm::Learned,17).is_err());
    assert!(build(&native,&layers,&[2],Arm::Learned,17).is_err());
}
