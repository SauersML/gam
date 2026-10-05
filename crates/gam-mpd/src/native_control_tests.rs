use super::*;
use crate::{artifact::Binding, operator_program::{Declarations,Interface,Operator,Provenance,Slot,exact_precision}};
use ndarray::array;
use std::sync::Arc;
fn fixture()->(OperatorProgram,Artifact) {
    let d=Interface::native(1).expect("one-coordinate interface");let matrix=array![[2.0]];
    let native=OperatorProgram{declarations:Declarations{domains:vec![],slots:vec![Slot::Raw{width:1}],parameters:0},bases:vec![],rules:vec![],
        operators:vec![Arc::new(Operator::identity("identity",d.clone())),Arc::new(Operator::dense("double",d.clone(),d,matrix.clone(),exact_precision(matrix.iter().copied()).expect("exact two"),Provenance::default()).expect("linear operator"))],
        nodes:vec![Node::Raw{slot:0},Node::Affine{terms:vec![(0,0)],bias:None},Node::Affine{terms:vec![(1,1)],bias:None}],output:2};
    let mut p=native.clone();p.nodes=vec![Node::Raw{slot:0},Node::Affine{terms:vec![(0,1)],bias:None}];p.output=1;
    let mut artifact=Artifact::native(&p).expect("compressed linear fixture");artifact.native_nodes=3;artifact.places=vec![(0,0),(2,1)];
    artifact.blocks=vec![Binding{name:"linear block".into(),native_reads:vec![0],native_write:2,reads:vec![0],write:1}];
    artifact.controls=vec![UniformScaleBinding{native_source:1,native_write:2,write:1,width:1}];
    (native,artifact)
}
#[test]
fn invalid_native_cuts_and_lineage_are_rejected() {
    let (native,artifact)=fixture();
    let mut a=artifact.clone();a.controls[0].width=2;assert!(validate(&a,&native).is_err());
    let mut n=native.clone();n.nodes[2]=Node::Affine{terms:vec![(1,1)],bias:Some(0)};assert!(validate(&artifact,&n).is_err());
    let mut n=native.clone();n.nodes[2]=Node::Affine{terms:vec![(1,1),(0,0)],bias:None};assert!(validate(&artifact,&n).is_err());
    let mut n=native.clone();n.nodes.push(Node::Affine{terms:vec![(1,0)],bias:None});let mut a=artifact.clone();a.native_nodes=4;assert!(validate(&a,&n).is_err());
    let mut a=artifact.clone();a.places.push((1,1));assert!(validate_shape(&a).is_err());
    let mut a=artifact.clone();a.places.push((0,1));assert!(validate_shape(&a).is_err());
    let mut a=artifact.clone();a.controls.push(a.controls[0].clone());assert!(validate_shape(&a).is_err());
    let mut a=artifact.clone();a.blocks[0].native_reads.push(1);assert!(validate_shape(&a).is_err());
}
