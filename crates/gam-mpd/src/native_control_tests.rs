use super::*;
use crate::{artifact::Binding, operator_program::{Declarations,FamilyInputs,Interface,Operator,Provenance,Slot,SlotValues,exact_precision}};
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
fn edit(node:usize,change:Change)->Edit {Edit{node,rows:None,columns:0..1,change}}
fn execute(p:&OperatorProgram,edits:&BTreeMap<usize,Vec<Edit>>)->f64 {
    let inputs=FamilyInputs{rows:1,slots:vec![SlotValues::Raw(array![[4.0]])],layout:None};
    let trace=p.execute_edited(&inputs,|node,value,_| {
        for e in edits.get(&node).into_iter().flatten() {
            for row in e.rows.clone().unwrap_or_else(||vec![0]) {for col in e.columns.clone() {value[[row,col]]=match e.change{Change::Scale(x)=>value[[row,col]]*x,Change::Add(x)=>value[[row,col]]+x};}}
        }
        Ok(())
    }).expect("edited linear execution");
    trace.values[p.output][[0,0]]
}
#[test]
fn source_controls_precede_write_add_regardless_episode_order() {
    let (native,artifact)=fixture();validate(&artifact,&native).expect("linear commutation cut");
    for scale in [0.0,2.0] {for reversed in [false,true] {
        let mut edits=vec![edit(1,Change::Scale(scale)),edit(2,Change::Add(3.0))];if reversed{edits.reverse();}
        let mut ordinary:BTreeMap<usize,Vec<Edit>>=BTreeMap::new();for e in &edits{ordinary.entry(e.node).or_default().push(e.clone());}
        let mapped=map_edits(&artifact,&edits).expect("mapped full-vector source scale");assert_eq!(mapped.unheld,0);
        assert_eq!(execute(&native,&ordinary),8.0*scale+3.0);assert_eq!(execute(&artifact.program,&mapped.edits),execute(&native,&ordinary));
    }}
}
#[test]
fn repeated_scales_keep_order_and_partial_controls_remain_unheld() {
    let (_,mut artifact)=fixture();artifact.controls[0].width=2;
    let mapped=map_edits(&artifact,&[edit(1,Change::Scale(0.0)),edit(1,Change::Add(3.0))]).expect("unsupported source controls remain explicit");assert_eq!(mapped.unheld,2);assert!(mapped.edits.is_empty());
    artifact.controls[0].width=1;
    let edits=[edit(1,Change::Scale(2.0)),edit(2,Change::Add(3.0)),edit(1,Change::Scale(0.0))];let mapped=map_edits(&artifact,&edits).expect("ordered scales");
    assert_eq!(mapped.edits[&1].iter().map(|e|e.change).collect::<Vec<_>>(),vec![Change::Scale(2.0),Change::Scale(0.0),Change::Add(3.0)]);
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
#[test]
fn empty_controls_preserve_direct_mapping_and_unheld_behavior() {
    let (native,mut artifact)=fixture();artifact.controls.clear();
    let mapped=map_edits(&artifact,&[edit(1,Change::Scale(0.0)),edit(2,Change::Add(3.0))]).expect("legacy direct mapping");assert_eq!(mapped.unheld,1);assert_eq!(execute(&artifact.program,&mapped.edits),11.0);
    artifact.places.clear();artifact.blocks.clear();validate_shape(&artifact).expect("ephemeral empty-control artifact");validate(&artifact,&native).expect("empty control native proof vacuous");
}
#[test]
fn transposed_linear_cut_supported_and_invalid_edits_rejected() {
    let (mut native,artifact)=fixture();native.nodes[2]=Node::Transposed{input:1,operator:1};validate(&artifact,&native).expect("transposed pointwise linear cut");
    assert!(map_edits(&artifact,&[edit(1,Change::Scale(f64::INFINITY))]).is_err());
    let mut invalid=edit(1,Change::Scale(1.0));invalid.columns=0..2;assert!(map_edits(&artifact,&[invalid]).is_err());
}
#[test]
fn rectangular_write_maps_full_source_columns_to_all_output_columns() {
    let source=Interface::native(2).expect("two-coordinate source");let target=Interface::native(1).expect("one-coordinate write");let matrix=array![[2.0,3.0]];
    let native=OperatorProgram{declarations:Declarations{domains:vec![],slots:vec![Slot::Raw{width:2}],parameters:0},bases:vec![],rules:vec![],operators:vec![Arc::new(Operator::identity("identity",source.clone())),Arc::new(Operator::dense("rectangular",target,source,matrix.clone(),exact_precision(matrix.iter().copied()).expect("exact rectangular weights"),Provenance::default()).expect("rectangular operator"))],nodes:vec![Node::Raw{slot:0},Node::Affine{terms:vec![(0,0)],bias:None},Node::Affine{terms:vec![(1,1)],bias:None}],output:2};
    let mut p=native.clone();p.nodes=vec![Node::Raw{slot:0},Node::Affine{terms:vec![(0,1)],bias:None}];p.output=1;
    let mut a=Artifact::native(&p).expect("rectangular artifact");a.native_nodes=3;a.places=vec![(0,0),(2,1)];a.blocks=vec![Binding{name:"rectangular".into(),native_reads:vec![0],native_write:2,reads:vec![0],write:1}];a.controls=vec![UniformScaleBinding{native_source:1,native_write:2,write:1,width:2}];validate(&a,&native).expect("rectangular linear cut");
    let mapped=map_edits(&a,&[Edit{node:1,rows:Some(vec![0]),columns:0..2,change:Change::Scale(2.0)}]).expect("full-source transport");assert_eq!(mapped.edits[&1][0].columns,0..1);assert_eq!(mapped.edits[&1][0].rows,Some(vec![0]));
    a.controls[0].width=usize::MAX;assert!(validate_shape(&a).is_err());
}
