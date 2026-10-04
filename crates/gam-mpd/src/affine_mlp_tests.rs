use super::*;
use ndarray::array;
#[test]
fn full_width_affine_fit_recovers_independent_outputs_and_offset(){
 let x=array![[0.,0.],[1.,0.],[0.,1.],[1.,1.],[-1.,2.]];let y=array![[3.,-2.],[5.,-3.],[2.,2.],[4.,1.],[-1.,7.]];
 let fit=AffineFit::fit(&x,&y).expect("full-rank augmented affine fit");assert_eq!(fit.numerical_rank,3);assert_eq!(fit.weights,array![[2.,-1.],[-1.,4.]]);assert_eq!(fit.offset,array![3.,-2.]);assert_eq!(fit.predict(&x).expect("affine prediction"),y);
}
#[test]
fn underdetermined_minimum_norm_includes_offset(){
 let fit=AffineFit::fit(&array![[1.]],&array![[2.]]).expect("minimum joint norm");assert_eq!(fit.numerical_rank,1);assert_eq!(fit.weights,array![[1.]]);assert_eq!(fit.offset,array![1.]);
}
#[test]
fn nonlinear_residual_spectrum_uses_native_denominator_and_one_rank_inventory(){
 let x=array![[-2.],[-1.],[0.],[1.],[2.]];let y=array![[4.],[1.],[0.],[1.],[4.]];let fit=AffineFit::fit(&x,&y).expect("nonlinear target affine proposal");let spectrum=residual_spectrum(&fit,&x,&y,&[0,1,8]).expect("residual diagnostic");assert_eq!(spectrum.fixed_base_curve.singular_values.len(),1);assert!((spectrum.native_frobenius-34_f64.sqrt()).abs()<1e-12);assert!((spectrum.residual_frobenius-14_f64.sqrt()).abs()<1e-12);assert!((spectrum.fixed_base_curve.corrections[0].affine_correction.necessary_local_lower.expect("nonzero native denominator")-(14_f64/34.).sqrt()).abs()<1e-10);assert_eq!(spectrum.fixed_base_curve.corrections[1].affine_correction.necessary_local_lower,Some(0.));
}
#[test]
fn malformed_nonfinite_and_zero_denominator_cases_explicit(){
 assert!(AffineFit::fit(&array![[f64::NAN]],&array![[1.]]).is_err());assert!(AffineFit::fit(&array![[1.]],&array![[1.],[2.]]).is_err());
 let x=array![[0.],[1.]];let y=array![[0.],[0.]];let fit=AffineFit::fit(&x,&y).expect("zero response");let spectrum=residual_spectrum(&fit,&x,&y,&[0]).expect("explicit zero denominator");assert_eq!(spectrum.fixed_base_curve.corrections[0].affine_correction.necessary_local_lower,None);
}
#[test]
fn compiled_full_affine_rule_preserves_lineage_and_complete_literal_price(){
 use crate::{acceptance::{CostCache,structural_cost},operator_program::{Declarations,Law,OperatorProgram,Slot}};use std::sync::Arc;
 let input=Interface::native(2).expect("input interface");let hidden=Interface::native(3).expect("hidden interface");let up=array![[1.,0.],[0.,1.],[1.,1.]];let down=array![[1.,0.,1.],[0.,1.,1.]];
 let op=|name,rows:Interface,cols:Interface,values:Array2<f64>|Arc::new(Operator::dense(name,rows,cols,values.clone(),exact_precision(values.iter().copied()).expect("exact fixture"),Provenance::default()).expect("fixture operator"));
 let native=OperatorProgram{declarations:Declarations{domains:vec![],slots:vec![Slot::Raw{width:2}],parameters:0},bases:vec![],rules:vec![],operators:vec![op("up",hidden.clone(),input.clone(),up),op("down",input.clone(),hidden,down)],nodes:vec![Node::Raw{slot:0},Node::Affine{terms:vec![(0,0)],bias:None},Node::Pointwise{input:1,laws:vec![Law::GeluTanh]},Node::Affine{terms:vec![(2,1)],bias:None}],output:3};
 let base=Artifact::native(&native).expect("native fixture");let fit=AffineFit::fit(&array![[0.,0.],[1.,0.],[0.,1.]],&array![[1.,2.],[3.,2.],[1.,5.]]).expect("affine fixture");
 let layer=LayerNodes{normed:0,active:2,mlp:3,..LayerNodes::default()};let candidate=fit.candidate(&base,&native,&layer,"full affine").expect("typed fullwidth rule");candidate.validate_coverage(&native).expect("original input/output lineage");assert!(candidate.place(2).is_none());assert_eq!(candidate.controls.len(),1);
 let bytes=candidate.to_bytes().expect("standalone paid encoding");let decoded=Artifact::from_bytes(&bytes,&native.declarations).expect("ordinary independent decoding");decoded.validate_coverage(&native).expect("decoded lineage");assert_eq!(decoded.to_bytes().expect("canonical replay"),bytes);
 let cost=structural_cost(&candidate,&mut CostCache::default()).expect("completeC32");assert_eq!(cost.literals,6);assert_eq!(cost,structural_cost(&decoded,&mut CostCache::default()).expect("decoded completeC32"));
}
