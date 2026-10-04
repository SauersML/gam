use super::*;
use crate::{
    acceptance::{Change, CostCache, Edit, Episode, FamilyRun, RunCheck, structural_cost},
    operator_program::{Slot, NativeOperatorCodec},
};
use ndarray::array;

fn fixture() -> (OperatorProgram, Artifact) {
    let d = Interface::native(2).expect("two logits");
    let values = array![[2.,0.],[0.,2.]];
    let native = OperatorProgram {
        declarations: Declarations { domains: vec![], slots: vec![Slot::Raw { width:2 }], parameters:0 },
        bases:vec![], rules:vec![],
        operators:vec![Arc::new(Operator::identity("I",d.clone())),Arc::new(Operator::dense("double",d.clone(),d,values.clone(),exact_precision(values.iter().copied()).expect("exact"),Provenance::default()).expect("matrix"))],
        nodes:vec![Node::Raw {slot:0},Node::Affine {terms:vec![(0,0)],bias:None},Node::Affine {terms:vec![(1,1)],bias:None}],output:2,
    };
    let mut compact = native.clone();
    compact.nodes = vec![Node::Raw {slot:0},Node::Affine {terms:vec![(0,1)],bias:None}];
    compact.output = 1;
    let mut artifact = Artifact::native(&compact).expect("artifact");
    artifact.native_nodes = 3;
    artifact.places = vec![(0,0),(2,1)];
    artifact.blocks = vec![Binding {name:"composed double".into(),native_reads:vec![0],native_write:2,reads:vec![0],write:1}];
    (native,artifact)
}

#[test]
fn paid_control_roundtrip_cost_and_counterfactual_prediction() {
    let (native,plain) = fixture();
    let artifact = plain.with_uniform_scale_control(&native,1,2).expect("native linear proof");
    artifact.validate_coverage(&native).expect("native coverage and control proof");
    let bytes = artifact.to_bytes().expect("standalone");
    assert_eq!(&bytes[8..16],&CONTROL_ARTIFACT_VERSION.to_le_bytes());
    let decoded = Artifact::from_bytes(&bytes,&native.declarations).expect("no native model needed to decode");
    assert_eq!(decoded.to_bytes().expect("canonical bytes"),bytes);
    decoded.validate_coverage(&native).expect("native proof checked before evaluation");
    let cache = NativeOperatorCodec::new(&plain.program,1<<20).expect("native operator cache");
    let cached = Artifact::decode_with_native_codec(&artifact.encode().expect("message"),&native.declarations,&cache).expect("borrowed cached decode");
    assert_eq!(cached.controls,decoded.controls);
    let plain_cost = structural_cost(&plain,&mut CostCache::default()).expect("plain cost");
    let control_cost = structural_cost(&decoded,&mut CostCache::default()).expect("complete control cost");
    assert_eq!(plain_cost.literals,control_cost.literals);
    assert!(control_cost.total()>plain_cost.total());
    assert_eq!(control_cost.total()-plain_cost.total(),artifact.encode().expect("control bits").len_bits()-plain.encode().expect("plain bits").len_bits());
    let family = FamilyInputs {rows:2,slots:vec![SlotValues::Raw(array![[4.,-4.],[-2.,2.]])],layout:None};
    let run = FamilyRun {model:&native,family,readouts:1,episodes:vec![
        Episode {id:"clean".into(),group:"clean".into(),edits:vec![]},
        Episode {id:"remove omitted state".into(),group:"remove".into(),edits:vec![Edit {node:1,rows:None,columns:0..2,change:Change::Scale(0.)}]},
    ]};
    let missing = run.episodes(&plain).expect("legacy omission behavior");
    assert_eq!(missing[1].unheld,1);
    assert!(missing[1].kl>1.0);
    let represented = run.episodes(&decoded).expect("decoded control execution");
    for score in represented {assert_eq!(score.unheld,0);assert!(score.kl.abs()<=score.numerical_error);}
    let truncated = artifact.truncated(2).expect("remapped compact artifact");
    assert_eq!(truncated.controls.len(),1);
    crate::native_control::validate(&truncated,&native).expect("proof survives compaction");
}

fn tail(a: &Artifact, count:u64, tag:u64, width_code:u64) -> BitString {
    let mut bits = BitString::new();
    encode_prefix_integer(&mut bits,count+1).expect("count");
    for _ in 0..count {
        encode_prefix_integer(&mut bits,tag).expect("tag");
        encode_fixed_index(&mut bits,1,a.native_nodes).expect("source");
        encode_fixed_index(&mut bits,2,a.native_nodes).expect("native write");
        encode_fixed_index(&mut bits,1,a.program.nodes.len()).expect("candidate write");
        encode_prefix_integer(&mut bits,width_code).expect("width");
    }
    bits
}

#[test]
fn malformed_control_messages_and_version_mismatches_are_refused() {
    let (native,plain) = fixture();
    let artifact = plain.with_uniform_scale_control(&native,1,2).expect("native linear proof");
    let full = artifact.encode().expect("message");
    let suffix = tail(&artifact,1,1,3);
    let prefix = full.reader().read_bit_string(full.len_bits()-suffix.len_bits()).expect("prefix before controls");
    for suffix in [tail(&artifact,0,1,3),tail(&artifact,1,2,3),tail(&artifact,1,1,1),tail(&artifact,2,1,3)] {
        let mut bad = prefix.clone();bad.append(&suffix);
        assert!(Artifact::decode(&bad,&native.declarations).is_err());
    }
    let mut bytes = artifact.to_bytes().expect("bytes");
    bytes[8..16].copy_from_slice(&MATRIX_ARTIFACT_VERSION.to_le_bytes());
    assert!(Artifact::from_bytes(&bytes,&native.declarations).is_err());
    let mut plain = plain;
    plain.controls = artifact.controls.clone();plain.controls[0].width=0;
    assert!(plain.encode().is_err());
}
