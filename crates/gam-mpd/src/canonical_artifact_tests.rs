use super::*;
use crate::{operator_program::{FamilyInputs, Interface, Node, Operator, OperatorProgram, Provenance, Slot, SlotValues}, operator_program::exact_precision};
use ndarray::array;
use std::sync::Arc;
fn native() -> Artifact {
    let interface = Interface::native(2).expect("interface");
    let matrix = |name, values: ndarray::Array2<f64>| Arc::new(Operator::dense(name, interface.clone(), interface.clone(), values.clone(), exact_precision(values.iter().copied()).expect("precision"), Provenance::default()).expect("operator"));
    let program = OperatorProgram {
        declarations: Declarations { domains: vec![], parameters: 0, slots: vec![Slot::Raw { width: 2 }] },
        bases: vec![], rules: vec![],
        operators: vec![matrix("named source A", array![[1.,0.],[0.,1.]]),matrix("named source B",array![[2.,0.5],[0.,3.]])],
        nodes: vec![Node::Raw { slot: 0 },Node::Affine { terms: vec![(0,0)], bias: None },Node::Affine { terms: vec![(1,1)], bias: None }], output: 2,
    };
    Artifact::native(&program).expect("native")
}
fn check(cache: &CanonicalArtifactCache, source: &Artifact) {
    let ordinary = source.f32_literals().expect("f32").to_bytes().expect("ordinary encoding");
    let ordinary_decoded = Artifact::from_bytes(&ordinary, &source.program.declarations).expect("ordinary decode");
    let cached = cache.canonical(source).expect("cached canonical");
    assert_eq!(cached.bytes, ordinary);
    assert_eq!(cached.decoded, ordinary_decoded);
    let saved = cache.decode_saved(&ordinary, &source.program.declarations).expect("saved decode");
    assert_eq!(saved.to_bytes().expect("saved bytes"),ordinary);
    let input = FamilyInputs { rows: 2, slots: vec![SlotValues::Raw(array![[0.25,-1.],[2.,3.]])], layout: None };
    let a = ordinary_decoded.execute(&input).expect("ordinary execution");
    let b = saved.execute(&input).expect("cached execution");
    assert_eq!(a.values,b.values);
}
#[test]
fn exact_native_reuse_changed_f32_literal_and_moved_indices_match_ordinary() -> Result<(), String> {
    let source = native();
    let cache = CanonicalArtifactCache::new(&source, 1<<20).expect("cache");
    check(&cache, &source);
    let before = cache.usage();
    let mut changed = source.clone();
    let op = Arc::make_mut(&mut changed.program.operators[0]);
    let OperatorBody::Dense { values, precision, .. } = &mut op.body else { return Err("fixture must be Dense".into()); };
    values[[0,0]] = f64::from(f32::from_bits(1f32.to_bits()+1));
    *precision = exact_precision(values.iter().copied()).expect("changed precision");
    check(&cache, &changed);
    assert_ne!(changed.to_bytes().expect("changed bytes"),source.to_bytes().expect("native bytes"));
    assert!(cache.usage().decoded_native_operator_hits > before.decoded_native_operator_hits);
    let mut moved = source.clone();
    moved.program.operators.swap(0,1);
    for node in &mut moved.program.nodes {
        if let Node::Affine { terms, .. } = node { for (_,operator) in terms { *operator=1-*operator; } }
    }
    check(&cache, &moved);
    let a=cache.canonical(&source).expect("first");
    let b=cache.canonical(&source).expect("second");
    assert!(Arc::ptr_eq(&a.decoded.program.operators[0],&b.decoded.program.operators[0]));
    Ok(())
}
#[test]
fn exact_candidate_codec_preserves_non_f32_changes_and_ordinary_saved_bytes() -> Result<(), String> {
    let source = native();
    let cache = CanonicalArtifactCache::new(&source, 1 << 20)?;
    let mut changed = source.clone();
    let value = 1.0 + 2.0_f64.powi(-40);
    let OperatorBody::Dense { values, precision, .. } = &mut Arc::make_mut(&mut changed.program.operators[0]).body
        else { return Err("fixture must be Dense".into()); };
    values[[0, 0]] = value;
    *precision = exact_precision(values.iter().copied()).map_err(|e| e.to_string())?;
    assert!(!changed.has_f32_literals());
    let ordinary_bytes = changed.to_bytes()?;
    let ordinary = Artifact::from_bytes(&ordinary_bytes, &changed.program.declarations)?;
    let cached = cache.canonical_exact(&changed)?;
    assert_eq!(cached.bytes, ordinary_bytes);
    assert_eq!(cached.decoded, ordinary);
    assert_eq!(cached.decoded.program.operators[0].matrix_cow()[[0, 0]].to_bits(), value.to_bits());
    assert_eq!(cached.timings.f32_seconds, 0.);
    assert_eq!(cache.decode_saved(&cached.bytes, &changed.program.declarations)?, ordinary);
    assert_ne!(cache.canonical(&changed)?.bytes, cached.bytes, "rounded and exact APIs must stay distinct");
    // A changed candidate never overwrites the immutable native codeword.
    assert_eq!(cache.canonical_exact(&source)?.bytes, source.to_bytes()?);
    assert!(cache.usage().encoded_native_operator_hits > 0);
    assert!(cache.usage().decoded_native_operator_hits > 0);
    Ok(())
}

#[test]
fn exact_preflight_refuses_small_budget_before_cache_and_rejects_non_f32_source() -> Result<(), String> {
    let source=native();
    let cache=CanonicalArtifactCache::new(&source,1<<20).expect("cache");
    let count=cache.preflight().packed_bytes+cache.preflight().decoded_numeric_bytes;
    assert!(CanonicalArtifactCache::new(&source,count-1).is_err());
    let bounded=CanonicalArtifactCache::new(&source,count).expect("exact budget");
    assert_eq!(bounded.stats().budget_bytes,count);
    let mut non_f32=source;
    let OperatorBody::Dense {values,..}=&mut Arc::make_mut(&mut non_f32.program.operators[0]).body else {return Err("fixture must be Dense".into());};
    values[[0,0]]=0.1;
    assert!(CanonicalArtifactCache::new(&non_f32,1<<20).is_err());
    Ok(())
}
