use super::*;
use crate::attention_map::AttentionLayerMap;
use crate::acceptance::{CostCache,structural_cost};
use crate::import::import_language_model;
use serde_json::json;
use std::{
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

struct Export(PathBuf);
impl Drop for Export {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).expect("remove temporary export fixture");
    }
}
fn export(layers: usize, heads: usize, kv: usize, qk: bool, gated: bool, well_conditioned: bool) -> Export {
    static SERIAL: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "mpd-mapped-copy-residual-{}-{}",
        std::process::id(),
        SERIAL.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&dir).expect("valid attention mapping fixture");
    let mut files = serde_json::Map::new();
    let mut write = |name: &str, rows: usize, cols: usize, gain: bool| {
        let values: Vec<f64> = (0..rows * cols)
            .map(|i| {
                if gain {
                    1. + 0.01 * (i as f64)
                } else if well_conditioned {
                    let diagonal = if i % cols == (i / cols) % cols { 0.25 } else { 0. };
                    diagonal + ((i * 7 + 3) % 31) as f64 / 1024.
                } else {
                    ((i * 7 + 3) as f64).sin() * 0.2
                }
            })
            .collect();
        // HF exports contain exactly widened f32 checkpoint literals.
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|x| f64::from(*x as f32).to_le_bytes())
            .collect();
        std::fs::write(dir.join(format!("{name}.f64")), bytes)
            .expect("valid attention mapping fixture");
        files.insert(name.into(), json!({"shape":[rows,cols]}));
    };
    // Odd head width and model width unequal to query_heads*head_width exercise
    // generic GQA dimensions rather than the old Decoder's reshaping assumptions.
    let (d, dh, hidden, vocab) = (20, 17, 23, 7);
    write("wte", vocab, d, false);
    for layer in 0..layers {
        for part in ["rms1.gain", "rms2.gain"] {
            write(&format!("blocks.{layer}.{part}"), 1, d, true);
        }
        for (part, rows, cols) in [
            ("attn.q_proj", heads * dh, d),
            ("attn.k_proj", kv * dh, d),
            ("attn.v_proj", kv * dh, d),
            ("attn.o_proj", d, heads * dh),
            ("mlp.c_fc", hidden, d),
            ("mlp.down_proj", d, hidden),
        ] {
            write(&format!("blocks.{layer}.{part}"), rows, cols, false);
        }
        if qk {
            for part in ["attn.q_norm.gain", "attn.k_norm.gain"] {
                write(&format!("blocks.{layer}.{part}"), 1, dh, true);
            }
        }
        if gated {
            write(&format!("blocks.{layer}.mlp.gate_proj"), hidden, d, false);
        }
    }
    write("final_norm.gain", 1, d, true);
    drop(write);
    std::fs::write(
        dir.join("tokens.f64"),
        [1_f64, 2., 3.]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .expect("valid attention mapping fixture");
    files.insert("tokens".into(), json!({"shape":[1,3]}));
    let record = json!({"config":{"n_layers":layers,"d_model":d,"n_heads":heads,"n_kv_heads":kv,"head_dim":dh,"vocab":vocab,"norm":"rms","norm_eps":0.00001,"rope_theta":10000.,"rotary_dims":2,"rope_pairing":"rotate_half","qk_norm":qk,"mlp_gated":gated,"mlp_act":if gated {"silu"}else {"gelu_tanh"},"tied_embeddings":true},"files":files});
    std::fs::write(
        dir.join("export.json"),
        serde_json::to_vec(&record).expect("valid attention mapping fixture"),
    )
    .expect("valid attention mapping fixture");
    Export(dir)
}

fn native27_with_conditioning(well_conditioned:bool) -> (Export,crate::import::Imported) {
    let dir=export(1,16,4,true,true,well_conditioned);
    let mut imported=import_language_model(&dir.0,1,3).expect("mapped proposal fixture");
    for op in &mut imported.program.operators {std::sync::Arc::make_mut(op).name=op.name.replace("blocks.0.","blocks.27.");}
    (dir,imported)
}
fn native27() -> (Export,crate::import::Imported) { native27_with_conditioning(true) }
#[test]
fn mapped_copy_residual_qwen_all16_rank16_complete_and_guarded() {
    let (_dir,imported)=native27();let p=&imported.program;
    let base=Artifact::native(p).unwrap().f32_literals().unwrap();
    assert!(MappedCopyResidualBank::new(&base,&[27],&[16],32).err().unwrap().contains("33"));
    assert!(MappedCopyResidualBank::new(&base,&[27,27],&[16],100).is_err());
    assert!(MappedCopyResidualBank::new(&base,&[0],&[16],100).is_err());
    assert!(MappedCopyResidualBank::new(&base,&[27],&[18],100).is_err());
    let bank=MappedCopyResidualBank::new(&base,&[27],&[16],33).unwrap();
    assert_eq!(bank.candidate_count,33);assert_eq!(bank.choices().count(),32);
    assert!(bank.cache.iter().all(|c|c.copy.get().is_none()&&c.native.get().is_none()));
    let map=AttentionLayerMap::of(p,27).unwrap();let mut seen=std::collections::BTreeSet::new();
    let mut cache=CostCache::default();let native_cost=structural_cost(&base,&mut cache).unwrap();
    for choice in bank.choices(){
        assert_eq!(choice.layer,27);assert_eq!(choice.rank,16);seen.insert(choice.head);
        let candidate=bank.candidate(choice).unwrap().f32_literals().unwrap();let h=&map.heads[choice.head];
        assert_eq!(candidate.places,base.places);assert_eq!(candidate.program.nodes.len(),p.nodes.len());
        assert_eq!(candidate.blocks[0].native_reads,map.native_reads());assert_eq!(candidate.blocks[0].native_write,map.output);
        for (n,node) in p.nodes.iter().enumerate(){if n!=map.output{assert_eq!(candidate.program.nodes[n],*node);}}
        for original in [h.raw_query,h.raw_key,h.query,h.key,h.value,h.read,map.skip,map.normed_input,map.output]{assert_eq!(candidate.place(original),Some(original));}
        let copy=choice.family==HeadApproximation::CopyResidual;
        assert_eq!(candidate.derived.len(),usize::from(copy));
        assert_eq!(candidate.program.operators.len(),p.operators.len()+usize::from(copy));
        if copy{assert_eq!(candidate.derived[0].law,map.copy_law(choice.head).unwrap());}
        let before=structural_cost(&candidate,&mut cache).unwrap();
        // Same declared factors plus one Copy scale, no hidden/free residual numbers.
        assert_eq!(before.literals,native_cost.literals-340+16*37+u64::from(copy));
        let wire=candidate.to_bytes().unwrap();let decoded=Artifact::from_bytes(&wire,&p.declarations).unwrap();
        decoded.validate_coverage(p).unwrap();assert!(decoded.has_f32_literals());assert_eq!(decoded.places,base.places);
        assert_eq!(decoded.to_bytes().unwrap(),wire);assert_eq!(structural_cost(&decoded,&mut Default::default()).unwrap(),before);
    }
    assert_eq!(seen,(0..16).collect());
}
#[test]
fn mapped_copy_residual_qk_norm_rotary_gqa_and_interventions_survive_full_rank() {
    let (_dir,imported)=native27();let p=&imported.program;let base=Artifact::native(p).unwrap().f32_literals().unwrap();
    let map=AttentionLayerMap::of(p,27).unwrap();assert_eq!(map.heads[0].key,map.heads[3].key);assert_ne!(map.heads[3].key,map.heads[4].key);
    assert_eq!(p.operators[map.heads[0].output_operator].matrix().dim(),(20,17));
    let bank=MappedCopyResidualBank::new(&base,&[27],&[17],33).unwrap();
    for family in [HeadApproximation::CopyResidual,HeadApproximation::NativeSvd]{
        let candidate=bank.candidate(CopyResidualChoice{layer:27,head:0,rank:17,family}).unwrap().f32_literals().unwrap();
        let wire=candidate.to_bytes().unwrap();let decoded=Artifact::from_bytes(&wire,&p.declarations).unwrap();decoded.validate_coverage(p).unwrap();
        let h=&map.heads[0];assert!(h.query_norm.is_some());assert!(h.key_norm.is_some());assert_eq!(h.rotary.as_ref().unwrap().dims,2);
        for edit in [h.raw_query,h.query,h.key,h.value,h.read,map.skip,map.output]{
            let run=|a:&Artifact|a.program.execute_edited(&imported.family,|n,value,_|{if n==edit{*value*=0.5;}Ok(())}).unwrap();
            let original=run(&base);let changed=run(&candidate);let rebuilt=run(&decoded);
            assert_eq!(changed.values[p.output],rebuilt.values[p.output]);
            let error=(&original.values[map.output]-&rebuilt.values[map.output]).iter().map(|x|x*x).sum::<f64>().sqrt();
            assert!(error<0.0002,"full-rank reconstruction is numerical, error={error}, edit={edit}");
        }
    }
}
#[test]
fn mapped_copy_residual_semantic_bypass_is_explicit_failure_not_missing_head() {
    let (_dir,mut imported)=native27();let map=AttentionLayerMap::of(&imported.program,27).unwrap();
    let Node::Attend{query,..}=&mut imported.program.nodes[map.heads[7].read]else{panic!("fixture attention")};*query=map.heads[7].raw_query;
    let base=Artifact::native(&imported.program).unwrap().f32_literals().unwrap();
    let result=MappedCopyResidualBank::new(&base,&[27],&[16],33);
    assert!(result.is_err());assert!(result.err().unwrap().contains("bypasses"));
}

#[test]
fn mapped_copy_residual_full_rank_does_not_certify_ill_conditioned_literal_rounding() {
    let (_dir,imported)=native27_with_conditioning(false);let p=&imported.program;
    let base=Artifact::native(p).unwrap().f32_literals().unwrap();let map=AttentionLayerMap::of(p,27).unwrap();
    let bank=MappedCopyResidualBank::new(&base,&[27],&[17],33).unwrap();
    let candidate=bank.candidate(CopyResidualChoice{layer:27,head:0,rank:17,family:HeadApproximation::CopyResidual}).unwrap().f32_literals().unwrap();
    candidate.validate_coverage(p).unwrap();
    let original=base.program.execute(&imported.family,false).unwrap();let changed=candidate.program.execute(&imported.family,false).unwrap();
    let error=(&original.values[map.output]-&changed.values[map.output]).iter().map(|x|x*x).sum::<f64>().sqrt();
    assert!(error.is_finite()&&error>0.001,"full rank must not be treated as a quality certificate: {error}");
}

#[test]
fn mapped_copy_residual_fits_actual_gain_operator_ids_not_name_aliases() {
    let (_dir,mut imported)=native27();let old=AttentionLayerMap::of(&imported.program,27).unwrap();
    let mut actual=(*imported.program.operators[old.input_gain]).clone();
    let crate::operator_program::OperatorBody::Diagonal { values,.. }=&mut actual.body else {panic!("gain fixture")};
    for (i,value) in values.iter_mut().enumerate(){*value=if i%2==0{2.}else{0.5};}
    let actual_gain=imported.program.operators.len();imported.program.operators.push(std::sync::Arc::new(actual));
    let Node::Affine {terms,..}=&mut imported.program.nodes[old.normed_input]else{panic!("gain fixture")};terms[0].1=actual_gain;
    let p=&imported.program;let map=AttentionLayerMap::of(p,27).unwrap();assert_eq!(map.input_gain,actual_gain);
    let base=Artifact::native(p).unwrap().f32_literals().unwrap();let bank=MappedCopyResidualBank::new(&base,&[27],&[16],33).unwrap();
    let candidate=bank.candidate(CopyResidualChoice{layer:27,head:0,rank:16,family:HeadApproximation::CopyResidual}).unwrap();
    let crate::operator_program::OperatorBody::Diagonal {values:gain,..}=&p.operators[actual_gain].body else{panic!("gain fixture")};
    let crate::operator_program::OperatorBody::Diagonal {values:final_gain,..}=&p.operators[map.final_gain].body else{panic!("gain fixture")};
    let h=&map.heads[0];let value=p.operators[h.value_operator].matrix();let prediction=crate::rules::copy_prediction(&value,gain,final_gain).unwrap();
    let expected=crate::rules::copy_scale(&p.operators[h.output_operator].matrix(),&value,&prediction) as f32;
    assert_eq!(candidate.derived[0].scale,expected);assert_eq!(candidate.derived[0].law,map.copy_law(0).unwrap());
    candidate.validate_coverage(p).unwrap();
}
