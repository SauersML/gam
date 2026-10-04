//! Saved-program response representability diagnostic, not a fidelity/frontier win.
//! EXPORT SAVED SPEC OUT SAVED_SHA SPEC_SHA TRACE_BYTES
use gam_mpd::{acceptance::{Change,CostCache,Edit,Local,RunCheck,RunMeasure,structural_cost},artifact::Artifact,
    coder_capture::sha256,counterfactual::{Action,Decoder,InputChange,Rows,Spec,passages},import::import_language_model,
    operator_program::{FamilyInputs,SequenceLayout,SlotValues},run_check::{LanguageRun,layer_nodes,split_sites}};
use serde_json::json;
use std::{collections::BTreeMap,path::Path,time::Instant};
fn write(path:&Path,value:&serde_json::Value)->Result<(),String>{std::fs::write(path,serde_json::to_vec_pretty(value).map_err(|e|e.to_string())?).map_err(|e|e.to_string())}
fn peak_rss()->Option<u64>{std::fs::read_to_string("/proc/self/status").ok()?.lines().find_map(|s|s.strip_prefix("VmHWM:")?.split_whitespace().next()?.parse::<u64>().ok()?.checked_mul(1024))}
fn main()->Result<(),String>{
    let args:Vec<String>=std::env::args().skip(1).collect();if args.len()!=7{return Err("EXPORT SAVED SPEC OUT SAVED_SHA SPEC_SHA TRACE_BYTES".into());}
    let(export,saved,spec_path,out)=(Path::new(&args[0]),Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]));
    if sha256(saved)?!=args[4] || sha256(spec_path)?!=args[5]{return Err("frozen saved bytes/spec changed".into());}
    let trace_bytes=args[6].parse::<usize>().map_err(|e|e.to_string())?;
    if out.exists(){return Err("fresh output directory required".into());}std::fs::create_dir_all(out).map_err(|e|e.to_string())?;
    let started=Instant::now();let decoder=Decoder::from_export(export)?;let spec=Spec::load(spec_path,&decoder)?;
    let imported=import_language_model(export,1,1)?;let native=split_sites(&imported.program)?;let layers=layer_nodes(&native,4)?;
    if spec.rows!=512 || spec.episodes.len()!=10{return Err("frozen two512passage clean+fourMLPremovals subset required".into());}
    let mut declared=BTreeMap::new();
    for episode in &spec.episodes {
        if episode.passage>=2{return Err("only fixed first two passages permitted".into());}
        let label=if episode.actions.is_empty(){"clean".to_string()}else{
            if episode.actions.len()!=1{return Err("one fullMLPremoval per episode required".into());}
            match &episode.actions[0]{Action::Input{site,change:InputChange::Scale{rows:Rows::All,cols,scale}} if site%6==5 && site/6<4 && *scale==0.0 && *cols==(0,native.node_interface(layers[site/6].active).map_err(|e|e.to_string())?.width())=>format!("L{}",site/6),_=>return Err("only wholeactivation zero controls in diagnostic panel".into())}
        };
        if declared.insert((episode.passage,label),episode.id.clone()).is_some(){return Err("duplicate response episode".into());}
    }
    if declared.len()!=10{return Err("incomplete allMLP panel".into());}
    let bytes=std::fs::read(saved).map_err(|e|e.to_string())?;let plain=Artifact::from_bytes(&bytes,&native.declarations)?;
    if !plain.controls.is_empty(){return Err("baseline must have no response bindings".into());}
    plain.validate_coverage(&native)?;if plain.to_bytes()?!=bytes{return Err("existing saved canonical replay mismatch".into());}drop(bytes);
    let mut paid=plain.clone();let mut bindings=Vec::new();
    for (layer,nodes) in layers.iter().enumerate(){
        if plain.blocks.iter().any(|b|b.native_write==nodes.mlp){
            paid=paid.with_uniform_scale_control(&native,nodes.active,nodes.mlp)?;
            bindings.push(json!({"layer":layer,"native_source":nodes.active,"native_write":nodes.mlp,"width":native.node_interface(nodes.active).map_err(|e|e.to_string())?.width()}));
        }
    }
    if bindings.is_empty(){return Err("saved program declares no wholeMLP block".into());}
    if paid.program!=plain.program || paid.places!=plain.places || paid.blocks!=plain.blocks || paid.derived!=plain.derived || paid.exceptions!=plain.exceptions{return Err("paid binding altered original executable program/lineage".into());}
    let path=out.join("paid.artifact");let bytes=paid.to_bytes()?;std::fs::write(&path,&bytes).map_err(|e|e.to_string())?;
    let decoded=Artifact::from_bytes(&bytes,&native.declarations)?;if decoded!=paid || decoded.to_bytes()?!=bytes{return Err("paid ordinary saved-byte replay mismatch".into());}drop(bytes);drop(paid);let paid=decoded;
    let mut cache=CostCache::default();let plain_cost=structural_cost(&plain,&mut cache)?;let paid_cost=structural_cost(&paid,&mut cache)?;
    if plain_cost.literals!=paid_cost.literals || plain_cost.structure_bits!=paid_cost.structure_bits || paid_cost.total()<=plain_cost.total(){return Err("control price must be positive explicit metadata without new numeric/computation payload".into());}
    let rows=passages(export,512)?;if rows.len()<2{return Err("two fixed native passages required".into());}
    let tokens:Vec<u32>=rows[..2].iter().flat_map(|r|r[..512].iter().copied()).collect();let family=FamilyInputs{rows:1024,slots:vec![SlotValues::Tokens(tokens)],layout:Some(SequenceLayout{sequence:(0..2).flat_map(|s|std::iter::repeat_n(s,512)).collect(),position:(0..2).flat_map(|_|0..512).collect()})};
    let provenance=json!({"scope":"distinct paid response-law program vs unchanged saved wholeMLP program; no refit/no overallMeetsclaim","saved_sha256":args[4],"spec_sha256":args[5],"saved_bytes":std::fs::metadata(saved).map_err(|e|e.to_string())?.len(),"paid_sha256":sha256(&path)?,"export":export,"export_json_sha256":sha256(&export.join("export.json"))?,"export_source":imported.record["source"],"binary_sha256":sha256(&std::env::current_exe().map_err(|e|e.to_string())?)?,"bindings":bindings,"episodes":10,"groups":5,"panel":"explicit10episode subset of frozen80; two512token passages; clean+allfourwholeMLPremovals","Local":"same full2x512inputs; measured once and shared by exact Localgraft equality","scope_limit":"only fullcolumn uniformscales; partial/additive omitted controls remainunheld; floatingexecutionmeasured, real-linearity law not floatingcertificate","numerical_model":"CPU KL conditional exp2ULP/log1ULP model, not full realarithmetic certificate","trace_bytes":trace_bytes,"source":include_str!("mpd_paid_mlp_control_2951.rs")});write(&out.join("PROVENANCE.json"),&provenance)?;
    let t=Instant::now();{
        let a=plain.execute(&family)?;let b=paid.execute(&family)?;if !a.values[plain.program.output].iter().map(|x|x.to_bits()).eq(b.values[paid.program.output].iter().map(|x|x.to_bits())){return Err("paid metadata changed clean executable outputs".into());}
    }let clean_equality_seconds=t.elapsed().as_secs_f64();
    if plain.local_artifact(&native)?.0!=paid.local_artifact(&native)?.0{return Err("control metadata altered Localgraft".into());}
    let device=gam_gpu::tensor::Device::accelerator(gam_gpu::GpuPolicy::Required).map_err(|e|e.to_string())?.ok_or("CUDA required")?;
    let local=Local::new(&native,family,None,16).with_cuda(device.clone(),trace_bytes)?;
    let t=Instant::now();let local_measure=local.measure(&plain)?;let local_seconds=t.elapsed().as_secs_f64();
    write(&out.join("LOCAL.json"),&json!({"measure":local_measure,"seconds":local_seconds,"exact_graft_equal":true,"direct_clean_outputs_equal":true,"clean_equality_seconds":clean_equality_seconds}))?;
    let run=LanguageRun::new(&decoder,&native,&spec,&rows,1)?.with_cuda(device,trace_bytes)?;
    let mut measurements=Vec::new();
    for (label,artifact,cost) in [("plain",&plain,&plain_cost),("paid",&paid,&paid_cost)]{
        let t=Instant::now();let measure=RunMeasure::of(run.episodes(artifact)?);let seconds=t.elapsed().as_secs_f64();
        let partial=artifact.controls.iter().map(|c|{let edit=Edit{node:c.native_source,rows:None,columns:0..c.width.saturating_sub(1),change:Change::Scale(0.0)};if edit.columns.is_empty(){return Ok(json!({"source":c.native_source,"partial_columns":"no nonempty strictsubset for width1"}));}let mapped=gam_mpd::native_control::map_edits(artifact,&[edit])?;Ok(json!({"source":c.native_source,"partial_scale_unheld":mapped.unheld}))}).collect::<Result<Vec<_>,String>>()?;
        let record=json!({"label":label,"C32":cost,"C32_bits":cost.total(),"run":measure,"seconds":seconds,"partial_scope":partial,"native_references":"same immutable LanguageRun/native/Spec/passages and teacher cache for bothprograms","run_timing":run.timing()});write(&out.join(format!("{label}.json")),&record)?;measurements.push(record);
    }
    write(&out.join("REPORT.json"),&json!({"provenance":provenance,"C32_increment":paid_cost.total()-plain_cost.total(),"Local_shared_by_exact_graft":true,"clean_outputs_exact_equal":true,"measurements":measurements,"seconds":started.elapsed().as_secs_f64(),"peak_rss_bytes":peak_rss(),"overall_acceptance_claim":false}))?;Ok(())
}
