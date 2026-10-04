//! Complete declared paired bank, streamed without singleton or cost pruning.
//! EXPORT SPEC OUT max_bank=193 start=0 count=193 backend=cuda cuda_local=0 trace_bytes=2147483648
use gam_mpd::acceptance::{Assessment, Constraint, CostCache, Local, assess_once, structural_cost};
use gam_mpd::artifact::Artifact;
use gam_mpd::coder_capture::sha256;
use gam_mpd::counterfactual::{Decoder, Spec, passages};
use gam_mpd::operator_program::{FamilyInputs, SequenceLayout, SlotValues};
use gam_mpd::precision::FidelityVerdict;
use gam_mpd::proposals::{CopyResidualBank, CopyResidualChoice};
use gam_mpd::run_check::{LanguageRun, layer_nodes, split_sites};
use serde_json::{Value, json};
use std::{collections::{BTreeMap,BTreeSet}, io::Write, path::Path, time::Instant};
const RANKS:[usize;4]=[8,32,64,96];
const DELTAS:[f64;5]=[0.05,0.1,0.2,0.5,1.0];
const EPSILONS:[f64;6]=[0.001,0.01,0.03,0.1,0.3,1.0];
const SPEC_SHA:&str="c74555341a75b1a6dcab4238c125f36c922c7683dd151b3ceb88ac399e963d0d";
fn write_json(path:&Path,value:&Value)->Result<(),String>{std::fs::write(path,serde_json::to_vec_pretty(value).map_err(|e|e.to_string())?).map_err(|e|e.to_string())}
fn family(rows:&[Vec<u32>])->Result<FamilyInputs,String>{
    if rows.len()<2 || rows[..2].iter().any(|r|r.len()<16){return Err("short screen requires two sequences of16 tokens".into());}
    let mut tokens=Vec::new();let mut sequence=Vec::new();let mut position=Vec::new();
    for (s,row) in rows[..2].iter().enumerate(){for (p,&t) in row[..16].iter().enumerate(){tokens.push(t);sequence.push(s as u32);position.push(p as u32);}}
    Ok(FamilyInputs{rows:32,slots:vec![SlotValues::Tokens(tokens)],layout:Some(SequenceLayout{sequence,position})})
}
fn states(a:&Assessment,grid:&[Constraint])->Result<Vec<&'static str>,String>{grid.iter().map(|c|{
    let (l,r)=(a.local.with_tolerance(c.local)?.verdict(),a.run.with_tolerance(c.run)?.verdict());
    Ok(if l==FidelityVerdict::Violates || r==FidelityVerdict::Violates {"Violates"} else if l==FidelityVerdict::Meets && r==FidelityVerdict::Meets {"Verified"} else {"Unresolved"})
}).collect()}
fn peak_rss()->Option<u64>{std::fs::read_to_string("/proc/self/status").ok()?.lines().find_map(|s|s.strip_prefix("VmHWM:")?.split_whitespace().next()?.parse::<u64>().ok()?.checked_mul(1024))}
fn main()->Result<(),String>{
    gam_mpd::engine::log_to_stderr();
    let args:Vec<String>=std::env::args().skip(1).collect();
    if args.len()<3{return Err("EXPORT SPEC OUT max_bank=193 start=0 count=193 backend=cuda cuda_local=0 trace_bytes=2147483648".into());}
    let (export,spec_path,out)=(Path::new(&args[0]),Path::new(&args[1]),Path::new(&args[2]));
    let mut keys=BTreeMap::new();
    for arg in &args[3..]{let(k,v)=arg.split_once('=').ok_or("expected KEY=VALUE")?;if !["max_bank","start","count","backend","cuda_local","trace_bytes"].contains(&k)||keys.insert(k,v).is_some(){return Err("unknown/duplicate option".into());}}
    let number=|k:&str|->Result<usize,String>{keys.get(k).ok_or_else(||format!("declare {k}"))?.parse().map_err(|e|format!("{e}"))};
    let(max_bank,start,count,cuda_local,trace_bytes)=(number("max_bank")?,number("start")?,number("count")?,number("cuda_local")?,number("trace_bytes")?);
    let backend=*keys.get("backend").ok_or("declare backend")?;
    if count==0 || cuda_local>1 || !["cpu","cuda"].contains(&backend) || (cuda_local==1&&backend!="cuda"){return Err("invalid scope/backend".into());}
    let config:Value=serde_json::from_slice(&std::fs::read(export.join("export.json")).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    let n=|k:&str|->Result<usize,String>{usize::try_from(config["config"][k].as_u64().ok_or("missing architecture integer")?).map_err(|e|e.to_string())};
    let(layers,heads,kv)=(n("n_layers")?,n("n_heads")?,n("n_kv_heads")?);
    let complete=CopyResidualBank::cardinality((0..layers).map(|_|heads),&RANKS)?;
    if complete>max_bank{return Err(format!("complete bank requires {complete}, max_bank={max_bank}"));}
    if layers!=4 || heads!=6 || kv==0 || heads%kv!=0 || start.checked_add(count).ok_or("scope overflow")?>complete{return Err("require4L6heads and bounded declared diagnostic range".into());}
    if sha256(spec_path)?!=SPEC_SHA{return Err("Run spec differs from frozen seven-episode control".into());}
    if out.exists(){return Err("fresh output directory required".into());}std::fs::create_dir_all(out).map_err(|e|e.to_string())?;
    let grid:Vec<Constraint>=DELTAS.iter().flat_map(|&local|EPSILONS.iter().map(move|&run|Constraint{local,run})).collect();
    let scope=json!({"all_heads":24,"ranks":RANKS,"families":["NativeSvd","CopyResidual"],"complete_count_including_native":complete,"range":{"start":start,"count":count},"native_control_always_measured":true,"grid":grid,"local":{"sequences":2,"context":16,"batch":16,"ascent":0,"scope":"short screen, not full context"},"run":{"episodes":7,"spec_sha256":SPEC_SHA,"rows":16},"backend":backend,"cuda_local":cuda_local,"trace_bytes":trace_bytes,"pruning":false,"unmeasured_and_failed_retained":true,"unknown_cost_lower_bound":0,"optimality":"declared193 single substitutions only; no global claim","selected_followup":"expanded512 context and80 episodes required before broader claim"});
    write_json(&out.join("SCOPE.json"),&scope)?;
    let mut inputs=BTreeMap::new();inputs.insert("spec".to_string(),sha256(spec_path)?);inputs.insert("export.json".into(),sha256(&export.join("export.json"))?);
    for name in config["files"].as_object().ok_or("missing export files")?.keys(){let file=format!("{name}.f64");inputs.insert(file.clone(),sha256(&export.join(file))?);}
    let mut sources=BTreeMap::new();
    for(name,text)in[("driver",include_str!("mpd_copy_residual_stream_2951.rs")),("proposals",include_str!("../src/proposals.rs")),("acceptance",include_str!("../src/acceptance.rs")),("artifact",include_str!("../src/artifact.rs")),("codec",include_str!("../src/codec.rs")),("rules",include_str!("../src/rules.rs"))]{let path=out.join(format!("{name}.source"));std::fs::write(&path,text).map_err(|e|e.to_string())?;sources.insert(name,sha256(&path)?);std::fs::remove_file(path).map_err(|e|e.to_string())?;}
    write_json(&out.join("PROVENANCE.json"),&json!({"inputs":inputs,"sources":sources,"binary":sha256(&std::env::current_exe().map_err(|e|e.to_string())?)?}))?;
    let begun=Instant::now();let decoder=Decoder::from_export(export)?;let spec=Spec::load(spec_path,&decoder)?;
    if spec.episodes.len()!=7 || spec.rows!=16{return Err("frozen seven episode sixteen-row control required".into());}
    let imported=import_language_model(export,1,1)?;let native=split_sites(&imported.program)?;let nodes=layer_nodes(&native,layers)?;let base=Artifact::native(&native)?.f32_literals()?;
    let bank=CopyResidualBank::new(&base,&nodes,heads/kv,&RANKS,max_bank)?;
    let run_passages=passages(export,spec.rows)?;
    let mut run=LanguageRun::new(&decoder,&native,&spec,&run_passages,1)?;
    let mut local=Local::new(&native,family(&passages(export,16)?)?,None,16);
    if backend=="cuda"{let device=gam_gpu::tensor::Device::accelerator(gam_gpu::GpuPolicy::Required).map_err(|e|e.to_string())?.ok_or("CUDA required")?;if cuda_local==1{local=local.with_cuda(device.clone(),trace_bytes)?;}run=run.with_cuda(device,trace_bytes)?;}
    let mut cache=CostCache::default();let mut records:Vec<Value>=bank.choices().enumerate().map(|(i,c)|json!({"index":i+1,"choice":c,"cost_bits":null,"cost_lower_bound":0,"states":vec!["Unevaluated";grid.len()]})).collect();
    records.insert(0,json!({"index":0,"label":"native","cost_bits":null,"cost_lower_bound":0,"states":vec!["Unevaluated";grid.len()]}));
    let choices:Vec<CopyResidualChoice>=bank.choices().collect();let mut journal=std::fs::OpenOptions::new().create_new(true).write(true).open(out.join("ASSESSMENTS.jsonl")).map_err(|e|e.to_string())?;
    for index in std::iter::once(0).chain((start..start+count).filter(|&i|i!=0)){
        let t=Instant::now();eprintln!("candidate {index}/{complete}: start");
        let result=(||->Result<(),String>{
            let artifact=if index==0{base.clone()}else{bank.candidate(choices[index-1])?};
            if artifact.places!=base.places{return Err("native places changed".into());}let artifact=artifact.f32_literals()?;
            let cost=structural_cost(&artifact,&mut cache)?;records[index]["cost_bits"]=json!(cost.total());records[index]["cost_lower_bound"]=json!(cost.total());
            let assessment=assess_once(&local,&run,&artifact,grid[0],&mut cache)?;
            if assessment.cost!=cost{return Err("assessment cost changed".into());}
            records[index]["states"]=json!(states(&assessment,&grid)?);records[index]["local"]=json!(assessment.local_measure);records[index]["run"]=json!(assessment.run_measure);
            Ok(())
        })();
        if let Err(error)=result{records[index]["states"]=json!(vec!["Failed";grid.len()]);records[index]["error"]=json!(error);}
        records[index]["seconds"]=json!(t.elapsed().as_secs_f64());records[index]["peak_rss_bytes"]=json!(peak_rss());
        writeln!(journal,"{}",records[index]).map_err(|e|e.to_string())?;journal.flush().map_err(|e|e.to_string())?;cache.clear_measurements();
    }
    let mut points=Vec::new();let mut selected=BTreeSet::new();
    for(g,c)in grid.iter().enumerate(){let winner=records.iter().filter(|r|r["states"][g].as_str()==Some("Verified")).min_by_key(|r|(r["cost_bits"].as_u64().unwrap_or(u64::MAX),r["index"].as_u64().unwrap_or(u64::MAX)));let upper=winner.and_then(|r|r["cost_bits"].as_u64());let index=winner.and_then(|r|r["index"].as_u64());if let Some(i)=index{selected.insert(i as usize);}let lower=records.iter().filter(|r|r["states"][g].as_str()!=Some("Violates")).filter_map(|r|r["cost_lower_bound"].as_u64()).min();points.push(json!({"constraint":c,"selected":index,"upper_cost":upper,"lower_cost":lower,"gap":upper.zip(lower).map(|(u,l)|u.saturating_sub(l))}));}
    let mut replays=Vec::new();
    for index in selected{
        let artifact=if index==0{base.clone()}else{bank.candidate(choices[index-1])?}.f32_literals()?;let bytes=artifact.to_bytes()?;let path=out.join(format!("selected.{index}.bin"));std::fs::write(&path,&bytes).map_err(|e|e.to_string())?;drop(bytes);
        let saved=std::fs::read(&path).map_err(|e|e.to_string())?;let decoded=Artifact::from_bytes(&saved,&native.declarations)?;
        if decoded.to_bytes()?!=saved || decoded.places!=base.places{return Err("selected saved-byte canonical/place check failed".into());}decoded.validate_coverage(&native)?;
        let replay=assess_once(&local,&run,&decoded,grid[0],&mut cache)?;
        if Some(replay.cost.total())!=records[index]["cost_bits"].as_u64() || json!(states(&replay,&grid)?)!=records[index]["states"]{return Err("selected replay changed cost or grid verdict".into());}
        replays.push(json!({"index":index,"file":path.file_name().and_then(|p|p.to_str()),"sha256":sha256(&path)?,"bytes":saved.len(),"cost_bits":replay.cost.total(),"local":replay.local_measure,"run":replay.run_measure}));cache.clear_measurements();
    }
    write_json(&out.join("REPORT.json"),&json!({"scope":scope,"records":records,"points":points,"selected_saved_byte_replays":replays,"seconds":begun.elapsed().as_secs_f64(),"peak_rss_bytes":peak_rss()}))?;Ok(())
}
