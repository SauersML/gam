//! Complete declared paired bank, streamed without singleton or cost pruning.
//! Default vpd4l: EXPORT SPEC OUT max_bank=193 start=0 count=193 backend=cuda cuda_local=0 trace_bytes=2147483648 codec_bytes=0
//! Mapped: add mode=mapped native_layers=27 ranks=16 local_sequences=1 local_context=16
//! spec_sha=SHA256 max_bank=33 start=0 count=33 cuda_share_native=0 codec_bytes=N.
//! Mapped mode compares declared Copy templates with explicit residuals; it is not automatic mechanism discovery.
use gam_mpd::acceptance::{Assessment, Constraint, CostCache, Local, RunCheck, FamilyRun, Change, Edit, Episode, assess_once, assess_once_with_native_codec, structural_cost};
use gam_mpd::artifact::{Artifact, EncodedArtifact};
use gam_mpd::import::import_language_model;
use gam_mpd::coder_capture::sha256;
use gam_mpd::counterfactual::{Decoder, Spec, passages};
use gam_mpd::operator_program::{FamilyInputs, NativeOperatorCodec, SequenceLayout, SlotValues};
use gam_mpd::precision::{DecodableArtifact, FidelityVerdict};
use gam_mpd::proposals::{CopyResidualBank, MappedCopyResidualBank, CopyResidualChoice};
use gam_mpd::run_check::{LanguageRun, layer_nodes, split_sites};
use serde_json::{Value, json};
use serde::Deserialize;
use gam_mpd::device_family_run::DeviceFamilyRun;
use std::{collections::{BTreeMap,BTreeSet}, io::Write, path::Path, time::Instant};
const RANKS:[usize;4]=[8,32,64,96];
const DELTAS:[f64;5]=[0.05,0.1,0.2,0.5,1.0];
const EPSILONS:[f64;6]=[0.001,0.01,0.03,0.1,0.3,1.0];
const SPEC_SHA:&str="c74555341a75b1a6dcab4238c125f36c922c7683dd151b3ceb88ac399e963d0d";
fn write_json(path:&Path,value:&Value)->Result<(),String>{std::fs::write(path,serde_json::to_vec_pretty(value).map_err(|e|e.to_string())?).map_err(|e|e.to_string())}
fn family(rows:&[Vec<u32>],count:usize,context:usize)->Result<FamilyInputs,String>{
    if count==0 || context==0 || rows.len()<count || rows[..count].iter().any(|r|r.len()<context){return Err("declared Local sequences/context unavailable".into());}
    let mut tokens=Vec::new();let mut sequence=Vec::new();let mut position=Vec::new();
    for (s,row) in rows[..count].iter().enumerate(){for (p,&t) in row[..context].iter().enumerate(){tokens.push(t);sequence.push(u32::try_from(s).map_err(|e|e.to_string())?);position.push(u32::try_from(p).map_err(|e|e.to_string())?);}}
    Ok(FamilyInputs{rows:tokens.len(),slots:vec![SlotValues::Tokens(tokens)],layout:Some(SequenceLayout{sequence,position})})
}
fn states(a:&Assessment,grid:&[Constraint])->Result<Vec<&'static str>,String>{grid.iter().map(|c|{
    let (l,r)=(a.local.with_tolerance(c.local)?.verdict(),a.run.with_tolerance(c.run)?.verdict());
    Ok(if l==FidelityVerdict::Violates || r==FidelityVerdict::Violates {"Violates"} else if l==FidelityVerdict::Meets && r==FidelityVerdict::Meets {"Verified"} else {"Unresolved"})
}).collect()}
fn peak_rss()->Option<u64>{std::fs::read_to_string("/proc/self/status").ok()?.lines().find_map(|s|s.strip_prefix("VmHWM:")?.split_whitespace().next()?.parse::<u64>().ok()?.checked_mul(1024))}
#[derive(Debug)]
struct Mode { mapped:bool,layers:Vec<usize>,ranks:Vec<usize>,sequences:usize,context:usize,spec_sha:String }
fn list(text:&str,positive:bool)->Result<Vec<usize>,String>{
    let values=text.split(',').map(|x|x.parse::<usize>().map_err(|e|e.to_string())).collect::<Result<Vec<_>,_>>()?;
    if values.is_empty() || (positive&&values.contains(&0)) || values.iter().collect::<BTreeSet<_>>().len()!=values.len(){return Err("declare nonempty distinct integers; ranks must be positive".into());}Ok(values)
}
fn mode(keys:&BTreeMap<&str,&str>)->Result<Mode,String>{
    let required=|k:&str|keys.get(k).copied().ok_or_else(||format!("mapped mode requires {k}"));
    match keys.get("mode").copied().unwrap_or("vpd4l") {
        "vpd4l"=>{if ["native_layers","ranks","local_sequences","local_context","spec_sha"].iter().any(|k|keys.contains_key(k)){return Err("mapped options require mode=mapped; frozen4L defaults unchanged".into());}
            Ok(Mode{mapped:false,layers:(0..4).collect(),ranks:RANKS.to_vec(),sequences:2,context:16,spec_sha:SPEC_SHA.into()})},
        "mapped"=>{let sha=required("spec_sha")?;if sha.len()!=64 || !sha.bytes().all(|b|b.is_ascii_hexdigit()){return Err("spec_sha must be an explicit64hex SHA256".into());}
            let sequences=required("local_sequences")?.parse::<usize>().map_err(|e|e.to_string())?;let context=required("local_context")?.parse::<usize>().map_err(|e|e.to_string())?;
            if sequences==0||context==0{return Err("positive Local family dimensions required".into());}
            Ok(Mode{mapped:true,layers:list(required("native_layers")?,false)?,ranks:list(required("ranks")?,true)?,sequences,context,spec_sha:sha.to_ascii_lowercase()})},
        _=>Err("mode must be vpd4l or mapped".into()),
    }
}
#[derive(Deserialize)]
struct Threshold { local:f64,run:f64 }
#[derive(Deserialize)]
struct NativeEdit {node:usize,width:usize,rows:Option<Vec<usize>>,columns:[usize;2],scale:Option<f64>,add:Option<f64>}
#[derive(Deserialize)]
struct NativeEpisode {id:String,group:String,edits:Vec<NativeEdit>}
#[derive(Deserialize)]
struct MappedRunSpec {checkpoint_sha256:String,sequences:usize,context:usize,nodes:usize,batch_rows:usize,constraints:Vec<Threshold>,episodes:Vec<NativeEpisode>}
fn episodes(spec:&MappedRunSpec,model:&gam_mpd::operator_program::OperatorProgram,rows:usize)->Result<Vec<Episode>,String>{
    let interfaces=model.interfaces().map_err(|e|e.to_string())?;let mut ids=BTreeSet::new();let mut out=Vec::new();
    for e in &spec.episodes {
        if e.id.is_empty()||e.group.is_empty()||!ids.insert(&e.id){return Err("empty/duplicate native episode ID/group".into());}
        let mut edits=Vec::new();for x in &e.edits {
            let width=interfaces.get(x.node).ok_or("native edit node absent")?.width();
            if width!=x.width||x.columns[0]>=x.columns[1]||x.columns[1]>width||x.rows.as_ref().is_some_and(|r|r.is_empty()||r.iter().any(|&i|i>=rows)){return Err("native edit width/range/rows invalid".into());}
            let change=match(x.scale,x.add){(Some(v),None)if v.is_finite()=>Change::Scale(v),(None,Some(v))if v.is_finite()=>Change::Add(v),_=>return Err("exactly one finite native scale/add required".into())};
            edits.push(Edit{node:x.node,rows:x.rows.clone(),columns:x.columns[0]..x.columns[1],change});
        }out.push(Episode{id:e.id.clone(),group:e.group.clone(),edits});
    }Ok(out)
}
enum LazyBank<'a> { Vpd(CopyResidualBank<'a>),Mapped(MappedCopyResidualBank<'a>) }
impl LazyBank<'_> {
    fn choices(&self)->Vec<CopyResidualChoice>{match self{Self::Vpd(b)=>b.choices().collect(),Self::Mapped(b)=>b.choices().collect()}}
    fn candidate(&self,c:CopyResidualChoice)->Result<Artifact,String>{match self{Self::Vpd(b)=>b.candidate(c),Self::Mapped(b)=>b.candidate(c)}}
}
fn main()->Result<(),String>{
    gam_mpd::engine::log_to_stderr();
    let args:Vec<String>=std::env::args().skip(1).collect();
    if args.len()<3{return Err("EXPORT SPEC OUT max_bank=193 start=0 count=193 backend=cuda cuda_local=0 trace_bytes=2147483648".into());}
    let (export,spec_path,out)=(Path::new(&args[0]),Path::new(&args[1]),Path::new(&args[2]));
    let mut keys=BTreeMap::new();
    for arg in &args[3..]{let(k,v)=arg.split_once('=').ok_or("expected KEY=VALUE")?;if !["max_bank","start","count","backend","cuda_local","trace_bytes","cuda_share_native","codec_bytes","mode","native_layers","ranks","local_sequences","local_context","spec_sha"].contains(&k)||keys.insert(k,v).is_some(){return Err("unknown/duplicate option".into());}}
    let mode=mode(&keys)?;
    let number=|k:&str|->Result<usize,String>{keys.get(k).ok_or_else(||format!("declare {k}"))?.parse().map_err(|e|format!("{e}"))};
    let(max_bank,start,count,cuda_local,trace_bytes)=(number("max_bank")?,number("start")?,number("count")?,number("cuda_local")?,number("trace_bytes")?);
    let backend=*keys.get("backend").ok_or("declare backend")?;
    let codec_bytes=keys.get("codec_bytes").unwrap_or(&"0").parse::<usize>().map_err(|e|e.to_string())?;
    if mode.mapped && (!keys.contains_key("codec_bytes") || !keys.contains_key("cuda_share_native")){return Err("mapped mode explicitly declares codec_bytes and cuda_share_native=0".into());}
    let share_native=keys.get("cuda_share_native").unwrap_or(&"0").parse::<usize>().map_err(|e|e.to_string())?;
    if share_native>1 || (share_native==1 && (backend!="cuda" || mode.mapped)){return Err("cuda_share_native requires explicit vpd4l CUDA; mapped backend lacks this API".into());}
    if count==0 || cuda_local>1 || !["cpu","cuda"].contains(&backend) || (cuda_local==1&&backend!="cuda"){return Err("invalid scope/backend".into());}
    let config:Value=serde_json::from_slice(&std::fs::read(export.join("export.json")).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    let n=|k:&str|->Result<usize,String>{usize::try_from(config["config"][k].as_u64().ok_or("missing architecture integer")?).map_err(|e|e.to_string())};
    let(layers,heads,kv)=(n("n_layers")?,n("n_heads")?,n("n_kv_heads")?);
    if mode.layers.iter().any(|&l|l>=layers){return Err("declared native layer ID outside original export".into());}
    let complete=CopyResidualBank::cardinality(mode.layers.iter().map(|_|heads),&mode.ranks)?;
    if complete>max_bank{return Err(format!("complete bank requires {complete}, max_bank={max_bank}"));}
    if (!mode.mapped&&(layers!=4 || heads!=6)) || kv==0 || heads%kv!=0 || start.checked_add(count).ok_or("scope overflow")?>complete{return Err("require4L6heads and bounded declared diagnostic range".into());}
    if sha256(spec_path)?!=mode.spec_sha{return Err("Run spec differs from frozen seven-episode control".into());}
    if out.exists(){return Err("fresh output directory required".into());}std::fs::create_dir_all(out).map_err(|e|e.to_string())?;
    let mapped_spec=if mode.mapped{Some(serde_json::from_slice::<MappedRunSpec>(&std::fs::read(spec_path).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?)}else{None};
    let grid:Vec<Constraint>=if let Some(spec)=&mapped_spec{spec.constraints.iter().map(|c|Constraint{local:c.local,run:c.run}).collect()}else{DELTAS.iter().flat_map(|&local|EPSILONS.iter().map(move|&run|Constraint{local,run})).collect()};
    if grid.is_empty()||grid.iter().any(|c|!c.local.is_finite()||!c.run.is_finite()||c.local<0.||c.run<0.){return Err("nonempty finite nonnegative tolerance grid required".into());}
    if mapped_spec.as_ref().is_some_and(|s|s.sequences==0||s.context==0||s.batch_rows==0||s.episodes.is_empty()){return Err("mapped spec requires positive family/batch and episodes".into());}
    let batch=mapped_spec.as_ref().map_or(16,|s|s.batch_rows);
    let run_rows=mapped_spec.as_ref().map(|s|s.sequences.checked_mul(s.context).ok_or("Run family size overflow")).transpose()?.unwrap_or(16);
    let scope=json!({"mode":if mode.mapped{"mapped"}else{"vpd4l"},"native_layers":mode.layers,"all_heads":mode.layers.len()*heads,"original_native_layers":layers,"ranks":mode.ranks,"families":["NativeSvd","CopyResidual"],"complete_count_including_native":complete,"range":{"start":start,"count":count},"native_control_always_measured":true,"grid":grid,"local":{"sequences":mode.sequences,"context":mode.context,"batch":batch,"denominator":if mode.mapped{gam_mpd::attention_map::LOCAL_DENOMINATOR}else{"RMS native split attention contribution row L2 norm"},"ascent":0,"scope":"short screen, not full context"},"run":{"episodes":mapped_spec.as_ref().map_or(7,|s|s.episodes.len()),"spec_sha256":mode.spec_sha,"rows":run_rows,"source_spec_bank_options":"candidate paths/budget/max_bank/backend in source spec are superseded by explicit streamed bank options"},"backend":backend,"cuda_local":cuda_local,"cuda_share_native":share_native,"trace_bytes":trace_bytes,"codec_bytes":codec_bytes,"pruning":false,"unmeasured_and_failed_retained":true,"unknown_cost_lower_bound":0,"optimality":"complete declared single-substitution bank only; no global claim","method":"fixed Copy template with priced residual vs same-rank native SVD; not automatic mechanism discovery","selected_followup":"expanded512 context and80 episodes required before broader claim"});
    write_json(&out.join("SCOPE.json"),&scope)?;
    std::fs::copy(spec_path,out.join("RUN_SPEC.json")).map_err(|e|e.to_string())?;
    let mut inputs=BTreeMap::new();inputs.insert("spec".to_string(),sha256(spec_path)?);inputs.insert("export.json".into(),sha256(&export.join("export.json"))?);
    for name in config["files"].as_object().ok_or("missing export files")?.keys(){let file=format!("{name}.f64");inputs.insert(file.clone(),sha256(&export.join(file))?);}
    let mut sources=BTreeMap::new();
    for(name,text)in[("driver",include_str!("mpd_copy_residual_stream_2951.rs")),("proposals",include_str!("../src/proposals.rs")),("acceptance",include_str!("../src/acceptance.rs")),("artifact",include_str!("../src/artifact.rs")),("codec",include_str!("../src/codec.rs")),("rules",include_str!("../src/rules.rs")),("attention_map",include_str!("../src/attention_map.rs")),("device_family_run",include_str!("../src/device_family_run.rs"))]{let path=out.join(format!("{name}.source"));std::fs::write(&path,text).map_err(|e|e.to_string())?;sources.insert(name,sha256(&path)?);std::fs::remove_file(path).map_err(|e|e.to_string())?;}
    write_json(&out.join("PROVENANCE.json"),&json!({"inputs":inputs,"sources":sources,"binary":sha256(&std::env::current_exe().map_err(|e|e.to_string())?)?}))?;
    let begun=Instant::now();
    let decoder=if mode.mapped{None}else{Some(Decoder::from_export(export)?)};
    let language_spec=if let Some(decoder)=&decoder{Some(Spec::load(spec_path,decoder)?)}else{None};
    if language_spec.as_ref().is_some_and(|s|s.episodes.len()!=7||s.rows!=16){return Err("frozen seven episode sixteen-row control required".into());}
    let imported=if let Some(spec)=&mapped_spec{import_language_model(export,spec.sequences,spec.context)?}else{import_language_model(export,1,1)?};
    if let Some(spec)=&mapped_spec{if imported.record["source"]["weights_sha256"].as_str()!=Some(spec.checkpoint_sha256.as_str())||imported.program.nodes.len()!=spec.nodes{return Err("original checkpoint/node-map identity mismatch".into());}}
    let native=if mode.mapped{imported.program.clone()}else{split_sites(&imported.program)?};
    let base=Artifact::native(&native)?.f32_literals()?;
    let bank=if mode.mapped{LazyBank::Mapped(MappedCopyResidualBank::new(&base,&mode.layers,&mode.ranks,max_bank)?)}else{LazyBank::Vpd(CopyResidualBank::new(&base,&layer_nodes(&native,layers)?,heads/kv,&mode.ranks,max_bank)?)};
    let choices=bank.choices();
    if choices.len().checked_add(1)!=Some(complete){return Err("mapped inventory differs from declared export cardinality".into());}
    let native_codec=if codec_bytes==0{None}else{Some(NativeOperatorCodec::new(&base.program,codec_bytes).map_err(|e|e.to_string())?)};
    let run_passages=if let Some(spec)=&language_spec{passages(export,spec.rows)?}else{Vec::new()};
    let mut language_run=if let (Some(decoder),Some(spec))=(&decoder,&language_spec){Some(LanguageRun::new(decoder,&native,spec,&run_passages,1)?)}else{None};
    let family_run=if let Some(spec)=&mapped_spec{Some(FamilyRun{model:&native,family:imported.contract.family.clone(),readouts:1,episodes:episodes(spec,&native,imported.contract.family.rows)?})}else{None};
    let local_family=family(&passages(export,mode.context)?,mode.sequences,mode.context)?;
    let mut local=Local::new(&native,local_family.clone(),None,batch);
    let device=if backend=="cuda"{Some(gam_gpu::tensor::Device::accelerator(gam_gpu::GpuPolicy::Required).map_err(|e|e.to_string())?.ok_or("CUDA required")?)}else{None};
    if let Some(device)=&device{
        if cuda_local==1{local=local.with_cuda(device.clone(),trace_bytes)?;}
        if let Some(run)=language_run.take(){let mut run=run.with_cuda(device.clone(),trace_bytes)?;if share_native==1{let source=if let Some(codec)=&native_codec{EncodedArtifact::of_with_native_codec(&base,codec)?.using_native_codec(codec).decode()?}else{Artifact::from_bytes(&base.to_bytes()?,&native.declarations)?};run=run.with_cuda_native_source(&source)?;}language_run=Some(run);}
    }
    let device_run=if let (Some(run),Some(device))=(&family_run,&device){Some(DeviceFamilyRun::new(run,device.clone(),trace_bytes)?)}else{None};
    let run:&dyn RunCheck=if let Some(run)=&device_run{run}else if let Some(run)=&family_run{run}else{language_run.as_ref().ok_or("missing Run backend")?};
    let mut cache=CostCache::default();let mut records:Vec<Value>=choices.iter().enumerate().map(|(i,c)|json!({"index":i+1,"choice":c,"cost_bits":null,"cost_lower_bound":0,"states":vec!["Unevaluated";grid.len()]})).collect();
    records.insert(0,json!({"index":0,"label":"native","cost_bits":null,"cost_lower_bound":0,"states":vec!["Unevaluated";grid.len()]}));
    let mut journal=std::fs::OpenOptions::new().create_new(true).write(true).open(out.join("ASSESSMENTS.jsonl")).map_err(|e|e.to_string())?;
    for index in std::iter::once(0).chain((start..start+count).filter(|&i|i!=0)){
        let t=Instant::now();eprintln!("candidate {index}/{complete}: start");
        let result=(||->Result<(),String>{
            let artifact=if index==0{base.clone()}else{bank.candidate(choices[index-1])?};
            if artifact.places!=base.places{return Err("native places changed".into());}let artifact=artifact.f32_literals()?;
            let cost=structural_cost(&artifact,&mut cache)?;records[index]["cost_bits"]=json!(cost.total());records[index]["cost_lower_bound"]=json!(cost.total());
            let assessment=if let Some(codec)=&native_codec {assess_once_with_native_codec(&local,run,&artifact,grid[0],&mut cache,codec)?} else {assess_once(&local,run,&artifact,grid[0],&mut cache)?};
            if assessment.cost!=cost{return Err("assessment cost changed".into());}
            records[index]["states"]=json!(states(&assessment,&grid)?);records[index]["local"]=json!(assessment.local_measure);records[index]["run"]=json!(assessment.run_measure);
            Ok(())
        })();
        if let Err(error)=result{records[index]["states"]=json!(vec!["Failed";grid.len()]);records[index]["error"]=json!(error);}
        records[index]["seconds"]=json!(t.elapsed().as_secs_f64());records[index]["codec_usage"]=json!(native_codec.as_ref().map(|c|c.usage()));records[index]["peak_rss_bytes"]=json!(peak_rss());
        writeln!(journal,"{}",records[index]).map_err(|e|e.to_string())?;journal.flush().map_err(|e|e.to_string())?;
    }
    let mut points=Vec::new();let mut selected=BTreeSet::new();
    for(g,c)in grid.iter().enumerate(){let winner=records.iter().filter(|r|r["states"][g].as_str()==Some("Verified")).min_by_key(|r|(r["cost_bits"].as_u64().unwrap_or(u64::MAX),r["index"].as_u64().unwrap_or(u64::MAX)));let upper=winner.and_then(|r|r["cost_bits"].as_u64());let index=winner.and_then(|r|r["index"].as_u64());if let Some(i)=index{selected.insert(i as usize);}let lower=records.iter().filter(|r|r["states"][g].as_str()!=Some("Violates")).filter_map(|r|r["cost_lower_bound"].as_u64()).min();points.push(json!({"constraint":c,"selected":index,"upper_cost":upper,"lower_cost":lower,"gap":upper.zip(lower).map(|(u,l)|u.saturating_sub(l))}));}
    let mut replays=Vec::new();
    for index in selected{
        let artifact=if index==0{base.clone()}else{bank.candidate(choices[index-1])?}.f32_literals()?;let bytes=artifact.to_bytes()?;let path=out.join(format!("selected.{index}.bin"));std::fs::write(&path,&bytes).map_err(|e|e.to_string())?;drop(bytes);
        let saved=std::fs::read(&path).map_err(|e|e.to_string())?;let decoded=Artifact::from_bytes(&saved,&native.declarations)?;
        if decoded.to_bytes()?!=saved || decoded.places!=base.places{return Err("selected saved-byte canonical/place check failed".into());}decoded.validate_coverage(&native)?;
        let replay=assess_once(&local,run,&decoded,grid[0],&mut cache)?;
        if Some(replay.cost.total())!=records[index]["cost_bits"].as_u64() || json!(states(&replay,&grid)?)!=records[index]["states"]{return Err("selected replay changed cost or grid verdict".into());}
        let isolated_kl=gam_mpd::local_kl::isolated_downstream_kl(&native,&decoded,&local_family,batch)?;
        replays.push(json!({"isolated_downstream_patch_kl":isolated_kl,"diagnostic_only":true,"index":index,"file":path.file_name().and_then(|p|p.to_str()),"sha256":sha256(&path)?,"bytes":saved.len(),"cost_bits":replay.cost.total(),"local":replay.local_measure,"run":replay.run_measure}));
    }
    write_json(&out.join("REPORT.json"),&json!({"scope":scope,"records":records,"points":points,"selected_saved_byte_replays":replays,"seconds":begun.elapsed().as_secs_f64(),"peak_rss_bytes":peak_rss(),"run_stage_seconds":language_run.as_ref().map(|r|r.timing()),"native_codec":native_codec.as_ref().map(|c|json!({"stats":c.stats(),"usage":c.usage(),"selected_replay":"ordinary uncached standalone decoding"}))}))?;Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn default_vpd_mode_keeps_frozen193_and_family() {
        let m=mode(&BTreeMap::new()).unwrap();assert!(!m.mapped);assert_eq!(m.layers,vec![0,1,2,3]);assert_eq!(m.ranks,RANKS);assert_eq!((m.sequences,m.context),(2,16));assert_eq!(m.spec_sha,SPEC_SHA);
        assert_eq!(CopyResidualBank::cardinality(m.layers.iter().map(|_|6),&m.ranks).unwrap(),193);
        let inputs=family(&[vec![1;16],vec![2;16]],m.sequences,m.context).unwrap();assert_eq!(inputs.rows,32);
        assert!(mode(&BTreeMap::from([("native_layers","27")])).is_err());
    }
    #[test]
    fn mapped_mode_explicit_native27_all16_rank16_and_no_implicit_options() {
        let keys=BTreeMap::from([("mode","mapped"),("native_layers","27"),("ranks","16"),("local_sequences","1"),("local_context","16"),("spec_sha",SPEC_SHA)]);
        let m=mode(&keys).unwrap();assert!(m.mapped);assert_eq!(m.layers,vec![27]);assert_eq!(m.ranks,vec![16]);assert_eq!(CopyResidualBank::cardinality(m.layers.iter().map(|_|16),&m.ranks).unwrap(),33);
        for missing in ["native_layers","ranks","local_sequences","local_context","spec_sha"]{let mut k=keys.clone();k.remove(missing);assert!(mode(&k).is_err());}
        let mut k=keys.clone();k.insert("native_layers","27,27");assert!(mode(&k).is_err());let mut k=keys.clone();k.insert("ranks","0");assert!(mode(&k).is_err());
    }
    #[test]
    fn mapped_spec_preserves_declared_episode_nodes_and_nine_point_grid() {
        let source=r#"{"checkpoint_sha256":"fixture","sequences":1,"context":16,"nodes":2974,"batch_rows":16,"constraints":[{"local":0.01,"run":0.001},{"local":0.01,"run":0.01},{"local":0.01,"run":0.1},{"local":0.1,"run":0.001},{"local":0.1,"run":0.01},{"local":0.1,"run":0.1},{"local":0.5,"run":0.001},{"local":0.5,"run":0.01},{"local":0.5,"run":0.1}],"episodes":[{"id":"clean","group":"clean","edits":[]},{"id":"gate","group":"gate-removal","edits":[{"node":2966,"width":3072,"columns":[0,3072],"rows":null,"scale":0.0,"add":null}]},{"id":"up","group":"up-scale","edits":[]},{"id":"norm","group":"norm-control","edits":[]},{"id":"combined","group":"combined","edits":[]}] }"#;
        let spec:MappedRunSpec=serde_json::from_str(source).unwrap();assert_eq!(spec.episodes.len(),5);assert_eq!(spec.constraints.len(),9);assert_eq!((spec.sequences,spec.context),(1,16));
        assert_eq!(spec.episodes[1].edits[0].node,2966);assert_eq!(spec.episodes[1].edits[0].width,3072);
    }
}
