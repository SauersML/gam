//! Parameterized controlled shared-geometry fit. EXPORT EXTRACT CONFIG OUT host|cuda
//! No held-out body transfer or whole-model acceptance is inferred from fitting.
use gam_mpd::{acceptance::{CostCache,structural_cost},artifact::Artifact,coder_capture::sha256,import::import_language_model,
    operator_program::OperatorProgram,resident_rule_fit::{self,OutputGroup,Settings},run_check::{layer_nodes,split_sites},shared_geometry_pilot::{self,Arm}};
use ndarray::{Array2,s};
use serde_json::{Value,json};
use std::{path::Path,time::Instant};
fn save(path:&Path,value:&Value)->Result<(),String>{std::fs::write(path,serde_json::to_vec_pretty(value).map_err(|e|e.to_string())?).map_err(|e|e.to_string())}
fn integer(value:&Value,key:&str)->Result<usize,String>{usize::try_from(value[key].as_u64().ok_or_else(||format!("missing integer {key}"))?).map_err(|e|e.to_string())}
fn load(root:&Path,panel:&Value,layer:usize,role:&str)->Result<(Array2<f64>,Value),String>{
    let values=panel["values"].as_array().ok_or("panel values absent")?;
    let found:Vec<_>=values.iter().filter(|v|v["layer"].as_u64()==Some(layer as u64)&&v["role"].as_str()==Some(role)).collect();
    if found.len()!=1{return Err("unique declared layer/role array required".into());}
    let descriptor=found[0];let file=descriptor["file"].as_str().ok_or("array filename absent")?;
    if Path::new(file).components().count()!=1 || !matches!(Path::new(file).components().next(),Some(std::path::Component::Normal(_))){return Err("sibling array filename required".into());}
    let path=root.join(file);let hash=sha256(&path)?;
    if descriptor["sha256"].as_str()!=Some(&hash){return Err("native archive array hash mismatch".into());}
    let rows=integer(panel,"rows")?;let width=integer(descriptor,"width")?;
    let bytes=std::fs::read(&path).map_err(|e|e.to_string())?;let count=rows.checked_mul(width).ok_or("array shape overflow")?;
    if bytes.len()!=count.checked_mul(8).ok_or("array bytes overflow")?{return Err("native array shape mismatch".into());}
    let data:Vec<_>=bytes.chunks_exact(8).map(|b|f64::from_le_bytes(b.try_into().expect("eight-byte chunk"))).collect();
    if data.iter().any(|x|!x.is_finite()){return Err("nonfinite native array".into());}
    Ok((Array2::from_shape_vec((rows,width),data).map_err(|e|e.to_string())?,json!({"descriptor":descriptor,"sha256":hash})))
}
fn panel<'a>(manifest:&'a Value,name:&str)->Result<&'a Value,String>{
    let panels=manifest["panels"].as_array().ok_or("panels absent")?;
    let found:Vec<_>=panels.iter().filter(|p|p["name"].as_str()==Some(name)).collect();
    if found.len()!=1{return Err("unique named native panel required".into());}Ok(found[0])
}
fn targets(rows:usize,parts:&[Array2<f64>])->Result<Array2<f64>,String>{
    let width=parts.iter().try_fold(0usize,|n,p|n.checked_add(p.ncols()).ok_or("target width overflow"))?;
    let mut joined=Array2::zeros((rows,width));let mut start=0;
    for p in parts{if p.nrows()!=rows{return Err("aligned native row count required".into());}let end=start+p.ncols();joined.slice_mut(s![..,start..end]).assign(p);start=end;}Ok(joined)
}
fn saved_program(path:&Path,program:&OperatorProgram)->Result<Artifact,String>{
    let artifact=Artifact::native(program)?.f32_literals()?;let bytes=artifact.to_bytes()?;
    std::fs::write(path,&bytes).map_err(|e|e.to_string())?;
    let replay=Artifact::from_bytes(&std::fs::read(path).map_err(|e|e.to_string())?,&program.declarations)?;
    if replay.to_bytes()?!=bytes{return Err("ordinary canonical saved-byte replay mismatch".into());}Ok(replay)
}
fn main()->Result<(),String>{
    let args:Vec<_>=std::env::args().skip(1).collect();if args.len()!=5{return Err("EXPORT EXTRACT CONFIG OUT host|cuda".into());}
    let (export,manifest_path,config_path,out)=(Path::new(&args[0]),Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]));
    if out.exists(){return Err("fresh output required".into());}std::fs::create_dir_all(out).map_err(|e|e.to_string())?;
    let started=Instant::now();let config:Value=serde_json::from_slice(&std::fs::read(config_path).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    let manifest:Value=serde_json::from_slice(&std::fs::read(manifest_path).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    if config["extract_sha256"].as_str()!=Some(&sha256(manifest_path)?){return Err("frozen extraction manifest hash mismatch".into());}
    if config["export_json_sha256"].as_str()!=Some(&sha256(&export.join("export.json"))?){return Err("frozen source export hash mismatch".into());}
    let arm=match config["arm"].as_str(){Some("learned")=>Arm::Learned,Some("frozen_native")=>Arm::FrozenNative,Some("frozen_random")=>Arm::FrozenRandom,Some("untied")=>Arm::Untied,_=>return Err("explicit declared control arm required".into())};
    let uses:Vec<usize>=config["uses"].as_array().ok_or("native uses absent")?.iter().map(|v|usize::try_from(v.as_u64().ok_or("native use must be integer")?).map_err(|e|e.to_string())).collect::<Result<_,_>>()?;
    if uses!=vec![0,1]{return Err("first pilot freezes discovery uses0/1; transfer is a separate declared stage".into());}
    let seed=config["seed"].as_u64().ok_or("seed absent")?;
    let settings:Settings=serde_json::from_value(config["settings"].clone()).map_err(|e|e.to_string())?;
    let imported=import_language_model(export,1,1)?;let native=split_sites(&imported.program)?;let layers=layer_nodes(&native,4)?;
    let mut proposal=shared_geometry_pilot::build(&native,&layers,&uses,arm,seed)?;
    let train=panel(&manifest,"train")?;let valid=panel(&manifest,"eval")?;
    if integer(train,"rows")?!=4096 || integer(valid,"rows")?!=1024{return Err("frozen4096train/1024eval native rows required".into());}
    let root=manifest_path.parent().ok_or("native manifest parent absent")?;
    let mut x=vec![];let mut y=vec![];let mut vx=vec![];let mut vy=vec![];let mut groups=vec![];let mut provenance=vec![];let mut column=0;
    for &use_id in &uses{
        let (a,ad)=load(root,train,use_id,"input")?;let (b,bd)=load(root,train,use_id,"write")?;
        let (c,cd)=load(root,valid,use_id,"input")?;let (d,dd)=load(root,valid,use_id,"write")?;
        if ad["sha256"]==cd["sha256"] || bd["sha256"]==dd["sha256"]{return Err("identical training/evaluation array rejected".into());}
        if a.ncols()!=768 || b.ncols()!=768 || c.ncols()!=768 || d.ncols()!=768{return Err("declared native full768 interfaces required".into());}
        groups.push(OutputGroup {label:format!("nativeMLP{use_id}"),start:column,end:column+768});column+=768;
        provenance.push(json!({"use":use_id,"train_input":ad,"train_write":bd,"eval_input":cd,"eval_write":dd}));x.push(a);y.push(b);vx.push(c);vy.push(d);
    }
    let y=targets(4096,&y)?;let vy=targets(1024,&vy)?;
    let source=saved_program(&out.join("source-pool.artifact"),&proposal.program)?;proposal.program=source.program;
    save(&out.join("PROVENANCE.json"),&json!({"config":config,"config_sha256":sha256(config_path)?,"extract_sha256":sha256(manifest_path)?,"export_record":imported.record,"arrays":provenance,"body_operators":proposal.body_operators,"trainable":proposal.trainable,"groups":groups,"source_sha256":sha256(&out.join("source-pool.artifact"))?,"binary_sha256":sha256(&std::env::current_exe().map_err(|e|e.to_string())?)?,"claim_scope":"supplied GELU architecture; learned overcomplete numerical geometry diagnostic, not interpreted algorithm recovery","initialization":"all arms native use0 reader geometry except declared seeded random; per-use native writers; identity input bindings; untied starts identical effective use0 readers","data_scope":"4096 unique aligned training tokens,8192site-row targets; eval1024 previously examined project tokens, not untouched confirmation","transfer_status":"not performed by this discovery-only invocation"}))?;
    let device=match args[4].as_str(){"host"=>gam_gpu::tensor::Device::host(),"cuda"=>gam_gpu::tensor::Device::accelerator(gam_gpu::GpuPolicy::Required).map_err(|e|e.to_string())?.ok_or("CUDA required")?,_=>return Err("host|cuda backend required".into())};
    let frozen:Vec<_>=proposal.body_operators.iter().map(|i|proposal.program.operators[*i].clone()).collect();
    let result=resident_rule_fit::fit_grouped(&device,&proposal.program,&x,&y,&vx,&vy,&groups,&proposal.trainable,settings.clone())?;
    if arm!=Arm::Learned{for (index,before) in proposal.body_operators.iter().zip(&frozen){if result.program.operators[*index]!=*before{return Err("frozen body changed during fitting".into());}}}
    let report=result.report;proposal.program=result.program;
    let replay=saved_program(&out.join("fitted-pool.artifact"),&proposal.program)?;
    let train_measure=resident_rule_fit::measure_grouped(&device,&replay.program,&x,&y,&groups,settings.numeric_bytes,settings.forward_rows)?;
    let valid_measure=resident_rule_fit::measure_grouped(&device,&replay.program,&vx,&vy,&groups,settings.numeric_bytes,settings.forward_rows)?;
    // Export from ONE decoded pool, preserving shared operator Arcs between functions.
    proposal.program=replay.program.clone();let mut exports=vec![];
    for slot in 0..uses.len(){let function=shared_geometry_pilot::function(&proposal,slot)?;let path=out.join(format!("use{}.artifact",uses[slot]));let standalone=saved_program(&path,&function)?;exports.push(json!({"native_use":uses[slot],"path":path,"sha256":sha256(&path)?,"standalone_cost_not_joint_cost":structural_cost(&standalone,&mut CostCache::default())?.total()}));}
    let pool_cost=structural_cost(&replay,&mut CostCache::default())?;
    save(&out.join("REPORT.json"),&json!({"optimizer":report,"saved_f32_F64_training":train_measure,"saved_f32_F64_evaluation":valid_measure,"pool_C32_not_full_model":pool_cost,"pool_C32_bits":pool_cost.total(),"fitted_pool_sha256":sha256(&out.join("fitted-pool.artifact"))?,"function_exports":exports,"seconds":started.elapsed().as_secs_f64(),"full_native_graft_acceptance":"not measured; must load pool once and preserve sharedbody during actualgraft","body_transfer":"not measured"}))?;Ok(())
}
