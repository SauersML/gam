//! EXPORT OUT CONTEXT TEACHER_BYTES TILE_BYTES TRACE_BYTES [SAVED_ARTIFACT]
//! Storage/backend parity only: fixed first/last head edits, no mechanism discovery.
//! SAVED_ARTIFACT may be `generated_svd16`: serialize/decode a fixed rank16 final-head fixture.
use gam_mpd::{
    acceptance::{Change,Edit,Episode,FamilyRun,RunCheck},
    artifact::Artifact,
    attention_map::AttentionLayerMap,
    coder_capture::sha256,
    device_family_run::{DeviceFamilyRun,StreamedBudget,StreamedFamilyRun},
    import::import_language_model,
};
use serde_json::json;
use std::{path::Path,time::Instant};

fn main()->Result<(),String>{
    let args:Vec<_>=std::env::args().skip(1).collect();
    if !(6..=7).contains(&args.len()){return Err("EXPORT OUT CONTEXT TEACHER_BYTES TILE_BYTES TRACE_BYTES [SAVED_ARTIFACT]".into());}
    let number=|i:usize|args[i].parse::<usize>().map_err(|e|e.to_string());
    let export=Path::new(&args[0]);let out=Path::new(&args[1]);
    if out.exists(){return Err("fresh probe directory required".into());}
    std::fs::create_dir_all(out).map_err(|e|e.to_string())?;
    let begun=Instant::now();let context=number(2)?;
    let imported=import_language_model(export,1,context)?;
    let layer_count=imported.record["config"]["n_layers"].as_u64().ok_or("native layer count missing")? as usize;
    let first=AttentionLayerMap::of(&imported.program,0)?;
    let last=AttentionLayerMap::of(&imported.program,layer_count.checked_sub(1).ok_or("empty model")?)?;
    let first_head=first.heads.first().ok_or("first native head missing")?;
    let last_head=last.heads.last().ok_or("last native head missing")?;
    let edit=|node|->Result<Edit,String>{Ok(Edit {node,rows:None,columns:0..imported.program.node_interface(node).map_err(|e|e.to_string())?.width(),change:Change::Scale(0.)})};
    let a=edit(first_head.read)?;let b=edit(last_head.read)?;
    let episodes=vec![
        Episode {id:"clean".into(),group:"clean".into(),edits:vec![]},
        Episode {id:"first-head-read-off".into(),group:"first-head".into(),edits:vec![a.clone()]},
        Episode {id:"last-head-read-off".into(),group:"last-head".into(),edits:vec![b.clone()]},
        Episode {id:"first-and-last-off".into(),group:"combined".into(),edits:vec![a,b]},
    ];
    let run=FamilyRun {model:&imported.program,family:imported.contract.family.clone(),readouts:1,episodes};
    let device=gam_gpu::tensor::Device::accelerator(gam_gpu::GpuPolicy::Required).map_err(|e|e.to_string())?.ok_or("CUDA required")?;
    let budget=StreamedBudget {teacher_bytes:number(3)?,tile_bytes:number(4)?,intermediate_bytes:number(5)?};
    let original=DeviceFamilyRun::new(&run,device.clone(),budget.intermediate_bytes)?;
    let streamed=StreamedFamilyRun::new(&run,device,budget)?;
    let base=Artifact::native(&imported.program)?.f32_literals()?;
    let mut results=Vec::new();
    for kind in 0..if args.len()==7{2}else{1} {
        let (artifact,source)=if kind==0 {(base.clone(),json!({"kind":"native"}))} else if args[6]=="generated_svd16" {
            let bank=gam_mpd::proposals::MappedCopyResidualBank::new(&base,&[layer_count-1],&[16],1+2*last.heads.len())?;
            let choice=bank.choices().find(|c| c.family==gam_mpd::proposals::HeadApproximation::NativeSvd).ok_or("mapped parity fixture absent")?;
            let candidate=bank.candidate(choice)?.f32_literals()?;
            let bytes=candidate.to_bytes()?;let path=out.join("fixture.bin");std::fs::write(&path,&bytes).map_err(|e|e.to_string())?;
            let candidate=Artifact::from_bytes(&bytes,&imported.program.declarations)?;
            if candidate.to_bytes()?!=bytes {return Err("generated fixture canonical roundtrip mismatch".into());}
            (candidate,json!({"kind":"serialized fixed final-layer head0 rank16 fixture","choice":choice,"sha256":sha256(&path)?,"bytes":bytes.len()}))
        } else {
            let path=Path::new(&args[6]);let bytes=std::fs::read(path).map_err(|e|e.to_string())?;
            let candidate=Artifact::from_bytes(&bytes,&imported.program.declarations)?;
            if candidate.to_bytes()?!=bytes {return Err("saved candidate canonical roundtrip mismatch".into());}
            (candidate,json!({"kind":"saved","path":path,"sha256":sha256(path)?,"bytes":bytes.len()}))
        };
        let start=Instant::now();let full=original.episodes(&artifact)?;let full_seconds=start.elapsed().as_secs_f64();
        let start=Instant::now();let tiled=streamed.episodes(&artifact)?;let streamed_seconds=start.elapsed().as_secs_f64();
        if full.len()!=tiled.len(){return Err("episode inventory changed".into());}
        let mut max_kl_difference=0.0_f64;let mut max_effect_difference=0.0_f64;
        for (f,t) in full.iter().zip(&tiled) {
            if f.id!=t.id||f.group!=t.group||f.unheld!=t.unheld||f.top1_agree!=t.top1_agree {return Err("streamed identity/coverage/top1 parity failed".into());}
            let difference=(f.kl-t.kl).abs();max_kl_difference=max_kl_difference.max(difference);
            max_effect_difference=max_effect_difference.max((f.native_effect-t.native_effect).abs());
            if difference>f.numerical_error+t.numerical_error {return Err(format!("{}: conditional comparison intervals do not overlap",f.id));}
        }
        results.push(json!({"source":source,"full_seconds":full_seconds,"streamed_seconds":streamed_seconds,"full":full,"streamed":tiled,"max_KL_difference":max_kl_difference,"max_native_effect_difference":max_effect_difference,"conditional_interval_overlap":true}));
        std::fs::write(out.join(format!("artifact{kind}.json")),serde_json::to_vec_pretty(results.last().ok_or("missing measured result")?).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    }
    let report=json!({"scope":"storage/backend parity on one declared causal sequence and fixed first/last native head interventions; no fidelity/discovery result","numerics":"CPU exp/log bands conditional; interval overlap is operational parity, not proof of neural arithmetic equivalence","source":imported.record["source"],"config":imported.record["config"],"export_json_sha256":sha256(&export.join("export.json"))?,"binary_sha256":sha256(&std::env::current_exe().map_err(|e|e.to_string())?)?,"context":context,"budget":budget,"retained_hidden_teacher_numeric_bytes":streamed.teacher_numeric_bytes(),"tile_rows":streamed.tile_rows(),"original_timing":original.timing(),"streamed_timing":streamed.timing(),"results":results,"seconds":begun.elapsed().as_secs_f64()});
    std::fs::write(out.join("COMPLETE.json"),serde_json::to_vec_pretty(&report).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    Ok(())
}
