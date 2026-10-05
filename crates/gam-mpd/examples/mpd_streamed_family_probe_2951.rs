//! EXPORT OUT CONTEXT TEACHER_BYTES TILE_BYTES TRACE_BYTES [SAVED_ARTIFACT [CUDA_NATIVE_BYTES]]
//! Storage/backend parity only: fixed first/last head edits, no mechanism discovery.
//! SAVED_ARTIFACT may be `generated_svd16`: serialize/decode a fixed rank16 final-head fixture.
use gam_mpd::{
    acceptance::{Change,Edit,Episode,FamilyRun,RunCheck},
    artifact::Artifact,
    attention_map::AttentionLayerMap,
    engine::sha256,
    device_family_run::{DeviceFamilyRun,StreamedBudget,StreamedFamilyRun},
    import::import_language_model,
};
use serde_json::json;
use std::{path::Path,time::Instant};

fn main()->Result<(),String>{
    let args:Vec<_>=std::env::args().skip(1).collect();
    if !(6..=8).contains(&args.len()){return Err("EXPORT OUT CONTEXT TEACHER_BYTES TILE_BYTES TRACE_BYTES [SAVED_ARTIFACT [CUDA_NATIVE_BYTES]]".into());}
    let number=|i:usize|args[i].parse::<usize>().map_err(|e|e.to_string());
    let export=Path::new(&args[0]);let out=Path::new(&args[1]);
    if out.exists(){return Err("fresh probe directory required".into());}
    std::fs::create_dir_all(out).map_err(|e|e.to_string())?;
    let begun=Instant::now();let context=number(2)?;
    if context==0 {return Err("positive context required".into());}
    let imported=import_language_model(export,1,context)?;
    let layer_count=imported.record["config"]["n_layers"].as_u64().ok_or("native layer count missing")? as usize;
    let first=AttentionLayerMap::of(&imported.program,0)?;
    let last=AttentionLayerMap::of(&imported.program,layer_count.checked_sub(1).ok_or("empty model")?)?;
    let first_head=first.heads.first().ok_or("first native head missing")?;
    let last_head=last.heads.last().ok_or("last native head missing")?;
    let edit=|node|->Result<Edit,String>{Ok(Edit {node,rows:None,columns:0..imported.program.node_interface(node).map_err(|e|e.to_string())?.width(),change:Change::Scale(0.)})};
    let a=edit(first_head.read)?;let b=edit(last_head.read)?;
    let mut episodes=vec![
        Episode {id:"clean".into(),group:"clean".into(),edits:vec![]},
        Episode {id:"first-head-read-off".into(),group:"first-head".into(),edits:vec![a.clone()]},
        Episode {id:"last-head-read-off".into(),group:"last-head".into(),edits:vec![b.clone()]},
        Episode {id:"first-and-last-off".into(),group:"combined".into(),edits:vec![a,b]},
    ];
    if args.len()==8 {
        episodes.push(Episode {id:"ordered-partial-first-read".into(),group:"ordered-partial".into(),edits:vec![
            Edit {node:first_head.read,rows:Some(vec![0,0,context-1]),columns:0..1,change:Change::Scale(0.5)},
            Edit {node:first_head.read,rows:Some(vec![0,context-1]),columns:0..1,change:Change::Add(0.125)},
        ]});
    }
    let run=FamilyRun {model:&imported.program,family:imported.family.clone(),readouts:1,episodes};
    let device=gam_gpu::tensor::Device::accelerator(gam_gpu::GpuPolicy::Required).map_err(|e|e.to_string())?.ok_or("CUDA required")?;
    let budget=StreamedBudget {teacher_bytes:number(3)?,tile_bytes:number(4)?,intermediate_bytes:number(5)?};
    let original=DeviceFamilyRun::new(&run,device.clone(),budget.intermediate_bytes)?;
    let control_device=device.clone();
    let streamed=StreamedFamilyRun::new(&run,device.clone(),budget)?;
    let cuda_native=if args.len()==8 {Some(StreamedFamilyRun::new(&run,device,budget)?.with_cuda_native(number(7)?)?)}else{None};
    let base=Artifact::native(&imported.program)?.f32_literals()?;
    let mut results=Vec::new();
    for kind in 0..if args.len()>=7{2}else{1} {
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
        let native_cuda_result=if let Some(cuda)=&cuda_native {
            let start=Instant::now();let values=cuda.episodes(&artifact)?;let seconds=start.elapsed().as_secs_f64();
            let mut max_kl_difference=0.0_f64;let mut max_effect_difference=0.0_f64;
            if values.len()!=tiled.len() {return Err("CUDA native episode inventory changed".into());}
            for (cpu,gpu) in tiled.iter().zip(&values) {
                if cpu.id!=gpu.id || cpu.group!=gpu.group || cpu.unheld!=gpu.unheld || cpu.top1_agree!=gpu.top1_agree {return Err("CUDA native identity/coverage/top1 parity failed".into());}
                let gap=(cpu.kl-gpu.kl).abs();max_kl_difference=max_kl_difference.max(gap);
                max_effect_difference=max_effect_difference.max((cpu.native_effect-gpu.native_effect).abs());
                if gap>cpu.numerical_error+gpu.numerical_error {return Err(format!("{} CUDA native conditional intervals do not overlap",cpu.id));}
            }
            Some(json!({"seconds":seconds,"episodes":values,"max_KL_difference":max_kl_difference,"max_native_effect_difference":max_effect_difference,"conditional_interval_overlap":true,"source":cuda.native_cuda_report(),"backend":cuda.backend_name()}))
        }else{None};
        results.push(json!({"native_cuda":native_cuda_result,"source":source,"full_seconds":full_seconds,"streamed_seconds":streamed_seconds,"full":full,"streamed":tiled,"max_KL_difference":max_kl_difference,"max_native_effect_difference":max_effect_difference,"conditional_interval_overlap":true}));
        std::fs::write(out.join(format!("artifact{kind}.json")),serde_json::to_vec_pretty(results.last().ok_or("missing measured result")?).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    }
    let control_fixture=if args.len()==8 {Some(paid_control_probe(control_device,budget,number(7)?,out)?)}else{None};
    let report=json!({"standalone_generic_paid_control_fixture":control_fixture,"scope":"storage/backend parity on one declared causal sequence and fixed first/last native head interventions; no fidelity/discovery result","numerics":"CPU exp/log bands conditional; interval overlap is operational parity, not proof of neural arithmetic equivalence","source":imported.record["source"],"config":imported.record["config"],"export_json_sha256":sha256(&export.join("export.json"))?,"binary_sha256":sha256(&std::env::current_exe().map_err(|e|e.to_string())?)?,"context":context,"budget":budget,"retained_hidden_teacher_numeric_bytes":streamed.teacher_numeric_bytes(),"tile_rows":streamed.tile_rows(),"original_timing":original.timing(),"streamed_timing":streamed.timing(),"results":results,"seconds":begun.elapsed().as_secs_f64()});
    std::fs::write(out.join("COMPLETE.json"),serde_json::to_vec_pretty(&report).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    Ok(())
}


/// Independent explicit two-dimensional linear control fixture; not the imported model.
fn paid_control_probe(device:gam_gpu::tensor::Device,budget:StreamedBudget,numeric_bytes:usize,out:&Path)->Result<serde_json::Value,String> {
    use gam_mpd::{operator_program::{Declarations,FamilyInputs,Interface,Node,Operator,OperatorProgram,Provenance,Slot,SlotValues,exact_precision},artifact::Binding};
    use std::sync::Arc;
    let interface=Interface::native(2).map_err(|e|e.to_string())?;
    let dense=|name:&str,values:ndarray::Array2<f64>|->Result<Arc<Operator>,String>{let precision=exact_precision(values.iter().copied()).map_err(|e|e.to_string())?;Ok(Arc::new(Operator::dense(name,interface.clone(),interface.clone(),values,precision,Provenance::default()).map_err(|e|e.to_string())?))};
    let native=OperatorProgram {declarations:Declarations {domains:vec![],slots:vec![Slot::Raw {width:2}],parameters:0},bases:vec![],rules:vec![],operators:vec![Arc::new(Operator::identity("source-identity",interface.clone())),dense("linear-write",ndarray::arr2(&[[2.,0.],[0.,0.5]]))?,dense("terminal-head",ndarray::arr2(&[[1.,-0.5],[-0.25,1.]]))?],nodes:vec![Node::Raw {slot:0},Node::Affine {terms:vec![(0,0)],bias:None},Node::Affine {terms:vec![(1,1)],bias:None},Node::Affine {terms:vec![(2,2)],bias:None}],output:3};
    let mut compressed=native.clone();compressed.nodes=vec![Node::Raw {slot:0},Node::Affine {terms:vec![(0,1)],bias:None},Node::Affine {terms:vec![(1,2)],bias:None}];compressed.output=2;
    let mut candidate=Artifact::native(&compressed)?;candidate.native_nodes=4;candidate.places=vec![(0,0),(2,1),(3,2)];candidate.blocks=vec![Binding {name:"linear control fixture".into(),native_reads:vec![0],native_write:2,reads:vec![0],write:1}];
    candidate=candidate.with_uniform_scale_control(&native,1,2)?.f32_literals()?;
    let bytes=candidate.to_bytes()?;let path=out.join("generic-control-fixture.bin");std::fs::write(&path,&bytes).map_err(|e|e.to_string())?;let decoded=Artifact::from_bytes(&bytes,&native.declarations)?;
    if decoded.to_bytes()?!=bytes || decoded.controls.len()!=1 {return Err("generic control saved-byte replay mismatch".into());}
    let scale=|value|Edit {node:1,rows:None,columns:0..2,change:Change::Scale(value)};
    let run=FamilyRun {model:&native,family:FamilyInputs {rows:2,slots:vec![SlotValues::Raw(ndarray::arr2(&[[0.75,-0.25],[0.125,1.5]]))],layout:None},readouts:1,episodes:vec![
        Episode {id:"clean".into(),group:"clean".into(),edits:vec![]},
        Episode {id:"source-off".into(),group:"off".into(),edits:vec![scale(0.)]},
        Episode {id:"write-add-before-source-scale".into(),group:"ordered".into(),edits:vec![Edit {node:2,rows:Some(vec![0]),columns:0..1,change:Change::Add(0.125)},scale(2.)]},
        Episode {id:"repeated-source-row".into(),group:"repeated".into(),edits:vec![Edit {rows:Some(vec![0,0]),..scale(0.5)}]},
        Episode {id:"unsupported-partial-source".into(),group:"partial".into(),edits:vec![Edit {columns:0..1,..scale(0.)}]},
    ]};
    let cpu=run.episodes(&decoded)?;
    let streamed=StreamedFamilyRun::new(&run,device,budget)?.with_cuda_native(numeric_bytes)?;let gpu=streamed.episodes(&decoded)?;
    if cpu.len()!=gpu.len() {return Err("generic control episode inventory changed".into());}
    for (c,g) in cpu.iter().zip(&gpu) {
        if c.id!=g.id || c.group!=g.group || c.unheld!=g.unheld || c.top1_agree!=g.top1_agree || (c.kl-g.kl).abs()>c.numerical_error+g.numerical_error {return Err("generic paid-control CUDA parity failed".into());}
    }
    if cpu[1].native_effect<=0.0 || cpu[1].unheld!=0 || cpu[4].unheld!=1 {return Err("generic control did not exercise held response and unsupported partial action".into());}
    Ok(json!({"scope":"independent explicit linear raw-input fixture; not imported model","cpu_reference":cpu,"cuda_native_candidate":gpu,"conditional_interval_overlap":true,"saved_byte_replay":true,"sha256":sha256(&path)?,"source":streamed.native_cuda_report()}))
}
