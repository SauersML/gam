//! One serialized native artifact of the entire export, with no learned edits or exceptions.
//! EXPORT OUT.json [CONTEXT=16] [BACKEND=cuda|cpu]. CUDA is required when requested.
//! Reports numerical discrepancies; no invented fidelity threshold or certificate.
use gam_gpu::{GpuPolicy, tensor::Device};
use gam_mpd::{artifact::Artifact, artifact_device::Resident, counterfactual::read_f64_matrix, import::import_language_model};
use ndarray::Array2;
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::Path, time::Instant};

fn compare(a: &Array2<f64>, b: &Array2<f64>) -> Result<Value, String> {
    if a.dim() != b.dim() { return Err(format!("dimensions {:?} versus {:?}", a.dim(), b.dim())); }
    if a.iter().chain(b.iter()).any(|v| !v.is_finite()) { return Err("nonfinite comparison value".into()); }
    let mut maximum = 0.0_f64; let mut sum = 0.0; let mut scale = 0.0_f64;
    for (&x,&y) in a.iter().zip(b.iter()) { let d=(x-y).abs();maximum=maximum.max(d);sum+=d*d;scale=scale.max(y.abs()); }
    Ok(json!({"shape":a.shape(),"max_absolute_error":maximum,"rms_error":if a.is_empty(){0.0}else{(sum/a.len() as f64).sqrt()},"reference_max_absolute":scale,"error_relative_to_reference_peak":if scale>0.0{Some(maximum/scale)}else{None}}))
}
fn peak_rss_bytes() -> Option<u64> {
    std::fs::read_to_string("/proc/self/status").ok()?.lines().find_map(|line| {
        line.strip_prefix("VmHWM:")?.split_whitespace().next()?.parse::<u64>().ok()?.checked_mul(1024)
    })
}
fn main() -> Result<(), String> {
    gam_mpd::engine::log_to_stderr();
    let args:Vec<String>=std::env::args().collect();
    if !(3..=5).contains(&args.len()) { return Err("EXPORT OUT.json [CONTEXT=16] [BACKEND=cuda|cpu]".into()); }
    let export=Path::new(&args[1]);let out=Path::new(&args[2]);
    if out.exists() { return Err(format!("refusing existing report {}",out.display())); }
    let context:usize=args.get(3).map_or(Ok(16),|s|s.parse().map_err(|e|format!("context:{e}")))?;
    if context==0 {return Err("context must be positive".into());}
    let backend=args.get(4).map(String::as_str).unwrap_or("cuda");
    let device=match backend {
        "cpu"=>None,
        "cuda"=>Some(Device::accelerator(GpuPolicy::Required).map_err(|e|e.to_string())?.ok_or("CUDA required")?),
        _=>return Err("backend must be cuda or cpu".into()),
    };
    let total=Instant::now();let mut times=BTreeMap::new();
    let t=Instant::now();let imported=import_language_model(export,1,context)?;
    let record=imported.record;let family=imported.contract.family;let model=imported.program;
    let layers=record["config"]["n_layers"].as_u64().ok_or("missing layer count")?;
    if record["source"]["layers_kept"].as_u64()!=Some(layers) {return Err("export layer provenance disagrees with configuration".into());}
    times.insert("import",t.elapsed().as_secs_f64());
    let t=Instant::now();let native=model.execute(&family,false).map_err(|e|e.to_string())?;
    times.insert("native_cpu",t.elapsed().as_secs_f64());
    let output=model.output;
    let hf=read_f64_matrix(&export.join("logits_row0.f64"),native.values[output].ncols())?;
    if hf.nrows()<context {return Err("stored HF reference shorter than context".into());}
    let hf=hf.slice(ndarray::s![..context,..]).to_owned();
    let hf_vs_native=compare(&native.values[output],&hf)?;drop(hf);
    let t=Instant::now();let artifact=Artifact::native(&model)?.f32_literals()?;drop(model);
    let message=artifact.encode()?;let message_bits=message.len_bits();let declarations=artifact.program.declarations.clone();drop(artifact);
    times.insert("serialize",t.elapsed().as_secs_f64());
    let t=Instant::now();let decoded=Artifact::decode(&message,&declarations)?;drop(message);
    times.insert("decode",t.elapsed().as_secs_f64());
    if !decoded.blocks.is_empty() || !decoded.exceptions.is_empty() || !decoded.derived.is_empty() || !decoded.program.rules.is_empty() {
        return Err("native control unexpectedly contains replacements, exceptions, derived operators, or rules".into());
    }
    let t=Instant::now();let cpu=decoded.execute_edited(&family,|_,_,_|Ok(()))?;
    times.insert("decoded_cpu",t.elapsed().as_secs_f64());
    if native.values.len()!=cpu.values.len(){return Err("decoded native node count changed".into());}
    let mut serialization_states=Vec::new();
    for (node,(a,b)) in cpu.values.iter().zip(&native.values).enumerate() {
        serialization_states.push(json!({"node":node,"error":compare(a,b)?}));
    }
    let serialization_logits=compare(&cpu.values[decoded.program.output],&native.values[output])?;
    drop(native);
    let mut cuda_states=Vec::new();let mut cuda_logits=None;let mut device_name=None;let mut estimated_bytes=None;
    if let Some(d)=device {
        if d.is_host() || !d.float64(){return Err("float64 CUDA required; no host fallback".into());}
        device_name=Some(d.name().to_string());
        let t=Instant::now();let resident=Resident::from_decoded(&d,&decoded)?;
        times.insert("cuda_compile_upload",t.elapsed().as_secs_f64());
        estimated_bytes=Some(resident.estimated_resident_bytes(family.rows)?);
        let t=Instant::now();let trace=resident.forward_edited(&family,|_,_|Ok(None))?;
        // Downloads synchronize CUDA, so the recorded phase covers completed computation.
        let logits=d.download(&resident.output(&trace)?).map_err(|e|e.to_string())?;
        times.insert("cuda_forward_and_logit_download",t.elapsed().as_secs_f64());
        cuda_logits=Some(compare(&logits,&cpu.values[decoded.program.output])?);
        let t=Instant::now();
        for (node,expected) in cpu.values.iter().enumerate() {
            if expected.is_empty(){continue;}
            let actual=d.download(resident.root_value(&trace,node)?).map_err(|e|e.to_string())?;
            cuda_states.push(json!({"node":node,"error":compare(&actual,expected)?}));
        }
        times.insert("cuda_state_download_comparison",t.elapsed().as_secs_f64());
    }
    times.insert("total",total.elapsed().as_secs_f64());
    let report=json!({"control":"single complete serialized decoded native artifact; no learned edit or plant","export":export,"source":record["source"],"config":record["config"],"backend":backend,"device":device_name,"sequences":1,"context":context,"rows":family.rows,"nodes":decoded.program.nodes.len(),"operators":decoded.program.operators.len(),"serialized_bits":message_bits,"hf_fp32_vs_native_f64_logits":hf_vs_native,"decoded_cpu_vs_native_logits":serialization_logits,"decoded_cpu_vs_native_states":serialization_states,"cuda_vs_decoded_cpu_logits":cuda_logits,"cuda_vs_decoded_cpu_states":cuda_states,"estimated_resident_bytes_incomplete":estimated_bytes,"peak_host_rss_bytes":peak_rss_bytes(),"seconds":times,"claim":"measured numerical discrepancies only; no fidelity tolerance or optimality certificate"});
    if let Some(parent)=out.parent(){std::fs::create_dir_all(parent).map_err(|e|e.to_string())?;}
    std::fs::write(out,serde_json::to_vec_pretty(&report).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    println!("{}",json!({"report":out,"layers":layers,"rows":family.rows,"seconds":times,"peak_host_rss_bytes":peak_rss_bytes(),"cuda_logits":report["cuda_vs_decoded_cpu_logits"],"cpu_logits":report["decoded_cpu_vs_native_logits"],"hf_logits":report["hf_fp32_vs_native_f64_logits"]}));
    Ok(())
}
