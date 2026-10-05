//! Compare CPU and CUDA execution of the same decoded native-parent Local graft.
//! EXPORT BANK.json OUT.json sequences=N context=N batch=N trace_bytes=N deltas=... [source_bytes=N]
//! This is backend validation/timing, never a new candidate family or quality score.
use gam_gpu::{GpuPolicy, tensor::Device};
use gam_mpd::{acceptance::Local, artifact::Artifact, engine::sha256, import::import_language_model, run_check::split_sites};
use serde::Deserialize;
use serde_json::json;
use std::{collections::BTreeMap, path::{Path, PathBuf}, time::Instant};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry { label: String, artifact: PathBuf }

fn norm_gate(device: &Device) -> Result<(), String> {
    let host = Device::host();
    let (rows, cols) = (35, 773);
    let values: Vec<f64> = (0..rows*cols).map(|i| if i/cols==0 { 0.0 } else {
        match i%5 { 0=>1.0, 1=>2.0_f64.powi(-26), 2=>-0.75, 3=>f64::from_bits(1), _=>(i%31) as f64/17.0 }
    }).collect();
    let reference = host.upload_vec(rows,cols,values.clone()).map_err(|e|e.to_string())?;
    let input = device.upload_vec(rows,cols,values).map_err(|e|e.to_string())?;
    for columns in [0..cols, 3..cols-2, 9..9] {
        for scale in [1.0,0.125,3.25] {
            let cpu = host.scaled_row_l2(&reference,columns.clone(),scale).map_err(|e|e.to_string())?;
            let gpu = device.scaled_row_l2(&input,columns.clone(),scale).map_err(|e|e.to_string())?;
            if cpu.len()!=gpu.len() || cpu.iter().zip(&gpu).any(|(a,b)|a.to_bits()!=b.to_bits()) { return Err("resident norm arithmetic fixture mismatch".into()); }
        }
    }
    for amplitude in [2.0_f64.powi(-1000),1e-200,1.0,1e200] {
        let values=vec![3.0*amplitude,4.0*amplitude];
        let reference=host.upload_vec(1,2,values.clone()).map_err(|e|e.to_string())?;
        let input=device.upload_vec(1,2,values).map_err(|e|e.to_string())?;
        let cpu=host.scaled_row_l2_enclosed(&reference,0..2,[amplitude;3]).map_err(|e|e.to_string())?[0];
        let gpu=device.scaled_row_l2_enclosed(&input,0..2,[amplitude;3]).map_err(|e|e.to_string())?[0];
        if cpu[0].to_bits()!=gpu[0].to_bits() || cpu[1]>5.0 || cpu[2]<5.0 || gpu[1]>5.0 || gpu[2]<5.0 {return Err(format!("extreme enclosed norm mismatch {amplitude}: {cpu:?} {gpu:?}"));}
    }
    let zero=device.zeros(1,2).map_err(|e|e.to_string())?;
    if device.scaled_row_l2_enclosed(&zero,0..2,[0.0;3]).map_err(|e|e.to_string())?!=vec![[0.0;3]] {return Err("zero/zero norm not exact".into());}
    let tiny=device.upload_vec(1,1,vec![f64::from_bits(1)]).map_err(|e|e.to_string())?;
    let tiny_bounds=device.scaled_row_l2_enclosed(&tiny,0..1,[f64::from_bits(1);3]).map_err(|e|e.to_string())?[0];
    if tiny_bounds[1]>1.0 || tiny_bounds[2]<1.0 || device.scaled_row_l2_enclosed(&tiny,0..1,[0.0;3]).is_ok() {return Err("subnormal or nonzero/zero norm semantics failed".into());}
    let bad = device.upload_vec(1,2,vec![f64::NAN,1.0]).map_err(|e|e.to_string())?;
    if device.scaled_row_l2(&bad,0..2,1.0).is_ok() { return Err("nonfinite norm accepted".into()); }
    Ok(())
}

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() != 8 && a.len() != 9 { return Err("EXPORT BANK.json OUT.json sequences=N context=N batch=N trace_bytes=N deltas=... [source_bytes=N]".into()); }
    let mut options = BTreeMap::new();
    for text in &a[3..] {
        let (key, value) = text.split_once('=').ok_or("expected key=value")?;
        if !["sequences", "context", "batch", "trace_bytes", "deltas", "source_bytes"].contains(&key) || options.insert(key, value).is_some() { return Err("unknown or duplicate option".into()); }
    }
    let number = |k| options.get(k).ok_or("missing option")?.parse::<usize>().map_err(|e| e.to_string());
    let (sequences, context, batch, trace_bytes) = (number("sequences")?, number("context")?, number("batch")?, number("trace_bytes")?);
    if [sequences, context, batch, trace_bytes].contains(&0) { return Err("positive dimensions and byte limit required".into()); }
    let deltas: Vec<f64> = options.get("deltas").ok_or("declare deltas")?.split(',').map(|s| s.parse::<f64>().map_err(|e| e.to_string())).collect::<Result<_, _>>()?;
    if deltas.is_empty() || deltas.iter().any(|v| !v.is_finite() || *v < 0.0) { return Err("finite nonnegative delta grid required".into()); }
    let (export, bank, out) = (Path::new(&a[0]), Path::new(&a[1]), Path::new(&a[2]));
    if out.exists() { return Err("output exists".into()); }
    let entries: Vec<Entry> = serde_json::from_slice(&std::fs::read(bank).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    if entries.is_empty() { return Err("nonempty explicit validation bank required".into()); }
    let device = Device::accelerator(GpuPolicy::Required).map_err(|e| e.to_string())?.ok_or("CUDA required")?;
    norm_gate(&device)?;
    let imported = import_language_model(export, sequences, context)?;
    let model = split_sites(&imported.program)?;
    let family = imported.contract.family;
    let cpu = Local::new(&model, family.clone(), None, batch);
    let cuda = Local::new(&model, family.clone(), None, batch).with_cuda(device.clone(), trace_bytes)?;
    let source_bytes = options.get("source_bytes").map(|v| v.parse::<usize>().map_err(|e| e.to_string())).transpose()?.unwrap_or(0);
    let init = Instant::now();
    let shared = if source_bytes == 0 { None } else {
        Some(Local::new(&model, family.clone(), None, batch).with_cuda(device.clone(), trace_bytes)?.with_cuda_native_sharing(source_bytes)?)
    };
    let source_initialization_seconds = init.elapsed().as_secs_f64();
    let retained_source_numeric_bytes = shared.as_ref().map(Local::cuda_native_source_numeric_bytes).transpose()?.flatten();
    let init = Instant::now();
    let mut resident_norms = Local::new(&model, family, None, batch).with_cuda(device, trace_bytes)?.with_cuda_resident_norms()?;
    if source_bytes != 0 { resident_norms = resident_norms.with_cuda_native_sharing(source_bytes)?; }
    let resident_norm_source_initialization_seconds = init.elapsed().as_secs_f64();
    let mut records = Vec::new();
    for entry in entries {
        let path = if entry.artifact.is_absolute() { entry.artifact } else { bank.parent().unwrap_or(Path::new(".")).join(entry.artifact) };
        let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
        let artifact = Artifact::from_bytes(&bytes, &model.declarations)?;
        artifact.validate_coverage(&model)?;
        let mut pairs = Vec::new();
        // Both backends cache the declared native scales independently. The second
        // pair separates that startup cost from repeated graft execution.
        for repeat in 0..2 {
            let start = Instant::now();
            let c = cpu.measure(&artifact)?;
            let cpu_seconds = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let g = cuda.measure(&artifact)?;
            let cuda_seconds = start.elapsed().as_secs_f64();
            let shared_measure = if let Some(shared) = &shared {
                let start = Instant::now();
                let measured = shared.measure(&artifact)?;
                let seconds = start.elapsed().as_secs_f64();
                if measured != g { return Err(format!("fresh/shared CUDA complete Local evidence mismatch: {} repeat {repeat}", entry.label)); }
                Some(json!({"seconds":seconds,"measure":measured,"complete_fresh_cuda_evidence_equal":true}))
            } else { None };
            let start = Instant::now();
            let reduced = resident_norms.measure(&artifact)?;
            let resident_norm_seconds = start.elapsed().as_secs_f64();
            if reduced.rows!=g.rows || reduced.blocks.len()!=g.blocks.len() || reduced.blocks.iter().zip(&g.blocks).any(|(a,b)|a.name!=b.name || a.row!=b.row || a.worst.to_bits()!=b.worst.to_bits() || a.scale.to_bits()!=b.scale.to_bits() || a.lower>b.upper || b.lower>a.upper) {
                return Err(format!("host/resident centers or valid interval overlap mismatch: {} repeat {repeat}",entry.label));
            }
            let difference_width: usize = artifact.blocks.iter().map(|b| model.node_interface(b.native_write).map(|i| i.width()).map_err(|e|e.to_string())).collect::<Result<Vec<_>,_>>()?.iter().sum();
            if c.blocks.len() != g.blocks.len() || c.rows != g.rows { return Err("backend block/row mismatch".into()); }
            let differences: Vec<_> = c.blocks.iter().zip(&g.blocks).map(|(c,g)| json!({"block":c.name,"absolute_worst_difference":(c.worst-g.worst).abs(),"same_worst_row":c.row==g.row,"same_scale_bits":c.scale.to_bits()==g.scale.to_bits()})).collect();
            let cs = c.status()?;
            let gs = g.status()?;
            let rs = reduced.status()?;
            let verdict = |s: &gam_mpd::supports::EvidenceStatus<String, String>, d| if s.refutes_at_most(d) { "Violates" } else if s.certifies_at_most(d) { "Meets" } else { "Unresolved" };
            let grid: Vec<_> = deltas.iter().map(|&d| json!({"delta":d,"cpu":verdict(&cs,d),"cuda":verdict(&gs,d),"resident_cuda":verdict(&rs,d)})).collect();
            if deltas.iter().any(|&d|verdict(&cs,d)!=verdict(&gs,d) || verdict(&gs,d)!=verdict(&rs,d)) { return Err(format!("declared-grid classification changed: {} repeat {repeat}: {grid:?}",entry.label)); }
            pairs.push(json!({"repeat":repeat,"cpu_seconds":cpu_seconds,"cuda_seconds":cuda_seconds,"shared_cuda":shared_measure,"resident_norms":{"seconds":resident_norm_seconds,"host_center_bits_equal":true,"sound_interval_overlap":true,"declared_grid_classifications_equal":true,"downloaded_norm_bytes":reduced.rows*reduced.blocks.len()*3*8,"old_downloaded_difference_bytes":reduced.rows*difference_width*8,"measure":reduced},"cpu":c,"cuda":g,"differences":differences,"grid":grid}));
        }
        records.push(json!({"label":entry.label,"artifact":path,"artifact_sha256":sha256(&path)?,"pairs":pairs}));
        eprintln!("Local CPU/CUDA pairs complete: {}", entry.label);
    }
    let report = json!({"scope":"same decoded native-parent graft; frozen CPU native scales; host/resident max-rescaled binary64 reductions compared for bitwise centers, interval overlap and declared-grid classification equality; intervals enclose native RMS denominator and final subtraction/norm only, not CPU/CUDA execution discrepancy; fixed CPU-then-CUDA-then-resident timing order","cpu_backend":cpu.backend_name(),"cuda_backend":cuda.backend_name(),"resident_norm_backend":resident_norms.backend_name(),"resident_norm_source_initialization_seconds":resident_norm_source_initialization_seconds,"sequences":sequences,"context":context,"batch":batch,"trace_bytes":trace_bytes,"source_numeric_bytes_limit":source_bytes,"retained_source_numeric_bytes":retained_source_numeric_bytes,"source_initialization_seconds":source_initialization_seconds,"source_scope":"actual f64 native model, no rounding; source never forwarded; same Arc and role only; numeric limit excludes indices/activations/workspaces/allocator/host; initialization excluded from repeated measurements; this comparison harness holds two separate native sources when sharing is enabled","export_sha256":sha256(&export.join("export.json"))?,"bank_sha256":sha256(bank)?,"binary_sha256":sha256(&std::env::current_exe().map_err(|e|e.to_string())?)?,"records":records});
    std::fs::write(out, serde_json::to_vec_pretty(&report).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    Ok(())
}
