//! Compare CPU and CUDA execution of the same decoded native-parent Local graft.
//! EXPORT BANK.json OUT.json sequences=N context=N batch=N trace_bytes=N deltas=...
//! This is backend validation/timing, never a new candidate family or quality score.
use gam_gpu::{GpuPolicy, tensor::Device};
use gam_mpd::{acceptance::Local, artifact::Artifact, coder_capture::sha256, import::import_language_model, run_check::split_sites};
use serde::Deserialize;
use serde_json::json;
use std::{collections::BTreeMap, path::{Path, PathBuf}, time::Instant};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry { label: String, artifact: PathBuf }

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() != 8 { return Err("EXPORT BANK.json OUT.json sequences=N context=N batch=N trace_bytes=N deltas=...".into()); }
    let mut options = BTreeMap::new();
    for text in &a[3..] {
        let (key, value) = text.split_once('=').ok_or("expected key=value")?;
        if !["sequences", "context", "batch", "trace_bytes", "deltas"].contains(&key) || options.insert(key, value).is_some() { return Err("unknown or duplicate option".into()); }
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
    let imported = import_language_model(export, sequences, context)?;
    let model = split_sites(&imported.program)?;
    let family = imported.contract.family;
    let cpu = Local::new(&model, family.clone(), None, batch);
    let cuda = Local::new(&model, family, None, batch).with_cuda(device, trace_bytes)?;
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
            if c.blocks.len() != g.blocks.len() || c.rows != g.rows { return Err("backend block/row mismatch".into()); }
            let differences: Vec<_> = c.blocks.iter().zip(&g.blocks).map(|(c,g)| json!({"block":c.name,"absolute_worst_difference":(c.worst-g.worst).abs(),"same_worst_row":c.row==g.row,"same_scale_bits":c.scale.to_bits()==g.scale.to_bits()})).collect();
            let cs = c.status()?;
            let gs = g.status()?;
            let verdict = |s: &gam_mpd::supports::EvidenceStatus<String, String>, d| if s.refutes_at_most(d) { "Violates" } else if s.certifies_at_most(d) { "Meets" } else { "Unresolved" };
            let grid: Vec<_> = deltas.iter().map(|&d| json!({"delta":d,"cpu":verdict(&cs,d),"cuda":verdict(&gs,d)})).collect();
            pairs.push(json!({"repeat":repeat,"cpu_seconds":cpu_seconds,"cuda_seconds":cuda_seconds,"cpu":c,"cuda":g,"differences":differences,"grid":grid}));
        }
        records.push(json!({"label":entry.label,"artifact":path,"artifact_sha256":sha256(&path)?,"pairs":pairs}));
        eprintln!("Local CPU/CUDA pairs complete: {}", entry.label);
    }
    let report = json!({"scope":"same decoded native-parent graft; CPU native scales and comparison; intervals bound comparison rounding only, not CPU/CUDA execution discrepancy; fixed CPU-then-CUDA timing order","cpu_backend":cpu.backend_name(),"cuda_backend":cuda.backend_name(),"sequences":sequences,"context":context,"batch":batch,"trace_bytes":trace_bytes,"export_sha256":sha256(&export.join("export.json"))?,"bank_sha256":sha256(bank)?,"binary_sha256":sha256(&std::env::current_exe().map_err(|e|e.to_string())?)?,"records":records});
    std::fs::write(out, serde_json::to_vec_pretty(&report).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    Ok(())
}
