//! Full-native graft of one saved standalone arithmetic proposal; no discovery claim.
//! EXPORT FITTED FITTED_SHA LAYER SPEC SPEC_SHA OUT TRACE_BYTES
use gam_mpd::{
    acceptance::{CostCache, Local, RunCheck, RunMeasure, structural_cost},
    artifact::Artifact,
    coder_capture::sha256,
    counterfactual::{Decoder, Spec, passages},
    import::import_language_model,
    operator_program::{Declarations, FamilyInputs, SequenceLayout, Slot, SlotValues},
    run_check::{LanguageRun, layer_nodes, split_sites},
};
use serde_json::json;
use std::{path::Path, time::Instant};
fn save(path: &Path, value: &serde_json::Value) -> Result<(), String> {
    std::fs::write(path, serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}
fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 8 { return Err("EXPORT FITTED FITTED_SHA LAYER SPEC SPEC_SHA OUT TRACE_BYTES".into()); }
    let export = Path::new(&args[0]); let fitted = Path::new(&args[1]); let spec_path = Path::new(&args[4]); let out = Path::new(&args[6]);
    if sha256(fitted)? != args[2] || sha256(spec_path)? != args[5] { return Err("frozen fitted/spec hash mismatch".into()); }
    if out.exists() { return Err("fresh scientific output required".into()); }
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let started = Instant::now();
    let layer: usize = args[3].parse().map_err(|e| format!("layer: {e}"))?;
    let trace_bytes: usize = args[7].parse().map_err(|e| format!("trace budget: {e}"))?;
    let imported = import_language_model(export, 1, 1)?;
    let native = split_sites(&imported.program)?;
    let layers = layer_nodes(&native, 4)?;
    let nodes = layers.get(layer).ok_or("declared layer outside native four-layer model")?;
    let width = native.node_interface(nodes.normed).map_err(|e| e.to_string())?.width();
    let declarations = Declarations { domains: vec![], slots: vec![Slot::Raw { width }], parameters: 0 };
    let source_bytes = std::fs::read(fitted).map_err(|e| e.to_string())?;
    let function = Artifact::from_bytes(&source_bytes, &declarations)?;
    if !function.exceptions.is_empty() || !function.controls.is_empty() || !function.derived.is_empty() { return Err("function metadata not represented by program graft".into()); }
    if function.to_bytes()? != source_bytes { return Err("function canonical saved-byte replay mismatch".into()); }
    drop(source_bytes);
    let base = Artifact::native(&native)?.f32_literals()?;
    let candidate = base.replace_function(&format!("arithmetic-mlp-{layer}"), &function.program, nodes.normed, nodes.mlp)?
        .with_uniform_scale_control(&native, nodes.active, nodes.mlp)?.f32_literals()?;
    candidate.validate_coverage(&native)?;
    let encoded = candidate.to_bytes()?;
    let artifact_path = out.join("candidate.artifact");
    std::fs::write(&artifact_path, &encoded).map_err(|e| e.to_string())?;
    // Read the actual saved bytes with the ordinary decoder before all fidelity work.
    let decoded = Artifact::from_bytes(&std::fs::read(&artifact_path).map_err(|e| e.to_string())?, &native.declarations)?;
    if decoded != candidate || decoded.to_bytes()? != encoded { return Err("full graft ordinary replay mismatch".into()); }
    decoded.validate_coverage(&native)?;
    drop(encoded); drop(candidate);
    let mut costs = CostCache::default();
    let native_cost = structural_cost(&base, &mut costs)?;
    let cost = structural_cost(&decoded, &mut costs)?;
    let decoder = Decoder::from_export(export)?;
    let spec = Spec::load(spec_path, &decoder)?;
    if spec.rows != 512 || spec.episodes.len() != 80 { return Err("frozen80 episode full512 protocol required".into()); }
    let rows = passages(export, 512)?;
    if rows.len() < 2 { return Err("two full512 passages required".into()); }
    let tokens = rows[..2].iter().flat_map(|r| r[..512].iter().copied()).collect();
    let family = FamilyInputs { rows: 1024, slots: vec![SlotValues::Tokens(tokens)], layout: Some(SequenceLayout {
        sequence: (0..2).flat_map(|s| std::iter::repeat_n(s, 512)).collect(),
        position: (0..2).flat_map(|_| 0..512).collect(),
    }) };
    save(&out.join("PROVENANCE.json"), &json!({"scope":"single fitted quadratic arithmetic proposal; no learned reuse, transfer or global optimum claim", "layer":layer,"input_width":width,"fitted_sha256":args[2],"candidate_sha256":sha256(&artifact_path)?,"spec_sha256":args[5],"export_json_sha256":sha256(&export.join("export.json"))?,"source_record":imported.record,"binary_sha256":sha256(&std::env::current_exe().map_err(|e|e.to_string())?)?,"native_C32":native_cost,"candidate_C32":cost,"candidate_C32_bits":cost.total(),"local_boundary":"native normalized MLP input to pure MLP contribution, fixed native parents; native contribution RMS","local_rows":1024,"run_episodes":80,"trace_bytes":trace_bytes,"native_control_scope":"paid whole native activation uniform Scale only; individual native units are not mapped to learned coordinates", "resident_budgets":{"aggregate":12884901888u64,"teacher":536870912,"operators":2147483648u64,"edits":536870912,"readout_resident":536870912,"readout_workspace":268435456,"metric_workspace":1073741824,"metric_rows":128},"numerical_scope":"Local final-write/RMS enclosures; resident fixed-logit metric arithmetic model, not full neural arithmetic certificate"}))?;
    let device = gam_gpu::tensor::Device::accelerator(gam_gpu::GpuPolicy::Required).map_err(|e|e.to_string())?.ok_or("CUDA required")?;
    let local = Local::new(&native, family, None, 16).with_cuda(device.clone(), trace_bytes)?.with_cuda_resident_norms()?;
    let timer = Instant::now(); let local_measure = local.measure(&decoded)?; let local_seconds = timer.elapsed().as_secs_f64();
    save(&out.join("LOCAL.json"), &json!({"measure":local_measure,"seconds":local_seconds}))?;
    // Run is a prespecified diagnostic even when Local is poor; no gate is weakened.
    let run = LanguageRun::new(&decoder, &native, &spec, &rows, 1)?.with_cuda(device, trace_bytes)?
        .with_cuda_readout(gam_mpd::native_readout::Budget { resident_bytes:536870912, workspace_bytes:268435456 })?
        .with_cuda_native_source(&base)?;
    let resident = run.resident_run(gam_mpd::run_check::ResidentBudget {
        aggregate_numeric_bytes:12884901888, teacher_bytes:536870912,
        operator_bytes:2147483648, edit_bytes:536870912,
        metric:gam_mpd::fixed_metric_device::Budget {batch_rows:128,workspace_bytes:1073741824},
    });
    let timer = Instant::now(); let run_measure = RunMeasure::of(resident.episodes(&decoded)?); let run_seconds = timer.elapsed().as_secs_f64();
    let peak_rss = std::fs::read_to_string("/proc/self/status").ok().and_then(|text| text.lines().find_map(|s|s.strip_prefix("VmHWM:")?.split_whitespace().next()?.parse::<u64>().ok()?.checked_mul(1024)));
    save(&out.join("REPORT.json"), &json!({"C32_bits":cost.total(),"native_C32_bits":native_cost.total(),"local":local_measure,"local_seconds":local_seconds,"run":run_measure,"run_seconds":run_seconds,"run_timing":run.timing(),"resident_telemetry":resident.telemetry()?,"seconds":started.elapsed().as_secs_f64(),"peak_host_rss_bytes":peak_rss,"acceptance_claim":false}))?;
    Ok(())
}
