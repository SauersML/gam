//! Joint full-native assessment from one decoded shared operator pool; no discovery claim.
//! EXPORT POOL POOL_SHA FIT_CONFIG SPEC SPEC_SHA OUT BUDGET_JSON
use gam_mpd::{
    acceptance::{CostCache, Local, RunCheck, structural_cost},
    artifact::Artifact,
    coder_capture::sha256,
    counterfactual::{Decoder, Spec, passages},
    import::import_language_model,
    operator_program::{Declarations, FamilyInputs, SequenceLayout, Slot, SlotValues},
    run_check::{LanguageRun, ResidentBudget, layer_nodes, split_sites},
    shared_geometry_pilot::{Proposal, function},
};
use serde_json::json;
use std::{path::Path, time::Instant};
fn save(path: &Path, value: &serde_json::Value) -> Result<(), String> {
    std::fs::write(path, serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}
fn graft(base:&Artifact, native:&gam_mpd::operator_program::OperatorProgram,layers:&[gam_mpd::run_check::LayerNodes],proposal:&Proposal)->Result<Artifact,String>{
    let mut candidate=base.clone();
    for (slot,&layer) in proposal.uses.iter().enumerate(){
        let nodes=layers.get(layer).ok_or("native use outside map")?;
        let body=function(proposal,slot)?;
        candidate=candidate.replace_function(&format!("shared-geometry-mlp-{layer}"),&body,nodes.normed,nodes.mlp)?
            .with_uniform_scale_control(native,nodes.active,nodes.mlp)?;
    }
    Ok(candidate)
}
fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 8 && !(args.len()==9 && args[8]=="prepare_only") { return Err("EXPORT POOL POOL_SHA FIT_CONFIG SPEC SPEC_SHA OUT BUDGET_JSON [prepare_only]".into()); }
    let export = Path::new(&args[0]); let fitted = Path::new(&args[1]); let spec_path = Path::new(&args[4]); let out = Path::new(&args[6]);
    if sha256(fitted)? != args[2] || sha256(spec_path)? != args[5] { return Err("frozen fitted/spec hash mismatch".into()); }
    if out.exists() { return Err("fresh scientific output required".into()); }
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let started = Instant::now();
    let config_bytes=std::fs::read(&args[3]).map_err(|e|e.to_string())?;
    let config:serde_json::Value=serde_json::from_slice(&config_bytes).map_err(|e|e.to_string())?;
    let uses:Vec<usize>=config["uses"].as_array().ok_or("declared uses absent")?.iter().map(|v|usize::try_from(v.as_u64().ok_or("use must be integer")?).map_err(|e|e.to_string())).collect::<Result<_,_>>()?;
    if uses!=vec![0,1] { return Err("this declared discovery assessment requires native uses0/1".into()); }
    let budget_bytes=std::fs::read(&args[7]).map_err(|e|e.to_string())?;
    let budgets:serde_json::Value=serde_json::from_slice(&budget_bytes).map_err(|e|e.to_string())?;
    let required=|key:&str|->Result<usize,String>{usize::try_from(budgets[key].as_u64().ok_or_else(||format!("explicit numeric budget {key} required"))?).map_err(|e|e.to_string())};
    let trace_bytes=required("trace_bytes")?;
    let resident_budget=ResidentBudget{aggregate_numeric_bytes:required("aggregate_numeric_bytes")?,teacher_bytes:required("teacher_bytes")?,operator_bytes:required("operator_bytes")?,edit_bytes:required("edit_bytes")?,metric:gam_mpd::fixed_metric_device::Budget{batch_rows:required("metric_batch_rows")?,workspace_bytes:required("metric_workspace_bytes")?}};
    if config["export_json_sha256"].as_str()!=Some(&sha256(&export.join("export.json"))?){return Err("fit native export hash mismatch".into());}
    let imported = import_language_model(export, 1, 1)?;
    let native = split_sites(&imported.program)?;
    let layers = layer_nodes(&native, 4)?;
    let nodes = layers.get(uses[0]).ok_or("declared native use outside four-layer model")?;
    let width = native.node_interface(nodes.normed).map_err(|e| e.to_string())?.width();
    let declarations = Declarations { domains: vec![], slots: uses.iter().map(|_|Slot::Raw { width }).collect(), parameters: 0 };
    let source_bytes = std::fs::read(fitted).map_err(|e| e.to_string())?;
    let pool = Artifact::from_bytes(&source_bytes, &declarations)?;
    if !pool.blocks.is_empty() || !pool.exceptions.is_empty() || !pool.controls.is_empty() || !pool.derived.is_empty() { return Err("function metadata not represented by program graft".into()); }
    if pool.to_bytes()? != source_bytes { return Err("function canonical saved-byte replay mismatch".into()); }
    drop(source_bytes);
    let pool_cost=structural_cost(&pool,&mut CostCache::default())?;
    let mut changed_native_operator_ids=std::collections::BTreeSet::new();
    for &use_id in &uses{
        for node in [layers[use_id].pre,layers[use_id].mlp]{
            let gam_mpd::operator_program::Node::Affine{terms,bias}=&native.nodes[node] else{return Err("declared native MLP reader/writer not affine".into())};
            changed_native_operator_ids.extend(terms.iter().map(|(_,op)|*op));
            changed_native_operator_ids.extend(bias.iter().copied());
        }
    }
    let changed_native_literals=changed_native_operator_ids.iter().try_fold(0usize,|total,&op|{
        let gam_mpd::operator_program::OperatorBody::Dense{values,present,..}=&native.operators[op].body else{return Err("coefficient accounting requires dense native operators".to_string())};
        if !present.iter().all(|p|*p){return Err("native coefficient accounting requires complete dense operator".into());}
        total.checked_add(values.len()).ok_or_else(||"native literal count overflow".to_string())
    })?;
    let native_message = Artifact::native(&native)?.f32_literals()?.to_bytes()?;
    let base = Artifact::from_bytes(&native_message,&native.declarations)?;
    if base.to_bytes()?!=native_message{return Err("native canonical wire replay mismatch".into());}
    drop(native_message);
    let outputs=match &pool.program.nodes[pool.program.output]{gam_mpd::operator_program::Node::Concat{parts} if parts.len()==uses.len()=>parts.clone(),_=>return Err("pool must output declared per-use Concat".into())};
    let proposal=Proposal{program:pool.program,trainable:vec![],body_operators:vec![],outputs,uses:uses.clone()};
    let candidate=graft(&base,&native,&layers,&proposal)?;
    let candidate=candidate.f32_literals()?;
    candidate.validate_coverage(&native)?;
    let encoded = candidate.to_bytes()?;
    let artifact_path = out.join("candidate.artifact");
    std::fs::write(&artifact_path, &encoded).map_err(|e| e.to_string())?;
    // Read the actual saved bytes with the ordinary decoder before all fidelity work.
    let decoded = Artifact::from_bytes(&std::fs::read(&artifact_path).map_err(|e| e.to_string())?, &native.declarations)?;
    if decoded.to_bytes()? != encoded { return Err("full graft ordinary replay mismatch".into()); }
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
    save(&out.join("PROVENANCE.json"), &json!({"scope":"joint supplied-GELU geometry proposal assessment; discovery uses0/1, no frozen-body transfer or global optimum claim", "uses":uses,"fit_config":config,"fit_config_sha256":sha256(Path::new(&args[3]))?,"budget_sha256":sha256(Path::new(&args[7]))?,"input_width":width,"fitted_sha256":args[2],"candidate_sha256":sha256(&artifact_path)?,"spec_sha256":args[5],"export_json_sha256":sha256(&export.join("export.json"))?,"source_record":imported.record,"binary_sha256":sha256(&std::env::current_exe().map_err(|e|e.to_string())?)?,"native_C32":native_cost,"joint_pool_C32":pool_cost,"native_changed_operator_ids":changed_native_operator_ids,"native_changed_coefficient_literals":changed_native_literals,"native_changed_coefficient_payload_bits":changed_native_literals.checked_mul(32).ok_or("payload bit count overflow")?,"subsystem_cost_scope":"native changed coefficient payload only: unique reader/writer/bias IDs at32bits per literal, excludes structure and binding; joint pool standalone C32, full candidate additionally pays surrounding native model and graft/control wiring","candidate_C32":cost,"candidate_C32_bits":cost.total(),"local_boundary":"native normalized MLP input to pure MLP contribution, fixed native parents; native contribution RMS","local_rows":1024,"run_episodes":80,"trace_bytes":trace_bytes,"native_control_scope":"paid whole native activation uniform Scale only; individual native units are not mapped to learned coordinates", "numerical_scope":"Local final-write/RMS enclosures; checked exact fixed-binary64 raw-logit KL intervals; forward/RMS/gain/GEMM rounding excluded; not full neural arithmetic certificate"}))?;
    if args.len()==9 {
        save(&out.join("REPORT.json"),&json!({"scope":"CPU preparation only: complete jointly grafted canonical artifact; fidelity unmeasured", "candidate_sha256":sha256(&artifact_path)?,"C32_bits":cost.total(),"native_C32_bits":native_cost.total(),"joint_pool_C32_bits":pool_cost.total(),"native_changed_coefficient_payload_bits":changed_native_literals.checked_mul(32).ok_or("payload bit count overflow")?,"seconds":started.elapsed().as_secs_f64(),"acceptance_claim":false}))?;
        return Ok(());
    }
    let device = gam_gpu::tensor::Device::accelerator(gam_gpu::GpuPolicy::Required).map_err(|e|e.to_string())?.ok_or("CUDA required")?;
    if device.is_host() || !device.float64() || !cfg!(target_os="linux") {return Err("CUDA f64 required".into());}
    let local = Local::new(&native, family, None, 16).with_cuda(device.clone(), trace_bytes)?.with_cuda_resident_norms()?;
    let native_local=local.measure(&base)?;
    let timer = Instant::now(); let local_measure = local.measure(&decoded)?; let local_seconds = timer.elapsed().as_secs_f64();
    save(&out.join("LOCAL.json"), &json!({"measure":local_measure,"seconds":local_seconds}))?;
    // Run is a prespecified diagnostic even when Local is poor; no gate is weakened.
    let run = LanguageRun::new(&decoder, &native, &spec, &rows, 1)?.with_cuda(device, trace_bytes)?
        .with_cuda_readout(gam_mpd::native_readout::Budget{resident_bytes:required("readout_resident_bytes")?,workspace_bytes:required("readout_workspace_bytes")?})?
        .with_cuda_native_source(&base)?;
    let resident=run.resident_run(resident_budget);
    let timer=Instant::now();let native_run=resident.measure(&base)?;let native_run_seconds=timer.elapsed().as_secs_f64();
    let timer=Instant::now();let run_measure=resident.measure(&decoded)?;let run_seconds=timer.elapsed().as_secs_f64();
    let peak_rss = std::fs::read_to_string("/proc/self/status").ok().and_then(|text| text.lines().find_map(|s|s.strip_prefix("VmHWM:")?.split_whitespace().next()?.parse::<u64>().ok()?.checked_mul(1024)));
    save(&out.join("REPORT.json"), &json!({"joint_pool_C32_bits":pool_cost.total(),"native_changed_coefficient_payload_bits":changed_native_literals.checked_mul(32).ok_or("payload bit count overflow")?,"C32_bits":cost.total(),"native_C32_bits":native_cost.total(),"native_local":native_local,"local":local_measure,"local_seconds":local_seconds,"run":run_measure,"run_seconds":run_seconds,"native_run":native_run,"native_run_seconds":native_run_seconds,"resident_run_telemetry":resident.telemetry()?,"numeric_budgets":budgets,"seconds":started.elapsed().as_secs_f64(),"peak_host_rss_bytes":peak_rss,"acceptance_claim":false}))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_mpd::{operator_program::{OperatorProgram,Operator,Interface,Node,Law,exact_precision},shared_geometry_pilot::{build,Arm},run_check::LayerNodes};
    use ndarray::{Array2,array};
    use std::sync::Arc;
    #[test]
    fn one_decoded_pool_keeps_shared_body_in_joint_graft_and_wire_replay()->Result<(),String>{
        let dense=|name:&str,values:Array2<f64>|->Result<Arc<Operator>,String>{
            Ok(Arc::new(Operator::dense(name,Interface::native(values.nrows()).map_err(|e|e.to_string())?,Interface::native(values.ncols()).map_err(|e|e.to_string())?,values.clone(),exact_precision(values.iter().copied()).map_err(|e|e.to_string())?,Default::default()).map_err(|e|e.to_string())?))
        };
        let mut operators=vec![];let mut nodes=vec![Node::Raw{slot:0}];let mut layers=vec![];
        for layer in 0..2 {
            let op=operators.len();operators.push(dense("reader",Array2::from_shape_fn((3,2),|(r,c)|0.125*(r+c+layer+1) as f64))?);
            operators.push(dense("writer",Array2::from_shape_fn((2,3),|(r,c)|0.125*(r+c+layer+1) as f64))?);
            let pre=nodes.len();nodes.push(Node::Affine{terms:vec![(0,op)],bias:None});
            let active=nodes.len();nodes.push(Node::Pointwise{input:pre,laws:vec![Law::GeluTanh]});
            let mlp=nodes.len();nodes.push(Node::Affine{terms:vec![(active,op+1)],bias:None});
            layers.push(LayerNodes{stream:0,normed_stream:0,queries:vec![],keys:vec![],values:vec![],reads:vec![],attention:0,attended:0,normed:0,pre,active,mlp,residual:mlp});
        }
        let joined=nodes.len();nodes.push(Node::Concat{parts:layers.iter().map(|l|l.mlp).collect()});
        let native=OperatorProgram{declarations:Declarations{domains:vec![],slots:vec![Slot::Raw{width:2}],parameters:0},bases:vec![],operators,rules:vec![],output:joined,nodes};
        let mut proposal=build(&native,&layers,&[0,1],Arm::Learned,17)?;
        let bytes=Artifact::native(&proposal.program)?.to_bytes()?;
        proposal.program=Artifact::from_bytes(&bytes,&proposal.program.declarations)?.program;
        let f0=function(&proposal,0)?;let f1=function(&proposal,1)?;
        assert!(Arc::ptr_eq(&f0.operators[0],&f1.operators[0]));
        let candidate=graft(&Artifact::native(&native)?,&native,&layers,&proposal)?;
        let shared=&proposal.program.operators[0];
        assert_eq!(candidate.program.operators.iter().filter(|op|Arc::ptr_eq(op,shared)).count(),1);
        let x=array![[0.5,-0.25],[1.0,0.75]];
        let input=FamilyInputs{rows:2,slots:vec![SlotValues::Raw(x.clone())],layout:None};
        let trace=candidate.program.execute(&input,false).map_err(|e|e.to_string())?;
        for (slot,layer) in layers.iter().enumerate(){
            let body=function(&proposal,slot)?;let expected=body.execute(&input,false).map_err(|e|e.to_string())?;
            assert_eq!(trace.values[candidate.place(layer.mlp).ok_or("missing held output")?],expected.values[body.output]);
        }
        let bytes=candidate.to_bytes()?;let replay=Artifact::from_bytes(&bytes,&native.declarations)?;
        assert_eq!(replay.to_bytes()?,bytes);replay.validate_coverage(&native)?;
        // Human-readable operator/rule labels are not part of the canonical wire.
        assert_ne!(replay,candidate);
        let decoded_native=Artifact::from_bytes(&Artifact::native(&native)?.f32_literals()?.to_bytes()?,&native.declarations)?;
        let interner=gam_mpd::decoded_intern::DecodedOperatorInterner::new(&decoded_native)?;
        drop(interner);
        assert_eq!(replay.program.execute(&input,false).map_err(|e|e.to_string())?.values[replay.program.output],trace.values[candidate.program.output]);
        Ok(())
    }
}
