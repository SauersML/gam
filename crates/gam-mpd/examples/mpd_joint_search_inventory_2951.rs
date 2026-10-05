//! Inspect the actual native search space before allocating fitting compute.
//! EXPORT SETTINGS.json FRESH_OUT. Accepts either legacy region-inventory settings
//! or the main composed driver's structural settings. No forwards or fitting.
use gam_mpd::{
    acceptance::{structural_cost, CostCache},
    artifact::Artifact, engine::sha256, import::import_language_model,
    operator_program::Node, program_joint_regions as joint, program_learned_dag as learned,
    program_structure_search as search, intervention_program::{self, Control},
    run_check::{split_sites, layer_nodes},
};
use serde::Deserialize;
use serde_json::json;
use std::{collections::{BTreeMap, BTreeSet}, io::Write, path::Path, time::Instant};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Settings {
    export_sha256: String,
    regions: joint::Limits,
    grammar: learned::Settings,
    max_region_enumerations: usize,
}

// Read only fields that affect graph construction and proposal scheduling.
// The untouched full configuration is archived; unsupported lowerings reject.
#[derive(Deserialize)]
struct StructuralInput {
    export_sha256: String,
    layers: usize,
    sequences: usize,
    context: usize,
    controls: Vec<NativeControl>,
    structural_search: StructuralInputSearch,
}
#[derive(Deserialize)]
struct StructuralInputSearch {
    settings: search::Settings,
    constraints: search::Constraints,
}
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum NativeControl {
    MlpOutput { layer: usize },
    AttentionOutput { layer: usize },
    RetainedOperator { name: String },
}

fn structural_preflight(export: &Path, bytes: &[u8], out: &Path) -> Result<(), String> {
    let raw: serde_json::Value = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
    for unsupported in ["native_parameter_edits", "down_edit_family", "frozen_shared_body", "joint_response_search"] {
        if raw.get(unsupported).is_some_and(|v| !v.is_null()) {
            return Err(format!("structural preflight does not implement {unsupported}; refusing a different controlled graph"));
        }
    }
    if raw["native_initialization"].as_bool() == Some(true)
        || raw["width"].as_u64().is_some_and(|w| w != 0)
        || ["uses", "expression_ids"].iter().any(|k| raw[*k].as_array().is_some_and(|v| !v.is_empty())) {
        return Err("structural preflight requires the main driver's automatic structural mode".into());
    }
    let config: StructuralInput = serde_json::from_value(raw.clone()).map_err(|e| e.to_string())?;
    if sha256(&export.join("export.json"))? != config.export_sha256 {
        return Err("export identity differs".into());
    }
    if config.layers == 0 || config.sequences == 0 || config.context == 0 || config.controls.is_empty() {
        return Err("positive dimensions and explicit native controls required".into());
    }
    let cases = raw["cases"].as_array().filter(|v| !v.is_empty()).ok_or("nonempty cases required")?;
    let mut labels = BTreeSet::new();
    for case in cases {
        let label = case["label"].as_str().filter(|s| !s.is_empty()).ok_or("case label absent")?;
        if !labels.insert(label) || case["group"].as_str().is_none_or(|s| s.is_empty()) {
            return Err("unique labels and nonempty case groups required".into());
        }
        let gains = case["gains"].as_array().ok_or("case gains absent")?;
        if gains.len() != config.controls.len() || gains.iter().any(|g| g.as_f64().is_none_or(|x| !x.is_finite())) {
            return Err("one finite gain per declared native control required".into());
        }
        if ["down_amplitudes", "parameter_amplitudes"].iter()
            .any(|k| case[*k].as_array().is_some_and(|v| !v.is_empty())) {
            return Err("native edit amplitudes require their unsupported lowering; refusing partial preflight".into());
        }
    }
    if !cases.iter().any(|c| c["gains"].as_array().is_some_and(|gs| gs.iter().all(|g| g.as_f64() == Some(1.)))) {
        return Err("explicit clean case required".into());
    }
    let start = Instant::now();
    let imported = import_language_model(export, config.sequences, config.context)?;
    let original = split_sites(&imported.program)?;
    let layers = layer_nodes(&original, config.layers)?;
    let controls = config.controls.iter().map(|c| match c {
        NativeControl::MlpOutput { layer } => Ok(Control::NodeScale {
            node: layers.get(*layer).ok_or("controlled native layer absent")?.mlp,
        }),
        NativeControl::AttentionOutput { layer } => Ok(Control::NodeScale {
            node: layers.get(*layer).ok_or("controlled native layer absent")?.attention,
        }),
        NativeControl::RetainedOperator { name } => {
            let ids = original.operators.iter().enumerate().filter(|(_, op)| op.name == *name)
                .map(|(i, _)| i).collect::<Vec<_>>();
            if ids.len() != 1 { return Err(format!("unique native operator required: {name}")); }
            Ok(Control::GlobalOperatorScale { operator: ids[0] })
        }
    }).collect::<Result<Vec<_>, String>>()?;
    let controlled = intervention_program::compile(&original, &controls)?;
    // Match structural_run: canonical f32-literal parent before enumeration.
    let native = Artifact::native(&controlled.program)?;
    let wire = native.f32_literals()?.to_bytes()?;
    let initial = Artifact::from_bytes(&wire, &controlled.program.declarations)?;
    if initial.to_bytes()? != wire { return Err("noncanonical controlled native replay".into()); }
    let mut settings = config.structural_search.settings;
    if let Some(observations) = &settings.joint_observation_places {
        if observations.iter().any(|n| *n >= original.nodes.len()) {
            return Err("joint observation absent from original native graph".into());
        }
        settings.joint_observation_places = Some(observations.iter().map(|n| controlled.root_mapping[*n]).collect());
    }
    let layer_places = layers.iter().enumerate().map(|(layer, l)| json!({
        "layer":layer,"original":{"normed":l.normed,"pre":l.pre,"active":l.active,"mlp":l.mlp},
        "controlled":{"normed":controlled.root_mapping[l.normed],"pre":controlled.root_mapping[l.pre],
            "active":controlled.root_mapping[l.active],"mlp":controlled.root_mapping[l.mlp]},
    })).collect::<Vec<_>>();
    let constraints = config.structural_search.constraints;
    let identity_axes = |axes: &[search::Metric]| axes.iter().map(|m| search::Metric { name:m.name.clone(), value:0. }).collect();
    let initial_c32 = structural_cost(&initial, &mut CostCache::default())?.total();
    // The seed's reference here is the canonical controlled native compared
    // with itself: its identity errors are zero by construction, not measured
    // original-versus-rounded scores. No child is admitted, so these axes never
    // select a fitted parent. Use the real ordinary C32 price, not a dummy cost.
    let initial = search::EvaluatedArtifact { artifact:initial, evaluation:search::Evaluation {
        fidelity:identity_axes(&constraints.max_fidelity),
        local_errors:identity_axes(&constraints.max_local_errors),
        intervention_errors:identity_axes(&constraints.max_intervention_errors), description_bits:initial_c32 as f64,
    }};
    std::fs::create_dir(out).map_err(|e| e.to_string())?;
    std::fs::write(out.join("SETTINGS.json"), bytes).map_err(|e| e.to_string())?;
    let mut journal = std::fs::File::create(out.join("CALLBACKS.jsonl")).map_err(|e| e.to_string())?;
    let mut callbacks = Vec::new();
    let mut layer_coverage = BTreeMap::<usize, usize>::new();
    let result = search::search(&controlled.program, initial, &settings, &constraints, |request| {
        let joint_region = match request.mutation {
            search::Mutation::ReuseNativeExpression { region, .. }
            | search::Mutation::SynthesizeLearnedDAG { region, .. }
            | search::Mutation::SynthesizeSharedDAG { region, .. } => Some(region),
            _ => None,
        };
        let covered = layers.iter().enumerate().filter_map(|(layer, l)| {
            joint_region.is_some_and(|r| [l.pre, l.active, l.mlp].iter()
                .any(|n| r.current_internal_nodes.contains(&controlled.root_mapping[*n])))
                .then_some(layer)
        }).collect::<Vec<_>>();
        for layer in &covered { *layer_coverage.entry(*layer).or_default() += 1; }
        let owners = match request.mutation {
            search::Mutation::ReuseNativeExpression { proposal, .. } => Some(proposal.owners.iter().map(|owner| {
                let op = &request.parent.artifact.program.operators[owner.parent_operator];
                json!({"parent_operator":owner.parent_operator,"name":op.name,
                    "rows":op.rows.width(),"cols":op.cols.width(),"trainable":owner.trainable,
                    "coefficient_elements":owner.coefficient_elements})
            }).collect::<Vec<_>>()),
            _ => None,
        };
        callbacks.push(json!({"scheduled_callback_index":callbacks.len(),"attempt_id":request.attempt_id,
            "parent_id":request.parent_id,"depth":request.depth,"mutation":request.mutation,
            "native_layer_coverage":covered,"native_coefficient_owners":owners,
            "initialization":request.initialization,"trainable_operator_ids":request.trainable_operator_ids,
            "candidate_operator_count":request.candidate.program.operators.len(),
            "candidate_trainable_operator_count":request.trainable_operator_ids.len(),
            "candidate_frozen_operator_count":request.candidate.program.operators.len()-request.trainable_operator_ids.len(),
            "status":"structurally_valid_callback_reached_unmeasured"}));
        serde_json::to_writer(&mut journal, callbacks.last().expect("just appended callback"))
            .map_err(|e| format!("preflight journal write failed: {e}"))?;
        journal.write_all(b"\n").and_then(|_| journal.flush())
            .map_err(|e| format!("preflight journal flush failed: {e}"))?;
        println!("preflight callback {} attempt {}: {} trainable owners", callbacks.len(),
            request.attempt_id, request.trainable_operator_ids.len());
        Err("inventory-only callback: no fitting or measurement; child deliberately not admitted".into())
    })?;
    let controls = result.report.learned_dag_enumerations.iter().filter_map(|entry| {
        entry.native_inventory.as_ref().map(|inventory| json!({"region":entry.region,
            "control":inventory.control,
            "trainable_owner_count":inventory.control.owners.iter().filter(|o| o.trainable).count(),
            "frozen_owner_count":inventory.control.owners.iter().filter(|o| !o.trainable).count(),
            "classification":"native-copy preservation control; excluded from discovery callbacks"}))
    }).collect::<Vec<_>>();
    let mut driver_report = serde_json::to_value(&result.report).map_err(|e| e.to_string())?;
    driver_report.as_object_mut().ok_or("search report object absent")?.remove("admitted_candidates");
    let report = json!({"mode":"main_structural_initial_parent_preflight","source":imported.record,
        "export_sha256":config.export_sha256,"settings_sha256":sha256(&out.join("SETTINGS.json"))?,
        "original_nodes":original.nodes.len(),"controlled_nodes":controlled.program.nodes.len(),
        "original_to_controlled_root_mapping":controlled.root_mapping,
        "control_slots":controlled.control_slots.iter().map(|s| json!({"control":s.control,"slot":s.slot,"width":s.width})).collect::<Vec<_>>(),
        "controlled_source_operators":controlled.program.operators.iter().enumerate().map(|(id,op)| json!({
            "id":id,"name":op.name,"rows":op.rows.width(),"cols":op.cols.width()})).collect::<Vec<_>>(),
        "declared_controls":raw["controls"],"declared_cases":raw["cases"],"layer_places":layer_places,
        "effective_search_settings":settings,"callbacks":callbacks,"callback_layer_coverage":layer_coverage,
        "initial_c32":initial_c32,
        "initial_error_reference":"Canonical controlled native compared with itself; zero identity axes by construction, not measured original-versus-rounded scores. Seed error records omitted.",
        "native_copy_controls":controls,"driver_report":driver_report,"seconds":start.elapsed().as_secs_f64(),
        "scope":"Exact public structural-search initial-parent schedule, applied proposals and canonical replay on the main driver's controlled native graph. No model forward, fitting, KL, Local, heldout reading, or discovery evidence. All callbacks deliberately return an unmeasured sentinel, so callback_failures count inspection stops, not failed fits; no child is admitted and deeper-parent coverage is unknown. Identity-baseline error records are omitted and must not be read as measurements. preserve_native_places retain the main driver's controlled-node interpretation."});
    std::fs::write(out.join("REPORT.json"), serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    println!("{} actual initial-parent callbacks inspected in {:.3}s", callbacks.len(), start.elapsed().as_secs_f64());
    Ok(())
}
fn kind(node: &Node) -> &'static str {
    match node {
        Node::Feature { .. } => "Feature",
        Node::Raw { .. } => "Raw",
        Node::Constant { .. } => "Constant",
        Node::Affine { .. } => "Affine",
        Node::Bilinear { .. } => "Bilinear",
        Node::Softmax { .. } => "Softmax",
        Node::Mix { .. } => "Mix",
        Node::Pointwise { .. } => "Pointwise",
        Node::Hadamard { .. } => "Hadamard",
        Node::Readout { .. } => "Readout",
        Node::Outer { .. } => "Outer",
        Node::Concat { .. } => "Concat",
        Node::Param { .. } => "Param",
        Node::Call { .. } => "Call",
        Node::Gain { .. } => "Gain",
        Node::Attend { .. } => "Attend",
        Node::RmsNorm { .. } => "RmsNorm",
        Node::Transposed { .. } => "Transposed",
    }
}
fn main() -> Result<(), String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 3 {
        return Err("EXPORT SETTINGS.json FRESH_OUT".into());
    }
    let export = Path::new(&args[0]);
    let bytes = std::fs::read(&args[1]).map_err(|e| e.to_string())?;
    let raw: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    if raw.get("structural_search").is_some_and(|v| !v.is_null()) {
        return structural_preflight(export, &bytes, Path::new(&args[2]));
    }
    let settings: Settings = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    if sha256(&export.join("export.json"))? != settings.export_sha256 {
        return Err("export identity differs".into());
    }
    if settings.max_region_enumerations == 0 {
        return Err("positive region enumeration budget required".into());
    }
    let out = Path::new(&args[2]);
    std::fs::create_dir(out).map_err(|e| e.to_string())?;
    std::fs::write(out.join("SETTINGS.json"), bytes).map_err(|e| e.to_string())?;
    let start = Instant::now();
    let imported = import_language_model(export, 1, 4)?;
    let native = split_sites(&imported.program)?;
    let artifact = Artifact::native(&native)?;
    let interfaces = native.interfaces().map_err(|e| e.to_string())?;
    let inventory = joint::propose_regions(&artifact, settings.regions)?;
    let mut rows = Vec::new();
    for (index, region) in inventory.regions.iter().enumerate() {
        let mut operations = BTreeMap::new();
        for &node in &region.current_internal_nodes {
            *operations
                .entry(kind(&native.nodes[node]))
                .or_insert(0usize) += 1;
        }
        let enumeration = if index < settings.max_region_enumerations {
            Some(
                match learned::enumerate(&artifact, region, &settings.grammar, &region.native_reads)
                {
                    Ok(proposals) => json!({
                        "status":"enumerated", "proposals":proposals.proposals.len(),
                        "truncated":proposals.truncated, "states":proposals.explored_states,
                        "tuple_work":proposals.checked_tuples, "rejections":proposals.rejection_counts,
                        "minimum_parameter_elements":proposals.proposals.iter().map(|p|p.parameter_elements).min(),
                        "first_proposals":proposals.proposals.iter().take(4).collect::<Vec<_>>(),
                    }),
                    Err(error) => json!({"status":"rejected", "error":error}),
                },
            )
        } else {
            None
        };
        rows.push(json!({
            "index":index, "region":region, "operations":operations,
            "input_interfaces":region.current_reads.iter().map(|&i|json!({"width":interfaces[i].width(),"groups":interfaces[i].group_count()})).collect::<Vec<_>>(),
            "output_interfaces":region.current_writes.iter().map(|&i|json!({"width":interfaces[i].width(),"groups":interfaces[i].group_count()})).collect::<Vec<_>>(),
            "source_operators":region.current_internal_nodes.iter().flat_map(|&i|native.nodes[i].operators())
                .map(|i|json!({"id":i,"name":native.operators[i].name,"rows":native.operators[i].rows.width(),"cols":native.operators[i].cols.width()})).collect::<Vec<_>>(),
            "learned_proposals":enumeration,
        }));
    }
    let report = json!({
        "source":imported.record,"nodes":native.nodes.len(),"regions":rows,
        "region_states":inventory.explored_states,"region_truncated":inventory.truncated,
        "attempted_expansions":inventory.attempted_expansions,
        "skipped":inventory.skipped,"seconds":start.elapsed().as_secs_f64(),
        "scope":"Native syntax and bounded proposal coverage only. No model forward or fitting; no heldout scores. Operation names are architectural primitives, not discovered mechanisms. An affine fit or reproducing known architectural composition does not establish learned computational organization.",
    });
    std::fs::write(
        out.join("REPORT.json"),
        serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!(
        "{} native regions inspected in {:.3}s",
        inventory.regions.len(),
        start.elapsed().as_secs_f64()
    );
    Ok(())
}
