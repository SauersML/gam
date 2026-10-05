//! Inspect the actual native search space before allocating fitting compute.
//! EXPORT SETTINGS.json FRESH_OUT. No model forward, fitting, or heldout scoring.
use gam_mpd::{
    artifact::Artifact, coder_capture::sha256, import::import_language_model,
    operator_program::Node, program_joint_regions as joint, program_learned_dag as learned,
    run_check::split_sites,
};
use serde::Deserialize;
use serde_json::json;
use std::{collections::BTreeMap, path::Path, time::Instant};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Settings {
    export_sha256: String,
    regions: joint::Limits,
    grammar: learned::Settings,
    max_region_enumerations: usize,
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
