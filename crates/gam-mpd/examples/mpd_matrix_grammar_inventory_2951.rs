//! Metadata-only inventory of a declared matrix-expression grammar on every native
//! attention output head. No numerical fit, fidelity ranking, or mechanistic claim.
//! EXPORT OUT MAX_INPUTS MAX_NODES MAX_PREFIXES MAX_BODIES MAX_BINDINGS MAX_C32
use gam_mpd::{
    artifact::Artifact,
    attention_map::AttentionLayerMap,
    coder_capture::sha256,
    import::import_language_model,
    matrix_rule::Type,
    matrix_rule_enumeration::{Budget, Source, Target, inventory},
    operator_program::OperatorBody,
};
use serde_json::json;
use std::{path::Path, time::Instant};

fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 8 {
        return Err("EXPORT OUT MAX_INPUTS MAX_NODES MAX_PREFIXES MAX_BODIES MAX_BINDINGS MAX_C32".into());
    }
    let integer = |i: usize| args[i].parse::<u64>().map_err(|e| e.to_string());
    let max_inputs = usize::try_from(integer(2)?).map_err(|e|e.to_string())?;
    let budget = Budget {
        max_nodes: usize::try_from(integer(3)?).map_err(|e|e.to_string())?,
        max_prefixes: integer(4)?,
        max_bodies: usize::try_from(integer(5)?).map_err(|e|e.to_string())?,
        scale_coefficients: vec![],
        max_c32: integer(7)?,
    };
    let max_bindings=integer(6)?;
    let export=Path::new(&args[0]);
    let output=Path::new(&args[1]);
    if output.exists() { return Err("fresh inventory report required".into()); }
    let start=Instant::now();
    let imported=import_language_model(export,1,1)?;
    let native=Artifact::native(&imported.program)?.f32_literals()?;
    let layers=usize::try_from(imported.record["config"]["n_layers"].as_u64().ok_or("native layer count absent")?).map_err(|e|e.to_string())?;
    let mut sources=Vec::new();
    let mut source_metadata=Vec::new();
    for (id,op) in native.program.operators.iter().enumerate() {
        let ty=match &op.body {
            OperatorBody::Identity=>continue,
            OperatorBody::Diagonal {values,..}=>Type::Vector {len:values.len()},
            OperatorBody::Dense {..}|OperatorBody::LowRank {..}=>Type::Matrix {rows:op.rows.width(),cols:op.cols.width()},
        };
        source_metadata.push(json!({"id":id,"name":op.name,"type":format!("{ty:?}")}));
        sources.push(Source {id,ty,dependencies:vec![]});
    }
    let mut targets=Vec::new();
    let mut target_metadata=Vec::new();
    for layer in 0..layers {
        let map=AttentionLayerMap::of(&native.program,layer)?;
        for head in map.heads {
            let op=&native.program.operators[head.output_operator];
            let ty=Type::Matrix {rows:op.rows.width(),cols:op.cols.width()};
            target_metadata.push(json!({"id":head.output_operator,"native_layer":layer,"head":head.head,"name":op.name,"type":format!("{ty:?}")}));
            targets.push(Target {id:head.output_operator,ty});
        }
    }
    let loading_seconds=start.elapsed().as_secs_f64();
    eprintln!("inventory {} targets, {} native learned sources, up to {max_inputs} inputs and {} nodes; all limits explicit",targets.len(),sources.len(),budget.max_nodes);
    let enumeration_start=Instant::now();
    let result=inventory(&targets,&sources,max_inputs,&budget,max_bindings)?;
    let enumeration_seconds=enumeration_start.elapsed().as_secs_f64();
    let families=result.families.iter().enumerate().map(|(id,f)| {
        let code=f.body.rule.encode()?;
        Ok(json!({"id":id,"target":f.target,"body_c32":f.body.rule.cost()?.c32(),"body_code_bits":code.len_bits(),"body_code_bytes":code.packed_bytes(),"input_permutation":f.body.input_permutation,"nodes":format!("{:?}",f.body.rule.nodes),"source_pools":f.source_pools,"binding_count":f.binding_count}))
    }).collect::<Result<Vec<_>,String>>()?;
    let report=json!({"scope":"metadata only; all native attention output heads and all learned native operators; target/transitive dependencies excluded; repeated source bindings permitted; no numerical scoring","max_inputs":max_inputs,"max_nodes":budget.max_nodes,"max_prefixes":budget.max_prefixes,"max_bodies":budget.max_bodies,"max_bindings":max_bindings,"max_c32":budget.max_c32,"internal_scale_coefficients":[],"complete":result.complete,"stop_reason":result.stop_reason,"binding_count":result.binding_count,"binding_count_meaning":if result.complete {"exact for declared grammar"} else {"lower bound only; unexplored cardinality unknown"},"visited_prefixes":result.visited_prefixes,"family_count":families.len(),"families":families,"targets":target_metadata,"sources":source_metadata,"export_sha256":sha256(&export.join("export.json"))?,"checkpoint":imported.record["source"],"binary_sha256":sha256(&std::env::current_exe().map_err(|e|e.to_string())?)?,"loading_seconds":loading_seconds,"enumeration_seconds":enumeration_seconds,"optimality":"none; this inventories a hypothesis language, not fitted or accepted programs"});
    let mut file=std::fs::OpenOptions::new().create_new(true).write(true).open(output).map_err(|e|e.to_string())?;
    serde_json::to_writer_pretty(&mut file,&report).map_err(|e|e.to_string())?;
    eprintln!("complete={} bindings={} prefixes={} families={} seconds={enumeration_seconds:.3}",result.complete,result.binding_count,result.visited_prefixes,result.families.len());
    Ok(())
}
