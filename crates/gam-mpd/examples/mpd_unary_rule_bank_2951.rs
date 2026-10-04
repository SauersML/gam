//! CPU-only complete generic unary weight-relation inventory and rank-zero fitting.
//! EXPORT OUT max_bank=6000 cache_bytes=268435456 workspace_bytes=67108864 max_seconds=600
use gam_mpd::{
    acceptance::{CostCache, structural_cost},
    artifact::Artifact,
    coder_capture::sha256,
    import::import_language_model,
    matrix_rule::Type,
    unary_rule_bank::{Limits, UnaryRuleBank},
};
use serde_json::{Value, json};
use std::{collections::BTreeMap, io::Write, path::Path, time::Instant};
fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    std::fs::write(
        path,
        serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}
fn peak_rss() -> Option<u64> {
    std::fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find_map(|l| {
            l.strip_prefix("VmHWM:")
                .and_then(|s| s.split_whitespace().next())
                .and_then(|s| s.parse::<u64>().ok())
                .and_then(|n| n.checked_mul(1024))
        })
}
fn ty(t: &Type) -> Value {
    match *t {
        Type::Matrix { rows, cols } => json!({"matrix":[rows,cols]}),
        Type::Vector { len } => json!({"vector":len}),
    }
}
fn main() -> Result<(), String> {
    let started = Instant::now();
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 6 {
        return Err("EXPORT OUT max_bank=N cache_bytes=N workspace_bytes=N max_seconds=N".into());
    }
    let mut options = BTreeMap::new();
    for arg in &args[2..] {
        let (k, v) = arg.split_once('=').ok_or("KEY=VALUE required")?;
        if !["max_bank", "cache_bytes", "workspace_bytes", "max_seconds"].contains(&k)
            || options.insert(k, v).is_some()
        {
            return Err("unknown/duplicate option".into());
        }
    }
    let integer = |k| -> Result<u64, String> {
        options
            .get(k)
            .ok_or("missing required option")?
            .parse::<u64>()
            .map_err(|e| e.to_string())
    };
    let (max_bank, cache_bytes, workspace_bytes, max_seconds) = (
        integer("max_bank")?,
        integer("cache_bytes")?,
        integer("workspace_bytes")?,
        integer("max_seconds")?,
    );
    if max_bank == 0 || max_seconds == 0 {
        return Err("positive bank/wall budgets required".into());
    }
    let (export, out) = (Path::new(&args[0]), Path::new(&args[1]));
    if out.exists() {
        return Err("fresh output directory required".into());
    }
    let record: Value = serde_json::from_slice(
        &std::fs::read(export.join("export.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let layers = usize::try_from(
        record["config"]["n_layers"]
            .as_u64()
            .ok_or("n_layers absent")?,
    )
    .map_err(|e| e.to_string())?;
    let native_layers: Vec<_> = (0..layers).collect();
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let scope = json!({"grammar":"typed MatrixRule version1; enumerator max_inputs1/max_nodes2, connected ordered DAGs; all applicable primitives; no internal Scale coefficient inventory",
        "amplitude":"one f32 least-squares coefficient fitted to native matrix entries and paid32bits", "residual_ranks":[0],
        "targets":"every O head of every imported native layer","native_layer_ids":native_layers,"sources":"ALL native Dense/LowRank/Diagonal learned operators and global parameters, excluding target; Identity excluded because it has no learned numeric payload; no weight-value source filters",
        "source_dependencies":"unmodified native operator literals have no operator dependencies","quality":"all Local/Run acceptance unresolved; weight fit diagnostic only, not a mechanism or global optimum",
        "max_candidates_including_native":max_bank,"cache_bytes":cache_bytes,"logical_matrix_workspace_bytes":workspace_bytes,"workspace_scope":"logical source/input/DAG/fit matrices only; library SVD scratch and model excluded; host limit and peakRSS separately recorded",
        "max_seconds":max_seconds,"timing":"driver clock begins before input hashing/import; expired budget retains every remaining inventory record as unmeasured","no_pruning":true,"artifacts_materialized_as_bank":0});
    write_json(&out.join("SCOPE.json"), &scope)?;
    let mut hashes = BTreeMap::new();
    hashes.insert(
        "export.json".to_string(),
        sha256(&export.join("export.json"))?,
    );
    for name in record["files"]
        .as_object()
        .ok_or("files metadata absent")?
        .keys()
    {
        let file = format!("{name}.f64");
        hashes.insert(file.clone(), sha256(&export.join(file))?);
    }
    let mut source_hashes = BTreeMap::new();
    for (name, text) in [
        ("driver", include_str!("mpd_unary_rule_bank_2951.rs")),
        ("bank", include_str!("../src/unary_rule_bank.rs")),
        (
            "enumerator",
            include_str!("../src/matrix_rule_enumeration.rs"),
        ),
        ("matrix_rule", include_str!("../src/matrix_rule.rs")),
        ("artifact", include_str!("../src/artifact.rs")),
        ("acceptance", include_str!("../src/acceptance.rs")),
    ] {
        let p = out.join(format!("{name}.source"));
        std::fs::write(&p, text).map_err(|e| e.to_string())?;
        source_hashes.insert(name, sha256(&p)?);
        std::fs::remove_file(p).map_err(|e| e.to_string())?;
    }
    write_json(
        &out.join("PROVENANCE.json"),
        &json!({"input_sha256":hashes,"source_sha256":source_hashes,"binary_sha256":sha256(&std::env::current_exe().map_err(|e|e.to_string())?)?,"export_source":record["source"],"config":record["config"],"tokens":"one row imported only to construct graph; never used by weight fitting","scope":scope}),
    )?;
    let imported = import_language_model(export, 1, 1)?;
    let native = Artifact::native(&imported.program)?.f32_literals()?;
    let mut bank = UnaryRuleBank::all_attention(
        &native,
        &native_layers,
        Limits {
            max_prefixes: 1_000_000,
            max_bodies: 100_000,
            max_candidates_including_native: max_bank,
            cache_bytes,
            matrix_workspace_bytes: workspace_bytes,
        },
    )?;
    let sources:Vec<_>=bank.inventory.sources.iter().map(|s|json!({"id":s.id,"name":native.program.operators[s.id].name,"type":ty(&s.ty),"dependencies":s.dependencies})).collect();
    let targets:Vec<_>=bank.targets.iter().map(|t|json!({"operator":t.operator,"name":native.program.operators[t.operator].name,"native_layer":t.native_layer,"head":t.head,"native_reads":t.reads,"native_write":t.write})).collect();
    let families:Vec<_>=bank.inventory.families.iter().enumerate().map(|(i,f)|{let code=f.body.rule.encode()?;Ok(json!({"index":i,"target":f.target,"inputs":f.body.rule.inputs.iter().map(ty).collect::<Vec<_>>(),"nodes":format!("{:?}",f.body.rule.nodes),"output":f.body.rule.output,"body_bits":code.len_bits(),"packed_bytes":code.packed_bytes(),"body_C32":f.body.rule.cost()?.c32(),"source_pools":f.source_pools,"binding_count":f.binding_count}))}).collect::<Result<_,String>>()?;
    write_json(
        &out.join("INVENTORY.json"),
        &json!({"complete":bank.inventory.complete,"candidate_count_including_native":bank.candidate_count,"visited_prefixes":bank.inventory.visited_prefixes,"targets":targets,"sources":sources,"families":families,"scope":scope}),
    )?;
    let choices: Vec<_> = bank.choices().collect(); // metadata only, never model copies.
    let mut costs = CostCache::default();
    let native_cost = structural_cost(&native, &mut costs)?;
    let mut journal = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(out.join("CANDIDATES.jsonl"))
        .map_err(|e| e.to_string())?;
    let native_record = json!({"index":0,"kind":"native_control","C32":native_cost,"cost_bits":native_cost.total(),"weight_relative_frobenius":0.0,"quality":"unmeasured; exact native fallback retained"});
    writeln!(journal, "{native_record}").map_err(|e| e.to_string())?;
    journal.flush().map_err(|e| e.to_string())?;
    let (mut fitted, mut unresolved, mut unmeasured, mut min_cost, mut savings) =
        (0u64, 0u64, 0u64, native_cost.total(), 0u64);
    for (ordinal, choice) in choices.into_iter().enumerate() {
        let start = Instant::now();
        let target = bank.target(choice)?.clone();
        let skeleton = bank.priced_skeleton(choice)?;
        let priced = structural_cost(&skeleton, &mut costs);
        drop(skeleton);
        let mut value = json!({"index":ordinal+1,"family":choice.family,"source":choice.source,"source_name":native.program.operators[choice.source].name,"target":target.operator,"native_layer":target.native_layer,"head":target.head,"residual_rank":0,"acceptance":"Local/Run unresolved"});
        match priced {
            Ok(cost) => {
                min_cost = min_cost.min(cost.total());
                if cost.total() < native_cost.total() {
                    savings += 1;
                }
                value["C32"] = json!(cost);
                value["cost_bits"] = json!(cost.total());
                value["saving_bits"] =
                    json!(i128::from(native_cost.total()) - i128::from(cost.total()));
            }
            Err(error) => {
                value["cost_unresolved"] = json!(error);
                value["cost_lower_bound"] = json!(0);
            }
        }
        if started.elapsed().as_secs() >= max_seconds {
            unmeasured += 1;
            value["unmeasured"] = json!("declared wall budget expired; retained without fit");
        } else {
            match bank.candidate(choice) {
                Ok((candidate, d)) => {
                    let actual = structural_cost(&candidate, &mut costs)?;
                    if value["cost_bits"].as_u64() != Some(actual.total()) {
                        return Err("fitted amplitude changed skeleton C32".into());
                    }
                    fitted += 1;
                    value["fit"] = json!({"amplitude":d.amplitude,"amplitude_bits":d.amplitude.to_bits(),"native_weight_squared":d.native_weight_squared,"residual_weight_squared":d.residual_weight_squared,"relative_weight_frobenius":d.relative_weight_frobenius,"logical_workspace_bytes":d.logical_workspace_bytes});
                    drop(candidate);
                }
                Err(error) => {
                    unresolved += 1;
                    value["unresolved"] = json!(error);
                }
            }
        }
        value["seconds"] = json!(start.elapsed().as_secs_f64());
        value["driver_seconds"] = json!(started.elapsed().as_secs_f64());
        value["peak_rss_bytes"] = json!(peak_rss());
        writeln!(journal, "{value}").map_err(|e| e.to_string())?;
        journal.flush().map_err(|e| e.to_string())?;
        if ordinal % 100 == 0 {
            eprintln!(
                "relation {}/{} fitted={fitted} unresolved={unresolved} unmeasured={unmeasured} elapsed={:.1}s",
                ordinal + 1,
                bank.candidate_count - 1,
                started.elapsed().as_secs_f64()
            );
        }
    }
    write_json(
        &out.join("REPORT.json"),
        &json!({"scope":scope,"candidate_count_including_native":bank.candidate_count,"inventory_complete":true,"fitted":fitted,"unresolved":unresolved,"unmeasured":unmeasured,"all_quality_unresolved":bank.candidate_count,"native_C32":native_cost,"minimum_declared_relation_C32":min_cost,"relations_with_positive_structural_savings":savings,"cache":{"bytes":bank.cache_stats.bytes,"hits":bank.cache_stats.hits,"misses":bank.cache_stats.misses,"unresolved":bank.cache_stats.unresolved},"driver_seconds":started.elapsed().as_secs_f64(),"peak_rss_bytes":peak_rss(),"claim":"priced weight proposals only; no acceptance/optimality/mechanism claim; native fallback retained"}),
    )
}
