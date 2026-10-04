//! Bounded backend diagnostic, not a head/rank quality selection or bank generation.
//! EXPORT OUT max_bank=193 layer=3 head=5 rank=64 repeats=2
use gam_mpd::acceptance::{CostCache, structural_cost};
use gam_mpd::artifact::{Artifact, EncodedArtifact};
use gam_mpd::operator_program::NativeOperatorCodec;
use gam_mpd::precision::DecodableArtifact;
use gam_mpd::coder_capture::sha256;
use gam_mpd::import::import_language_model;
use gam_mpd::proposals::{CopyResidualBank, CopyResidualChoice, HeadApproximation};
use gam_mpd::run_check::{layer_nodes, split_sites};
use serde_json::{Value, json};
use std::{collections::BTreeMap, io::Write, path::Path, time::Instant};

const RANKS: [usize;4] = [8,32,64,96];
fn peak_rss_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| line.strip_prefix("VmHWM:").and_then(|s| s.split_whitespace().next())
        .and_then(|s| s.parse::<u64>().ok()).and_then(|k| k.checked_mul(1024)))
}
fn write_json(path: &Path, value: &Value) -> Result<(),String> {
    std::fs::write(path,serde_json::to_vec_pretty(value).map_err(|e|e.to_string())?).map_err(|e|e.to_string())
}
fn main() -> Result<(),String> {
    gam_mpd::engine::log_to_stderr();
    let args:Vec<String>=std::env::args().skip(1).collect();
    if !(7..=8).contains(&args.len()) { return Err("EXPORT OUT max_bank=N layer=N head=N rank=N repeats=N [codec_bytes=N]".into()); }
    let mut options=BTreeMap::new();
    for arg in &args[2..] {
        let (k,v)=arg.split_once('=').ok_or("expected KEY=VALUE")?;
        if !["max_bank","layer","head","rank","repeats","codec_bytes"].contains(&k) || options.insert(k,v).is_some() { return Err("unknown/duplicate option".into()); }
    }
    let number=|k|->Result<usize,String>{options.get(k).ok_or("missing option")?.parse().map_err(|e|format!("{e}"))};
    let (max_bank,layer,head,rank,repeats)=(number("max_bank")?,number("layer")?,number("head")?,number("rank")?,number("repeats")?);
    let codec_bytes=options.get("codec_bytes").unwrap_or(&"0").parse::<usize>().map_err(|e|e.to_string())?;
    if repeats==0 || repeats>2 || !RANKS.contains(&rank) { return Err("declare one/two diagnostic repetitions and a rank in8,32,64,96".into()); }
    let (export,out)=(Path::new(&args[0]),Path::new(&args[1]));
    let export_json=export.join("export.json");
    let record:Value=serde_json::from_slice(&std::fs::read(&export_json).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    let integer=|k|->Result<usize,String>{usize::try_from(record["config"][k].as_u64().ok_or("missing config integer")?).map_err(|e|e.to_string())};
    let (layers,heads,kv_heads)=(integer("n_layers")?,integer("n_heads")?,integer("n_kv_heads")?);
    let count=CopyResidualBank::cardinality((0..layers).map(|_|heads),&RANKS)?;
    // Cardinality guard precedes tensor import, output allocation, and SVDs.
    if count>max_bank { return Err(format!("complete bank needs {count} including native, max_bank={max_bank}")); }
    if layers!=4 || heads!=6 || kv_heads==0 || heads%kv_heads!=0 || layer>=layers || head>=heads { return Err("diagnostic requires4L6head export and valid explicit head indices".into()); }
    if out.exists() { return Err("use a fresh diagnostic output directory".into()); }
    std::fs::create_dir_all(out).map_err(|e|e.to_string())?;
    let scope=json!({"kind":"backend profiling only; no quality evaluation or head selection","all_heads":24,"declared_ranks":RANKS,
        "families":["CopyResidual","NativeSvd"],"complete_cardinality_including_native":count,"selected_diagnostic":{"layer":layer,"head":head,"rank":rank},
        "repetitions":repeats,"codec_bytes":codec_bytes,"artifacts_materialized_as_bank":0,"quality_measured":false,"quality_unresolved_including_native":count,
        "retention":"every declared alternative remains unmeasured for quality; no singleton or rank pruning","timing":"host composition/f32, coverage, encode, independent decode, coverage, reencode, pre/post exact C32; no Local/Run metrics"});
    write_json(&out.join("SCOPE.json"),&scope)?;
    let mut source_hashes=BTreeMap::new();
    for (name,text) in [("profile",include_str!("mpd_copy_residual_profile_2951.rs")),("proposals",include_str!("../src/proposals.rs")),
        ("rules",include_str!("../src/rules.rs")),("artifact",include_str!("../src/artifact.rs")),("codec",include_str!("../src/codec.rs")),
        ("operator_program",include_str!("../src/operator_program.rs")),("acceptance",include_str!("../src/acceptance.rs"))] {
        let path=out.join(format!("{name}.source"));std::fs::write(&path,text).map_err(|e|e.to_string())?;
        source_hashes.insert(name,sha256(&path)?); std::fs::remove_file(path).map_err(|e|e.to_string())?;
    }
    let mut data_hashes=BTreeMap::new();data_hashes.insert("export.json".into(),sha256(&export_json)?);
    for name in record["files"].as_object().ok_or("missing export files")?.keys() {
        let file=format!("{name}.f64");data_hashes.insert(file.clone(),sha256(&export.join(file))?);
    }
    write_json(&out.join("PROVENANCE.json"),&json!({"scope":scope,"sources":source_hashes,"inputs":data_hashes,
        "binary_sha256":sha256(&std::env::current_exe().map_err(|e|e.to_string())?)?}))?;
    let start=Instant::now();
    let imported=import_language_model(export,1,1)?;let native=split_sites(&imported.program)?;
    let nodes=layer_nodes(&native,layers)?;let base=Artifact::native(&native)?.f32_literals()?;
    let native_codec=if codec_bytes==0 {None} else {Some(NativeOperatorCodec::new(&base.program,codec_bytes).map_err(|e|e.to_string())?)};
    let bank=CopyResidualBank::new(&base,&nodes,heads/kv_heads,&RANKS,max_bank)?;
    let mut cache=CostCache::default();let native_cost=structural_cost(&base,&mut cache)?;
    write_json(&out.join("NATIVE_COST.json"),&json!({"cost":native_cost,"C32_bits":native_cost.total(),"import_and_native_cost_seconds":start.elapsed().as_secs_f64(),"peak_rss_bytes":peak_rss_bytes()}))?;
    let mut file=std::fs::OpenOptions::new().create_new(true).write(true).open(out.join("PROFILE.jsonl")).map_err(|e|e.to_string())?;
    for repeat in 0..repeats {
        // Alternate the diagnostic pair; nothing is accepted or pruned.
        let order=if repeat%2==0 {[HeadApproximation::NativeSvd,HeadApproximation::CopyResidual]} else {[HeadApproximation::CopyResidual,HeadApproximation::NativeSvd]};
        for family in order {
            let choice=CopyResidualChoice{layer,head,rank,family};eprintln!("diagnostic start repeat={repeat} choice={choice:?}");
            let started=Instant::now();let (decoded,cost,wire,timing)=bank.checked_profiled(choice,&mut cache)?;
            let original_seconds=started.elapsed().as_secs_f64();
            let codec_profile=if let Some(codec)=&native_codec {
                let t=Instant::now();let candidate=bank.candidate(choice)?.f32_literals()?;
                let prepare=t.elapsed().as_secs_f64();
                let t=Instant::now();let encoded=EncodedArtifact::of_with_native_codec(&candidate,codec)?;
                let encode=t.elapsed().as_secs_f64();
                let t=Instant::now();let reused=encoded.using_native_codec(codec).decode()?;
                let decode=t.elapsed().as_secs_f64();
                let t=Instant::now();let post_cost=structural_cost(&reused,&mut CostCache::default())?;
                let decoded_cost=t.elapsed().as_secs_f64();
                if post_cost!=cost || reused!=decoded {return Err("cached profile changed decoded artifact or C32".into());}
                let t=Instant::now();let ordinary=EncodedArtifact::of(&candidate)?;
                let ordinary_encode=t.elapsed().as_secs_f64();
                if ordinary.message!=encoded.message {return Err("cached profile changed standalone message".into());}
                Some(json!({"candidate_prepare_seconds":prepare,"encode_seconds":encode,"decode_seconds":decode,"decoded_cost_seconds":decoded_cost,"ordinary_encode_seconds":ordinary_encode,"exact_message_and_decoded_artifact_equal":true,"stats":codec.stats(),"usage":codec.usage()}))
            } else {None};
            let value=json!({"codec_profile":codec_profile,"repeat":repeat,"choice":choice,"C32":cost,"C32_bits":cost.total(),"exact_wire_bytes":wire,"staged_seconds":timing,
                "total_checked_seconds":original_seconds,"peak_rss_bytes":peak_rss_bytes(),"retained_native_places":decoded.places.len(),
                "decoded_coverage":true,"numeric_projection":"independent f32 literals; exact architecture epsilon preserved","quality_assessed":false});
            writeln!(file,"{value}").map_err(|e|e.to_string())?;file.flush().map_err(|e|e.to_string())?;
            println!("{value}");drop(decoded);
        }
    }
    write_json(&out.join("COMPLETE.json"),&json!({"state":"backend diagnostic complete","elapsed_seconds":start.elapsed().as_secs_f64(),"peak_rss_bytes":peak_rss_bytes(),
        "candidate_count":bank.candidate_count,"quality_assessed":false,"full_bank_materialized":false}))?;
    Ok(())
}
