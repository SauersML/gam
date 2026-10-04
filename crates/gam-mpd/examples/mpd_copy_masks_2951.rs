//! Exhaustive layer-local Copy mask measurements; no joint materialization or Run pruning.
//! mpd_copy_masks_2951 EXPORT OUT max_layer=N max_joint=N [local=4 context=512 batch=1024 deltas=0.1,0.2 budget=N]
use gam_mpd::acceptance::Local;
use gam_mpd::artifact::Artifact;
use gam_mpd::counterfactual::passages;
use gam_mpd::import::import_language_model;
use gam_mpd::operator_program::{FamilyInputs, SequenceLayout, SlotValues};
use gam_mpd::proposals::CopyMasks;
use gam_mpd::run_check::{layer_nodes, split_sites};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Instant;
fn digest(path: Option<&Path>, text: Option<&str>) -> Result<String, String> {
    for (program, options) in [("sha256sum", vec![]), ("shasum", vec!["-a", "256"])] {
        let mut command = Command::new(program);
        command.args(options).stdout(Stdio::piped()).stderr(Stdio::piped());
        if let Some(path) = path {
            command.arg("--").arg(path);
        } else {
            command.stdin(Stdio::piped());
        }
        let mut child = match command.spawn() {
            Ok(child) => child,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(format!("{program}: {e}")),
        };
        if let Some(text) = text {
            child.stdin.take().ok_or("hash process has no stdin")?.write_all(text.as_bytes()).map_err(|e| e.to_string())?;
        }
        let output = child.wait_with_output().map_err(|e| e.to_string())?;
        if !output.status.success() {
            return Err(format!("{program}: {}", String::from_utf8_lossy(&output.stderr)));
        }
        let stdout = String::from_utf8(output.stdout).map_err(|e| e.to_string())?;
        let hash = stdout.split_whitespace().next().ok_or("empty SHA-256 output")?;
        if hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(format!("invalid SHA-256 output from {program}"));
        }
        return Ok(hash.to_ascii_lowercase());
    }
    Err("provenance needs sha256sum or shasum on PATH".into())
}


fn main() -> Result<(), String> {
    gam_mpd::engine::log_to_stderr();
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 { return Err("mpd_copy_masks_2951 EXPORT OUT max_layer=N max_joint=N [local=4 context=512 batch=1024 deltas=0.1,0.2 budget=N]".into()); }
    let (export, out) = (Path::new(&args[1]), Path::new(&args[2]));
    let mut keys = BTreeMap::new();
    for argument in &args[3..] {
        let (k, v) = argument.split_once('=').ok_or("expected KEY=VALUE")?;
        if !["max_layer", "max_joint", "local", "context", "batch", "deltas", "budget"].contains(&k) || keys.insert(k, v).is_some() {
            return Err(format!("unknown or duplicate option {k}"));
        }
    }
    let number = |k: &str, default: Option<usize>| -> Result<usize, String> {
        match keys.get(k) { Some(v) => v.parse().map_err(|e| format!("{k}: {e}")), None => default.ok_or_else(|| format!("declare {k}=N")) }
    };
    let (max_layer, max_joint) = (number("max_layer", None)?, number("max_joint", None)?);
    let record: Value = serde_json::from_slice(&std::fs::read(export.join("export.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let config = &record["config"];
    let cfg = |k: &str| -> Result<usize, String> { usize::try_from(config[k].as_u64().ok_or_else(|| format!("missing {k}"))?).map_err(|e| e.to_string()) };
    let (layers, heads, kv) = (cfg("n_layers")?, cfg("n_heads")?, cfg("n_kv_heads")?);
    // Complete implicit cardinality guard precedes import, hashing, outputs, and rule fitting.
    let (local_count, joint_count) = CopyMasks::cardinalities(std::iter::repeat_n(heads, layers), max_layer, max_joint)?;
    let budget = number("budget", Some(local_count))?;
    if budget > local_count { return Err("budget exceeds complete layer bank".into()); }
    if kv == 0 || heads % kv != 0 { return Err("invalid KV grouping".into()); }
    if out.exists() { return Err("output directory must be fresh".into()); }
    let (count, context, batch) = (number("local", Some(4))?, number("context", Some(512))?, number("batch", Some(1024))?);
    if count == 0 || context == 0 || batch == 0 { return Err("local/context/batch must be positive".into()); }
    let deltas: Vec<f64> = keys.get("deltas").copied().unwrap_or("0.1,0.2").split(',').map(|s| s.parse::<f64>().map_err(|e| e.to_string())).collect::<Result<_, _>>()?;
    if deltas.is_empty() || deltas.iter().any(|v| !v.is_finite() || *v < 0.0) { return Err("invalid deltas".into()); }
    let mut input_hashes = BTreeMap::new();
    input_hashes.insert("export.json".to_string(), digest(Some(&export.join("export.json")), None)?);
    for name in record["files"].as_object().ok_or("missing files")?.keys() {
        let filename = format!("{name}.f64");
        input_hashes.insert(filename.clone(), digest(Some(&export.join(filename)), None)?);
    }
    let mut sources = BTreeMap::new();
    for (name, source) in [
        ("mpd_copy_masks_2951.rs", include_str!("mpd_copy_masks_2951.rs")),
        ("proposals.rs", include_str!("../src/proposals.rs")),
        ("artifact.rs", include_str!("../src/artifact.rs")),
        ("acceptance.rs", include_str!("../src/acceptance.rs")),
        ("operator_program.rs", include_str!("../src/operator_program.rs")),
        ("rules.rs", include_str!("../src/rules.rs")),
        ("run_check.rs", include_str!("../src/run_check.rs")),
        ("import.rs", include_str!("../src/import.rs")),
        ("counterfactual.rs", include_str!("../src/counterfactual.rs")),
    ] { sources.insert(name, digest(None, Some(source))?); }
    let binary_hash = digest(Some(&std::env::current_exe().map_err(|e| e.to_string())?), None)?;
    let imported = import_language_model(export, 1, 1)?;
    let native = split_sites(&imported.program)?;
    let nodes = layer_nodes(&native, layers)?;
    let start = Artifact::native(&native)?.f32_literals()?;
    let bank = CopyMasks::new(&start, &nodes, heads / kv, max_layer, max_joint)?;
    let preparation = Instant::now();
    bank.prepare()?;
    let preparation_seconds = preparation.elapsed().as_secs_f64();
    let sequences = passages(export, context)?;
    if sequences.len() < count || sequences.iter().take(count).any(|r| r.len() < context) { return Err("insufficient local sequences/context".into()); }
    let (mut tokens, mut sequence, mut position) = (Vec::new(), Vec::new(), Vec::new());
    for (s, row) in sequences.iter().take(count).enumerate() {
        for (p, &token) in row.iter().take(context).enumerate() {
            tokens.push(token); sequence.push(u32::try_from(s).map_err(|e| e.to_string())?); position.push(u32::try_from(p).map_err(|e| e.to_string())?);
        }
    }
    let family = FamilyInputs { rows: tokens.len(), slots: vec![SlotValues::Tokens(tokens)], layout: Some(SequenceLayout { sequence, position }) };
    let local = Local::new(&native, family.clone(), None, batch);
    std::fs::create_dir(out).map_err(|e| e.to_string())?;
    let mut log = std::fs::File::create(out.join("LOCAL.jsonl")).map_err(|e| e.to_string())?;
    let mut retained = vec![vec![Vec::<usize>::new(); layers]; deltas.len()];
    let started = Instant::now();
    for (index, (layer, mask)) in bank.layer_masks().enumerate() {
        if index >= budget {
            for grid in &mut retained { grid[layer].push(mask); }
            serde_json::to_writer(&mut log, &json!({"layer":layer,"mask":mask,"state":"unmeasured_retained"})).map_err(|e| e.to_string())?;
            writeln!(log).map_err(|e| e.to_string())?;
            continue;
        }
        let candidate_started = Instant::now();
        let mut masks = vec![0; layers]; masks[layer] = mask;
        let measured = bank.checked(&masks, &native).and_then(|(artifact, cost, bytes)| {
            let checked_seconds = candidate_started.elapsed().as_secs_f64();
            let local_started = Instant::now();
            local.screen(&artifact).map(|measure| (measure, cost, bytes, checked_seconds, local_started.elapsed().as_secs_f64()))
        });
        let entry = match measured {
            Ok((measure, cost, bytes, checked_seconds, local_seconds)) => {
                let lower = measure.blocks.iter().map(|b| b.lower).fold(0.0_f64, f64::max);
                let upper = measure.blocks.iter().map(|b| b.upper).fold(0.0_f64, f64::max);
                let states: Vec<_> = deltas.iter().enumerate().map(|(i, &delta)| {
                    let state = if lower > delta { "violates" } else if upper <= delta { "passes_declared_family" } else { "unresolved" };
                    if state != "violates" { retained[i][layer].push(mask); }
                    json!({"delta":delta,"state":state})
                }).collect();
                json!({"layer":layer,"mask":mask,"local":measure,"states":states,"C32":cost,"C32_bits":cost.total(),"wire_bytes":bytes,"checked_seconds":checked_seconds,"local_seconds":local_seconds})
            }
            Err(error) => {
                for grid in &mut retained { grid[layer].push(mask); }
                json!({"layer":layer,"mask":mask,"state":"failed_unresolved_retained","error":error})
            }
        };
        serde_json::to_writer(&mut log, &entry).map_err(|e| e.to_string())?;
        writeln!(log).map_err(|e| e.to_string())?;
        log.flush().map_err(|e| e.to_string())?;
        eprintln!("layer {layer} mask {mask} measured; {:.3}s", started.elapsed().as_secs_f64());
    }
    let grids: Vec<_> = deltas.iter().zip(&retained).map(|(&delta, masks)| {
        let remaining = masks.iter().map(Vec::len).product::<usize>();
        json!({"delta":delta,"retained_layer_masks":masks,"unexcluded_joint_candidates":remaining})
    }).collect();
    let manifest = json!({"schema":"copy-layer-mask-local-v1","scope":{"laws":"Copy only","complete_layer_masks":local_count,"implicit_joint_candidates":joint_count,
        "native":"empty mask at each layer","measurement_budget":budget,"unmeasured_retained":true,"max_layer":max_layer,"max_joint":max_joint,"single_failure_pruning":false,"joint_artifacts_emitted":0,
        "local":"fixed native parents; declared family only; ascent disabled","run":"not measured; no run pruning","global_optimum_claim":false},
        "local_sequences":count,"context":context,"rows":family.rows,"batch":batch,"input_sha256":input_hashes,"compiled_source_sha256":sources,
        "binary_sha256":binary_hash,"literal_projection":"f32 learned literals; exact architecture epsilon retained", "price":"C32 distinct from exact wire bytes",
        "Copy_derivation_preparation_seconds":preparation_seconds,"derivation_memoization":"one native-source fit per head; checked codec still recomputes derived values",
        "grids":grids,"seconds":started.elapsed().as_secs_f64(),"results":"LOCAL.jsonl"});
    std::fs::write(out.join("MANIFEST.json"), serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    Ok(())
}
